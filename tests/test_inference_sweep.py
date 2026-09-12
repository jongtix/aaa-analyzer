"""밴드 스윕 + signal_price_bands INSERT + 학습 잡 레이스 방어 테스트
(SPEC-ANALYZER-INFER-001 M6, REQ-AIF-110/111/060, design.md §5/§7,
AC-AIF-018/019/011).

실 DB 접속 없이 `analyzer.data.repository`의 fetch 함수를 모킹한 단위
테스트다 — 모델 파일은 실제 LightGBM/XGBoost 부스터를 임시 디렉토리에
학습해 사용한다(predict.py 테스트와 동일 패턴).
"""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from analyzer.data.models import TradingCalendar
from analyzer.inference.features import LOOKBACK_TRADING_DAYS
from analyzer.inference.resolution import ServingPlan, SkipReason
from analyzer.inference.sweep import (
    DOMESTIC_GRID_RANGE_PCT,
    GRID_STEP_PCT,
    MERGE_THRESHOLD_PCT,
    OVERSEAS_GRID_RANGE_PCT,
    PriceBand,
    assemble_sweep_feature_matrix,
    build_price_grid,
    merge_adjacent_bands,
    sweep_and_write_price_bands,
)
from analyzer.training.campaign_metrics import sidecar_path_for


def _weekdays(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _calendar(start: date, end: date) -> TradingCalendar:
    return TradingCalendar(calendar_code="TEST", trading_days=frozenset(_weekdays(start, end)))


def _ohlcv(stock_code: str, dates: list[date], *, last_close: float = 50_000.0) -> pd.DataFrame:
    n = len(dates)
    df = pd.DataFrame(
        {
            "stock_code": [stock_code] * n,
            "trade_date": dates,
            "open_price": [100.0 + i * 0.1 for i in range(n)],
            "high_price": [101.0 + i * 0.1 for i in range(n)],
            "low_price": [99.0 + i * 0.1 for i in range(n)],
            "close_price": [100.5 + i * 0.1 for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        }
    )
    df.loc[df.index[-1], "close_price"] = last_close
    df.loc[df.index[-1], "open_price"] = last_close
    df.loc[df.index[-1], "high_price"] = last_close * 1.01
    df.loc[df.index[-1], "low_price"] = last_close * 0.99
    return df


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_type",
            "event_date",
            "stock_rate",
            "cash_amount",
            "event_subtype",
            "ex_dividend_date",
            "currency_code",
        ]
    )


def _empty_trend() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "stock_code",
            "trade_date",
            "foreign_net_value",
            "institution_net_value",
            "individual_net_value",
            "total_trading_value",
        ]
    )


def _trend(stock_code: str, dates: list[date]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "stock_code": [stock_code] * n,
            "trade_date": dates,
            "foreign_net_value": [1_000_000 + i * 100 for i in range(n)],
            "institution_net_value": [-500_000 + i * 50 for i in range(n)],
            "individual_net_value": [-500_000 - i * 50 for i in range(n)],
            "total_trading_value": [10_000_000 + i * 1000 for i in range(n)],
        }
    )


class TestBuildPriceGrid:
    """AC-AIF-018: 국내 그리드는 ±30%/0.5% 간격을 사용해야 한다."""

    def test_domestic_grid_matches_worked_example(self):
        grid = build_price_grid(50_000.0, "domestic")

        assert grid.min() == pytest.approx(35_000.0)
        assert grid.max() == pytest.approx(65_000.0)
        assert len(grid) == round(2 * DOMESTIC_GRID_RANGE_PCT / GRID_STEP_PCT) + 1
        step = grid[1] - grid[0]
        assert step == pytest.approx(250.0)

    def test_domestic_grid_is_symmetric_around_prev_close(self):
        grid = build_price_grid(50_000.0, "domestic")

        center = grid[len(grid) // 2]
        assert center == pytest.approx(50_000.0)

    def test_overseas_grid_uses_provisional_range(self):
        grid = build_price_grid(100.0, "overseas")

        assert grid.min() == pytest.approx(100.0 * (1 - OVERSEAS_GRID_RANGE_PCT))
        assert grid.max() == pytest.approx(100.0 * (1 + OVERSEAS_GRID_RANGE_PCT))

    def test_unknown_market_raises_key_error(self):
        with pytest.raises(KeyError):
            build_price_grid(100.0, "unknown")


class TestMergeAdjacentBands:
    """AC-AIF-019 첫 worked example의 병합 규칙 — 연속 동일 등급 그리드
    포인트를 하나의 밴드로 접고, 폭이 임계 미만인 조각은 인접 밴드에
    병합한다(TECHSPEC §6.6)."""

    def test_run_length_collapses_consecutive_same_grade(self):
        grid = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        grades = np.array(["HOLD", "HOLD", "BUY", "BUY", "BUY"])

        bands = merge_adjacent_bands(grid, grades, prev_close=12.0, threshold=0.0)

        assert bands == [
            PriceBand(price_low=10.0, price_high=11.0, signal_class="HOLD"),
            PriceBand(price_low=12.0, price_high=14.0, signal_class="BUY"),
        ]

    def test_single_grade_returns_one_band(self):
        grid = np.array([10.0, 11.0, 12.0])
        grades = np.array(["HOLD", "HOLD", "HOLD"])

        bands = merge_adjacent_bands(grid, grades, prev_close=11.0, threshold=0.0)

        assert bands == [PriceBand(price_low=10.0, price_high=12.0, signal_class="HOLD")]

    def test_narrow_band_merges_into_following_neighbor(self):
        # 100 근방 폭 1(1% 미만)인 고립 조각 — 다음 밴드(BUY)로 흡수돼야 한다.
        grid = np.array([90.0, 99.0, 100.0, 101.0, 110.0])
        grades = np.array(["HOLD", "HOLD", "STRONG_BUY", "BUY", "BUY"])

        bands = merge_adjacent_bands(grid, grades, prev_close=100.0, threshold=0.02)

        assert bands == [
            PriceBand(price_low=90.0, price_high=99.0, signal_class="HOLD"),
            PriceBand(price_low=100.0, price_high=110.0, signal_class="BUY"),
        ]

    def test_narrow_last_band_merges_into_preceding_neighbor(self):
        grid = np.array([90.0, 99.0, 100.0])
        grades = np.array(["HOLD", "HOLD", "STRONG_BUY"])

        bands = merge_adjacent_bands(grid, grades, prev_close=99.0, threshold=0.02)

        assert bands == [PriceBand(price_low=90.0, price_high=100.0, signal_class="HOLD")]

    def test_default_threshold_is_one_percent(self):
        assert MERGE_THRESHOLD_PCT == pytest.approx(0.01)


class TestAssembleSweepFeatureMatrix:
    """AC-AIF-018: PRICE_DERIVED 컬럼은 그리드 가격마다 재계산되고 FROZEN
    컬럼은 전일 실제값으로 동결돼야 한다."""

    def test_returns_grid_matrix_and_prev_close(self):
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        assert len(dates) >= LOOKBACK_TRADING_DAYS
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()
        feature_columns = ["ROC_60", "foreign_net_ratio"]

        with (
            patch(
                "analyzer.inference.sweep.fetch_daily_ohlcv",
                return_value=_ohlcv("A1", dates, last_close=50_000.0),
            ),
            patch("analyzer.inference.sweep.fetch_corporate_events", return_value=_empty_events()),
            patch(
                "analyzer.inference.sweep.fetch_investor_trend",
                return_value=_trend("A1", dates),
            ),
        ):
            result = assemble_sweep_feature_matrix(
                engine, calendar, "A1", as_of_date, "domestic", feature_columns
            )

        assert not isinstance(result, SkipReason)
        grid, matrix, prev_close = result
        assert prev_close == pytest.approx(50_000.0)
        assert len(matrix) == len(grid)
        assert list(matrix.columns) == feature_columns

    def test_price_derived_column_varies_across_grid(self):
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()
        feature_columns = ["KMID"]

        with (
            patch(
                "analyzer.inference.sweep.fetch_daily_ohlcv",
                return_value=_ohlcv("A1", dates, last_close=50_000.0),
            ),
            patch("analyzer.inference.sweep.fetch_corporate_events", return_value=_empty_events()),
            patch("analyzer.inference.sweep.fetch_investor_trend", return_value=_empty_trend()),
        ):
            result = assemble_sweep_feature_matrix(
                engine, calendar, "A1", as_of_date, "domestic", feature_columns
            )

        assert not isinstance(result, SkipReason)
        _grid, matrix, _prev_close = result
        assert matrix["KMID"].nunique() > 1

    def test_frozen_column_is_constant_across_grid(self):
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()
        feature_columns = ["ROC_60", "foreign_net_ratio"]

        with (
            patch(
                "analyzer.inference.sweep.fetch_daily_ohlcv",
                return_value=_ohlcv("A1", dates, last_close=50_000.0),
            ),
            patch("analyzer.inference.sweep.fetch_corporate_events", return_value=_empty_events()),
            patch(
                "analyzer.inference.sweep.fetch_investor_trend",
                return_value=_trend("A1", dates),
            ),
        ):
            result = assemble_sweep_feature_matrix(
                engine, calendar, "A1", as_of_date, "domestic", feature_columns
            )

        assert not isinstance(result, SkipReason)
        _grid, matrix, _prev_close = result
        assert matrix["foreign_net_ratio"].nunique() == 1

    def test_insufficient_history_returns_feature_insufficient(self):
        dates = _weekdays(date(2026, 3, 1), date(2026, 3, 20))
        assert len(dates) < LOOKBACK_TRADING_DAYS
        as_of_date = dates[-1]
        calendar = _calendar(date(2026, 1, 1), date(2026, 4, 10))
        engine = MagicMock()

        with patch(
            "analyzer.inference.sweep.fetch_daily_ohlcv",
            return_value=_ohlcv("A1", dates),
        ):
            result = assemble_sweep_feature_matrix(
                engine, calendar, "A1", as_of_date, "domestic", ["ROC_60"]
            )

        assert result is SkipReason.FEATURE_INSUFFICIENT

    def test_frozen_column_required_but_trend_empty_returns_feature_insufficient(self):
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()

        with (
            patch(
                "analyzer.inference.sweep.fetch_daily_ohlcv",
                return_value=_ohlcv("A1", dates),
            ),
            patch("analyzer.inference.sweep.fetch_corporate_events", return_value=_empty_events()),
            patch("analyzer.inference.sweep.fetch_investor_trend", return_value=_empty_trend()),
        ):
            result = assemble_sweep_feature_matrix(
                engine, calendar, "A1", as_of_date, "domestic", ["foreign_net_ratio"]
            )

        assert result is SkipReason.FEATURE_INSUFFICIENT


def _write_sidecar(model_path: Path, feature_columns: list[str]) -> None:
    sidecar_path = sidecar_path_for(model_path)
    sidecar_path.write_text(json.dumps({"feature_columns": feature_columns}), encoding="utf-8")


def _train_lightgbm(tmp_path: Path, name: str, feature_columns: list[str], seed: int) -> Path:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(60, len(feature_columns))), columns=feature_columns)
    y = x.sum(axis=1).to_numpy() * 0.01 + rng.normal(scale=0.001, size=60)
    model = lgb.LGBMRegressor(n_estimators=10, verbosity=-1)
    model.fit(x, y)
    model_path = tmp_path / f"{name}.txt"
    model.booster_.save_model(str(model_path))
    _write_sidecar(model_path, feature_columns)
    return model_path


def _train_xgboost(tmp_path: Path, name: str, feature_columns: list[str], seed: int) -> Path:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(60, len(feature_columns))), columns=feature_columns)
    y = x.sum(axis=1).to_numpy() * 0.01 + rng.normal(scale=0.001, size=60)
    model = xgb.XGBRegressor(n_estimators=10, verbosity=0)
    model.fit(x, y)
    model_path = tmp_path / f"{name}.json"
    model.get_booster().save_model(str(model_path))
    _write_sidecar(model_path, feature_columns)
    return model_path


class TestSweepAndWritePriceBands:
    """AC-AIF-011 세 번째 시나리오 + design.md §7: 모델 로드 직후 매니페스트
    재확인으로 학습 잡 레이스를 감지하고, 감지되면 밴드를 기록하지 않는다."""

    def test_manifest_race_detected_skips_without_writing(self, tmp_path: Path):
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()
        feature_columns = ["ROC_60"]
        model_path = _train_xgboost(tmp_path, "xgb_race", feature_columns, seed=30)
        plan = ServingPlan(
            market="domestic",
            horizon=20,
            active_strategy="xgboost",
            algorithms=("xgboost",),
            manifests={},
            model_paths={"xgboost": model_path},
        )

        with (
            patch(
                "analyzer.inference.sweep.fetch_daily_ohlcv",
                return_value=_ohlcv("A1", dates),
            ),
            patch("analyzer.inference.sweep.fetch_corporate_events", return_value=_empty_events()),
            patch("analyzer.inference.sweep.fetch_investor_trend", return_value=_empty_trend()),
            patch("analyzer.inference.sweep.detect_manifest_race", return_value=True),
            patch("analyzer.inference.sweep.insert_signal_price_bands") as mock_insert,
        ):
            result = sweep_and_write_price_bands(
                engine=engine,
                calendar=calendar,
                boundaries_artifact=MagicMock(),
                serving_plan=plan,
                models_root=tmp_path,
                stock_id=1,
                stock_code="A1",
                market="domestic",
                horizon=20,
                trade_date=as_of_date,
                model_version="domestic_20_xgboost_2026-08-19",
            )

        assert result is SkipReason.MANIFEST_RACE
        mock_insert.assert_not_called()

    def test_successful_sweep_writes_promote_and_demote(self, tmp_path: Path):
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()
        feature_columns = ["ROC_60"]
        model_path = _train_xgboost(tmp_path, "xgb_sweep", feature_columns, seed=31)
        plan = ServingPlan(
            market="domestic",
            horizon=20,
            active_strategy="xgboost",
            algorithms=("xgboost",),
            manifests={},
            model_paths={"xgboost": model_path},
        )
        boundaries_artifact = MagicMock()
        boundaries_artifact.boundaries_for.return_value = {
            "STRONG_SELL_SELL": -0.10,
            "SELL_HOLD": -0.03,
            "HOLD_BUY": 0.03,
            "BUY_STRONG_BUY": 0.10,
        }

        with (
            patch(
                "analyzer.inference.sweep.fetch_daily_ohlcv",
                return_value=_ohlcv("A1", dates),
            ),
            patch("analyzer.inference.sweep.fetch_corporate_events", return_value=_empty_events()),
            patch("analyzer.inference.sweep.fetch_investor_trend", return_value=_empty_trend()),
            patch("analyzer.inference.sweep.detect_manifest_race", return_value=False),
            patch(
                "analyzer.inference.sweep.insert_signal_price_bands",
                return_value="inserted",
            ) as mock_insert,
        ):
            result = sweep_and_write_price_bands(
                engine=engine,
                calendar=calendar,
                boundaries_artifact=boundaries_artifact,
                serving_plan=plan,
                models_root=tmp_path,
                stock_id=1,
                stock_code="A1",
                market="domestic",
                horizon=20,
                trade_date=as_of_date,
                model_version="domestic_20_xgboost_2026-08-19",
            )

        assert result == {"PROMOTE": "inserted", "DEMOTE": "inserted"}
        assert mock_insert.call_count == 2
        boundary_sets_called = {call.args[1][0].boundary_set for call in mock_insert.call_args_list}
        assert boundary_sets_called == {"PROMOTE", "DEMOTE"}

    def test_insufficient_history_returns_feature_insufficient_before_predicting(
        self, tmp_path: Path
    ):
        dates = _weekdays(date(2026, 3, 1), date(2026, 3, 20))
        as_of_date = dates[-1]
        calendar = _calendar(date(2026, 1, 1), date(2026, 4, 10))
        engine = MagicMock()
        model_path = _train_xgboost(tmp_path, "xgb_insufficient", ["ROC_60"], seed=32)
        plan = ServingPlan(
            market="domestic",
            horizon=20,
            active_strategy="xgboost",
            algorithms=("xgboost",),
            manifests={},
            model_paths={"xgboost": model_path},
        )

        with patch(
            "analyzer.inference.sweep.fetch_daily_ohlcv",
            return_value=_ohlcv("A1", dates),
        ):
            result = sweep_and_write_price_bands(
                engine=engine,
                calendar=calendar,
                boundaries_artifact=MagicMock(),
                serving_plan=plan,
                models_root=tmp_path,
                stock_id=1,
                stock_code="A1",
                market="domestic",
                horizon=20,
                trade_date=as_of_date,
                model_version="domestic_20_xgboost_2026-08-19",
            )

        assert result is SkipReason.FEATURE_INSUFFICIENT

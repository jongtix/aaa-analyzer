"""밴드 스윕 + signal_price_bands INSERT + 학습 잡 레이스 방어 테스트
(SPEC-ANALYZER-INFER-001 M6, REQ-AIF-110/111/060, design.md §5/§7,
AC-AIF-018/019/011) + 격자 칸 안 경계 정련·연속 파티션
(SPEC-ANALYZER-INFER-002, AC-AIR-001~010).

실 DB 접속 없이 `analyzer.data.repository`의 fetch 함수를 모킹한 단위
테스트다 — 모델 파일은 실제 LightGBM/XGBoost 부스터를 임시 디렉토리에
학습해 사용한다(predict.py 테스트와 동일 패턴). 정련 알고리즘 테스트는
"가격 → score"를 결정적으로 반환하는 대역 평가 함수(호출 기록)를 쓴다.
"""

import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from analyzer.data.models import TradingCalendar
from analyzer.inference import sweep as sweep_module
from analyzer.inference.boundaries_store import (
    BoundarySet,
    classify_grades,
    load_grade_boundaries,
    shift_boundaries,
)
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

    def test_overseas_grid_uses_confirmed_range(self):
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

    def test_grid_and_merge_constants_are_unchanged(self):
        # AC-AIR-010: 정련은 격자 범위·간격·병합 임계를 바꾸지 않는다(REQ-AIR-009).
        assert (
            DOMESTIC_GRID_RANGE_PCT,
            OVERSEAS_GRID_RANGE_PCT,
            GRID_STEP_PCT,
            MERGE_THRESHOLD_PCT,
        ) == (0.30, 0.215, 0.005, 0.01)

    @pytest.mark.parametrize("seed", range(20))
    def test_non_last_band_ends_on_its_own_grade_and_boundary_cell_ends_differ(self, seed: int):
        # plan.md §B — 정련 탐색의 시작 조건(아래 끝 참·위 끝 거짓)을 보장하는 병합 불변식.
        # Arrange
        rng = np.random.default_rng(seed)
        grid = build_price_grid(10_000.0, "domestic")
        grades = rng.choice(["SELL", "HOLD", "BUY"], size=len(grid), p=[0.15, 0.7, 0.15])

        # Act
        bands = merge_adjacent_bands(grid, grades, prev_close=10_000.0)

        # Assert
        index_of = {float(p): i for i, p in enumerate(grid)}
        for prev, nxt in zip(bands, bands[1:], strict=False):
            last_i = index_of[prev.price_high]
            first_i = index_of[nxt.price_low]
            assert first_i == last_i + 1
            assert grades[last_i] == prev.signal_class
            assert grades[first_i] != grades[last_i]


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
        # AC-AIR-001: 기록된 파티션은 구멍 없이 연속이고 격자 양 끝을 덮는다.
        grid = build_price_grid(50_000.0, "domestic")
        for call in mock_insert.call_args_list:
            rows = call.args[1]
            assert rows[0].price_low == grid[0]
            assert rows[-1].price_high == grid[-1]
            for prev, nxt in zip(rows, rows[1:], strict=False):
                assert prev.price_high == nxt.price_low
            assert all(row.price_low < row.price_high for row in rows)

    def _roc_scored_setup(
        self, tmp_path: Path
    ) -> tuple[list[date], TradingCalendar, ServingPlan, np.ndarray, pd.DataFrame]:
        """ROC_60(가격에 단조 증가)을 그대로 score로 쓰는 대역 예측 픽스처 — 격자 칸
        안에서 등급이 실제로 바뀌는 상황을 실 피처 조립 경로 위에서 재현한다."""
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        model_path = _train_xgboost(tmp_path, "xgb_roc", ["ROC_60"], seed=33)
        plan = ServingPlan(
            market="domestic",
            horizon=60,
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
        ):
            assembled = assemble_sweep_feature_matrix(
                MagicMock(), calendar, "A1", dates[-1], "domestic", ["ROC_60"]
            )
        assert not isinstance(assembled, SkipReason)
        grid, matrix, _prev_close = assembled
        return dates, calendar, plan, grid, matrix

    def _run_roc_sweep(
        self,
        tmp_path: Path,
        *,
        race: bool,
        events: list[str],
    ) -> tuple[dict[str, str] | SkipReason, MagicMock, np.ndarray]:
        dates, calendar, plan, grid, matrix = self._roc_scored_setup(tmp_path)
        # 칸 하단에서 30% 지점(이분 탐색 중간점과 겹치지 않는 위치)에서 HOLD → BUY로 바뀐다.
        roc_low, roc_high = matrix["ROC_60"].iloc[60], matrix["ROC_60"].iloc[61]
        threshold = float(roc_low + 0.3 * (roc_high - roc_low))
        boundaries_artifact = MagicMock()
        boundaries_artifact.boundaries_for.return_value = {
            "STRONG_SELL_SELL": -1e9,
            "SELL_HOLD": -1e8,
            "HOLD_BUY": threshold,
            "BUY_STRONG_BUY": 1e9,
        }

        def fake_predict(_plan: ServingPlan, feature_matrix: pd.DataFrame) -> dict[str, np.ndarray]:
            events.append("predict")
            return {"xgboost": feature_matrix["ROC_60"].to_numpy(dtype=float)}

        def fake_race(_root: Path, _plan: ServingPlan) -> bool:
            events.append("race")
            return race

        with (
            patch(
                "analyzer.inference.sweep.fetch_daily_ohlcv",
                return_value=_ohlcv("A1", dates),
            ),
            patch("analyzer.inference.sweep.fetch_corporate_events", return_value=_empty_events()),
            patch("analyzer.inference.sweep.fetch_investor_trend", return_value=_empty_trend()),
            patch("analyzer.inference.sweep.predict_point_models_batch", side_effect=fake_predict),
            patch("analyzer.inference.sweep.detect_manifest_race", side_effect=fake_race),
            patch(
                "analyzer.inference.sweep.insert_signal_price_bands",
                return_value="inserted",
            ) as mock_insert,
        ):
            result = sweep_and_write_price_bands(
                engine=MagicMock(),
                calendar=calendar,
                boundaries_artifact=boundaries_artifact,
                serving_plan=plan,
                models_root=tmp_path,
                stock_id=1,
                stock_code="A1",
                market="domestic",
                horizon=60,
                trade_date=dates[-1],
                model_version="domestic_60_xgboost_2026-08-19",
            )
        return result, mock_insert, grid

    def test_written_boundary_is_refined_inside_grid_cell(self, tmp_path: Path):
        # AC-AIR-001/003: 내부 경계는 격자점이 아니라 경계 칸 안의 정련 가격이며 두 밴드가 공유한다.
        events: list[str] = []

        result, mock_insert, grid = self._run_roc_sweep(tmp_path, race=False, events=events)

        assert result == {"PROMOTE": "inserted", "DEMOTE": "inserted"}
        cell = grid[61] - grid[60]
        crossing = grid[60] + 0.3 * cell
        for call in mock_insert.call_args_list:
            rows = call.args[1]
            assert [row.signal_class for row in rows] == ["HOLD", "BUY"]
            boundary = rows[0].price_high
            assert boundary == rows[1].price_low
            assert grid[60] < boundary < grid[61]
            assert crossing <= boundary <= crossing + cell / 256 + 0.0001

    def test_manifest_race_rechecked_after_last_refinement_evaluation(self, tmp_path: Path):
        # AC-AIR-008: 레이스 재확인은 정련 평가(마지막 모델 로드) 뒤에 하고, 감지되면 기록 0건.
        events: list[str] = []

        result, mock_insert, _grid = self._run_roc_sweep(tmp_path, race=True, events=events)

        assert result is SkipReason.MANIFEST_RACE
        mock_insert.assert_not_called()
        assert events.count("predict") > 1
        assert events.count("race") == 1
        assert events[-1] == "race"

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


class _RecordingEvaluator:
    """가격 → score 대역 평가 함수 — 호출마다 평가 가격과 score를 기록한다
    (acceptance.md 공통 전제). 첫 호출은 격자 평가, 이후 호출은 정련 평가다."""

    def __init__(self, curve: Callable[[float], float]) -> None:
        self._curve = curve
        self.calls: list[np.ndarray] = []
        self.records: list[tuple[float, float]] = []

    def __call__(self, prices: np.ndarray) -> np.ndarray:
        evaluated = np.asarray(prices, dtype=float).copy()
        self.calls.append(evaluated)
        scores = np.array([self._curve(float(p)) for p in evaluated], dtype=float)
        self.records.extend(zip(evaluated.tolist(), scores.tolist(), strict=True))
        return scores

    @property
    def refinement_prices(self) -> np.ndarray:
        if len(self.calls) <= 1:
            return np.array([], dtype=float)
        return np.concatenate(self.calls[1:])


def _notifier_lookup(bands: list[PriceBand], price: float) -> tuple[int, bool]:
    """notifier `BandPartition.lookup` 규칙 이식 — "하한이 가격 이하인 마지막 밴드",
    첫 하한 미만·마지막 상한 초과는 클램프, 마지막 상한과 같은 가격은 클램프 아님."""
    if price < bands[0].price_low:
        return 0, True
    if price > bands[-1].price_high:
        return len(bands) - 1, True
    index = max(i for i, band in enumerate(bands) if band.price_low <= price)
    return index, False


_SYNTHETIC_BASE_BOUNDARIES = {
    "STRONG_SELL_SELL": -0.2,
    "SELL_HOLD": -0.1,
    "HOLD_BUY": 0.1,
    "BUY_STRONG_BUY": 0.2,
}
_PARTITION_SETS = (BoundarySet.PROMOTE, BoundarySet.DEMOTE)


def _synthetic_boundaries() -> dict[BoundarySet, dict[str, float]]:
    return {
        boundary_set: shift_boundaries(
            _SYNTHETIC_BASE_BOUNDARIES, delta=0.01, boundary_set=boundary_set
        )
        for boundary_set in _PARTITION_SETS
    }


def _real_boundaries(market: str, horizon: int) -> dict[BoundarySet, dict[str, float]]:
    artifact = load_grade_boundaries()
    return {
        boundary_set: artifact.boundaries_for(market, horizon, boundary_set)
        for boundary_set in _PARTITION_SETS
    }


def _sweep(
    grid: np.ndarray,
    boundaries_by_set: dict[BoundarySet, dict[str, float]],
    prev_close: float,
    curve: Callable[[float], float],
) -> tuple[dict[BoundarySet, list[PriceBand]], _RecordingEvaluator]:
    evaluator = _RecordingEvaluator(curve)
    partitions = sweep_module.sweep_price_partitions(
        grid, boundaries_by_set, prev_close=prev_close, evaluate=evaluator
    )
    return partitions, evaluator


def _grade(score: float, boundaries: dict[str, float]) -> str:
    return str(classify_grades([score], boundaries)[0])


def _internal_boundaries(bands: list[PriceBand]) -> list[float]:
    return [band.price_high for band in bands[:-1]]


class TestContinuousPartition:
    """AC-AIR-001/002: 인접 밴드는 경계 가격을 공유하고, 반개구간 해석이 notifier 조회와 같다."""

    @staticmethod
    def _three_transition_curve(prev_close: float, range_pct: float) -> Callable[[float], float]:
        # -0.15 → +0.25 선형 — SELL | HOLD | BUY | STRONG_BUY (병합 후 밴드 4개)
        def curve(price: float) -> float:
            return -0.15 + 0.4 * ((price / prev_close - 1) / range_pct + 1) / 2

        return curve

    @pytest.mark.parametrize(
        ("market", "range_pct"),
        [("domestic", DOMESTIC_GRID_RANGE_PCT), ("overseas", OVERSEAS_GRID_RANGE_PCT)],
    )
    def test_adjacent_bands_share_boundary_and_cover_whole_grid(
        self, market: str, range_pct: float
    ):
        prev_close = 10_000.0
        grid = build_price_grid(prev_close, market)
        curve = self._three_transition_curve(prev_close, range_pct)

        partitions, _ = _sweep(grid, _synthetic_boundaries(), prev_close, curve)

        assert set(partitions) == set(_PARTITION_SETS)
        for bands in partitions.values():
            assert [band.signal_class for band in bands] == ["SELL", "HOLD", "BUY", "STRONG_BUY"]
            assert bands[0].price_low == grid[0]
            assert bands[-1].price_high == grid[-1]
            for prev, nxt in zip(bands, bands[1:], strict=False):
                assert prev.price_high == nxt.price_low
            assert all(band.price_low < band.price_high for band in bands)

    def test_half_open_reading_matches_notifier_lookup(self):
        # Arrange
        prev_close = 10_000.0
        grid = build_price_grid(prev_close, "domestic")
        curve = self._three_transition_curve(prev_close, DOMESTIC_GRID_RANGE_PCT)

        # Act
        partitions, _ = _sweep(grid, _synthetic_boundaries(), prev_close, curve)

        # Assert
        for bands in partitions.values():
            for i, boundary in enumerate(_internal_boundaries(bands)):
                assert _notifier_lookup(bands, boundary) == (i + 1, False)
                assert _notifier_lookup(bands, round(boundary - 0.0001, 4)) == (i, False)
            assert _notifier_lookup(bands, float(grid[-1])) == (len(bands) - 1, False)
            assert _notifier_lookup(bands, float(grid[0]) - 1.0) == (0, True)
            assert _notifier_lookup(bands, float(grid[-1]) + 1.0) == (len(bands) - 1, True)
            last = len(bands) - 1
            for index, band in enumerate(bands):
                inside = (band.price_low + band.price_high) / 2
                owners = [
                    j
                    for j, other in enumerate(bands)
                    if other.price_low <= inside
                    and (inside < other.price_high or (j == last and inside <= other.price_high))
                ]
                assert owners == [index]
                assert _notifier_lookup(bands, inside) == (index, False)


class TestCliffRefinement:
    """AC-AIR-003(004020형 절벽 재현)과 AC-AIR-009 (나)(폭 0 경계 그대로 저장)."""

    PREV_CLOSE = 30_000.0
    CLIFF = 30_003.0
    SCORE_BELOW = 0.18538
    SCORE_ABOVE = 0.08669

    def _curve(self, price: float) -> float:
        return self.SCORE_BELOW if price < self.CLIFF else self.SCORE_ABOVE

    def test_premise_cliff_scores_are_buy_and_hold_in_both_sets(self):
        for boundaries in _real_boundaries("domestic", 60).values():
            assert _grade(self.SCORE_BELOW, boundaries) == "BUY"
            assert _grade(self.SCORE_ABOVE, boundaries) == "HOLD"

    def test_cliff_is_located_within_eight_bisection_bound(self):
        grid = build_price_grid(self.PREV_CLOSE, "domestic")
        cell = self.PREV_CLOSE * GRID_STEP_PCT

        partitions, _ = _sweep(grid, _real_boundaries("domestic", 60), self.PREV_CLOSE, self._curve)

        for bands in partitions.values():
            assert [band.signal_class for band in bands] == ["BUY", "HOLD"]
            boundary = bands[0].price_high
            assert self.CLIFF <= boundary <= self.CLIFF + cell / 256
            assert boundary == round(boundary, 4)
            for stale_price in (30_100.0, 30_150.0):
                index, clamped = _notifier_lookup(bands, stale_price)
                assert bands[index].signal_class == "HOLD"
                assert not clamped

    def test_zero_width_promote_demote_pair_is_stored_without_minimum_width(self):
        grid = build_price_grid(self.PREV_CLOSE, "domestic")

        partitions, _ = _sweep(grid, _real_boundaries("domestic", 60), self.PREV_CLOSE, self._curve)

        promote = partitions[BoundarySet.PROMOTE][0].price_high
        demote = partitions[BoundarySet.DEMOTE][0].price_high
        assert promote == demote


class TestBoundaryPriceMeaning:
    """AC-AIR-004: 경계 가격은 평가된 "앞 밴드 등급이 아닌" 가격이고, 바로 아래
    평가 가격은 앞 밴드 등급이며 둘 사이는 격자 칸/256 이내다."""

    @pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9])
    def test_boundary_is_evaluated_price_just_past_prior_grade(self, fraction: float):
        # Arrange
        prev_close = 10_000.0
        grid = build_price_grid(prev_close, "domestic")
        cell = prev_close * GRID_STEP_PCT
        switch = prev_close + fraction * cell
        boundaries = _synthetic_boundaries()

        # Act
        partitions, evaluator = _sweep(
            grid, boundaries, prev_close, lambda price: 0.0 if price < switch else 0.15
        )

        # Assert
        for boundary_set, bands in partitions.items():
            assert [band.signal_class for band in bands] == ["HOLD", "BUY"]
            prior = bands[0].signal_class
            boundary = bands[0].price_high
            graded = {
                price: _grade(score, boundaries[boundary_set]) for price, score in evaluator.records
            }
            assert boundary in graded
            assert graded[boundary] != prior
            below = [p for p, grade in graded.items() if grade == prior and p < boundary]
            assert below
            assert 0 < boundary - max(below) <= cell / 256 + 0.0001


class TestMergeThenRefine:
    """AC-AIR-005(PD-1): 1% 조각 병합 뒤 살아남은 경계만 정련하며, 밴드 개수·등급 순서는
    병합만 한 결과와 같고, 병합으로 사라진 조각 경계는 평가하지 않는다."""

    PREV_CLOSE = 10_000.0
    # 국내 격자(7,000~13,000원, 50원 간격): 8,500 = BUY 1점 조각, 10,000·10,050 =
    # STRONG_BUY 2점 조각, 11,500 = SELL 1점 조각 — 병합 전 run-length 7개 → 병합 후 4개.
    EATEN_CELLS = ((8_500.0, 8_550.0), (10_050.0, 10_100.0), (11_500.0, 11_550.0))

    @staticmethod
    def _curve(price: float) -> float:
        if 8_490.0 <= price < 8_510.0:
            return 0.15
        if 9_980.0 <= price < 10_075.0:
            return 0.25
        if 10_075.0 <= price < 11_480.0:
            return 0.15
        if 11_480.0 <= price < 11_520.0:
            return -0.15
        return 0.0

    def test_band_structure_equals_merge_only_result_with_boundaries_inside_cells(self):
        # Arrange
        grid = build_price_grid(self.PREV_CLOSE, "domestic")
        boundaries = _synthetic_boundaries()
        grid_scores = np.array([self._curve(float(price)) for price in grid])

        # Act
        partitions, _ = _sweep(grid, boundaries, self.PREV_CLOSE, self._curve)

        # Assert
        for boundary_set, bands in partitions.items():
            grades = classify_grades(grid_scores, boundaries[boundary_set])
            assert int(np.count_nonzero(grades[1:] != grades[:-1])) + 1 == 7
            merged = merge_adjacent_bands(grid, grades, prev_close=self.PREV_CLOSE)
            assert len(merged) == 4
            assert [band.signal_class for band in bands] == [b.signal_class for b in merged]
            for i, boundary in enumerate(_internal_boundaries(bands)):
                assert merged[i].price_high < boundary <= merged[i + 1].price_low

    def test_boundaries_eaten_by_merge_are_never_evaluated(self):
        grid = build_price_grid(self.PREV_CLOSE, "domestic")

        _, evaluator = _sweep(grid, _synthetic_boundaries(), self.PREV_CLOSE, self._curve)

        refinement = evaluator.refinement_prices
        assert refinement.size > 0
        for low, high in self.EATEN_CELLS:
            assert not np.any((refinement > low) & (refinement < high))


class TestRefinementBudget:
    """AC-AIR-006 + PD-2(최대 8회·양자화 소진 조기 종료) + PD-3(반복 단위 일괄 평가)."""

    def test_each_boundary_evaluates_at_most_eight_prices(self):
        # (가) 전환이 칸 상단 바로 아래 — 매 반복 조건이 참이라 상한 8회를 모두 쓴다.
        prev_close = 10_000.0
        grid = build_price_grid(prev_close, "domestic")
        switch = 10_049.99

        partitions, evaluator = _sweep(
            grid,
            _synthetic_boundaries(),
            prev_close,
            lambda price: 0.0 if price < switch else 0.15,
        )

        refinement = evaluator.refinement_prices
        assert sweep_module.REFINE_MAX_ITERATIONS == 8
        assert len(refinement) == 8
        assert np.all((refinement > grid[60]) & (refinement < grid[61]))
        for bands in partitions.values():
            assert bands[0].price_high == grid[61]

    def test_single_grade_curve_skips_refinement_and_stores_one_band(self):
        # (나) 전 구간 단일 등급 — 정련 평가 0건, 밴드 1개.
        prev_close = 10_000.0
        grid = build_price_grid(prev_close, "domestic")

        partitions, evaluator = _sweep(grid, _synthetic_boundaries(), prev_close, lambda _: 0.0)

        assert len(evaluator.calls) == 1
        for bands in partitions.values():
            assert bands == [
                PriceBand(price_low=float(grid[0]), price_high=float(grid[-1]), signal_class="HOLD")
            ]

    def test_sets_with_and_without_internal_boundary_are_refined_independently(self):
        # §D.1: PROMOTE는 밴드 1개(HOLD), DEMOTE는 HOLD|BUY 2개 — 세트별로 독립 처리한다.
        prev_close = 10_000.0
        grid = build_price_grid(prev_close, "domestic")
        switch = prev_close + 0.3 * prev_close * GRID_STEP_PCT

        partitions, evaluator = _sweep(
            grid,
            _synthetic_boundaries(),
            prev_close,
            lambda price: 0.085 if price < switch else 0.105,
        )

        assert partitions[BoundarySet.PROMOTE] == [
            PriceBand(price_low=float(grid[0]), price_high=float(grid[-1]), signal_class="HOLD")
        ]
        demote = partitions[BoundarySet.DEMOTE]
        assert [band.signal_class for band in demote] == ["HOLD", "BUY"]
        assert grid[60] < demote[0].price_high == demote[1].price_low <= grid[61]
        assert 0 < len(evaluator.refinement_prices) <= sweep_module.REFINE_MAX_ITERATIONS

    def test_quantization_exhaustion_stops_before_iteration_cap(self):
        # §D.1: 전일 종가 1원 — 격자 칸(0.005원)이 0.0001 × 2⁸보다 작아 양자화 격자가 먼저 소진된다.
        prev_close = 1.0
        grid = build_price_grid(prev_close, "domestic")
        switch = 1.0049

        partitions, evaluator = _sweep(
            grid,
            _synthetic_boundaries(),
            prev_close,
            lambda price: 0.0 if price < switch else 0.15,
        )

        refinement = evaluator.refinement_prices
        assert 0 < len(refinement) < sweep_module.REFINE_MAX_ITERATIONS
        assert all(price == round(price, 4) for price in refinement)
        for bands in partitions.values():
            assert switch <= bands[0].price_high <= grid[61]

    def test_refinement_is_batched_per_round_across_boundaries_and_sets(self):
        # PD-3: 경계 세트·경계를 묶어 반복마다 평가 1회 — 경계×반복만큼 평가가 늘지 않는다.
        grid = build_price_grid(TestMergeThenRefine.PREV_CLOSE, "domestic")

        _, evaluator = _sweep(
            grid,
            _synthetic_boundaries(),
            TestMergeThenRefine.PREV_CLOSE,
            TestMergeThenRefine._curve,
        )

        refinement_calls = evaluator.calls[1:]
        assert 0 < len(refinement_calls) <= sweep_module.REFINE_MAX_ITERATIONS
        assert any(len(call) > 1 for call in refinement_calls)
        for call in refinement_calls:
            assert len(np.unique(call)) == len(call)


class TestPriceAxisDeadZone:
    """AC-AIR-009 (가): 완만한 단조 곡선에서는 PROMOTE·DEMOTE 경계 사이 가격 폭이
    해석적 기대 폭(경계 ± δ를 곡선 역함수로 환산)과 2 × 격자 칸/256 이내로 일치한다."""

    def test_gentle_monotonic_curve_keeps_analytic_dead_zone_width(self):
        # Arrange — 칸당 Δscore = 0.75 × 0.5% = 0.00375 < δ(≈0.01)
        prev_close = 10_000.0
        slope = 0.75
        grid = build_price_grid(prev_close, "domestic")
        boundaries = _real_boundaries("domestic", 60)
        cell = prev_close * GRID_STEP_PCT

        # Act
        partitions, _ = _sweep(
            grid, boundaries, prev_close, lambda price: slope * (price / prev_close - 1)
        )

        # Assert
        def hold_buy_boundary(bands: list[PriceBand]) -> float:
            for prev, nxt in zip(bands, bands[1:], strict=False):
                if (prev.signal_class, nxt.signal_class) == ("HOLD", "BUY"):
                    return prev.price_high
            raise AssertionError("HOLD|BUY 경계가 없다")

        c_promote = hold_buy_boundary(partitions[BoundarySet.PROMOTE])
        c_demote = hold_buy_boundary(partitions[BoundarySet.DEMOTE])
        expected = (
            (
                boundaries[BoundarySet.PROMOTE]["HOLD_BUY"]
                - boundaries[BoundarySet.DEMOTE]["HOLD_BUY"]
            )
            / slope
            * prev_close
        )
        assert c_promote - c_demote > 0
        assert abs((c_promote - c_demote) - expected) <= 2 * cell / 256


class TestSharedEvaluationPath:
    """AC-AIR-007(REQ-AIR-006): 정련 평가 경로로 격자점 가격을 평가하면 격자 경로와
    피처 행·score가 비트 단위로 같다."""

    def test_refinement_path_reproduces_grid_features_and_scores_bitwise(self, tmp_path: Path):
        # Arrange
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        feature_columns = ["ROC_60", "KMID", "foreign_net_ratio"]
        model_path = _train_xgboost(tmp_path, "xgb_shared", feature_columns, seed=34)
        plan = ServingPlan(
            market="domestic",
            horizon=60,
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
            patch(
                "analyzer.inference.sweep.fetch_investor_trend",
                return_value=_trend("A1", dates),
            ),
        ):
            context = sweep_module.prepare_sweep_context(
                MagicMock(), calendar, "A1", dates[-1], "domestic", feature_columns
            )
            assembled = assemble_sweep_feature_matrix(
                MagicMock(), calendar, "A1", dates[-1], "domestic", feature_columns
            )
        assert not isinstance(context, SkipReason)
        assert not isinstance(assembled, SkipReason)
        grid, grid_matrix, _prev_close = assembled
        picks = [0, len(grid) // 2, len(grid) - 1]

        # Act
        refine_matrix = sweep_module.feature_matrix_at_prices(context, grid[picks])

        # Assert
        pd.testing.assert_frame_equal(
            refine_matrix, grid_matrix.iloc[picks].reset_index(drop=True), check_exact=True
        )
        np.testing.assert_array_equal(
            sweep_module.score_prices(plan, refine_matrix),
            sweep_module.score_prices(plan, grid_matrix)[picks],
        )

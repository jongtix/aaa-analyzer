"""src/analyzer/training/campaign.py 캠페인 오케스트레이션 골격 테스트.

SPEC-ANALYZER-TRAIN-EVAL-001 M3. 합성 축소 데이터만 사용한다 —
실 DB 접속·라이브 캠페인 실행은 수행하지 않는다(F15, plan.md §F M3
순서 의존성 경고). 검증 대상:
1. 시장당 데이터셋 조립 호출 횟수 정확히 1회(REQ-ATE-022, 모킹).
2. 1회 튜닝이 초기 학습 이력 구간만 사용(평가 구간 데이터 미노출, REQ-ATE-027).
3. 동결 하이퍼파라미터가 모든 폴드 학습 호출에 동일하게 전달(REQ-ATE-028).
4. `purged_walk_forward_split()`가 실제로 호출됨(REQ-ATE-027, 모킹 확인).
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from analyzer.training import campaign as campaign_module
from analyzer.training.campaign import (
    CAMPAIGN_OOS_EVALUATION_SPAN_YEARS,
    CAMPAIGN_OPTUNA_TRIALS,
    CAMPAIGN_TIMEOUT_HOURS,
    CAMPAIGN_VAL_SIZE_TRADING_DAYS,
    POINT_ALGORITHMS,
    POINT_COMBOS,
    _initial_train_end_index,
    main,
    resolve_tuning_split,
    run_campaign_for_market_horizon,
    run_walk_forward_campaign,
)
from analyzer.training.models import HORIZONS, MARKETS


def _make_synthetic_panel(n_dates: int, n_stocks: int = 1, seed: int = 42) -> pd.DataFrame:
    """합성 단일/다종목 패널 — `KMID` 피처 1개 + `label_D20`/`label_D60` 레이블."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n_dates)

    frames = []
    for stock_index in range(n_stocks):
        stock_code = f"S{stock_index}"
        frames.append(
            pd.DataFrame(
                {
                    "stock_code": stock_code,
                    "trade_date": dates,
                    "KMID": rng.normal(size=n_dates),
                    "label_D20": rng.normal(scale=0.02, size=n_dates),
                    "label_D60": rng.normal(scale=0.03, size=n_dates),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestConstantsAndPointCombos:
    def test_named_constants_match_spec_initial_values(self):
        assert CAMPAIGN_OOS_EVALUATION_SPAN_YEARS == 10
        assert CAMPAIGN_TIMEOUT_HOURS == 12.0
        assert CAMPAIGN_OPTUNA_TRIALS == 50
        assert CAMPAIGN_VAL_SIZE_TRADING_DAYS == 5

    def test_point_combos_covers_8_combinations(self):
        assert len(POINT_COMBOS) == len(MARKETS) * len(HORIZONS) * len(POINT_ALGORITHMS)
        assert set(POINT_COMBOS) == {
            (market, horizon, algorithm)
            for market in MARKETS
            for horizon in HORIZONS
            for algorithm in POINT_ALGORITHMS
        }


class TestInitialTrainEndIndex:
    def test_clamped_to_at_least_1_when_span_exceeds_available_dates(self):
        assert _initial_train_end_index(n_dates=100, oos_span_years=10) == 1

    def test_subtracts_span_days_when_sufficient_history(self):
        # 252*1=252일 span, n_dates=1000 -> initial_train_end=748
        assert _initial_train_end_index(n_dates=1000, oos_span_years=1) == 748


class TestResolveTuningSplitUsesOnlyInitialHistory:
    """REQ-ATE-027: 튜닝은 평가 구간 시작 이전 데이터만 사용해야 한다."""

    def test_train_and_val_never_touch_evaluation_span_dates(self):
        panel = _make_synthetic_panel(n_dates=400)
        trade_dates = campaign_module.extract_global_trade_date_axis(panel)
        initial_history_end_idx = _initial_train_end_index(len(trade_dates), oos_span_years=0) + 200
        # oos_span_years=0 -> initial_history_end=len(trade_dates); 의도적으로 작은
        # 초기 이력 구간(200일)만 남기도록 인위적으로 절단해 평가 구간 존재를 보장한다.
        initial_history_end_idx = min(initial_history_end_idx, len(trade_dates) - 50)
        eval_span_start_date = trade_dates[initial_history_end_idx]

        train_df, val_df = resolve_tuning_split(
            panel, trade_dates, initial_history_end_idx, horizon=20
        )

        assert not train_df.empty
        assert not val_df.empty
        assert train_df["trade_date"].max() < eval_span_start_date
        assert val_df["trade_date"].max() < eval_span_start_date

    def test_purged_walk_forward_split_is_actually_invoked(self):
        panel = _make_synthetic_panel(n_dates=400)
        trade_dates = campaign_module.extract_global_trade_date_axis(panel)
        initial_history_end_idx = 300

        with patch.object(
            campaign_module,
            "purged_walk_forward_split",
            wraps=campaign_module.purged_walk_forward_split,
        ) as spy:
            resolve_tuning_split(panel, trade_dates, initial_history_end_idx, horizon=20)

        spy.assert_called_once()
        # 초기 이력 구간의 날짜 수(300)만이 n_samples로 전달되어야 한다.
        assert spy.call_args.args[0] == initial_history_end_idx


class TestRunCampaignForMarketHorizonThreadsFrozenParams:
    """REQ-ATE-028: 동결 하이퍼파라미터가 모든 폴드 학습 호출에 동일하게 전달된다."""

    def test_frozen_params_identical_across_all_folds(self):
        panel = _make_synthetic_panel(n_dates=260, n_stocks=2)
        frozen_params_by_algorithm = {
            "lightgbm": {"n_estimators": 15, "learning_rate": 0.1, "num_leaves": 7},
            "xgboost": {"n_estimators": 15, "learning_rate": 0.1, "max_depth": 3},
        }

        with patch.object(
            campaign_module, "_fit_point_model", wraps=campaign_module._fit_point_model
        ) as spy:
            records = run_campaign_for_market_horizon(
                panel,
                market="domestic",
                horizon=20,
                initial_train_end_idx=150,
                n_folds=3,
                frozen_params_by_algorithm=frozen_params_by_algorithm,
            )

        assert len(records) == 3
        lgbm_calls = [c for c in spy.call_args_list if c.args[0] == "lightgbm"]
        xgb_calls = [c for c in spy.call_args_list if c.args[0] == "xgboost"]
        assert len(lgbm_calls) == 3
        assert len(xgb_calls) == 3
        assert all(c.args[1] == frozen_params_by_algorithm["lightgbm"] for c in lgbm_calls)
        assert all(c.args[1] == frozen_params_by_algorithm["xgboost"] for c in xgb_calls)

    def test_fold_train_end_sequence_advances_by_val_size(self):
        panel = _make_synthetic_panel(n_dates=260, n_stocks=2)
        frozen_params_by_algorithm = {
            "lightgbm": {"n_estimators": 10},
            "xgboost": {"n_estimators": 10},
        }

        records = run_campaign_for_market_horizon(
            panel,
            market="domestic",
            horizon=20,
            initial_train_end_idx=150,
            n_folds=3,
            frozen_params_by_algorithm=frozen_params_by_algorithm,
        )

        train_ends = [r.train_end for r in records]
        assert train_ends == sorted(train_ends)
        assert len(set(train_ends)) == 3

    def test_timeout_deadline_in_the_past_stops_before_first_fold(self):
        panel = _make_synthetic_panel(n_dates=260, n_stocks=2)
        frozen_params_by_algorithm = {
            "lightgbm": {"n_estimators": 10},
            "xgboost": {"n_estimators": 10},
        }

        records = run_campaign_for_market_horizon(
            panel,
            market="domestic",
            horizon=20,
            initial_train_end_idx=150,
            n_folds=3,
            frozen_params_by_algorithm=frozen_params_by_algorithm,
            timeout_deadline=0.0,
        )

        assert records == []


class TestRunWalkForwardCampaignAssemblesDatasetOncePerMarket:
    """REQ-ATE-022: 시장당 정확히 1회만 데이터셋을 조립한다(모킹 검증)."""

    def test_assemble_market_dataset_called_exactly_once_per_market(self, tmp_path: Path):
        synthetic_panels = {
            market: _make_synthetic_panel(n_dates=50, seed=idx)
            for idx, market in enumerate(MARKETS)
        }

        with (
            patch.object(
                campaign_module.train_module,
                "_assemble_market_dataset",
                side_effect=lambda engine, calendar, market, *a, **kw: synthetic_panels[market],
            ) as assemble_spy,
            patch.object(campaign_module, "fetch_market_calendar", return_value=MagicMock()),
            patch.object(campaign_module, "tune_initial_history_hyperparameters", return_value={}),
            patch.object(campaign_module, "run_campaign_for_market_horizon", return_value=[]),
        ):
            result = run_walk_forward_campaign(
                trainer_engine=MagicMock(),
                calendar_code="KRX",
                cache_dir=tmp_path / "cache",
                data_as_of=date(2026, 8, 17),
                feature_code_version="v1",
                optuna_storage_dir=tmp_path / "optuna",
            )

        assert result.success is True
        assert assemble_spy.call_count == len(MARKETS)
        called_markets = {c.args[2] for c in assemble_spy.call_args_list}
        assert called_markets == set(MARKETS)

    def test_single_combo_failure_does_not_abort_other_combos(self, tmp_path: Path):
        panel = _make_synthetic_panel(n_dates=50)

        def _fail_or_empty(panel_arg, market, horizon, *a, **kw):
            if market == "domestic" and horizon == 20:
                raise RuntimeError("합성 실패")
            return []

        with (
            patch.object(
                campaign_module.train_module, "_assemble_market_dataset", return_value=panel
            ),
            patch.object(campaign_module, "fetch_market_calendar", return_value=MagicMock()),
            patch.object(campaign_module, "tune_initial_history_hyperparameters", return_value={}),
            patch.object(
                campaign_module, "run_campaign_for_market_horizon", side_effect=_fail_or_empty
            ),
        ):
            result = run_walk_forward_campaign(
                trainer_engine=MagicMock(),
                calendar_code="KRX",
                cache_dir=tmp_path / "cache",
                data_as_of=date(2026, 8, 17),
                feature_code_version="v1",
                optuna_storage_dir=tmp_path / "optuna",
            )

        assert result.success is True
        assert ("domestic", 20) in result.errors_by_market_horizon
        assert ("domestic", 60) in result.records_by_market_horizon
        assert ("overseas", 20) in result.records_by_market_horizon


class TestMainCli:
    def test_main_returns_0_on_success(self, tmp_path: Path):
        with (
            patch.object(campaign_module, "build_trainer_engine", return_value=MagicMock()),
            patch.object(
                campaign_module,
                "run_walk_forward_campaign",
                return_value=campaign_module.CampaignResult(success=True),
            ),
        ):
            exit_code = main(
                [
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--models-root",
                    str(tmp_path / "models"),
                    "--data-as-of",
                    "2026-08-17",
                    "--feature-code-version",
                    "v1",
                    "--optuna-storage-dir",
                    str(tmp_path / "optuna"),
                ]
            )

        assert exit_code == 0

    def test_main_returns_1_on_failure(self, tmp_path: Path):
        with (
            patch.object(campaign_module, "build_trainer_engine", return_value=MagicMock()),
            patch.object(
                campaign_module,
                "run_walk_forward_campaign",
                return_value=campaign_module.CampaignResult(success=False, error="boom"),
            ),
        ):
            exit_code = main(
                [
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--models-root",
                    str(tmp_path / "models"),
                    "--data-as-of",
                    "2026-08-17",
                    "--feature-code-version",
                    "v1",
                    "--optuna-storage-dir",
                    str(tmp_path / "optuna"),
                ]
            )

        assert exit_code == 1

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
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from analyzer.training import campaign as campaign_module
from analyzer.training.backtest import BacktestMetrics
from analyzer.training.campaign import (
    CAMPAIGN_OOS_EVALUATION_SPAN_YEARS,
    CAMPAIGN_OPTUNA_TRIALS,
    CAMPAIGN_TIMEOUT_HOURS,
    CAMPAIGN_VAL_SIZE_TRADING_DAYS,
    POINT_ALGORITHMS,
    POINT_COMBOS,
    MarketHorizonFoldRecord,
    _initial_train_end_index,
    main,
    resolve_tuning_split,
    run_campaign_for_market_horizon,
    run_walk_forward_campaign,
    run_walk_forward_campaign_and_activate,
)
from analyzer.training.models import HORIZONS, MARKETS


def _crafted_fold_records(
    n_folds: int,
    lgbm_mean: float = 0.20,
    xgb_mean: float = -0.10,
    jitter: float = 0.01,
) -> list[MarketHorizonFoldRecord]:
    """GATE-1/2/3(12/전체/52주 롤링)를 통과시키는 lgbm 안정 시계열 +
    GATE-2에서 탈락하는 xgb 시계열을 합성한다 — 52개 이상의 폴드로
    `activate_market_horizon_combo()`의 실제 배포 분기(711~778줄)를
    실행시키기 위한 헬퍼(GATE-3 CAMPAIGN_GATE3_ROLLING_WEEKS=52 충족)."""
    base = pd.Timestamp("2020-01-06")
    records: list[MarketHorizonFoldRecord] = []
    for i in range(n_folds):
        sign = 1.0 if i % 2 == 0 else -1.0
        lgbm_ic = lgbm_mean + sign * jitter
        xgb_ic = xgb_mean + sign * jitter
        ens_ic = (lgbm_ic + xgb_ic) / 2
        records.append(
            MarketHorizonFoldRecord(
                fold_index=i,
                train_end=cast(pd.Timestamp, base + pd.Timedelta(weeks=i)),
                val_start=cast(pd.Timestamp, base + pd.Timedelta(weeks=i, days=1)),
                val_end=cast(pd.Timestamp, base + pd.Timedelta(weeks=i, days=6)),
                lightgbm_metrics=BacktestMetrics(
                    hit_rate=0.55,
                    pearson_ic=lgbm_ic,
                    rank_ic=lgbm_ic,
                    precision=0.5,
                    sharpe_ratio=0.1,
                    max_drawdown=-0.05,
                    confidence_calibration=0.5,
                ),
                xgboost_metrics=BacktestMetrics(
                    hit_rate=0.50,
                    pearson_ic=xgb_ic,
                    rank_ic=xgb_ic,
                    precision=0.5,
                    sharpe_ratio=0.0,
                    max_drawdown=-0.05,
                    confidence_calibration=0.5,
                ),
                ensemble_metrics=BacktestMetrics(
                    hit_rate=0.52,
                    pearson_ic=ens_ic,
                    rank_ic=ens_ic,
                    precision=0.5,
                    sharpe_ratio=0.05,
                    max_drawdown=-0.05,
                    confidence_calibration=0.5,
                ),
            )
        )
    return records


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


class TestComputeFoldEnsembleDegenerateQuantileFallback:
    """리뷰 지적사항: `_compute_fold_ensemble()`의 축퇴 분위수 분포
    (p10==p90) 폴백 confidence=0.5 분기가 기존에 테스트 커버리지가
    없었다(src/analyzer/training/campaign.py `_compute_fold_ensemble`).
    프로덕션 폴백 로직 자체는 수정하지 않는다(테스트 공백 보완만)."""

    def test_degenerate_quantile_row_falls_back_to_neutral_confidence(self):
        # 0번 행: p10==p90(축퇴, sigma=0) → ensemble.compute_confidence()가
        # ValueError를 내며, _compute_fold_ensemble()이 이를 잡아 0.5로 대체.
        # 1번 행: 정상 분위수 분포(비교 대조군).
        lgbm_preds = np.array([0.5, 0.1])
        xgb_preds = np.array([0.3, 0.1])
        p10_preds = np.array([0.2, 0.2])
        p90_preds = np.array([0.2, 0.4])

        _, confidences = campaign_module._compute_fold_ensemble(
            lgbm_preds, xgb_preds, p10_preds, p90_preds
        )

        assert confidences[0] == 0.5
        assert confidences[1] != 0.5


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
    """M6 Part 0: `main()`은 `run_walk_forward_campaign_and_activate()`를
    호출하도록 배선이 갱신되었다(1차 배포까지 end-to-end 연결) — 신규
    `--summary-report-path` 인자가 추가되었다."""

    def test_main_returns_0_on_success(self, tmp_path: Path):
        with (
            patch.object(campaign_module, "build_trainer_engine", return_value=MagicMock()),
            patch.object(
                campaign_module,
                "run_walk_forward_campaign_and_activate",
                return_value=campaign_module.CampaignActivationResult(
                    campaign_result=campaign_module.CampaignResult(success=True)
                ),
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
                    "--summary-report-path",
                    str(tmp_path / "summary.md"),
                ]
            )

        assert exit_code == 0

    def test_main_returns_1_on_failure(self, tmp_path: Path):
        with (
            patch.object(campaign_module, "build_trainer_engine", return_value=MagicMock()),
            patch.object(
                campaign_module,
                "run_walk_forward_campaign_and_activate",
                return_value=campaign_module.CampaignActivationResult(
                    campaign_result=campaign_module.CampaignResult(success=False, error="boom")
                ),
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
                    "--summary-report-path",
                    str(tmp_path / "summary.md"),
                ]
            )

        assert exit_code == 1


class TestPart0IntegrationWiring:
    """M6 Part 0: 캠페인 실행 후 `campaign_metrics.py`(JSONL/사이드카)와
    `stabilization.py`(안정화 게이트 + 챔피언 선정)가 실제로 배선되어
    호출됨을 확인한다 — M4 산출물이 M5 산출물과 순환 의존 없이 연결됨."""

    def test_activate_market_horizon_combo_writes_jsonl_and_invokes_stabilization(
        self, tmp_path: Path
    ):
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
            n_folds=15,
            frozen_params_by_algorithm=frozen_params_by_algorithm,
        )
        assert len(records) == 15

        models_root = tmp_path / "models"
        jsonl_dir = models_root / "domestic" / "20"

        with (
            patch.object(
                campaign_module.stabilization,
                "evaluate_combo_stabilization",
                wraps=campaign_module.stabilization.evaluate_combo_stabilization,
            ) as stab_spy,
            patch.object(
                campaign_module.stabilization,
                "select_champion_strategy",
                wraps=campaign_module.stabilization.select_champion_strategy,
            ) as champ_spy,
        ):
            outcome = campaign_module.activate_market_horizon_combo(
                panel=panel,
                market="domestic",
                horizon=20,
                fold_records=records,
                jsonl_dir=jsonl_dir,
                models_root=models_root,
                initial_train_end_idx=150,
                n_folds=15,
                frozen_params_by_algorithm=frozen_params_by_algorithm,
                trained_date=date(2026, 8, 17),
            )

        # stabilization.py의 안정화 게이트 함수가 실제로(lgbm+xgb) 2회 호출됨.
        assert stab_spy.call_count == 2
        champ_spy.assert_called_once()

        # campaign_metrics.py의 JSONL 스트림 3개(lgbm/xgb/ensemble)가 실제로
        # 생성되고 15줄씩 기록됨(12개 스트림 중 이 (시장,horizon) 조합분 3개).
        for algorithm in ("lightgbm", "xgboost", "ensemble"):
            jsonl_path = jsonl_dir / campaign_module.campaign_metrics.fold_metrics_jsonl_filename(
                "domestic", 20, algorithm
            )
            assert jsonl_path.exists()
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 15

        assert isinstance(outcome, campaign_module.ComboActivationOutcome)
        assert outcome.market == "domestic"
        assert outcome.horizon == 20

    def test_deployment_prohibited_combo_skips_persistence_and_manifest(self, tmp_path: Path):
        """REQ-ATE-047: 안정화 미통과(자격 있는 전략 0개)면 어떤 형태로도
        배포하지 않는다 — 모델 파일/사이드카/활성화 매니페스트 미생성."""
        panel = _make_synthetic_panel(n_dates=260, n_stocks=2)
        frozen_params_by_algorithm = {
            "lightgbm": {"n_estimators": 5},
            "xgboost": {"n_estimators": 5},
        }
        records = run_campaign_for_market_horizon(
            panel,
            market="domestic",
            horizon=20,
            initial_train_end_idx=150,
            n_folds=3,  # 롤링 윈도우(12주/52주) 미충족 — 모든 게이트 FAIL 유도
            frozen_params_by_algorithm=frozen_params_by_algorithm,
        )
        models_root = tmp_path / "models"

        outcome = campaign_module.activate_market_horizon_combo(
            panel=panel,
            market="domestic",
            horizon=20,
            fold_records=records,
            jsonl_dir=models_root / "domestic" / "20",
            models_root=models_root,
            initial_train_end_idx=150,
            n_folds=3,
            frozen_params_by_algorithm=frozen_params_by_algorithm,
            trained_date=date(2026, 8, 17),
        )

        assert outcome.champion_selection.deployment_prohibited is True
        assert outcome.persisted_algorithms == ()
        # 모델 파일이 전혀 생성되지 않아야 한다(models_root 자체는 jsonl로
        # 인해 생성되었을 수 있으나, algorithm 서브디렉토리는 없어야 한다).
        assert not (models_root / "domestic" / "20" / "lightgbm").exists()
        assert not (models_root / "domestic" / "20" / "xgboost").exists()


class TestActivateMarketHorizonComboDeploymentBranch:
    """sync-auditor BLOCKING 결함 수정: `activate_market_horizon_combo()`의
    711~778줄(실제 배포 분기 — 최종 재학습+저장+사이드카+활성화 매니페스트+
    전략 매니페스트)을 GATE-3(52주 롤링) 요구를 충족하는 52개 이상의 합성
    폴드로 실제 실행시킨다."""

    def test_deployment_success_path_persists_model_and_manifests(self, tmp_path: Path):
        panel = _make_synthetic_panel(n_dates=520)
        n_folds = 52
        initial_train_end_idx = 200
        records = _crafted_fold_records(n_folds)
        frozen_params_by_algorithm = {
            "lightgbm": {"n_estimators": 5},
            "xgboost": {"n_estimators": 5},
        }
        models_root = tmp_path / "models"
        jsonl_dir = models_root / "domestic" / "20"

        with (
            patch.object(
                campaign_module.activation_module,
                "promote_activation_manifest",
                wraps=campaign_module.activation_module.promote_activation_manifest,
            ) as promote_spy,
            patch.object(
                campaign_module.activation_module,
                "write_strategy_manifest",
                wraps=campaign_module.activation_module.write_strategy_manifest,
            ) as strategy_spy,
        ):
            outcome = campaign_module.activate_market_horizon_combo(
                panel=panel,
                market="domestic",
                horizon=20,
                fold_records=records,
                jsonl_dir=jsonl_dir,
                models_root=models_root,
                initial_train_end_idx=initial_train_end_idx,
                n_folds=n_folds,
                frozen_params_by_algorithm=frozen_params_by_algorithm,
                trained_date=date(2026, 8, 17),
            )

        # lgbm만 안정화(gate2 실패로 xgb 탈락) -> 챔피언은 lgbm 단독, xgb는
        # 활성화 매니페스트 대상에서 제외된다(REQ-ATE-046).
        assert outcome.champion_selection.deployment_prohibited is False
        assert outcome.champion_selection.champion_algorithm == "lightgbm"
        assert outcome.persisted_algorithms == ("lightgbm",)
        assert outcome.champion_selection.excluded_artifact_algorithms == ("xgboost",)

        # 711줄 이후 실제 배포 분기(모델 저장+사이드카+매니페스트)가 실행됨.
        promote_spy.assert_called_once()
        strategy_spy.assert_called_once()

        model_dir = campaign_module.persistence_module.model_dir(
            models_root, "domestic", 20, "lightgbm"
        )
        model_filename = campaign_module.persistence_module.model_filename(
            "domestic", 20, "lightgbm", date(2026, 8, 17)
        )
        model_path = model_dir / model_filename
        assert model_path.exists()
        assert (model_dir / f"{model_filename}.sha256").exists()
        assert campaign_module.campaign_metrics.sidecar_path_for(model_path).exists()

        activation_manifest_path = campaign_module.activation_module.activation_manifest_path(
            models_root, "domestic", 20, "lightgbm"
        )
        assert activation_manifest_path.exists()

        strategy_manifest_path = campaign_module.activation_module.strategy_manifest_path(
            models_root, "domestic", 20
        )
        assert strategy_manifest_path.exists()

        # xgboost는 배포 대상에서 제외되었으므로 아티팩트 디렉터리가 없다.
        assert not campaign_module.persistence_module.model_dir(
            models_root, "domestic", 20, "xgboost"
        ).exists()


class TestRunWalkForwardCampaignAndActivateEntrypoint:
    """sync-auditor BLOCKING 결함 수정: 최상위 진입점
    `run_walk_forward_campaign_and_activate()`(M6 Part 0) 자체가 실제로
    실행됨을 확인한다 — 데이터셋 조립/튜닝은 최소 모킹하되, 캠페인 결과
    소비(`activate_market_horizon_combo()` 포함)는 실코드 경로로 실행한다."""

    def test_entrypoint_executes_and_activates_all_combos(self, tmp_path: Path):
        panel = _make_synthetic_panel(n_dates=900)

        def _run_campaign_side_effect(
            _panel, _market, _horizon, _initial_train_end_idx, folds, *_a, **_kw
        ):
            return _crafted_fold_records(folds)

        with (
            patch.object(
                campaign_module.train_module,
                "_assemble_market_dataset",
                return_value=panel,
            ) as assemble_spy,
            patch.object(campaign_module, "fetch_market_calendar", return_value=MagicMock()),
            patch.object(
                campaign_module,
                "tune_initial_history_hyperparameters",
                return_value={"n_estimators": 5},
            ),
            patch.object(
                campaign_module,
                "run_campaign_for_market_horizon",
                side_effect=_run_campaign_side_effect,
            ) as run_campaign_spy,
        ):
            result = run_walk_forward_campaign_and_activate(
                trainer_engine=MagicMock(),
                calendar_code="KRX",
                cache_dir=tmp_path / "cache",
                models_root=tmp_path / "models",
                data_as_of=date(2026, 8, 17),
                feature_code_version="v1",
                optuna_storage_dir=tmp_path / "optuna",
                summary_report_path=tmp_path / "summary.md",
                oos_span_years=2,
            )

        assert assemble_spy.call_count == len(MARKETS)
        assert run_campaign_spy.call_count == len(MARKETS) * len(HORIZONS)
        assert result.campaign_result.success is True
        assert result.campaign_result.errors_by_market_horizon == {}

        # 시장x horizon 4개 조합 모두 activate_market_horizon_combo()를
        # 실제로 거쳤다(0% 커버리지였던 최상위 진입점 실행 확인).
        assert len(result.activation_outcomes) == len(MARKETS) * len(HORIZONS)
        for (market, horizon), outcome in result.activation_outcomes.items():
            assert outcome.market == market
            assert outcome.horizon == horizon
            # lgbm 시계열은 GATE-1/2/3을 통과하도록 합성했으므로 조합마다
            # 배포가 실제로 실행된다(실 배포 분기 커버리지 확보).
            assert outcome.persisted_algorithms == ("lightgbm",)
            model_path = campaign_module.persistence_module.model_dir(
                tmp_path / "models", market, horizon, "lightgbm"
            )
            assert model_path.exists()

        assert result.summary_report_path is not None
        assert result.summary_report_path.exists()

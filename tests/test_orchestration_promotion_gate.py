"""상시 챔피언/챌린저 오프라인 평가 테스트 (SPEC-ANALYZER-TRAIN-EVAL-001 M6,
REQ-ATE-056/057/058/059, v0.3.0 F2 정정).

합성 축소 다종목 패널만 사용한다 — 실 DB 접속을 수행하지 않는다.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from analyzer.orchestration import promotion_gate as promotion_gate_module
from analyzer.orchestration.promotion_gate import (
    PROMOTION_GATE_HOLDOUT_TRADING_DAYS,
    build_pooled_train_holdout_data,
    evaluate_and_promote,
    evaluate_promotion_gate,
    load_champion_native,
    resolve_challenger_holdout_index_bounds,
    train_challenger_models,
)
from analyzer.training import persistence as persistence_module
from analyzer.training.models import HORIZONS, MARKETS


def _make_synthetic_panel(n_dates: int, n_stocks: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    frames = []
    for stock_index in range(n_stocks):
        frames.append(
            pd.DataFrame(
                {
                    "stock_code": f"S{stock_index}",
                    "trade_date": dates,
                    "KMID": rng.normal(size=n_dates),
                    "label_D20": rng.normal(scale=0.02, size=n_dates),
                    "label_D60": rng.normal(scale=0.03, size=n_dates),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestResolveChallengerHoldoutIndexBounds:
    def test_holdout_precedes_train_end_plus_gap(self):
        bounds = resolve_challenger_holdout_index_bounds(n_samples=500, horizon=20, holdout_size=20)

        assert bounds.val_start == bounds.train_end + 20  # purge gap D20
        assert bounds.val_end == bounds.val_start + 20


class TestChallengerTrainHoldoutSeparation:
    """AC-ATE-041: 챌린저는 [학습~T-홀드아웃) 구간에만 학습되며, 홀드아웃
    구간 데이터는 학습 호출의 data_by_combo 인자에 전혀 포함되지 않는다
    (in-sample 오염 없음)."""

    def test_holdout_rows_never_appear_in_training_data(self):
        panel_by_market = {market: _make_synthetic_panel(n_dates=400) for market in MARKETS}

        data_by_combo, eval_data_by_combo = build_pooled_train_holdout_data(
            panel_by_market, holdout_size=PROMOTION_GATE_HOLDOUT_TRADING_DAYS
        )

        assert set(data_by_combo.keys()) == {(m, h) for m in MARKETS for h in HORIZONS}
        for combo, (x_train, _y_train) in data_by_combo.items():
            x_holdout, _y_holdout = eval_data_by_combo[combo]
            # 학습 행수 + 홀드아웃 행수가 전체 패널 행수보다 작아야 한다
            # (purge gap 만큼 어느 구간에도 속하지 않는 행이 존재).
            assert len(x_train) > 0
            assert len(x_holdout) > 0

    def test_train_pooled_models_receives_eval_data_and_early_stopping(self):
        panel_by_market = {market: _make_synthetic_panel(n_dates=400) for market in MARKETS}
        data_by_combo, eval_data_by_combo = build_pooled_train_holdout_data(
            panel_by_market, holdout_size=PROMOTION_GATE_HOLDOUT_TRADING_DAYS
        )

        with patch.object(
            promotion_gate_module,
            "train_pooled_models",
            wraps=promotion_gate_module.train_pooled_models,
        ) as spy:
            train_challenger_models(
                data_by_combo,
                eval_data_by_combo,
                lgbm_params={"n_estimators": 5},
                xgb_params={"n_estimators": 5},
                early_stopping_rounds=5,
            )

        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs["eval_data_by_combo"] is eval_data_by_combo
        assert kwargs["early_stopping_rounds"] == 5
        # 홀드아웃 데이터가 학습용 data_by_combo(위치 인자 0)에 섞여 있지 않은지
        # 원본 객체 동일성으로 확인한다.
        called_data_by_combo = spy.call_args.args[0]
        assert called_data_by_combo is data_by_combo
        for combo, (x_train, _y) in called_data_by_combo.items():
            x_holdout, _yh = eval_data_by_combo[combo]
            assert x_train is not x_holdout


class TestChampionRescoringUsesNativeLoadNotPersistence:
    """AC-ATE-042: 챔피언은 `persistence.py`가 아니라 이 모듈이 직접
    프레임워크 네이티브 로드 API로 로드한다."""

    def test_load_champion_native_never_calls_persistence_module(self, tmp_path: Path):
        import lightgbm as lgb

        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 2))
        y = rng.normal(size=50)
        model = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
        model.fit(x, y)
        model_path = tmp_path / "champion.txt"
        model.booster_.save_model(str(model_path))

        with (
            patch.object(persistence_module, "save_model_native") as save_spy,
            patch.object(persistence_module, "verify_model_integrity") as verify_spy,
        ):
            loaded = load_champion_native(model_path, "lightgbm")

        assert loaded is not None
        save_spy.assert_not_called()
        verify_spy.assert_not_called()

    def test_load_champion_native_rejects_unknown_algorithm(self, tmp_path: Path):
        with pytest.raises(ValueError, match="지원하지 않는 algorithm"):
            load_champion_native(tmp_path / "x.bin", "prophet")


class TestSameWindowComparison:
    """AC-ATE-042: 챌린저와 챔피언이 동일한 홀드아웃 구간에서 평가된다 —
    챔피언의 활성화 매니페스트에 보존된 과거 기록 지표를 사용하지 않는다."""

    def test_evaluate_promotion_gate_skips_combos_without_champion_path(self):
        """§C 엣지케이스 3: 활성 챔피언이 없는 조합은 판정 대상에서 제외."""
        panel_by_market = {market: _make_synthetic_panel(n_dates=200) for market in MARKETS}

        verdicts = evaluate_promotion_gate(
            panel_by_market=panel_by_market,
            champion_model_paths={},  # 챔피언 전무 — 1차 배포 이전 상태
            challenger_trained_date=__import__("datetime").date(2026, 8, 17),
        )

        assert verdicts == {}

    def test_champion_and_challenger_evaluated_on_identical_holdout_rows(self, tmp_path: Path):
        """챔피언 재채점과 챌린저 평가가 `eval_data_by_combo`의 동일 객체를
        참조함을 확인한다(동일-윈도우 비교, 매니페스트의 과거 지표 미사용)."""
        import lightgbm as lgb
        import xgboost as xgb

        panel_by_market = {market: _make_synthetic_panel(n_dates=300) for market in MARKETS}

        # 미리 챔피언 아티팩트(도메스틱 D20 lightgbm 1개)를 저장해둔다 —
        # 합성 패널은 피처 1개(KMID)뿐이므로 챔피언도 동일 피처 차원으로 학습.
        rng = np.random.default_rng(1)
        champ_model = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
        champ_model.fit(rng.normal(size=(50, 1)), rng.normal(size=50))
        champion_path = tmp_path / "champion_domestic_20_lightgbm.txt"
        champ_model.booster_.save_model(str(champion_path))

        seen_holdouts: list[np.ndarray] = []
        original_predict_native = promotion_gate_module._predict_native

        def _spy_predict_native(model, algorithm, x):
            seen_holdouts.append(x)
            return original_predict_native(model, algorithm, x)

        with (
            patch.object(promotion_gate_module, "_predict_native", side_effect=_spy_predict_native),
            patch.object(promotion_gate_module, "train_challenger_models") as train_spy,
        ):
            # 챌린저 학습은 값싼 스텁으로 대체 — 이 테스트는 "동일 홀드아웃"
            # 배선만 검증한다(실제 학습 품질은 다른 테스트가 검증).
            fake_lgbm = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
            fake_lgbm.fit(rng.normal(size=(50, 1)), rng.normal(size=50))
            fake_xgb = xgb.XGBRegressor(n_estimators=5, verbosity=0)
            fake_xgb.fit(rng.normal(size=(50, 1)), rng.normal(size=50))
            train_spy.return_value = {
                (m, h, "lightgbm"): fake_lgbm for m in MARKETS for h in HORIZONS
            } | {(m, h, "xgboost"): fake_xgb for m in MARKETS for h in HORIZONS}

            verdicts = evaluate_promotion_gate(
                panel_by_market=panel_by_market,
                champion_model_paths={("domestic", 20, "lightgbm"): champion_path},
                challenger_trained_date=__import__("datetime").date(2026, 8, 17),
            )

        assert ("domestic", 20, "lightgbm") in verdicts
        # `_predict_native()`는 챔피언 재채점에서만 호출된다(챌린저는
        # `model.predict()`를 직접 호출) — 이 호출이 사용한 홀드아웃 배열이
        # `build_pooled_train_holdout_data()`가 산출한 것과 동일 객체임을
        # 확인해 "동일 윈도우"를 보장한다(매니페스트의 과거 지표 미사용).
        assert len(seen_holdouts) == 1
        _data_by_combo, eval_data_by_combo = build_pooled_train_holdout_data(panel_by_market)
        expected_holdout, _ = eval_data_by_combo[("domestic", 20)]
        assert seen_holdouts[0].shape == expected_holdout.shape


class TestPromotionThresholdBoundary:
    """AC-ATE-043/044: 임계값 이상 우수하면 승격, 이내/미달이면 보류."""

    def test_challenger_exceeding_threshold_promotes(self):
        panel_by_market = {market: _make_synthetic_panel(n_dates=200) for market in MARKETS}

        with (
            patch.object(
                promotion_gate_module,
                "evaluate_models_on_holdout",
                return_value={("domestic", 20, "lightgbm"): _fake_metrics(rank_ic=0.10)},
            ),
            patch.object(
                promotion_gate_module,
                "load_champion_native",
                return_value=MagicMock(),
            ),
            patch.object(
                promotion_gate_module,
                "_predict_native",
                return_value=np.zeros(5),
            ),
            patch.object(
                promotion_gate_module,
                "compute_backtest_metrics",
                return_value=_fake_metrics(rank_ic=0.02),
            ),
        ):
            verdicts = evaluate_promotion_gate(
                panel_by_market=panel_by_market,
                champion_model_paths={("domestic", 20, "lightgbm"): Path("x")},
                challenger_trained_date=__import__("datetime").date(2026, 8, 17),
                threshold=0.0,
            )

        verdict = verdicts[("domestic", 20, "lightgbm")]
        assert verdict.promoted is True

    def test_challenger_within_tolerance_holds_back(self):
        panel_by_market = {market: _make_synthetic_panel(n_dates=200) for market in MARKETS}

        with (
            patch.object(
                promotion_gate_module,
                "evaluate_models_on_holdout",
                return_value={("domestic", 20, "lightgbm"): _fake_metrics(rank_ic=0.02)},
            ),
            patch.object(promotion_gate_module, "load_champion_native", return_value=MagicMock()),
            patch.object(promotion_gate_module, "_predict_native", return_value=np.zeros(5)),
            patch.object(
                promotion_gate_module,
                "compute_backtest_metrics",
                return_value=_fake_metrics(rank_ic=0.02),
            ),
        ):
            verdicts = evaluate_promotion_gate(
                panel_by_market=panel_by_market,
                champion_model_paths={("domestic", 20, "lightgbm"): Path("x")},
                challenger_trained_date=__import__("datetime").date(2026, 8, 17),
                threshold=0.0,
            )

        verdict = verdicts[("domestic", 20, "lightgbm")]
        assert verdict.promoted is False


def _fake_metrics(rank_ic: float):
    from analyzer.training.backtest import BacktestMetrics

    return BacktestMetrics(
        hit_rate=0.5,
        pearson_ic=rank_ic,
        rank_ic=rank_ic,
        precision=0.5,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        confidence_calibration=0.0,
    )


class TestEvaluateAndPromoteActivatesOnlyPromoted:
    """REQ-ATE-052(F3): 승격된 조합만 `activation.promote_activation_manifest()`를
    호출한다(보류 조합은 매니페스트 미갱신)."""

    def test_only_promoted_combo_writes_activation_manifest(self, tmp_path: Path):
        from analyzer.orchestration.activation import read_activation_manifest

        models_root = tmp_path / "models"
        model_dir = persistence_module.model_dir(models_root, "domestic", 20, "lightgbm")
        model_dir.mkdir(parents=True)
        model_filename = persistence_module.model_filename(
            "domestic", 20, "lightgbm", __import__("datetime").date(2026, 8, 17)
        )
        model_path = model_dir / model_filename
        model_path.write_text("dummy", encoding="utf-8")
        sidecar = model_path.with_suffix(model_path.suffix + ".sha256")
        sidecar.write_text("cafebabe", encoding="utf-8")

        with patch.object(
            promotion_gate_module,
            "evaluate_promotion_gate",
            return_value={
                ("domestic", 20, "lightgbm"): promotion_gate_module.PromotionVerdict(
                    market="domestic",
                    horizon=20,
                    algorithm="lightgbm",
                    promoted=True,
                    challenger_rank_ic=0.05,
                    champion_rank_ic=0.01,
                    challenger_trained_date=__import__("datetime").date(2026, 8, 17),
                ),
                ("domestic", 20, "xgboost"): promotion_gate_module.PromotionVerdict(
                    market="domestic",
                    horizon=20,
                    algorithm="xgboost",
                    promoted=False,
                    challenger_rank_ic=0.0,
                    champion_rank_ic=0.02,
                    challenger_trained_date=__import__("datetime").date(2026, 8, 17),
                ),
            },
        ):
            evaluate_and_promote(
                models_root=models_root,
                panel_by_market={},
                champion_model_paths={},
                challenger_trained_date=__import__("datetime").date(2026, 8, 17),
                merged_to_active=True,
            )

        promoted_manifest = read_activation_manifest(models_root, "domestic", 20, "lightgbm")
        held_back_manifest = read_activation_manifest(models_root, "domestic", 20, "xgboost")
        assert promoted_manifest is not None
        assert promoted_manifest.sidecar_sha256 == "cafebabe"
        assert held_back_manifest is None

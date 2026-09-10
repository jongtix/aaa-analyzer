"""포인트/분위수 모델 예측 실행 테스트 (SPEC-ANALYZER-INFER-001 M5, 신규
갭 해소 — design.md §1 단계 5, REQ-AIF-040/041/050).

`resolve_serving_targets()`/`resolve_latest_quantile_manifest()`가 반환한
경로에서 실제로 `.predict()`를 호출해 `resolution.compute_score_columns()`·
`scoring.resolve_confidence_for_stock()`이 기대하는 입력 계약(예측값
Mapping/p10·p90 튜플)을 그대로 만족하는지 검증한다.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from analyzer.inference.predict import predict_point_models, predict_quantile_models
from analyzer.inference.resolution import (
    QuantileManifest,
    ServingPlan,
    compute_score_columns,
)
from analyzer.inference.scoring import resolve_confidence_for_stock
from analyzer.training.campaign_metrics import sidecar_path_for


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


def _feature_row(feature_columns: list[str], extra: dict[str, float] | None = None) -> pd.DataFrame:
    data = {col: [0.1 * (i + 1)] for i, col in enumerate(feature_columns)}
    if extra:
        data.update({k: [v] for k, v in extra.items()})
    return pd.DataFrame(data)


def _serving_plan(active_strategy: str, model_paths: dict[str, Path]) -> ServingPlan:
    return ServingPlan(
        market="domestic",
        horizon=20,
        active_strategy=active_strategy,
        algorithms=tuple(model_paths.keys()),
        manifests={},
        model_paths=model_paths,
    )


class TestPredictPointModelsEnsemble:
    """앙상블 전략은 lightgbm/xgboost 두 알고리즘 모두 예측해야 하며, 반환값은
    `resolution.compute_score_columns()`의 `predictions` 인자 계약과 정확히
    일치해야 한다(REQ-AIF-040)."""

    def test_returns_mapping_with_both_algorithms(self, tmp_path: Path):
        feature_columns = ["roc_60", "ma_60", "std_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm", feature_columns, seed=0)
        xgb_path = _train_xgboost(tmp_path, "xgb", feature_columns, seed=1)
        plan = _serving_plan("ensemble", {"lightgbm": lgbm_path, "xgboost": xgb_path})
        feature_row = _feature_row(feature_columns)

        predictions = predict_point_models(plan, feature_row)

        assert set(predictions.keys()) == {"lightgbm", "xgboost"}
        assert isinstance(predictions["lightgbm"], float)
        assert isinstance(predictions["xgboost"], float)

    def test_predictions_feed_directly_into_compute_score_columns(self, tmp_path: Path):
        """boundary-verification: predict.py의 출력이 실제로
        resolution.compute_score_columns()의 입력 계약을 만족하는지 양쪽을
        함께 검증한다."""
        feature_columns = ["roc_60", "ma_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm", feature_columns, seed=2)
        xgb_path = _train_xgboost(tmp_path, "xgb", feature_columns, seed=3)
        plan = _serving_plan("ensemble", {"lightgbm": lgbm_path, "xgboost": xgb_path})
        feature_row = _feature_row(feature_columns)

        predictions = predict_point_models(plan, feature_row)
        columns = compute_score_columns(plan.active_strategy, predictions)

        assert columns.lgbm_score == predictions["lightgbm"]
        assert columns.xgb_score == predictions["xgboost"]


class TestPredictPointModelsSingleStrategy:
    """단독 전략(G3)은 배포된 알고리즘 1개만 예측한다(REQ-AIF-041)."""

    def test_lightgbm_only_strategy_predicts_single_algorithm(self, tmp_path: Path):
        feature_columns = ["roc_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm_solo", feature_columns, seed=4)
        plan = _serving_plan("lightgbm", {"lightgbm": lgbm_path})
        feature_row = _feature_row(feature_columns)

        predictions = predict_point_models(plan, feature_row)

        assert set(predictions.keys()) == {"lightgbm"}
        columns = compute_score_columns(plan.active_strategy, predictions)
        assert columns.lgbm_score == predictions["lightgbm"]
        assert columns.xgb_score is None

    def test_xgboost_only_strategy_predicts_single_algorithm(self, tmp_path: Path):
        feature_columns = ["ma_60"]
        xgb_path = _train_xgboost(tmp_path, "xgb_solo", feature_columns, seed=5)
        plan = _serving_plan("xgboost", {"xgboost": xgb_path})
        feature_row = _feature_row(feature_columns)

        predictions = predict_point_models(plan, feature_row)

        assert set(predictions.keys()) == {"xgboost"}
        columns = compute_score_columns(plan.active_strategy, predictions)
        assert columns.xgb_score == predictions["xgboost"]
        assert columns.lgbm_score is None


class TestPredictPointModelsFeatureColumnSelection:
    """각 모델의 `.meta.json` 사이드카가 선언한 feature_columns만 정확한
    순서로 선택해야 하며(모델별로 다를 수 있음), 누락 시 조용히 0으로
    채우지 않고 명시적으로 실패해야 한다."""

    def test_distinct_feature_columns_per_model_are_each_independently_resolved(
        self, tmp_path: Path
    ):
        lgbm_columns = ["roc_60", "ma_60"]
        xgb_columns = ["std_60", "rank_60", "corr_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm_distinct", lgbm_columns, seed=6)
        xgb_path = _train_xgboost(tmp_path, "xgb_distinct", xgb_columns, seed=7)
        plan = _serving_plan("ensemble", {"lightgbm": lgbm_path, "xgboost": xgb_path})
        # 두 모델의 feature_columns 합집합 + 무관한 여분 컬럼을 포함한 피처 행.
        feature_row = _feature_row(
            list(dict.fromkeys(lgbm_columns + xgb_columns)), extra={"irrelevant_col": 99.0}
        )

        predictions = predict_point_models(plan, feature_row)

        assert set(predictions.keys()) == {"lightgbm", "xgboost"}

    def test_missing_required_feature_column_raises_value_error(self, tmp_path: Path):
        feature_columns = ["roc_60", "ma_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm_missing", feature_columns, seed=8)
        plan = _serving_plan("lightgbm", {"lightgbm": lgbm_path})
        # ma_60 컬럼이 빠진 피처 행 — 조용한 0-fill 금지, 명시적 실패 요구.
        feature_row = _feature_row(["roc_60"])

        try:
            predict_point_models(plan, feature_row)
            raise AssertionError("누락된 피처 컬럼에 대해 ValueError가 발생해야 한다")
        except ValueError as exc:
            assert "ma_60" in str(exc)


class TestPredictPointModelsUnsupportedAlgorithm:
    """`ServingPlan.model_paths`에 lightgbm/xgboost 이외의 알고리즘 키가
    섞여 들어오는 방어적 가드 — `resolve_serving_targets()`가 정상 경로에서
    생성하지 않는 조합이지만, 조용히 무시하지 않고 명시적으로 실패한다."""

    def test_unknown_algorithm_key_raises_value_error(self, tmp_path: Path):
        feature_columns = ["roc_60"]
        dummy_path = _train_lightgbm(tmp_path, "dummy", feature_columns, seed=15)
        plan = _serving_plan("ensemble", {"unknown_algo": dummy_path})
        feature_row = _feature_row(feature_columns)

        try:
            predict_point_models(plan, feature_row)
            raise AssertionError("알 수 없는 algorithm 키에 대해 ValueError가 발생해야 한다")
        except ValueError as exc:
            assert "unknown_algo" in str(exc)


class TestPredictQuantileModels:
    """p10/p90 분위수 예측 결과는 `scoring.resolve_confidence_for_stock()`의
    입력 계약(p10: float, p90: float)과 정확히 일치해야 한다(REQ-AIF-050)."""

    def test_returns_p10_p90_float_tuple(self, tmp_path: Path):
        feature_columns = ["roc_60", "ma_60"]
        p10_path = _train_lightgbm(tmp_path, "q10", feature_columns, seed=9)
        p90_path = _train_lightgbm(tmp_path, "q90", feature_columns, seed=10)
        manifest = QuantileManifest(
            market="domestic",
            horizon=20,
            trained_date=__import__("datetime").date(2026, 8, 25),
            p10_path=p10_path,
            p90_path=p90_path,
        )
        feature_row = _feature_row(feature_columns)

        p10, p90 = predict_quantile_models(manifest, feature_row)

        assert isinstance(p10, float)
        assert isinstance(p90, float)

    def test_prediction_output_feeds_directly_into_resolve_confidence_for_stock(
        self, tmp_path: Path
    ):
        """boundary-verification: predict.py의 (p10, p90) 출력이 scoring.py의
        입력 계약을 실제로 만족하는지 양쪽을 함께 검증한다."""
        feature_columns = ["roc_60"]
        p10_path = _train_lightgbm(tmp_path, "q10b", feature_columns, seed=11)
        p90_path = _train_lightgbm(tmp_path, "q90b", feature_columns, seed=12)
        manifest = QuantileManifest(
            market="domestic",
            horizon=20,
            trained_date=__import__("datetime").date(2026, 8, 25),
            p10_path=p10_path,
            p90_path=p90_path,
        )
        feature_row = _feature_row(feature_columns)

        p10, p90 = predict_quantile_models(manifest, feature_row)
        result = resolve_confidence_for_stock(manifest, score=0.04, p10=p10, p90=p90)

        assert isinstance(result, float) or hasattr(result, "value")

    def test_p10_and_p90_models_may_have_distinct_feature_columns(self, tmp_path: Path):
        p10_columns = ["roc_60", "ma_60"]
        p90_columns = ["std_60"]
        p10_path = _train_lightgbm(tmp_path, "q10c", p10_columns, seed=13)
        p90_path = _train_lightgbm(tmp_path, "q90c", p90_columns, seed=14)
        manifest = QuantileManifest(
            market="domestic",
            horizon=20,
            trained_date=__import__("datetime").date(2026, 8, 25),
            p10_path=p10_path,
            p90_path=p90_path,
        )
        feature_row = _feature_row(list(dict.fromkeys(p10_columns + p90_columns)))

        p10, p90 = predict_quantile_models(manifest, feature_row)

        assert isinstance(p10, float)
        assert isinstance(p90, float)

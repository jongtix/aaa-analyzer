"""포인트/분위수 모델 예측 실행 테스트 (SPEC-ANALYZER-INFER-001 M5, 신규
갭 해소 — design.md §1 단계 5, REQ-AIF-040/041/050).

`resolve_serving_targets()`/`resolve_latest_quantile_manifest()`가 반환한
경로에서 실제로 `.predict()`를 호출해 `resolution.compute_score_columns()`·
`scoring.resolve_confidence_for_stock()`이 기대하는 입력 계약(예측값
Mapping/p10·p90 튜플)을 그대로 만족하는지 검증한다.
"""

import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from analyzer.inference.predict import (
    predict_point_models,
    predict_point_models_batch,
    predict_quantile_models,
)
from analyzer.inference.resolution import (
    QuantileManifest,
    ServingPlan,
    SkipReason,
    compute_score_columns,
)
from analyzer.inference.scoring import resolve_confidence_for_stock
from analyzer.training.campaign_metrics import sidecar_path_for


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


def _trend(stock_code: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": [stock_code],
            "trade_date": [date(2026, 1, 5)],
            "foreign_net_value": [1_000_000],
            "institution_net_value": [-500_000],
            "individual_net_value": [-500_000],
            "total_trading_value": [10_000_000],
        }
    )


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


def _feature_matrix(feature_columns: list[str], n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: [0.1 * (i + 1) + 0.01 * row for row in range(n_rows)]
            for i, col in enumerate(feature_columns)
        }
    )


class TestPredictPointModelsBatch:
    """SPEC-ANALYZER-INFER-001 M6: 밴드 스윕은 booster를 그리드 크기만큼
    반복 로드하지 않고 한 번만 로드해 배치로 예측해야 한다(design.md §5)."""

    def test_returns_array_per_algorithm_matching_matrix_row_count(self, tmp_path: Path):
        feature_columns = ["roc_60", "ma_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm_batch", feature_columns, seed=20)
        xgb_path = _train_xgboost(tmp_path, "xgb_batch", feature_columns, seed=21)
        plan = _serving_plan("ensemble", {"lightgbm": lgbm_path, "xgboost": xgb_path})
        matrix = _feature_matrix(feature_columns, n_rows=5)

        predictions = predict_point_models_batch(plan, matrix)

        assert set(predictions.keys()) == {"lightgbm", "xgboost"}
        assert predictions["lightgbm"].shape == (5,)
        assert predictions["xgboost"].shape == (5,)

    def test_batch_prediction_matches_row_by_row_single_prediction(self, tmp_path: Path):
        """boundary-verification: 배치 예측 결과가 행 단위 `predict_point_models()`
        호출 결과와 정확히 일치해야 한다 — booster 로드 방식만 다를 뿐 같은
        모델·같은 입력이면 같은 출력이어야 한다."""
        feature_columns = ["roc_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm_match", feature_columns, seed=22)
        plan = _serving_plan("lightgbm", {"lightgbm": lgbm_path})
        matrix = _feature_matrix(feature_columns, n_rows=3)

        batch_predictions = predict_point_models_batch(plan, matrix)

        for i in range(3):
            row = matrix.iloc[[i]].reset_index(drop=True)
            single = predict_point_models(plan, row)
            assert batch_predictions["lightgbm"][i] == single["lightgbm"]

    def test_unknown_algorithm_key_raises_value_error(self, tmp_path: Path):
        feature_columns = ["roc_60"]
        dummy_path = _train_lightgbm(tmp_path, "dummy_batch", feature_columns, seed=23)
        plan = _serving_plan("ensemble", {"unknown_algo": dummy_path})
        matrix = _feature_matrix(feature_columns, n_rows=2)

        try:
            predict_point_models_batch(plan, matrix)
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


class TestPredictPointModelsSupplyDemandGapGuard:
    """SPEC-ANALYZER-TRAIN-META-001 M5(REQ-TM-007/008, plan.md §E M5,
    research.md §6-3): 단일종목 스코어링 경로(`predict_point_models()`)에도
    `inference/sweep.py`와 동일한 조기 스킵 방어를 제공한다 — `investor_trend`가
    결측인데 FROZEN 수급 피처가 필요하면 `_select_feature_columns()`의
    무방비 `ValueError` 대신 `SkipReason.FEATURE_INSUFFICIENT`를 반환한다.

    `investor_trend` 키워드 인자는 생략 시(기본값 `None`) 기존 동작을 완전히
    보존한다 — `pipeline.py` 등 기존 호출자는 이 milestone으로 영향받지
    않는다(REQ-TM-006 방향의 회귀 최소화 원칙과 동일 취지 — 실제 프로덕션
    오케스트레이션 배선은 INFER-001 M9 Gap 소관으로 이 SPEC의 범위 밖이다)."""

    def test_investor_trend_omitted_preserves_existing_behavior(self, tmp_path: Path):
        """회귀 가드: `investor_trend` 인자 없이 호출하면 FROZEN 컬럼이
        feature_row에 없을 때 여전히 무방비 `ValueError`가 발생해야 한다
        (기존 동작 완전 보존, REQ-TM-006 방향과 동일 취지)."""
        feature_columns = ["foreign_net_ratio", "ROC_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm_gap_omitted", feature_columns, seed=30)
        plan = _serving_plan("lightgbm", {"lightgbm": lgbm_path})
        # investor_trend 결측 상황을 흉내낸 feature_row — FROZEN 컬럼 없음.
        feature_row = _feature_row(["ROC_60"])

        try:
            predict_point_models(plan, feature_row)
            raise AssertionError("investor_trend 미지정 시 기존 ValueError가 발생해야 한다")
        except ValueError as exc:
            assert "foreign_net_ratio" in str(exc)

    def test_empty_investor_trend_with_frozen_requirement_routes_to_skip_reason(
        self, tmp_path: Path
    ):
        """재현(RED→GREEN, research.md §6-3): `investor_trend`가 명시적으로
        제공되고 비어 있으며 feature_columns가 FROZEN 컬럼을 요구하면
        `ValueError` 대신 `SkipReason.FEATURE_INSUFFICIENT`로 라우팅해야
        한다(REQ-TM-007)."""
        feature_columns = ["foreign_net_ratio", "ROC_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm_gap_skip", feature_columns, seed=31)
        plan = _serving_plan("lightgbm", {"lightgbm": lgbm_path})
        # investor_trend 결측이므로 FROZEN 컬럼이 조립되지 않은 상태 재현.
        feature_row = _feature_row(["ROC_60"])

        result = predict_point_models(plan, feature_row, investor_trend=_empty_trend())

        assert result is SkipReason.FEATURE_INSUFFICIENT

    def test_non_empty_investor_trend_still_predicts_normally(self, tmp_path: Path):
        """회귀 가드: `investor_trend`가 존재하면(도메스틱처럼) 갭이 아니므로
        기존과 동일하게 정상 예측을 수행해야 한다."""
        feature_columns = ["foreign_net_ratio", "ROC_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm_gap_present", feature_columns, seed=32)
        plan = _serving_plan("lightgbm", {"lightgbm": lgbm_path})
        feature_row = _feature_row(feature_columns)

        predictions = predict_point_models(plan, feature_row, investor_trend=_trend("A1"))

        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == {"lightgbm"}
        assert isinstance(predictions["lightgbm"], float)

    def test_genuinely_missing_non_frozen_column_still_raises_value_error(self, tmp_path: Path):
        """REQ-TM-008 회귀 가드: `investor_trend`가 제공되고 비어 있어도,
        누락된 컬럼이 FROZEN이 아닌 진짜 예상 밖 컬럼 누락이면 기존
        `ValueError` 가드가 여전히 발동해야 한다 — 신규 스킵 가드가 이
        경로를 가로채지 않는다."""
        feature_columns = ["ROC_60", "MA_60"]
        lgbm_path = _train_lightgbm(tmp_path, "lgbm_genuine_missing", feature_columns, seed=33)
        plan = _serving_plan("lightgbm", {"lightgbm": lgbm_path})
        # MA_60이 빠진 피처 행 — FROZEN과 무관한 진짜 컬럼 누락.
        feature_row = _feature_row(["ROC_60"])

        try:
            predict_point_models(plan, feature_row, investor_trend=_empty_trend())
            raise AssertionError("FROZEN과 무관한 컬럼 누락은 여전히 ValueError여야 한다")
        except ValueError as exc:
            assert "MA_60" in str(exc)


class TestPredictQuantileModelsSupplyDemandGapGuard:
    """`predict_quantile_models()`에도 동일한 조기 스킵 방어를 제공한다
    (REQ-TM-007/008) — p10/p90 두 모델 중 하나라도 FROZEN 컬럼을 요구하고
    `investor_trend`가 결측이면 스킵한다."""

    def test_empty_investor_trend_with_frozen_requirement_routes_to_skip_reason(
        self, tmp_path: Path
    ):
        feature_columns = ["foreign_net_ratio", "ROC_60"]
        p10_path = _train_lightgbm(tmp_path, "q10_gap_skip", feature_columns, seed=34)
        p90_path = _train_lightgbm(tmp_path, "q90_gap_skip", feature_columns, seed=35)
        manifest = QuantileManifest(
            market="overseas",
            horizon=20,
            trained_date=date(2026, 9, 5),
            p10_path=p10_path,
            p90_path=p90_path,
        )
        feature_row = _feature_row(["ROC_60"])

        result = predict_quantile_models(manifest, feature_row, investor_trend=_empty_trend())

        assert result is SkipReason.FEATURE_INSUFFICIENT

    def test_investor_trend_omitted_preserves_existing_behavior(self, tmp_path: Path):
        feature_columns = ["foreign_net_ratio", "ROC_60"]
        p10_path = _train_lightgbm(tmp_path, "q10_gap_omitted", feature_columns, seed=36)
        p90_path = _train_lightgbm(tmp_path, "q90_gap_omitted", feature_columns, seed=37)
        manifest = QuantileManifest(
            market="overseas",
            horizon=20,
            trained_date=date(2026, 9, 5),
            p10_path=p10_path,
            p90_path=p90_path,
        )
        feature_row = _feature_row(["ROC_60"])

        try:
            predict_quantile_models(manifest, feature_row)
            raise AssertionError("investor_trend 미지정 시 기존 ValueError가 발생해야 한다")
        except ValueError as exc:
            assert "foreign_net_ratio" in str(exc)

    def test_non_empty_investor_trend_still_predicts_normally(self, tmp_path: Path):
        feature_columns = ["foreign_net_ratio", "ROC_60"]
        p10_path = _train_lightgbm(tmp_path, "q10_gap_present", feature_columns, seed=38)
        p90_path = _train_lightgbm(tmp_path, "q90_gap_present", feature_columns, seed=39)
        manifest = QuantileManifest(
            market="overseas",
            horizon=20,
            trained_date=date(2026, 9, 5),
            p10_path=p10_path,
            p90_path=p90_path,
        )
        feature_row = _feature_row(feature_columns)

        p10, p90 = predict_quantile_models(manifest, feature_row, investor_trend=_trend("A1"))

        assert isinstance(p10, float)
        assert isinstance(p90, float)

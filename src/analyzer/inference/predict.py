"""포인트/분위수 모델 예측 실행 — feature_columns 정렬 선택 + 네이티브
`.predict()` 호출로 실제 예측값을 산출한다 (SPEC-ANALYZER-INFER-001 M5,
신규 갭 해소 — design.md §1 단계 5, REQ-AIF-040/041/050).

이 모듈이 산출하는 값이 M2의 `resolution.compute_score_columns()`(G3 score
분기)와 M3의 `scoring.resolve_confidence_for_stock()`(confidence 가드)의
실제 입력이 된다 — 두 함수는 이미 존재하며(PRESERVE, 무수정), 이 모듈은
그 함수들이 기대하는 계약(예측값 `Mapping[str, float]`, `(p10, p90)`
float 튜플)을 정확히 만족하는 값을 만들어 전달하는 역할만 한다.

`inference/features.py`의 `resolve_feature_columns()`를 재사용해 모델별로
서로 다를 수 있는 feature_columns 순서를 얻은 뒤, 조립된 피처 행에서 그
순서 그대로 선택한다 — 필수 컬럼이 없으면 조용히 0으로 채우지 않고
`ValueError`로 명시적으로 실패한다(데이터 무결성 가드).

모델 캐싱은 이번 사이클의 범위 밖이다 — 매 호출마다 디스크에서 booster를
새로 로드한다(§ Self-Verification 잔여 위험 참조).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from analyzer.inference.features import resolve_feature_columns
from analyzer.inference.resolution import QuantileManifest, ServingPlan


class _PointPredictor(Protocol):
    def __call__(self, model_path: Path, feature_row: pd.DataFrame) -> float: ...


def _select_feature_columns(
    feature_row: pd.DataFrame, feature_columns: list[str], model_path: Path
) -> pd.DataFrame:
    """`feature_columns` 순서 그대로 선택한다 — 누락 컬럼이 있으면 조용한
    0-fill 대신 `ValueError`로 실패한다(데이터 무결성 가드)."""
    missing = [c for c in feature_columns if c not in feature_row.columns]
    if missing:
        raise ValueError(f"{model_path} 예측에 필요한 피처 컬럼이 누락되었다: {missing}")
    return feature_row.loc[:, feature_columns]


def _predict_lightgbm_array(model_path: Path, feature_matrix: pd.DataFrame) -> np.ndarray:
    feature_columns = resolve_feature_columns(model_path)
    selected = _select_feature_columns(feature_matrix, feature_columns, model_path)
    booster = lgb.Booster(model_file=str(model_path))
    return np.asarray(booster.predict(selected))


def _predict_xgboost_array(model_path: Path, feature_matrix: pd.DataFrame) -> np.ndarray:
    feature_columns = resolve_feature_columns(model_path)
    selected = _select_feature_columns(feature_matrix, feature_columns, model_path)
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmatrix = xgb.DMatrix(selected)
    return np.asarray(booster.predict(dmatrix))


def _predict_lightgbm(model_path: Path, feature_row: pd.DataFrame) -> float:
    return float(_predict_lightgbm_array(model_path, feature_row)[0])


def _predict_xgboost(model_path: Path, feature_row: pd.DataFrame) -> float:
    return float(_predict_xgboost_array(model_path, feature_row)[0])


class _BatchPredictor(Protocol):
    def __call__(self, model_path: Path, feature_matrix: pd.DataFrame) -> np.ndarray: ...


_POINT_PREDICTORS: Mapping[str, _PointPredictor] = {
    "lightgbm": _predict_lightgbm,
    "xgboost": _predict_xgboost,
}

_POINT_PREDICTORS_BATCH: Mapping[str, _BatchPredictor] = {
    "lightgbm": _predict_lightgbm_array,
    "xgboost": _predict_xgboost_array,
}


def predict_point_models(serving_plan: ServingPlan, feature_row: pd.DataFrame) -> dict[str, float]:
    """`serving_plan.model_paths`에 담긴 각 알고리즘의 포인트 모델로 실제
    예측을 수행한다(design.md §1 단계 5, REQ-AIF-040/041).

    반환값은 `resolution.compute_score_columns(active_strategy, predictions)`
    의 `predictions` 인자 계약과 정확히 일치한다 — 알고리즘명을 키로,
    원 예측값(스칼라 float)을 값으로 갖는 매핑이다.
    """
    predictions: dict[str, float] = {}
    for algorithm, model_path in serving_plan.model_paths.items():
        predictor = _POINT_PREDICTORS.get(algorithm)
        if predictor is None:
            raise ValueError(f"지원하지 않는 algorithm: {algorithm!r}")
        predictions[algorithm] = predictor(model_path, feature_row)
    return predictions


def predict_point_models_batch(
    serving_plan: ServingPlan, feature_matrix: pd.DataFrame
) -> dict[str, np.ndarray]:
    """`serving_plan.model_paths`에 담긴 각 알고리즘의 booster를 **한 번만**
    로드해 `feature_matrix`(그리드 크기만큼의 행) 전체를 배치로 예측한다
    (SPEC-ANALYZER-INFER-001 M6, design.md §5, REQ-AIF-110/111).

    `predict_point_models()`와 달리 booster를 매 그리드 가격마다 다시
    로드하지 않는다 — 밴드 스윕은 종목당 수십~수백 개의 그리드 가격을
    가지므로, 행 단위 반복 대신 이 함수로 한 번에 예측해야 한다. 반환값은
    알고리즘명을 키로, `feature_matrix`와 동일한 행 수의 예측값 배열을
    값으로 갖는 매핑이다.
    """
    predictions: dict[str, np.ndarray] = {}
    for algorithm, model_path in serving_plan.model_paths.items():
        predictor = _POINT_PREDICTORS_BATCH.get(algorithm)
        if predictor is None:
            raise ValueError(f"지원하지 않는 algorithm: {algorithm!r}")
        predictions[algorithm] = predictor(model_path, feature_matrix)
    return predictions


def predict_quantile_models(
    quantile_manifest: QuantileManifest, feature_row: pd.DataFrame
) -> tuple[float, float]:
    """분위수(p10/p90) 모델 쌍으로 실제 예측을 수행한다(design.md §1 단계 5,
    REQ-AIF-050). p10/p90은 항상 LightGBM 네이티브 모델이다(REQ-AT-061/062).

    반환값은 `scoring.resolve_confidence_for_stock(quantile_manifest, score,
    p10, p90)`의 `p10`/`p90` 인자 계약과 정확히 일치한다. 두 모델은 서로
    다른 `feature_columns` 사이드카를 가질 수 있으므로(각기 다른 재학습
    이력), 동일 피처 컬럼 집합이라고 가정하지 않고 각각 독립적으로
    `resolve_feature_columns()`를 호출한다.
    """
    p10 = _predict_lightgbm(quantile_manifest.p10_path, feature_row)
    p90 = _predict_lightgbm(quantile_manifest.p90_path, feature_row)
    return p10, p90

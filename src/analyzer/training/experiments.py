"""단조 제약(Monotone Constraints) 비교 실험 (SPEC-ANALYZER-TRAIN-001 M7).

REQ-AT-110: 가격 파생 피처(FEATURE-001 `PRICE_DERIVED` 분류)에 단조
제약을 적용한 경우와 적용하지 않은 경우의 성능을 비교하는 실험을
제공한다(TECHSPEC §6.6, 1180행).

REQ-AT-111: 초기 기본값은 단조 제약 미적용이며, 비교 실험이 명확한
성능 이득을 보이는 경우에만 채택한다 — 이 모듈은 채택 여부를 자동으로
결정하지 않는다. `compare_monotone_constraints()`는 두 전략의 검증
MAE를 산출할 뿐이며, "명확한 이득"의 판단 기준과 최종 채택 결정·근거
문서화는 호출자(구현 실행 시점)의 책임이다.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from analyzer.features.classification import FeatureClass, classify_feature


@dataclass(frozen=True, slots=True)
class MonotoneConstraintExperimentResult:
    """단조 제약 적용/미적용 비교 실험 결과 — 자동 채택 결정은 포함하지 않는다."""

    unconstrained_val_mae: float
    constrained_val_mae: float

    @property
    def constrained_is_better(self) -> bool:
        """참고용 비교 신호일 뿐(REQ-AT-111) — 이 값만으로 자동 채택하지 않는다."""
        return self.constrained_val_mae < self.unconstrained_val_mae


def build_monotone_constraints(feature_names: Sequence[str]) -> list[int]:
    """`feature_names` 순서에 맞춰 LightGBM `monotone_constraints` 벡터를 만든다(REQ-AT-110).

    `PRICE_DERIVED`로 분류된 피처는 `+1`(단조 증가 제약), 그 외(`FROZEN`)는
    `0`(제약 없음)으로 지정한다. `FEATURE_REGISTRY`에 없는 이름이 섞이면
    `classify_feature()`가 `ValueError`를 그대로 전파한다(FEATURE-001
    REQ-AF-043 계승).
    """
    return [
        1 if classify_feature(name) is FeatureClass.PRICE_DERIVED else 0 for name in feature_names
    ]


def compare_monotone_constraints(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    random_state: int,
    test_size: float = 0.2,
    lgbm_params: Mapping[str, Any] | None = None,
) -> MonotoneConstraintExperimentResult:
    """단조 제약 적용/미적용 두 모델을 동일 분할로 학습해 검증 MAE를 비교한다(REQ-AT-110).

    `random_state`가 고정되면 재현 가능한 결과를 반환한다. 이 함수는
    비교 지표만 산출한다(REQ-AT-111) — 채택 여부는 결정하지 않는다.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    base_params = dict(lgbm_params or {"verbosity": -1})

    unconstrained = lgb.LGBMRegressor(**base_params)
    unconstrained.fit(x_train, y_train)
    unconstrained_pred = np.asarray(unconstrained.predict(x_val))
    unconstrained_mae = float(np.mean(np.abs(y_val - unconstrained_pred)))

    constraints = build_monotone_constraints(feature_names)
    constrained = lgb.LGBMRegressor(**base_params, monotone_constraints=constraints)
    constrained.fit(x_train, y_train)
    constrained_pred = np.asarray(constrained.predict(x_val))
    constrained_mae = float(np.mean(np.abs(y_val - constrained_pred)))

    return MonotoneConstraintExperimentResult(
        unconstrained_val_mae=unconstrained_mae, constrained_val_mae=constrained_mae
    )

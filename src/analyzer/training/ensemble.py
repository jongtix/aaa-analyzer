"""앙상블 score + confidence 산출 (SPEC-ANALYZER-TRAIN-001 M5).

REQ-AT-080/081: LightGBM/XGBoost 포인트 모델의 score 부호가 일치하면
`sign × min(|score_lgbm|, |score_xgb|)`, 불일치하면 `0`을 반환한다
(ADR-033 앙상블 규칙 그대로 구현, 재설계 금지). 개별 `lgbm_score`/
`xgb_score`도 앙상블 score와 함께 보존한다(`trading_signals.lgbm_score`/
`xgb_score` 컬럼 계약, SCHEMA-001).

REQ-AT-082/083/084: confidence = `Φ(|score_ensemble| / σ)`,
`σ = (p90 - p10) / 2.563`, `Φ`는 표준정규 누적분포함수(ADR-033 확정
공식, 재설계 금지). `score_ensemble == 0`이면 confidence는 `0.5`(방향
무정보). 분위수 예측이 크로싱(p10 > p90)을 일으키면 정렬 스왑으로
보정한다(plan.md §B 리스크5 채택안).

이 모듈은 순수 함수만 제공한다 — 신규 의존성 없이 `math.erf`로 표준정규
CDF를 계산하며, LightGBM/XGBoost 모델 자체(models.py)와는 독립적이다.
"""

import math
from dataclasses import dataclass

_SIGMA_DIVISOR = 2.563
"""ADR-033 확정 상수 — σ = (p90 - p10) / 2.563."""


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """앙상블 score와 함께 보존되는 개별 모델 score(REQ-AT-081)."""

    lgbm_score: float
    xgb_score: float
    score_ensemble: float


def compute_ensemble_score(lgbm_score: float, xgb_score: float) -> EnsembleResult:
    """LightGBM/XGBoost score를 앙상블한다(REQ-AT-080/081, AC-AT-006).

    부호가 일치하면(둘 다 양수 또는 둘 다 음수) `sign × min(|lgbm|, |xgb|)`,
    불일치하거나 둘 중 하나라도 정확히 0이면(부호 미정) `0.0`을 반환한다.
    """
    if lgbm_score > 0 and xgb_score > 0:
        score_ensemble = min(lgbm_score, xgb_score)
    elif lgbm_score < 0 and xgb_score < 0:
        score_ensemble = -min(abs(lgbm_score), abs(xgb_score))
    else:
        score_ensemble = 0.0

    return EnsembleResult(lgbm_score=lgbm_score, xgb_score=xgb_score, score_ensemble=score_ensemble)


def resolve_quantile_crossing(p10: float, p90: float) -> tuple[float, float]:
    """분위수 크로싱(p10 > p90)을 정렬 스왑으로 보정한다(REQ-AT-084).

    크로싱이 없으면 입력을 그대로 반환한다.
    """
    if p10 > p90:
        return p90, p10
    return p10, p90


def compute_confidence(score_ensemble: float, p10: float, p90: float) -> float:
    """confidence = Φ(|score_ensemble| / σ)를 계산한다(REQ-AT-082/083/084, AC-AT-007).

    `score_ensemble == 0`이면 `0.5`(방향 무정보)를 반환한다. `p10`/`p90`은
    계산 전에 `resolve_quantile_crossing()`으로 크로싱을 보정한다.
    `p10 == p90`(축퇴 분포, σ=0)이면 0-나눗셈 대신 `ValueError`를
    명시적으로 발생시킨다.
    """
    if score_ensemble == 0.0:
        return 0.5

    p10, p90 = resolve_quantile_crossing(p10, p90)
    sigma = (p90 - p10) / _SIGMA_DIVISOR
    if sigma <= 0:
        raise ValueError(f"축퇴 분위수 분포(p10={p10}, p90={p90})로는 confidence를 계산할 수 없다")

    z = abs(score_ensemble) / sigma
    return _standard_normal_cdf(z)


def _standard_normal_cdf(z: float) -> float:
    """표준정규 누적분포함수 Φ(z) — `math.erf` 기반, 신규 의존성 없이 계산."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

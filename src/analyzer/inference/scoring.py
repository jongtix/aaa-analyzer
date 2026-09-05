"""confidence 산출 가드 — `compute_confidence()`의 축퇴 분포 ValueError를
호출부에서 흡수해 배치 전체가 아니라 종목 단위 스킵으로 전환한다
(SPEC-ANALYZER-INFER-001 M3, REQ-AIF-051, design.md §3).

`training/ensemble.py`(`compute_confidence()`/`resolve_quantile_crossing()`)는
PRESERVE 대상이다 — 이 모듈은 그 함수들의 시그니처·내부 로직을 전혀
수정하지 않고 순수하게 호출부에서 예외를 흡수한다.

분위수 모델 자체가 배포되지 않은 경우(`resolve_latest_quantile_manifest()`가
`None`)와, 모델은 있지만 특정 종목의 예측 결과가 축퇴 분포(p10==p90)인
경우는 서로 다른 스킵 사유로 구분된다(REQ-AIF-051, AC-AIF-010) —
전자는 (시장,horizon) 조합 단위, 후자는 종목 단위 스킵이다.
"""

from __future__ import annotations

from analyzer.inference.resolution import QuantileManifest, SkipReason
from analyzer.training.ensemble import compute_confidence


def resolve_confidence_for_stock(
    quantile_manifest: QuantileManifest | None,
    score: float,
    p10: float,
    p90: float,
) -> float | SkipReason:
    """분위수 매니페스트 부재(조합 단위)와 축퇴 분포(종목 단위)를 구분해
    스킵 사유를 라우팅한다(REQ-AIF-051, AC-AIF-010).

    `quantile_manifest`가 `None`이면 그 (시장,horizon) 조합의 모든 종목이
    `SkipReason.QUANTILE_MISSING`으로 스킵된다 — confidence 산출 자체를
    시도하지 않는다(폴백 모델 금지, G2 원칙 계승).

    매니페스트가 있어도 개별 종목의 예측 결과가 축퇴 분포(p10==p90)면
    `compute_confidence()`가 던지는 `ValueError`를 흡수해 그 종목만
    `SkipReason.DEGENERATE_QUANTILE`로 스킵한다 — 이 함수는 순수 함수이므로
    한 종목의 축퇴 분포가 같은 배치 내 다른 종목의 호출 결과에 어떤 영향도
    주지 않는다.
    """
    if quantile_manifest is None:
        return SkipReason.QUANTILE_MISSING

    try:
        return compute_confidence(score, p10, p90)
    except ValueError:
        return SkipReason.DEGENERATE_QUANTILE

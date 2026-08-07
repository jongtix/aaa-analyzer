"""피처 가격파생/동결 분류 레지스트리 및 표준 윈도 집합.

SPEC-ANALYZER-FEATURE-001 REQ-AF-040~043(M1): 이 SPEC이 정의하는 모든 피처를
`PRICE_DERIVED`(§6.6 가격 밴드 스윕 재계산 대상) 또는 `FROZEN`(동결) 중
정확히 하나로 분류하는 조회 가능한 레지스트리를 제공한다. 분류는 피처
개별 판단이 아니라 소속 REQ 블록(020번대=PRICE_DERIVED, 030번대=FROZEN)
단위로 일괄 적용한다(spec.md §4.5) — `CORR_{w}`처럼 거래량에도 의존하는
피처라도 020번대 소속이면 예외 없이 PRICE_DERIVED로 분류한다.

레지스트리에 없는 피처명을 조회하면 `classify_feature`가 명시적으로
`ValueError`를 발생시킨다(REQ-AF-043) — 신규 피처 추가 시 분류 등록이
구조적으로 강제된다.
"""

from enum import Enum

WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

_KBAR_FEATURES: tuple[str, ...] = ("KMID", "KLEN", "KUP", "KLOW", "KSFT")
_ROLLING_STATS: tuple[str, ...] = ("ROC", "MA", "STD", "RANK", "CORR")
_INVESTOR_TYPES: tuple[str, ...] = ("foreign", "institution", "individual")


class FeatureClass(Enum):
    """§6.6 가격 밴드 스윕 재계산 여부에 따른 피처 분류(REQ-AF-040)."""

    PRICE_DERIVED = "PRICE_DERIVED"
    FROZEN = "FROZEN"


def _build_registry() -> dict[str, FeatureClass]:
    registry: dict[str, FeatureClass] = {}

    for name in _KBAR_FEATURES:
        registry[name] = FeatureClass.PRICE_DERIVED

    for stat in _ROLLING_STATS:
        for window in WINDOWS:
            registry[f"{stat}_{window}"] = FeatureClass.PRICE_DERIVED

    for investor in _INVESTOR_TYPES:
        registry[f"{investor}_net_ratio"] = FeatureClass.FROZEN
        for window in WINDOWS:
            registry[f"{investor}_net_cum_{window}"] = FeatureClass.FROZEN

    return registry


FEATURE_REGISTRY: dict[str, FeatureClass] = _build_registry()


def classify_feature(name: str) -> FeatureClass:
    """피처명을 `FEATURE_REGISTRY`에서 조회한다(REQ-AF-040).

    미등록 피처명은 조용히 누락시키지 않고 명시적으로 `ValueError`를
    발생시킨다(REQ-AF-043) — 신규 피처가 분류 없이 노출되는 것을 구조적으로
    방지한다.
    """
    try:
        return FEATURE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unregistered feature name: {name!r}") from exc

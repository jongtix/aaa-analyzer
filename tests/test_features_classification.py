"""src/analyzer/features/classification.py 분류 레지스트리 테스트
(SPEC-ANALYZER-FEATURE-001 M1).

REQ-AF-040/041/042/043(AC-AF-008/009/010)을 검증한다: 020번대(기술적) 25개
전부 PRICE_DERIVED, 030번대(수급) 15개 전부 FROZEN, 미등록 피처명은
`classify_feature`가 명시적으로 에러를 발생시킨다.
"""

import pytest

from analyzer.features.classification import (
    WINDOWS,
    FeatureClass,
    classify_feature,
)


class TestWindowsConstant:
    def test_windows_is_5_10_20_60(self):
        assert WINDOWS == (5, 10, 20, 60)


class TestTechnicalFeaturesAllPriceDerived:
    """AC-AF-008 (REQ-AF-040/041): 020번대 25개 전부 PRICE_DERIVED."""

    def _expected_names(self) -> list[str]:
        kbar = ["KMID", "KLEN", "KUP", "KLOW", "KSFT"]
        rolling = [f"{stat}_{w}" for stat in ("ROC", "MA", "STD", "RANK", "CORR") for w in WINDOWS]
        return kbar + rolling

    def test_exactly_25_technical_features_classified_price_derived(self):
        names = self._expected_names()
        assert len(names) == 25
        for name in names:
            assert classify_feature(name) is FeatureClass.PRICE_DERIVED, name


class TestSupplyDemandFeaturesAllFrozen:
    """AC-AF-009 (REQ-AF-040/042): 030번대 15개 전부 FROZEN."""

    def _expected_names(self) -> list[str]:
        ratios = [f"{investor}_net_ratio" for investor in ("foreign", "institution", "individual")]
        cumulative = [
            f"{investor}_net_cum_{w}"
            for investor in ("foreign", "institution", "individual")
            for w in WINDOWS
        ]
        return ratios + cumulative

    def test_exactly_15_supply_demand_features_classified_frozen(self):
        names = self._expected_names()
        assert len(names) == 15
        for name in names:
            assert classify_feature(name) is FeatureClass.FROZEN, name


class TestUnregisteredFeatureRaises:
    """REQ-AF-043: 레지스트리 미등록 피처명은 노출되어서는 안 된다 — 조회 시 명시적 에러."""

    def test_unknown_feature_name_raises_value_error(self):
        with pytest.raises(ValueError, match="unregistered"):
            classify_feature("NOT_A_REAL_FEATURE")

"""src/analyzer/features/supply_demand.py 수급 피처 테스트
(SPEC-ANALYZER-FEATURE-001 M3).

REQ-AF-030(순매수 비율 3종)/REQ-AF-031(N일 누적 순매수)/REQ-AF-032
(0-분모 방어)을 검증한다(AC-AF-005~007).
"""

import math

import pandas as pd
import pytest

from analyzer.features.classification import FEATURE_REGISTRY, WINDOWS
from analyzer.features.supply_demand import compute_supply_demand_features


def _investor_trend_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestNetBuyRatios:
    """AC-AF-005 (REQ-AF-030): 투자자별 순매수 비율 3종."""

    def test_ratios_match_worked_example(self):
        df = _investor_trend_frame(
            [
                {
                    "trade_date": "2026-01-02",
                    "foreign_net_value": 500_000_000,
                    "institution_net_value": -200_000_000,
                    "individual_net_value": -300_000_000,
                    "total_trading_value": 10_000_000_000,
                }
            ]
        )

        result = compute_supply_demand_features(df)
        row = result.iloc[0]

        assert row["foreign_net_ratio"] == pytest.approx(0.05, abs=1e-4)
        assert row["institution_net_ratio"] == pytest.approx(-0.02, abs=1e-4)
        assert row["individual_net_ratio"] == pytest.approx(-0.03, abs=1e-4)

    def test_original_columns_preserved(self):
        df = _investor_trend_frame(
            [
                {
                    "trade_date": "2026-01-02",
                    "foreign_net_value": 100,
                    "institution_net_value": 200,
                    "individual_net_value": -300,
                    "total_trading_value": 1000,
                }
            ]
        )

        result = compute_supply_demand_features(df)

        assert result.loc[0, "trade_date"] == "2026-01-02"


class TestCumulativeNetBuy:
    """AC-AF-006 (REQ-AF-031): N일 누적 순매수, w=5 worked example."""

    def test_foreign_net_cum_5_at_day5(self):
        values = [100, 200, -50, 300, 150]
        df = _investor_trend_frame(
            [
                {
                    "trade_date": f"2026-01-0{i + 1}",
                    "foreign_net_value": v,
                    "institution_net_value": 0,
                    "individual_net_value": 0,
                    "total_trading_value": 1_000_000,
                }
                for i, v in enumerate(values)
            ]
        )

        result = compute_supply_demand_features(df)

        assert result.iloc[4]["foreign_net_cum_5"] == pytest.approx(700)


class TestZeroDenominatorDefensiveNaN:
    """AC-AF-007 (REQ-AF-032): total_trading_value=0 → NaN, 예외/inf 없음."""

    def test_zero_total_trading_value_returns_nan_not_inf_not_exception(self):
        df = _investor_trend_frame(
            [
                {
                    "trade_date": "2026-01-02",
                    "foreign_net_value": 500_000_000,
                    "institution_net_value": 0,
                    "individual_net_value": 0,
                    "total_trading_value": 0,
                }
            ]
        )

        result = compute_supply_demand_features(df)
        ratio = result.iloc[0]["foreign_net_ratio"]

        assert math.isnan(ratio)
        assert not math.isinf(ratio)

    def test_zero_over_zero_also_returns_nan(self):
        df = _investor_trend_frame(
            [
                {
                    "trade_date": "2026-01-02",
                    "foreign_net_value": 0,
                    "institution_net_value": 0,
                    "individual_net_value": 0,
                    "total_trading_value": 0,
                }
            ]
        )

        result = compute_supply_demand_features(df)

        assert math.isnan(result.iloc[0]["foreign_net_ratio"])


class TestAllSupplyDemandFeaturesClassifiable:
    """AC-AF-010 (REQ-AF-043): 산출 컬럼 전부가 분류 레지스트리에 등록되어야 한다."""

    def test_output_columns_subset_of_registry_and_all_frozen(self):
        from analyzer.features.classification import FeatureClass

        values = [100, 200, -50, 300, 150]
        df = _investor_trend_frame(
            [
                {
                    "trade_date": f"2026-01-0{i + 1}",
                    "foreign_net_value": v,
                    "institution_net_value": v,
                    "individual_net_value": v,
                    "total_trading_value": 1_000_000,
                }
                for i, v in enumerate(values)
            ]
        )

        result = compute_supply_demand_features(df)
        produced = set(result.columns) - set(df.columns)

        assert produced.issubset(FEATURE_REGISTRY.keys())
        assert len(produced) == 15
        for col in produced:
            assert FEATURE_REGISTRY[col] is FeatureClass.FROZEN, col


class TestWindowsSharedConstant:
    def test_uses_shared_windows_constant(self):
        assert WINDOWS == (5, 10, 20, 60)

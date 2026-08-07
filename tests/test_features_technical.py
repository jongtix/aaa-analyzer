"""src/analyzer/features/technical.py 기술적/가격-거래량 피처 테스트
(SPEC-ANALYZER-FEATURE-001 M2).

REQ-AF-021(KBAR)/REQ-AF-022(ROC/MA/STD/RANK)/REQ-AF-023(CORR)/REQ-AF-024
(관측치 부족 시 NaN 전파)을 검증한다(AC-AF-001~004).
"""

import math

import pandas as pd
import pytest

from analyzer.features.classification import WINDOWS
from analyzer.features.technical import compute_technical_features


def _adjusted_price_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestKbarRatios:
    """AC-AF-001 (REQ-AF-021): KBAR 비율 5종, open_price 정규화."""

    def test_kbar_ratios_match_worked_example(self):
        df = _adjusted_price_frame(
            [
                {
                    "trade_date": "2026-01-02",
                    "open_price": 10000.0,
                    "high_price": 10500.0,
                    "low_price": 9800.0,
                    "close_price": 10300.0,
                    "volume": 1000,
                }
            ]
        )

        result = compute_technical_features(df)
        row = result.iloc[0]

        assert row["KMID"] == pytest.approx(0.03, abs=1e-4)
        assert row["KLEN"] == pytest.approx(0.07, abs=1e-4)
        assert row["KUP"] == pytest.approx(0.02, abs=1e-4)
        assert row["KLOW"] == pytest.approx(0.02, abs=1e-4)
        assert row["KSFT"] == pytest.approx(0.03, abs=1e-4)

    def test_original_columns_preserved(self):
        df = _adjusted_price_frame(
            [
                {
                    "trade_date": "2026-01-02",
                    "open_price": 10000.0,
                    "high_price": 10500.0,
                    "low_price": 9800.0,
                    "close_price": 10300.0,
                    "volume": 1000,
                }
            ]
        )

        result = compute_technical_features(df)

        assert "trade_date" in result.columns
        assert result.loc[0, "trade_date"] == "2026-01-02"


class TestRollingRocMaStdRank:
    """AC-AF-002 (REQ-AF-022): ROC/MA/STD/RANK, w=5 worked example."""

    def _six_day_frame(self) -> pd.DataFrame:
        closes = [100, 105, 110, 115, 120, 125]
        return _adjusted_price_frame(
            [
                {
                    "trade_date": f"2026-01-0{i + 1}",
                    "open_price": c,
                    "high_price": c,
                    "low_price": c,
                    "close_price": c,
                    "volume": c * 10,
                }
                for i, c in enumerate(closes)
            ]
        )

    def test_roc_ma_std_rank_at_day6(self):
        df = self._six_day_frame()

        result = compute_technical_features(df)
        row = result.iloc[5]

        assert row["ROC_5"] == pytest.approx(0.25, abs=1e-4)
        assert row["MA_5"] == pytest.approx(-0.08, abs=1e-4)
        assert row["STD_5"] == pytest.approx(0.06325, abs=1e-3)
        assert row["RANK_5"] == pytest.approx(1.0, abs=1e-4)


class TestRollingCorr:
    """AC-AF-003 (REQ-AF-023): 가격-거래량 롤링 상관계수."""

    def test_corr_5_is_perfectly_correlated(self):
        closes = [100, 105, 110, 115, 120, 125]
        volumes = [1000, 1050, 1100, 1150, 1200, 1250]
        df = _adjusted_price_frame(
            [
                {
                    "trade_date": f"2026-01-0{i + 1}",
                    "open_price": c,
                    "high_price": c,
                    "low_price": c,
                    "close_price": c,
                    "volume": v,
                }
                for i, (c, v) in enumerate(zip(closes, volumes, strict=True))
            ]
        )

        result = compute_technical_features(df)

        assert result.iloc[5]["CORR_5"] == pytest.approx(1.0, abs=1e-4)


class TestNaNPropagationOnInsufficientObservations:
    """AC-AF-004 (REQ-AF-024): 관측치 부족 시 NaN, 0/전방채움 금지."""

    def test_all_rows_nan_when_history_shorter_than_window(self):
        closes = [100, 101, 102]
        df = _adjusted_price_frame(
            [
                {
                    "trade_date": f"2026-01-0{i + 1}",
                    "open_price": c,
                    "high_price": c,
                    "low_price": c,
                    "close_price": c,
                    "volume": c * 10,
                }
                for i, c in enumerate(closes)
            ]
        )

        result = compute_technical_features(df)

        for col in ("ROC_5", "MA_5", "STD_5", "RANK_5", "CORR_5"):
            assert bool(result[col].isna().all()), col


class TestAllTechnicalFeaturesClassifiable:
    """AC-AF-010 (REQ-AF-043): 산출 컬럼 전부가 분류 레지스트리에 등록되어야 한다."""

    def test_output_columns_subset_of_registry(self):
        from analyzer.features.classification import FEATURE_REGISTRY

        closes = [100 + i for i in range(6)]
        df = _adjusted_price_frame(
            [
                {
                    "trade_date": f"2026-01-0{i + 1}",
                    "open_price": c,
                    "high_price": c,
                    "low_price": c,
                    "close_price": c,
                    "volume": c * 10,
                }
                for i, c in enumerate(closes)
            ]
        )

        result = compute_technical_features(df)
        original_columns = set(df.columns)
        produced = set(result.columns) - original_columns

        assert produced, "no new columns produced"
        assert produced.issubset(FEATURE_REGISTRY.keys())
        assert len(produced) == 25


class TestWindowsSharedConstant:
    def test_uses_shared_windows_constant(self):
        assert WINDOWS == (5, 10, 20, 60)


class TestKbarZeroRangeEdgeCase:
    """spec.md §B edge case: high_price == low_price (무변동일)."""

    def test_high_equals_low_gives_zero_klen_and_equal_kup_klow(self):
        df = _adjusted_price_frame(
            [
                {
                    "trade_date": "2026-01-02",
                    "open_price": 10000.0,
                    "high_price": 10000.0,
                    "low_price": 10000.0,
                    "close_price": 10000.0,
                    "volume": 500,
                }
            ]
        )

        result = compute_technical_features(df)
        row = result.iloc[0]

        assert row["KLEN"] == pytest.approx(0.0, abs=1e-9)
        assert row["KUP"] == pytest.approx(row["KLOW"], abs=1e-9)
        assert not math.isnan(row["KUP"])

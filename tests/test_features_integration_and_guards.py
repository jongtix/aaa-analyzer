"""SPEC-ANALYZER-FEATURE-001 M5: as-of 통합 테스트 + 순수성/스코프 가드.

AC-AF-011(as-of 무결성)/AC-AF-013(순수 함수)/§C 품질 게이트(의존성 가드)를
검증한다.
"""

import re
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from analyzer.features.supply_demand import compute_supply_demand_features
from analyzer.features.technical import compute_technical_features

_PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _investor_trend_frame(values: list[int], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": dates,
            "foreign_net_value": values,
            "institution_net_value": values,
            "individual_net_value": values,
            "total_trading_value": [1_000_000] * len(values),
        }
    )


class TestAsOfIntegrity:
    """AC-AF-011 (REQ-AF-050): 미래 investor_trend 행이 T 시점 결과에 영향을 주지 않는다."""

    def test_future_rows_do_not_change_result_at_t(self):
        dates = [f"2026-01-0{i + 1}" for i in range(8)]  # T = day5(index 4), T+3 = day8
        values = [100, 200, -50, 300, 150, 999, -999, 555]

        only_up_to_t = _investor_trend_frame(values[:5], dates[:5])
        up_to_t_plus_3 = _investor_trend_frame(values, dates)

        result_a = compute_supply_demand_features(only_up_to_t)
        result_b = compute_supply_demand_features(up_to_t_plus_3)

        row_a = result_a.iloc[4]
        row_b = result_b.iloc[4]

        for col in ("foreign_net_ratio", "foreign_net_cum_5", "institution_net_cum_5"):
            assert row_a[col] == row_b[col] or (pd.isna(row_a[col]) and pd.isna(row_b[col])), col


class TestPurity:
    """AC-AF-013 (REQ-AF-061): 순수 함수 — 동일 입력에 동일 출력, 파일 I/O 없음."""

    def test_technical_features_call_twice_identical_no_file_io(self, tmp_path):
        df = pd.DataFrame(
            {
                "trade_date": [f"2026-01-0{i + 1}" for i in range(6)],
                "open_price": [100.0 + i for i in range(6)],
                "high_price": [101.0 + i for i in range(6)],
                "low_price": [99.0 + i for i in range(6)],
                "close_price": [100.5 + i for i in range(6)],
                "volume": [1000 + i * 10 for i in range(6)],
            }
        )

        before = set(tmp_path.iterdir())
        result_1 = compute_technical_features(df)
        result_2 = compute_technical_features(df)
        after = set(tmp_path.iterdir())

        pdt.assert_frame_equal(result_1, result_2)
        assert before == after

    def test_supply_demand_features_call_twice_identical_no_file_io(self, tmp_path):
        df = _investor_trend_frame(
            [100, 200, -50, 300, 150], [f"2026-01-0{i + 1}" for i in range(5)]
        )

        before = set(tmp_path.iterdir())
        result_1 = compute_supply_demand_features(df)
        result_2 = compute_supply_demand_features(df)
        after = set(tmp_path.iterdir())

        pdt.assert_frame_equal(result_1, result_2)
        assert before == after

    def test_technical_features_does_not_mutate_input(self):
        df = pd.DataFrame(
            {
                "trade_date": ["2026-01-02"],
                "open_price": [10000.0],
                "high_price": [10500.0],
                "low_price": [9800.0],
                "close_price": [10300.0],
                "volume": [1000],
            }
        )
        original_columns = list(df.columns)

        compute_technical_features(df)

        assert list(df.columns) == original_columns


class TestDependencyGuard:
    """§C 품질 게이트: pyproject.toml에 외부 TA 라이브러리 의존성이 없어야 한다."""

    def test_no_external_ta_library_dependency(self):
        content = _PYPROJECT_PATH.read_text(encoding="utf-8")
        assert not re.search(r"ta-lib|pandas-ta|talib", content, re.IGNORECASE)

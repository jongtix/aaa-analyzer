"""src/analyzer/training/dataset.py 데이터셋 조립 테스트 (SPEC-ANALYZER-TRAIN-001 M1).

REQ-AT-020(피처+레이블 조인)/REQ-AT-022(등급 유니버스 필터)/
REQ-AT-023(시장별 시작일 필터)/REQ-AT-025(is_delisted 파생)을 검증한다.
AC-AT-002의 worked example(합성 종목 3개: A등급/C등급/상장폐지)을 그대로
구현한다.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from analyzer.data.models import TradingCalendar
from analyzer.training.dataset import assemble_dataset


def _weekdays(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _calendar(start: date, end: date) -> TradingCalendar:
    return TradingCalendar(calendar_code="TEST", trading_days=frozenset(_weekdays(start, end)))


def _ohlcv(stock_code: str, dates: list[date]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "stock_code": [stock_code] * n,
            "trade_date": dates,
            "open_price": [100.0 + i for i in range(n)],
            "high_price": [101.0 + i for i in range(n)],
            "low_price": [99.0 + i for i in range(n)],
            "close_price": [100.5 + i for i in range(n)],
            "volume": [1000] * n,
        }
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_type",
            "event_date",
            "stock_rate",
            "cash_amount",
            "event_subtype",
            "ex_dividend_date",
            "currency_code",
        ]
    )


class TestAssembleDataset:
    """AC-AT-002: 합성 종목 3개(A등급/C등급/상장폐지), 국내 시장, 2005-01-01 경계."""

    def test_ac_at_002_grade_and_start_date_filter_and_is_delisted(self):
        calendar = _calendar(date(2004, 12, 1), date(2005, 3, 1))

        a_grade_dates = _weekdays(date(2004, 12, 27), date(2005, 2, 15))
        c_grade_dates = _weekdays(date(2005, 1, 3), date(2005, 2, 15))
        delisted_dates = _weekdays(date(2005, 1, 3), date(2005, 1, 10))

        stocks = pd.DataFrame(
            {
                "stock_code": ["AGRADE", "CGRADE", "DELISTED"],
                "grade": ["A", "C", "A"],
                "delisted_at": [None, None, date(2005, 1, 11)],
            }
        )
        ohlcv_by_stock = {
            "AGRADE": _ohlcv("AGRADE", a_grade_dates),
            "CGRADE": _ohlcv("CGRADE", c_grade_dates),
            "DELISTED": _ohlcv("DELISTED", delisted_dates),
        }
        events_by_stock = {
            "AGRADE": _empty_events(),
            "CGRADE": _empty_events(),
            "DELISTED": _empty_events(),
        }

        result = assemble_dataset(
            stocks=stocks,
            ohlcv_by_stock=ohlcv_by_stock,
            events_by_stock=events_by_stock,
            investor_trend_by_stock={},
            calendar=calendar,
            market="domestic",
        )

        # REQ-AT-022: C등급 종목은 결과에서 완전히 제외되어야 한다.
        assert "CGRADE" not in set(result["stock_code"])

        # REQ-AT-023: A등급 종목의 2005-01-01 이전 행은 제외되어야 한다.
        a_rows = result.loc[result["stock_code"] == "AGRADE"]
        assert not a_rows.empty
        assert a_rows["trade_date"].min() >= date(2005, 1, 1)
        pre_2005_count = sum(1 for d in a_grade_dates if d < date(2005, 1, 1))
        assert pre_2005_count > 0  # 합성 데이터가 실제로 경계 이전 행을 포함하는지 자체 점검
        assert len(a_rows) == len(a_grade_dates) - pre_2005_count

        # REQ-AT-025: 상장폐지 종목은 is_delisted=True가 compute_labels()에
        # 전달되어 exclude_reason="delisted"가 반영되어야 한다.
        delisted_rows = result.loc[result["stock_code"] == "DELISTED"].sort_values("trade_date")
        assert not delisted_rows.empty
        first_row = delisted_rows.iloc[0]
        assert first_row["label_D20_exclude_reason"] == "delisted"

    def test_stock_without_ohlcv_data_is_skipped(self):
        calendar = _calendar(date(2005, 1, 1), date(2005, 3, 1))
        stocks = pd.DataFrame({"stock_code": ["NODATA"], "grade": ["A"], "delisted_at": [None]})

        result = assemble_dataset(
            stocks=stocks,
            ohlcv_by_stock={},
            events_by_stock={},
            investor_trend_by_stock={},
            calendar=calendar,
            market="domestic",
        )

        assert result.empty

    def test_overseas_market_uses_overseas_start_date(self):
        """REQ-AT-023: market="overseas"는 2007-08-20 기준으로 필터링한다."""
        calendar = _calendar(date(2007, 8, 1), date(2007, 10, 1))
        dates = _weekdays(date(2007, 8, 13), date(2007, 9, 14))
        stocks = pd.DataFrame({"stock_code": ["OVERSEAS1"], "grade": ["B"], "delisted_at": [None]})

        result = assemble_dataset(
            stocks=stocks,
            ohlcv_by_stock={"OVERSEAS1": _ohlcv("OVERSEAS1", dates)},
            events_by_stock={"OVERSEAS1": _empty_events()},
            investor_trend_by_stock={},
            calendar=calendar,
            market="overseas",
        )

        assert result["trade_date"].min() >= date(2007, 8, 20)


class TestAssembleDatasetSupplyDemand:
    """REQ-AT-020: investor_trend가 있는 종목은 수급 피처가 병합되어야 한다."""

    def test_merges_supply_demand_features_when_investor_trend_present(self):
        calendar = _calendar(date(2005, 1, 1), date(2005, 3, 1))
        dates = _weekdays(date(2005, 1, 3), date(2005, 2, 1))
        stocks = pd.DataFrame({"stock_code": ["A1"], "grade": ["A"], "delisted_at": [None]})
        n = len(dates)
        trend = pd.DataFrame(
            {
                "stock_code": ["A1"] * n,
                "trade_date": dates,
                "foreign_net_value": [1000.0] * n,
                "institution_net_value": [500.0] * n,
                "individual_net_value": [-1500.0] * n,
                "total_trading_value": [10000.0] * n,
            }
        )

        result = assemble_dataset(
            stocks=stocks,
            ohlcv_by_stock={"A1": _ohlcv("A1", dates)},
            events_by_stock={"A1": _empty_events()},
            investor_trend_by_stock={"A1": trend},
            calendar=calendar,
            market="domestic",
        )

        assert "foreign_net_ratio" in result.columns

    def test_no_investor_trend_data_skips_supply_demand_features(self):
        calendar = _calendar(date(2005, 1, 1), date(2005, 3, 1))
        dates = _weekdays(date(2005, 1, 3), date(2005, 2, 1))
        stocks = pd.DataFrame({"stock_code": ["A1"], "grade": ["A"], "delisted_at": [None]})

        result = assemble_dataset(
            stocks=stocks,
            ohlcv_by_stock={"A1": _ohlcv("A1", dates)},
            events_by_stock={"A1": _empty_events()},
            investor_trend_by_stock={},
            calendar=calendar,
            market="domestic",
        )

        assert "foreign_net_ratio" not in result.columns
        assert "label_D20" in result.columns
        assert "label_D60" in result.columns


if __name__ == "__main__":
    pytest.main([__file__])

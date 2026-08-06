"""src/analyzer/data/models.py 타입 인터페이스 테스트 (SPEC-ANALYZER-DATA-001 M1).

daily_ohlcv/corporate_events/market_calendar 읽기 전용 표현(raw SQL + pandas
파이프라인 대상)과, 조정 엔진 시그니처가 받는 TradingCalendar 컨테이너 타입을
검증한다. 이 M1 단계에서는 타입 구조만 확정하며 거래일 판정 로직(prevTradingDay 등)
은 아직 구현하지 않는다(M2 소관).
"""

from datetime import date

import pytest

from analyzer.data.models import (
    CorporateEventRow,
    DailyOhlcvRow,
    MarketCalendarRow,
    TradingCalendar,
)


class TestDailyOhlcvRow:
    def test_constructs_with_expected_fields(self):
        row = DailyOhlcvRow(
            stock_code="AAPL",
            trade_date=date(2020, 8, 28),
            open_price=499.23,
            high_price=500.00,
            low_price=498.00,
            close_price=499.23,
            volume=1000,
        )

        assert row.stock_code == "AAPL"
        assert row.trade_date == date(2020, 8, 28)
        assert row.close_price == 499.23
        assert row.volume == 1000

    def test_is_frozen(self):
        row = DailyOhlcvRow(
            stock_code="AAPL",
            trade_date=date(2020, 8, 28),
            open_price=499.23,
            high_price=500.00,
            low_price=498.00,
            close_price=499.23,
            volume=1000,
        )

        with pytest.raises(AttributeError):
            row.close_price = 0.0  # type: ignore[misc]


class TestCorporateEventRow:
    def test_constructs_split_event(self):
        row = CorporateEventRow(
            stock_code="AAPL",
            event_type="SPLIT",
            event_date=date(2020, 8, 31),
            stock_rate=4.0,
            cash_amount=None,
            event_subtype=None,
            ex_dividend_date=None,
            currency_code="USD",
        )

        assert row.event_type == "SPLIT"
        assert row.stock_rate == 4.0
        assert row.cash_amount is None

    def test_constructs_dividend_event(self):
        row = CorporateEventRow(
            stock_code="005930",
            event_type="DIVIDEND",
            event_date=date(2023, 12, 31),
            stock_rate=None,
            cash_amount=500.0,
            event_subtype="결산",
            ex_dividend_date=None,
            currency_code="KRW",
        )

        assert row.event_type == "DIVIDEND"
        assert row.cash_amount == 500.0
        assert row.ex_dividend_date is None


class TestMarketCalendarRow:
    def test_constructs_with_expected_fields(self):
        row = MarketCalendarRow(calendar_code="KRX", cal_date=date(2023, 12, 27), is_open=True)

        assert row.calendar_code == "KRX"
        assert row.is_open is True


class TestTradingCalendar:
    """M1은 컨테이너 타입만 확정한다 — 거래일 판정 메서드는 M2에서 추가된다."""

    def test_constructs_with_calendar_code_and_trading_days(self):
        calendar = TradingCalendar(
            calendar_code="KRX",
            trading_days=frozenset({date(2023, 12, 26), date(2023, 12, 27)}),
        )

        assert calendar.calendar_code == "KRX"
        assert date(2023, 12, 27) in calendar.trading_days

    def test_is_frozen(self):
        calendar = TradingCalendar(calendar_code="KRX", trading_days=frozenset())

        with pytest.raises(AttributeError):
            calendar.calendar_code = "NYSE"  # type: ignore[misc]

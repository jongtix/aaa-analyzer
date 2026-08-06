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


class TestTradingCalendarJudgmentMethods:
    """M2(spec.md §4.2): 거래일 판정 메서드 — DB 비의존, 순수 함수."""

    @staticmethod
    def _build_calendar() -> TradingCalendar:
        # 2023-12: 26(화)/27(수)/28(목) 거래, 29(금)/30(토)/31(일) 휴장(연말 폐장).
        # 2024-02: 7(수)/8(목)/9(금) 거래, 10~12(토~월) 연휴 휴장, 13(화) 거래.
        trading_days = frozenset(
            {
                date(2023, 12, 26),
                date(2023, 12, 27),
                date(2023, 12, 28),
                date(2024, 2, 7),
                date(2024, 2, 8),
                date(2024, 2, 9),
                date(2024, 2, 13),
            }
        )
        return TradingCalendar(calendar_code="KRX", trading_days=trading_days)

    def test_is_trading_day_true_for_trading_day(self):
        calendar = self._build_calendar()

        assert calendar.is_trading_day(date(2023, 12, 27)) is True

    def test_is_trading_day_false_for_closed_day(self):
        calendar = self._build_calendar()

        assert calendar.is_trading_day(date(2023, 12, 31)) is False

    def test_last_trading_day_on_or_before_returns_self_when_trading_day(self):
        calendar = self._build_calendar()

        assert calendar.last_trading_day_on_or_before(date(2023, 12, 28)) == date(2023, 12, 28)

    def test_last_trading_day_on_or_before_skips_year_end_closure(self):
        calendar = self._build_calendar()

        assert calendar.last_trading_day_on_or_before(date(2023, 12, 31)) == date(2023, 12, 28)

    def test_prev_trading_day_excludes_self(self):
        calendar = self._build_calendar()

        assert calendar.prev_trading_day(date(2023, 12, 28)) == date(2023, 12, 27)

    def test_prev_trading_day_skips_consecutive_holiday_cluster(self):
        calendar = self._build_calendar()

        assert calendar.prev_trading_day(date(2024, 2, 13)) == date(2024, 2, 9)

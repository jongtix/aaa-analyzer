"""src/analyzer/data/dividend.py 배당락일(ex_date) 파생 함수 테스트 (SPEC-ANALYZER-DATA-001 M2).

REQ-AD-031(파생 공식)·REQ-AD-032(T+N config 구동)·spec.md §4.2 worked example
(AC-AD-004/AC-AD-010)을 검증한다. DB 접근 없이 `TradingCalendar` 순수 함수로만
동작한다.
"""

from datetime import date

from analyzer.data.dividend import derive_ex_date
from analyzer.data.models import TradingCalendar


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


class TestDeriveExDateWorkedExample:
    """AC-AD-004: 2023-12-31(일, 연말 결산배당) → 2023-12-27."""

    def test_year_end_sunday_record_date_derives_to_2023_12_27(self):
        calendar = _build_calendar()

        ex_date = derive_ex_date(record_date=date(2023, 12, 31), calendar=calendar, market="KRX")

        assert ex_date == date(2023, 12, 27)

    def test_naive_one_business_day_before_formula_would_be_wrong(self):
        """naive `record_date - 1영업일` 공식(2023-12-28)과 달라야 한다(spec.md §4.2)."""
        calendar = _build_calendar()

        ex_date = derive_ex_date(record_date=date(2023, 12, 31), calendar=calendar, market="KRX")

        assert ex_date != date(2023, 12, 28)


class TestDeriveExDateWeekdayReference:
    """기준일 자체가 거래일인 평일 케이스."""

    def test_weekday_trading_day_record_date(self):
        calendar = _build_calendar()

        ex_date = derive_ex_date(record_date=date(2024, 2, 9), calendar=calendar, market="KRX")

        # lastTradingDayOnOrBefore(2024-02-09) = 2024-02-09(자기 자신, 거래일).
        # prevTradingDay(2024-02-09) = 2024-02-08.
        assert ex_date == date(2024, 2, 8)


class TestDeriveExDateConsecutiveHolidayCluster:
    """연속 휴장(설 연휴 등) 케이스."""

    def test_record_date_inside_holiday_cluster(self):
        calendar = _build_calendar()

        ex_date = derive_ex_date(record_date=date(2024, 2, 12), calendar=calendar, market="KRX")

        # lastTradingDayOnOrBefore(2024-02-12, 월/연휴) = 2024-02-09(금).
        # prevTradingDay(2024-02-09) = 2024-02-08.
        assert ex_date == date(2024, 2, 8)


class TestDeriveExDateConfigDriven:
    """AC-AD-010: 결제주기 config N을 변경하면 파생 ex_date가 달라져야 한다(REQ-AD-032)."""

    def test_t_plus_2_default_matches_ac_ad_004(self, monkeypatch):
        monkeypatch.delenv("DIVIDEND_SETTLEMENT_DAYS_KRX", raising=False)
        calendar = _build_calendar()

        ex_date = derive_ex_date(record_date=date(2023, 12, 31), calendar=calendar, market="KRX")

        assert ex_date == date(2023, 12, 27)

    def test_t_plus_1_override_simplifies_to_last_trading_day_on_or_before(self, monkeypatch):
        monkeypatch.setenv("DIVIDEND_SETTLEMENT_DAYS_KRX", "1")
        calendar = _build_calendar()

        ex_date = derive_ex_date(record_date=date(2023, 12, 31), calendar=calendar, market="KRX")

        assert ex_date == date(2023, 12, 28)

    def test_different_n_produces_different_ex_date(self, monkeypatch):
        """같은 픽스처에서 N만 바꾸면 결과가 달라져야 한다 — 하드코딩이면 이 테스트가 FAIL한다."""
        calendar = _build_calendar()

        monkeypatch.setenv("DIVIDEND_SETTLEMENT_DAYS_KRX", "2")
        ex_date_t2 = derive_ex_date(record_date=date(2023, 12, 31), calendar=calendar, market="KRX")

        monkeypatch.setenv("DIVIDEND_SETTLEMENT_DAYS_KRX", "1")
        ex_date_t1 = derive_ex_date(record_date=date(2023, 12, 31), calendar=calendar, market="KRX")

        assert ex_date_t2 != ex_date_t1
        assert ex_date_t2 == date(2023, 12, 27)
        assert ex_date_t1 == date(2023, 12, 28)

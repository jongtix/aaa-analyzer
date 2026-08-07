"""src/analyzer/labels/core.py 핵심 계산 테스트 (SPEC-ANALYZER-LABEL-001 M2).

REQ-AL-020/021/022(레이블 핵심 계산)·REQ-AL-030/031(영업일 카운팅)을
검증한다. AC-AL-001(정상 케이스)·AC-AL-002(T+H 기준 조정가)·
AC-AL-003(영업일 기준 T+H 산출)·AC-AL-004(캘린더 범위 밖 NaN)·
AC-AL-012(as_of_date 비율 불변성)의 대상이다.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from analyzer.data.adjustment import adjust_prices
from analyzer.data.models import TradingCalendar
from analyzer.labels.core import compute_labels, nth_trading_day_on_or_after


def _weekdays(start: date, end: date) -> list[date]:
    """[start, end] 구간(포함)의 월~금 날짜 목록을 반환한다(주말 제외 합성 캘린더)."""
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _weekday_calendar(start: date, end: date, calendar_code: str = "TEST") -> TradingCalendar:
    return TradingCalendar(
        calendar_code=calendar_code, trading_days=frozenset(_weekdays(start, end))
    )


def _ohlcv_df(dates_prices: list[tuple[date, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [d for d, _ in dates_prices],
            "close_price": [p for _, p in dates_prices],
            "volume": [100] * len(dates_prices),
        }
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["event_type", "event_date", "stock_rate"])


class TestNthTradingDayOnOrAfter:
    """REQ-AL-030/031: nth_trading_day_on_or_after 직접 단위 테스트."""

    def test_start_itself_counts_as_first_when_trading_day(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 1, 31))

        assert nth_trading_day_on_or_after(cal, date(2024, 1, 1), 1) == date(2024, 1, 1)

    def test_ac_al_003_five_trading_days_skips_weekend(self):
        """AC-AL-003: T=월요일, H=5영업일 → T+H는 정확히 7 달력일 후(다음 주 월요일)."""
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 1, 31))
        t = date(2024, 1, 1)  # Monday

        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 5)

        assert target is not None
        assert target == date(2024, 1, 8)  # 다음 주 월요일
        assert (target - t).days == 7  # 단순 +5 달력일(토요일)이 아님을 실증

    def test_ac_al_004_out_of_calendar_range_returns_none(self):
        """AC-AL-004: 캘린더 마지막 개장일이 T로부터 10영업일 후까지만 존재하면 H=20은 None."""
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 1, 15))
        t = date(2024, 1, 1)

        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)

        assert target is None

    def test_rejects_non_positive_n(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 1, 31))

        with pytest.raises(ValueError):
            nth_trading_day_on_or_after(cal, date(2024, 1, 1), 0)

    def test_empty_calendar_returns_none(self):
        cal = TradingCalendar(calendar_code="TEST", trading_days=frozenset())

        assert nth_trading_day_on_or_after(cal, date(2024, 1, 1), 1) is None


class TestAcAl001NormalCaseLabel:
    """AC-AL-001: 정지·상장폐지·미완결 없는 정상 케이스 레이블 계산."""

    def test_normal_case_label_matches_ratio_formula(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 1))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)  # sanity: 20영업일 후

        df = _ohlcv_df([(t, 10000.0), (target, 10500.0)])
        events = _empty_events()

        result = compute_labels(df, events, cal, horizons=(20,))

        row = result.loc[result["trade_date"] == t].iloc[0]
        assert row["label_D20"] == pytest.approx(0.05, abs=1e-4)
        assert row["label_D20_exclude_reason"] is None


class TestAcAl002SplitAdjustedAsOfTargetLabel:
    """AC-AL-002: T~T+H 사이 SPLIT 이벤트 → 양쪽 모두 as_of_date=T+H 조정가 사용."""

    def test_split_between_t_and_target_is_reflected_in_label(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 1))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)

        df = _ohlcv_df([(t, 20000.0), (target, 11000.0)])
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT"],
                "event_date": [date(2024, 1, 15)],  # T < event_date <= target
                "stock_rate": [2.0],
            }
        )

        result = compute_labels(df, events, cal, horizons=(20,))

        row = result.loc[result["trade_date"] == t].iloc[0]
        # 원주가 그대로 계산했다면 11000/20000-1=-0.45(왜곡값) — 조정가 기준은 0.10.
        assert row["label_D20"] == pytest.approx(0.10, abs=1e-4)
        assert row["label_D20"] != pytest.approx(-0.45, abs=1e-4)
        assert row["label_D20_exclude_reason"] is None


class TestAcAl012AsOfDateRatioInvariance:
    """AC-AL-012: as_of_date를 T+H로 정확히 고정하든, 그 이후 임의 날짜로 하든 비율은 동일.

    plan.md §B 리스크 1 채택안(정확히 T+H 고정)은 core.py의 실제 구현
    선택이므로, 이 테스트는 `adjust_prices()`를 직접 호출해 두 as_of_date
    선택이 수학적으로 동치임을 실증한다(spec.md §4.1 근거).
    """

    def test_as_of_target_vs_as_of_later_date_yields_identical_ratio(self):
        t = date(2024, 1, 1)
        target = date(2024, 1, 29)
        later_as_of = date(2024, 3, 1)  # target 이후의 임의 날짜(종목 최신 관측일 상정)
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 1))

        df = _ohlcv_df([(t, 20000.0), (target, 11000.0)])
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT", "SPLIT"],
                # event1: T~target 사이. event2: target 이후 — 양쪽 가격에
                # 동일하게 적용되어 비율에서 상쇄되어야 한다(spec.md §4.1).
                "event_date": [date(2024, 1, 15), date(2024, 2, 5)],
                "stock_rate": [2.0, 3.0],
            }
        )

        adjusted_a = adjust_prices(df, events, as_of_date=target, calendar=cal)
        adjusted_b = adjust_prices(df, events, as_of_date=later_as_of, calendar=cal)

        def _ratio(adjusted: pd.DataFrame) -> float:
            start = adjusted.loc[adjusted["trade_date"] == t, "close_price"].iloc[0]
            end = adjusted.loc[adjusted["trade_date"] == target, "close_price"].iloc[0]
            return float(end / start - 1)

        ratio_a = _ratio(adjusted_a)
        ratio_b = _ratio(adjusted_b)

        assert ratio_a == pytest.approx(0.10, abs=1e-4)
        assert ratio_b == pytest.approx(ratio_a, abs=1e-6)


class TestComputeLabelsEmptyInput:
    def test_empty_df_returns_expected_columns_without_error(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 1, 31))
        df = pd.DataFrame(columns=["trade_date", "close_price", "volume"])
        events = _empty_events()

        result = compute_labels(df, events, cal, horizons=(20, 60))

        assert list(result.columns) == [
            "trade_date",
            "close_price",
            "volume",
            "label_D20",
            "label_D20_exclude_reason",
            "label_D60",
            "label_D60_exclude_reason",
        ]
        assert result.empty

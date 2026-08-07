"""src/analyzer/labels/halts.py 거래정지 판정 테스트 (SPEC-ANALYZER-LABEL-001 M3).

REQ-AL-040(결측 기반 정지 판정)·REQ-AL-041(롤포워드)·REQ-AL-042/043(NaN +
exclude_reason)을 검증한다. AC-AL-005(롤포워드)·AC-AL-006(장기 정지 NaN)·
acceptance.md §B 경계 사례(정지==horizon, 상장일 이전 오판 방지)의 대상이다.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from analyzer.data.models import TradingCalendar
from analyzer.labels.core import compute_labels, nth_trading_day_on_or_after
from analyzer.labels.halts import analyze_halt


def _weekdays(start: date, end: date) -> list[date]:
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


class TestAnalyzeHaltTargetAlreadyObserved:
    def test_target_in_observed_dates_returns_empty_segment(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 1, 31))
        target = date(2024, 1, 10)
        observed = frozenset({date(2024, 1, 1), target})

        info = analyze_halt(cal, observed, target, first_obs=date(2024, 1, 1), last_obs=target)

        assert info.segment == ()
        assert info.resumed_at == target


class TestAnalyzeHaltShortGap:
    def test_three_day_gap_resolves_at_fourth_day(self):
        """AC-AL-005 worked example 축소판: D~D+2 결측, D+3에 재개."""
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 2, 1))
        d = date(2024, 1, 29)  # Monday
        d_plus_3 = date(2024, 2, 1)  # Thursday (D+1,D+2,D+3영업일 이동)
        observed = frozenset({date(2024, 1, 26), d_plus_3})  # D 직전 관측일 + 재개일

        info = analyze_halt(cal, observed, d, first_obs=date(2024, 1, 1), last_obs=d_plus_3)

        assert info.segment == (date(2024, 1, 29), date(2024, 1, 30), date(2024, 1, 31))
        assert len(info.segment) == 3
        assert info.resumed_at == d_plus_3


class TestAnalyzeHaltPreListingBoundary:
    """acceptance.md §B: 상장일 이전 구간을 정지로 오판하지 않아야 한다(plan.md §B 리스크 2)."""

    def test_backward_walk_stops_at_first_obs_not_pre_listing_days(self):
        # 2023-12-25~29(상장 전, 시장은 개장) + 2024-01-01(first_obs, 관측)
        # + 01-02~05, 01-08(결측=정지) + 01-09(재개, last_obs).
        pre_listing = _weekdays(date(2023, 12, 25), date(2023, 12, 29))
        cal = TradingCalendar(
            calendar_code="TEST",
            trading_days=frozenset(
                [
                    *pre_listing,
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 4),
                    date(2024, 1, 5),
                    date(2024, 1, 8),
                    date(2024, 1, 9),
                ]
            ),
        )
        first_obs = date(2024, 1, 1)
        last_obs = date(2024, 1, 9)
        observed = frozenset({first_obs, last_obs})  # 01-02~05, 01-08은 결측(정지)
        target = date(2024, 1, 8)

        info = analyze_halt(cal, observed, target, first_obs=first_obs, last_obs=last_obs)

        assert date(2023, 12, 29) not in info.segment  # 상장 전 구간은 정지가 아니다
        assert date(2023, 12, 22) not in info.segment
        assert info.segment == (
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
            date(2024, 1, 8),
        )
        assert info.resumed_at == last_obs


class TestAnalyzeHaltResumedAtNoneWhenCalendarOmitsLastObs:
    """방어적 경계 — `market_calendar`가 `last_obs`를 개장일로 인식하지 못하는
    (캘린더 데이터가 최신 관측일을 아직 반영하지 못한) 데이터 불일치 상황에서도
    예외 없이 `resumed_at=None`을 반환해야 한다(core.py의 방어적 fallback 대상)."""

    def test_resumed_at_is_none_when_last_obs_is_outside_calendar_range(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 1, 31))  # Feb1 미포함
        continuous = _weekdays(date(2024, 1, 1), date(2024, 1, 26))
        observed = frozenset({*continuous, date(2024, 2, 1)})
        target = date(2024, 1, 29)
        last_obs = date(2024, 2, 1)  # 캘린더 범위 밖이지만 실제로는 관측된 날

        info = analyze_halt(cal, observed, target, first_obs=date(2024, 1, 1), last_obs=last_obs)

        assert info.segment == (date(2024, 1, 29), date(2024, 1, 30), date(2024, 1, 31))
        assert info.resumed_at is None


class TestAcAl005ComputeLabelsRollforward:
    """AC-AL-005: 정지 < horizon → 롤포워드된 실제 가격으로 레이블 계산."""

    def test_short_halt_rolls_forward_to_next_actual_trade(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 2, 29))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)

        continuous = _weekdays(date(2024, 1, 1), date(2024, 1, 26))  # T~정지 직전
        resumption = date(2024, 2, 1)  # target(01-29)+3영업일 후 재개
        dates_prices = [(d, 10000.0) for d in continuous] + [(resumption, 9800.0)]
        df = _ohlcv_df(dates_prices)
        events = _empty_events()

        result = compute_labels(df, events, cal, horizons=(20,))

        row = result.loc[result["trade_date"] == t].iloc[0]
        assert row["label_D20"] == pytest.approx(9800.0 / 10000.0 - 1, abs=1e-6)
        assert row["label_D20_exclude_reason"] is None


class TestAcAl006ComputeLabelsLongHaltNaN:
    """AC-AL-006: 정지 >= horizon → NaN + exclude_reason='halted', 롤포워드하지 않는다."""

    def test_25_day_halt_with_horizon_20_returns_nan_halted(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 5, 31))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)

        continuous = _weekdays(date(2024, 1, 1), date(2024, 1, 26))
        halt_run = _weekdays(date(2024, 1, 29), date(2024, 5, 31))
        halt_days = halt_run[:25]
        resumption = halt_run[25]
        dates_prices = [(d, 10000.0) for d in continuous] + [(resumption, 9000.0)]
        df = _ohlcv_df(dates_prices)
        events = _empty_events()

        result = compute_labels(df, events, cal, horizons=(20,))

        row = result.loc[result["trade_date"] == t].iloc[0]
        assert pd.isna(row["label_D20"])
        assert row["label_D20_exclude_reason"] == "halted"
        assert len(halt_days) == 25  # sanity: 결측 구간이 실제로 20 이상

    def test_halt_length_exactly_equals_horizon_is_nan_not_rollforward(self):
        """acceptance.md §B: 정지 기간 == horizon → '미만'이 아니므로 NaN(경계값 <= 오적용 방지)."""
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 5, 31))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)

        continuous = _weekdays(date(2024, 1, 1), date(2024, 1, 26))
        halt_run = _weekdays(date(2024, 1, 29), date(2024, 5, 31))
        halt_days = halt_run[:20]  # 정확히 horizon과 동일한 길이
        resumption = halt_run[20]
        dates_prices = [(d, 10000.0) for d in continuous] + [(resumption, 9500.0)]
        df = _ohlcv_df(dates_prices)
        events = _empty_events()

        result = compute_labels(df, events, cal, horizons=(20,))

        row = result.loc[result["trade_date"] == t].iloc[0]
        assert pd.isna(row["label_D20"])
        assert row["label_D20_exclude_reason"] == "halted"
        assert len(halt_days) == 20

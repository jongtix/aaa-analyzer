"""src/analyzer/labels/core.py 데이터 가용범위 제한 통합 테스트 (SPEC-ANALYZER-LABEL-001 M4).

REQ-AL-050(단일 NaN 규칙)·REQ-AL-051(상장폐지/최근미완결 사유 구분)·
REQ-AL-052(CRSP식 페널티 미적용)을 검증한다. AC-AL-007(상장폐지)·
AC-AL-008(최근 미완결)·acceptance.md §B 경계 사례(롤포워드 결과가 마지막
관측일보다 미래 → REQ-AL-050 적용, plan.md §B 리스크 3)의 대상이다.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from analyzer.data.models import TradingCalendar
from analyzer.labels.core import compute_labels, nth_trading_day_on_or_after


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


class TestAcAl007DelistedNoPenalty:
    """AC-AL-007: T+H가 상장폐지 종목의 마지막 관측일보다 미래 → NaN + 'delisted', 페널티 미적용."""

    def test_delisted_returns_nan_with_delisted_reason_not_penalty_value(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 1))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)

        # 마지막 관측일이 T 하루 뿐 — target(T+H)보다 훨씬 이전.
        df = _ohlcv_df([(t, 10000.0)])
        events = _empty_events()

        result = compute_labels(df, events, cal, horizons=(20,), is_delisted=True)

        row = result.loc[result["trade_date"] == t].iloc[0]
        assert pd.isna(row["label_D20"])
        assert row["label_D20_exclude_reason"] == "delisted"
        # CRSP식 페널티 값이 아님을 명시 검증(REQ-AL-052).
        assert row["label_D20"] != pytest.approx(-0.30, abs=1e-6)
        assert row["label_D20"] != pytest.approx(-0.55, abs=1e-6)


class TestAcAl008InsufficientFutureDataDistinctFromDelisted:
    """AC-AL-008: 상장폐지가 아니면 동일 NaN이되 exclude_reason으로 구분된다."""

    def test_not_delisted_returns_nan_with_insufficient_future_data_reason(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 1))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)

        df = _ohlcv_df([(t, 10000.0)])
        events = _empty_events()

        result = compute_labels(df, events, cal, horizons=(20,), is_delisted=False)

        row = result.loc[result["trade_date"] == t].iloc[0]
        assert pd.isna(row["label_D20"])
        assert row["label_D20_exclude_reason"] == "insufficient_future_data"

    def test_delisted_and_not_delisted_share_nan_but_differ_in_reason(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 1))
        t = date(2024, 1, 1)
        df = _ohlcv_df([(t, 10000.0)])
        events = _empty_events()

        delisted_result = compute_labels(df, events, cal, horizons=(20,), is_delisted=True)
        pending_result = compute_labels(df, events, cal, horizons=(20,), is_delisted=False)

        delisted_row = delisted_result.loc[delisted_result["trade_date"] == t].iloc[0]
        pending_row = pending_result.loc[pending_result["trade_date"] == t].iloc[0]

        assert pd.isna(delisted_row["label_D20"]) and pd.isna(pending_row["label_D20"])
        assert delisted_row["label_D20_exclude_reason"] != pending_row["label_D20_exclude_reason"]


class TestRollforwardBeyondLastObsRoutesToAvailabilityLimit:
    """acceptance.md §B / plan.md §B 리스크 3: 정지 해제가 관측되지 않고 데이터가
    끝나는 경우(target > last_obs) 롤포워드가 아니라 REQ-AL-050이 적용되어야 한다.
    """

    def test_halt_never_resolved_before_data_ends_is_availability_limit_not_rollforward(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 5, 31))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)

        # T~정지 직전(01-26)까지는 정상 관측되지만, 그 이후로는 데이터가
        # 아예 존재하지 않는다(정지 해제가 관측 안 됨 → 상장폐지/데이터 종료).
        continuous = _weekdays(date(2024, 1, 1), date(2024, 1, 26))
        df = _ohlcv_df([(d, 10000.0) for d in continuous])
        events = _empty_events()

        result_delisted = compute_labels(df, events, cal, horizons=(20,), is_delisted=True)
        result_pending = compute_labels(df, events, cal, horizons=(20,), is_delisted=False)

        row_delisted = result_delisted.loc[result_delisted["trade_date"] == t].iloc[0]
        row_pending = result_pending.loc[result_pending["trade_date"] == t].iloc[0]

        assert pd.isna(row_delisted["label_D20"])
        assert row_delisted["label_D20_exclude_reason"] == "delisted"
        assert pd.isna(row_pending["label_D20"])
        assert row_pending["label_D20_exclude_reason"] == "insufficient_future_data"


class TestComputeLabelsDefensiveFallbackWhenCalendarOmitsLastObs:
    """core.py의 `halt_info.resumed_at is None` 방어적 fallback — `market_calendar`가
    아직 최신 거래일(`last_obs`)을 개장일로 반영하지 못한 데이터 불일치 상황에서도
    예외 없이 NaN + 적절한 사유를 반환해야 한다(target <= last_obs가 이론상 보장되므로
    정상 데이터에서는 도달하지 않는 경로).
    """

    def test_returns_nan_without_raising_when_last_obs_outside_calendar_range(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 1, 31))  # Feb1 미포함
        t = date(2024, 1, 1)
        continuous = _weekdays(date(2024, 1, 1), date(2024, 1, 26))
        dates_prices = [(d, 10000.0) for d in continuous] + [(date(2024, 2, 1), 9500.0)]
        df = _ohlcv_df(dates_prices)
        events = _empty_events()

        result = compute_labels(df, events, cal, horizons=(20,), is_delisted=False)

        row = result.loc[result["trade_date"] == t].iloc[0]
        assert pd.isna(row["label_D20"])
        assert row["label_D20_exclude_reason"] == "insufficient_future_data"

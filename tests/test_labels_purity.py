"""src/analyzer/labels/core.py as-of purity + 순수성 통합 테스트 (SPEC-ANALYZER-LABEL-001 M6).

REQ-AL-070(순수 함수)·REQ-AL-071(비영속화)·REQ-AL-072(as-of purity)을
검증한다. AC-AL-010(미래 행이 T 시점 레이블에 영향 없음)·AC-AL-011(순수성 —
동일 입력 → 동일 출력, 파일 시스템 부수효과 없음)의 대상이다.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

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


class TestAcAl010AsOfPurity:
    """AC-AL-010: T+H보다 미래의 daily_ohlcv/corporate_events 행이 있어도 T 시점 레이블은 불변."""

    def test_future_rows_and_events_do_not_change_past_label(self):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 1))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)

        # (b) 미래 행 없는 버전.
        df_b = _ohlcv_df([(t, 20000.0), (target, 11000.0)])
        events_b = pd.DataFrame(columns=["event_type", "event_date", "stock_rate"])

        # (a) target보다 미래의 합성 daily_ohlcv 행 + 미래 SPLIT 이벤트를 추가한 버전.
        future_date = date(2024, 3, 1)
        df_a = _ohlcv_df([(t, 20000.0), (target, 11000.0), (future_date, 99999.0)])
        events_a = pd.DataFrame(
            {
                "event_type": ["SPLIT"],
                "event_date": [date(2024, 2, 15)],  # target(01-29)보다 미래
                "stock_rate": [5.0],
            }
        )

        result_a = compute_labels(df_a, events_a, cal, horizons=(20,))
        result_b = compute_labels(df_b, events_b, cal, horizons=(20,))

        row_a = result_a.loc[result_a["trade_date"] == t].iloc[0]
        row_b = result_b.loc[result_b["trade_date"] == t].iloc[0]

        assert row_a["label_D20"] == row_b["label_D20"]
        assert row_a["label_D20_exclude_reason"] == row_b["label_D20_exclude_reason"]


class TestAcAl011Purity:
    """AC-AL-011: 순수 함수 — 동일 입력 → 동일 출력, 파일 시스템 부수효과 없음."""

    def test_same_input_twice_yields_identical_output_and_no_filesystem_side_effects(
        self, tmp_path: Path
    ):
        cal = _weekday_calendar(date(2024, 1, 1), date(2024, 3, 1))
        t = date(2024, 1, 1)
        target = nth_trading_day_on_or_after(cal, t + timedelta(days=1), 20)
        assert target is not None
        assert target == date(2024, 1, 29)

        df = _ohlcv_df([(t, 20000.0), (target, 11000.0)])
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT"],
                "event_date": [date(2024, 1, 15)],
                "stock_rate": [2.0],
            }
        )

        before_files = set(tmp_path.iterdir())

        result_1 = compute_labels(df, events, cal, horizons=(20, 60))
        result_2 = compute_labels(df, events, cal, horizons=(20, 60))

        after_files = set(tmp_path.iterdir())

        pd.testing.assert_frame_equal(result_1, result_2)
        assert before_files == after_files

"""src/analyzer/data/split.py SPLIT 조정 핸들러 테스트 (SPEC-ANALYZER-DATA-001 M3).

REQ-AD-020(배율 조정 소급 적용)·REQ-AD-021(다중 SPLIT 누적곱)·REQ-AD-022(방향
무관 통일 공식)을 검증한다. `adjust_split`을 직접 호출하지 않고 `adjust_prices`
경유로 호출해 `HANDLER_REGISTRY` 디스패치 배선 자체도 함께 검증한다(REQ-AD-041,
디스패치 우회 금지).
"""

from datetime import date

import pandas as pd
import pytest

from analyzer.data.adjustment import HANDLER_REGISTRY, adjust_prices
from analyzer.data.models import TradingCalendar
from analyzer.data.split import adjust_split


def _calendar() -> TradingCalendar:
    # SPLIT 조정은 캘린더에 의존하지 않는다 — 시그니처 계약 충족용 빈 캘린더.
    return TradingCalendar(calendar_code="NYSE", trading_days=frozenset())


class TestSplitHandlerRegistration:
    def test_split_handler_registered_via_dispatch_registry(self):
        """REQ-AD-041: `register_handler` 데코레이터로 배선되어야 한다(우회 금지)."""
        assert HANDLER_REGISTRY["SPLIT"] is adjust_split

    def test_direct_call_with_empty_events_returns_df_unchanged(self):
        """방어적 분기: `adjust_prices` 경유 시엔 도달하지 않지만(빈 subset을 넘기지
        않음), 핸들러를 직접 호출하는 경로에 대한 방어 코드다."""
        df = pd.DataFrame(
            {"trade_date": [date(2024, 1, 1)], "close_price": [100.0], "volume": [10]}
        )
        events = pd.DataFrame(columns=["event_type", "event_date", "stock_rate"])

        result = adjust_split(df, events, date(2024, 1, 1), _calendar())

        pd.testing.assert_frame_equal(result, df)


class TestAcAd001ForwardSplitWorkedExample:
    """AC-AD-001: AAPL 2020-08-31 4:1 분할 — REQ-AD-020/022."""

    def test_price_before_event_is_divided_by_stock_rate(self):
        df = pd.DataFrame(
            {
                "trade_date": [date(2020, 8, 27), date(2020, 8, 28), date(2020, 8, 31)],
                "close_price": [500.04, 499.23, 129.04],
                "volume": [100, 200, 400],
            }
        )
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT"],
                "event_date": [date(2020, 8, 31)],
                "stock_rate": [4.0],
            }
        )

        result = adjust_prices(df, events, as_of_date=date(2020, 9, 1), calendar=_calendar())

        adjusted = result.loc[result["trade_date"] == date(2020, 8, 28)].iloc[0]
        assert adjusted["close_price"] == pytest.approx(124.8075, abs=0.01)
        assert adjusted["volume"] == pytest.approx(200 * 4.0)

    def test_price_on_event_date_is_unaffected(self):
        """이벤트 당일 행은 이미 분할 후 가격 기준이므로 추가 조정되지 않아야 한다."""
        df = pd.DataFrame(
            {
                "trade_date": [date(2020, 8, 31)],
                "close_price": [129.04],
                "volume": [400],
            }
        )
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT"],
                "event_date": [date(2020, 8, 31)],
                "stock_rate": [4.0],
            }
        )

        result = adjust_prices(df, events, as_of_date=date(2020, 9, 1), calendar=_calendar())

        assert result["close_price"].iloc[0] == pytest.approx(129.04)
        assert result["volume"].iloc[0] == pytest.approx(400)


class TestAcAd002MergerBidirectionalFormula:
    """AC-AD-002: 병합(rate<1) — REQ-AD-020/022 방향 무관 동일 공식."""

    def test_merger_rate_below_one_scales_price_up_and_volume_down(self):
        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 1, 1)],
                "close_price": [100.0],
                "volume": [1000],
            }
        )
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT"],
                "event_date": [date(2024, 2, 1)],
                "stock_rate": [0.2],
            }
        )

        result = adjust_prices(df, events, as_of_date=date(2024, 3, 1), calendar=_calendar())

        assert result["close_price"].iloc[0] == pytest.approx(500.0)
        assert result["volume"].iloc[0] == pytest.approx(200.0)

    def test_merger_preserves_turnover_price_times_volume(self):
        """defect-regression guard: price*volume(거래대금)은 조정 전후 불변이어야 한다."""
        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 1, 1)],
                "close_price": [100.0],
                "volume": [1000],
            }
        )
        original_turnover = df["close_price"].iloc[0] * df["volume"].iloc[0]
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT"],
                "event_date": [date(2024, 2, 1)],
                "stock_rate": [0.2],
            }
        )

        result = adjust_prices(df, events, as_of_date=date(2024, 3, 1), calendar=_calendar())

        adjusted_turnover = result["close_price"].iloc[0] * result["volume"].iloc[0]
        assert adjusted_turnover == pytest.approx(original_turnover)


class TestAcAd009CumulativeMultiplierAcrossBoundaries:
    """AC-AD-009: 3개 이상 SPLIT 이벤트 누적곱, event_date 경계 전환 — REQ-AD-021."""

    def _fixture(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame(
            {
                "trade_date": [date(2020, 12, 1)],
                "close_price": [1200.0],
                "volume": [500],
            }
        )
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT", "SPLIT", "SPLIT"],
                "event_date": [date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)],
                "stock_rate": [2.0, 3.0, 0.5],
            }
        )
        return df, events

    def test_before_any_event_price_is_unchanged(self):
        df, events = self._fixture()

        result = adjust_prices(df, events, as_of_date=date(2020, 12, 31), calendar=_calendar())

        assert result["close_price"].iloc[0] == pytest.approx(1200.0)

    def test_one_event_elapsed(self):
        df, events = self._fixture()

        result = adjust_prices(df, events, as_of_date=date(2021, 6, 1), calendar=_calendar())

        assert result["close_price"].iloc[0] == pytest.approx(600.0)

    def test_two_events_elapsed(self):
        df, events = self._fixture()

        result = adjust_prices(df, events, as_of_date=date(2022, 6, 1), calendar=_calendar())

        assert result["close_price"].iloc[0] == pytest.approx(200.0)

    def test_three_events_elapsed(self):
        df, events = self._fixture()

        result = adjust_prices(df, events, as_of_date=date(2023, 6, 1), calendar=_calendar())

        assert result["close_price"].iloc[0] == pytest.approx(400.0)


class TestAsOfBoundaryEqualsEventDate:
    """AC-AD-005b 스타일 경계값: `as_of_date == event_date`이면 반영되어야 한다."""

    def test_boundary_as_of_equals_event_date_is_applied(self):
        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 5, 1)],
                "close_price": [100.0],
                "volume": [10],
            }
        )
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT"],
                "event_date": [date(2024, 6, 1)],
                "stock_rate": [2.0],
            }
        )

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 1), calendar=_calendar())

        assert result["close_price"].iloc[0] == pytest.approx(50.0)

    def test_boundary_as_of_before_event_date_is_not_applied(self):
        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 5, 1)],
                "close_price": [100.0],
                "volume": [10],
            }
        )
        events = pd.DataFrame(
            {
                "event_type": ["SPLIT"],
                "event_date": [date(2024, 6, 1)],
                "stock_rate": [2.0],
            }
        )

        result = adjust_prices(df, events, as_of_date=date(2024, 5, 15), calendar=_calendar())

        assert result["close_price"].iloc[0] == pytest.approx(100.0)

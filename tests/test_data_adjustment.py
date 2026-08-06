"""src/analyzer/data/adjustment.py 공개 시그니처 + 이벤트 디스패치 구조 테스트
(SPEC-ANALYZER-DATA-001 M1, REQ-AD-041).

M1은 `adjust_prices()`의 공개 시그니처와 이벤트 타입(SPLIT/DIVIDEND)별 확장 가능
디스패치 레지스트리만 확정한다. 실제 SPLIT/DIVIDEND 조정 수식은 M3/M4에서 구현되므로,
이 테스트는 합성 이벤트 타입으로 디스패치 메커니즘 자체(등록 → 호출 → 미등록 타입
무시)만 검증한다.
"""

from datetime import date

import pandas as pd

from analyzer.data.adjustment import (
    HANDLER_REGISTRY,
    adjust_prices,
    register_handler,
)
from analyzer.data.models import TradingCalendar


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date(2020, 8, 27), date(2020, 8, 28)],
            "close_price": [500.04, 499.23],
            "volume": [100, 200],
        }
    )


def _sample_calendar() -> TradingCalendar:
    return TradingCalendar(calendar_code="NYSE", trading_days=frozenset())


class TestRegisterHandler:
    def setup_method(self):
        # 각 테스트가 레지스트리 오염 없이 독립적으로 동작하도록 스냅샷/복원한다.
        self._registry_snapshot = dict(HANDLER_REGISTRY)

    def teardown_method(self):
        HANDLER_REGISTRY.clear()
        HANDLER_REGISTRY.update(self._registry_snapshot)

    def test_register_handler_adds_to_registry(self):
        @register_handler("TEST_EVENT_TYPE")
        def _handler(df, events, as_of_date, calendar):
            return df

        assert HANDLER_REGISTRY["TEST_EVENT_TYPE"] is _handler

    def test_register_handler_returns_function_unchanged(self):
        def _handler(df, events, as_of_date, calendar):
            return df

        decorated = register_handler("ANOTHER_TEST_EVENT_TYPE")(_handler)

        assert decorated is _handler


class TestAdjustPricesDispatch:
    def setup_method(self):
        self._registry_snapshot = dict(HANDLER_REGISTRY)

    def teardown_method(self):
        HANDLER_REGISTRY.clear()
        HANDLER_REGISTRY.update(self._registry_snapshot)

    def test_dispatches_registered_handler_with_matching_event_subset(self):
        calls = []

        @register_handler("TEST_SPLIT_LIKE")
        def _handler(df, events, as_of_date, calendar):
            calls.append((df, events, as_of_date, calendar))
            return df

        df = _sample_df()
        calendar = _sample_calendar()
        events = pd.DataFrame(
            {
                "event_type": ["TEST_SPLIT_LIKE"],
                "event_date": [date(2020, 8, 31)],
            }
        )

        result = adjust_prices(df, events, as_of_date=date(2020, 9, 1), calendar=calendar)

        assert len(calls) == 1
        called_df, called_events, called_as_of, called_calendar = calls[0]
        assert called_as_of == date(2020, 9, 1)
        assert called_calendar is calendar
        assert list(called_events["event_type"]) == ["TEST_SPLIT_LIKE"]
        assert result is df

    def test_unregistered_event_type_is_skipped_without_error(self):
        """REQ-AD-050/051: RIGHTS_ISSUE 등 미등록 타입은 방어적으로 skip한다."""
        df = _sample_df()
        events = pd.DataFrame(
            {
                "event_type": ["RIGHTS_ISSUE"],
                "event_date": [date(2020, 8, 31)],
            }
        )

        result = adjust_prices(df, events, as_of_date=date(2020, 9, 1), calendar=_sample_calendar())

        pd.testing.assert_frame_equal(result, df)

    def test_no_events_returns_df_unchanged(self):
        df = _sample_df()
        events = pd.DataFrame(columns=["event_type", "event_date"])

        result = adjust_prices(df, events, as_of_date=date(2020, 9, 1), calendar=_sample_calendar())

        pd.testing.assert_frame_equal(result, df)

    def test_multiple_registered_handlers_are_each_invoked_once(self):
        call_order = []

        @register_handler("TEST_TYPE_A")
        def _handler_a(df, events, as_of_date, calendar):
            call_order.append("A")
            return df

        @register_handler("TEST_TYPE_B")
        def _handler_b(df, events, as_of_date, calendar):
            call_order.append("B")
            return df

        df = _sample_df()
        events = pd.DataFrame(
            {
                "event_type": ["TEST_TYPE_A", "TEST_TYPE_B"],
                "event_date": [date(2020, 8, 31), date(2020, 9, 1)],
            }
        )

        adjust_prices(df, events, as_of_date=date(2020, 9, 2), calendar=_sample_calendar())

        assert sorted(call_order) == ["A", "B"]

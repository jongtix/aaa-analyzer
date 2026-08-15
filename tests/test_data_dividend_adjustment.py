"""src/analyzer/data/dividend_adjustment.py DIVIDEND 조정 핸들러 테스트 (SPEC-ANALYZER-DATA-001 M4).

REQ-AD-030(ex_date COALESCE 저장값 우선)·REQ-AD-033(디플레이터 누적곱)·
REQ-AD-034(0/unknown skip)·REQ-AD-035(통화 불변 전제, 방어적 skip)·
REQ-AD-040(DIVIDEND as-of 컷오프는 ex_date)을 검증한다. `adjust_dividend`를
직접 호출하지 않고 `adjust_prices` 경유로 호출해 `HANDLER_REGISTRY` 디스패치
배선 자체도 함께 검증한다(REQ-AD-041, 디스패치 우회 금지).
"""

from datetime import date

import pandas as pd
import pytest

from analyzer.data.adjustment import HANDLER_REGISTRY, adjust_prices
from analyzer.data.dividend_adjustment import adjust_dividend
from analyzer.data.models import TradingCalendar


def _calendar(
    trading_days: frozenset[date] = frozenset(), calendar_code: str = "KRX"
) -> TradingCalendar:
    return TradingCalendar(calendar_code=calendar_code, trading_days=trading_days)


def _dividend_event(
    *,
    event_date: date = date(2024, 6, 5),
    cash_amount: float = 500.0,
    event_subtype: object = "결산",
    ex_dividend_date: object = date(2024, 6, 10),
    currency_code: str = "KRW",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_type": ["DIVIDEND"],
            "event_date": [event_date],
            "cash_amount": [cash_amount],
            "event_subtype": [event_subtype],
            "ex_dividend_date": [ex_dividend_date],
            "currency_code": [currency_code],
        }
    )


class TestDividendHandlerRegistration:
    def test_dividend_handler_registered_via_dispatch_registry(self):
        """REQ-AD-041: `register_handler` 데코레이터로 배선되어야 한다(우회 금지)."""
        assert HANDLER_REGISTRY["DIVIDEND"] is adjust_dividend

    def test_direct_call_with_empty_events_returns_df_unchanged(self):
        df = pd.DataFrame(
            {"trade_date": [date(2024, 1, 1)], "close_price": [100.0], "volume": [10]}
        )
        events = pd.DataFrame(
            columns=[
                "event_type",
                "event_date",
                "cash_amount",
                "event_subtype",
                "ex_dividend_date",
                "currency_code",
            ]
        )

        result = adjust_dividend(df, events, date(2024, 1, 1), _calendar())

        pd.testing.assert_frame_equal(result, df)


class TestAcAd003NormalDeflatorWorkedExample:
    """AC-AD-003: 배당 디플레이터 정상 계산 — REQ-AD-033."""

    def test_deflator_applied_cumulatively_to_prices_before_ex_date(self):
        cal = _calendar(frozenset({date(2024, 6, 5), date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 6, 5), date(2024, 6, 7), date(2024, 6, 10)],
                "close_price": [40000.0, 50000.0, 50500.0],
                "volume": [100, 200, 300],
            }
        )
        events = _dividend_event(cash_amount=500.0, ex_dividend_date=date(2024, 6, 10))

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 20), calendar=cal)

        deflator = 1 - 500.0 / 50000.0
        assert result.loc[result["trade_date"] == date(2024, 6, 5), "close_price"].iloc[
            0
        ] == pytest.approx(40000.0 * deflator)
        assert result.loc[result["trade_date"] == date(2024, 6, 7), "close_price"].iloc[
            0
        ] == pytest.approx(50000.0 * deflator)
        assert result.loc[result["trade_date"] == date(2024, 6, 10), "close_price"].iloc[
            0
        ] == pytest.approx(50500.0)

    def test_volume_is_not_adjusted_by_dividend(self):
        """SPLIT과 달리 배당은 거래량을 조정하지 않는다."""
        cal = _calendar(frozenset({date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 6, 7)],
                "close_price": [50000.0],
                "volume": [200],
            }
        )
        events = _dividend_event(cash_amount=500.0, ex_dividend_date=date(2024, 6, 10))

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 20), calendar=cal)

        assert result["volume"].iloc[0] == 200


class TestAcAd006SkipInvalidDividendRows:
    """AC-AD-006: 0/unknown 배당 행 skip — REQ-AD-034."""

    def test_zero_cash_amount_is_skipped_without_error(self):
        cal = _calendar(frozenset({date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {"trade_date": [date(2024, 6, 5)], "close_price": [40000.0], "volume": [100]}
        )
        events = _dividend_event(cash_amount=0.0)

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 20), calendar=cal)

        pd.testing.assert_frame_equal(result, df)

    def test_empty_string_subtype_is_skipped_without_error(self):
        cal = _calendar(frozenset({date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {"trade_date": [date(2024, 6, 5)], "close_price": [40000.0], "volume": [100]}
        )
        events = _dividend_event(event_subtype="")

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 20), calendar=cal)

        pd.testing.assert_frame_equal(result, df)

    def test_unknown_none_subtype_is_skipped_without_error(self):
        """빈 문자열과 별개인 '알려지지 않은 값' 케이스(결측/None)를 검증한다."""
        cal = _calendar(frozenset({date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {"trade_date": [date(2024, 6, 5)], "close_price": [40000.0], "volume": [100]}
        )
        events = _dividend_event(event_subtype=None)

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 20), calendar=cal)

        pd.testing.assert_frame_equal(result, df)

    def test_mixed_valid_and_invalid_rows_only_valid_applied(self):
        cal = _calendar(frozenset({date(2024, 6, 5), date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 6, 5), date(2024, 6, 7)],
                "close_price": [40000.0, 50000.0],
                "volume": [100, 200],
            }
        )
        events = pd.concat(
            [
                _dividend_event(cash_amount=500.0, ex_dividend_date=date(2024, 6, 10)),
                _dividend_event(cash_amount=0.0, ex_dividend_date=date(2024, 6, 10)),
                _dividend_event(event_subtype="", ex_dividend_date=date(2024, 6, 10)),
            ],
            ignore_index=True,
        )

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 20), calendar=cal)

        deflator = 1 - 500.0 / 50000.0
        assert result.loc[result["trade_date"] == date(2024, 6, 5), "close_price"].iloc[
            0
        ] == pytest.approx(40000.0 * deflator)


class TestAcAd007StoredExDateTakesPriority:
    """AC-AD-007: 해외 종목 저장된 ex_dividend_date 우선 사용 — REQ-AD-030."""

    def test_stored_ex_dividend_date_used_without_deriving(self):
        cal = _calendar(
            frozenset({date(2024, 3, 6), date(2024, 3, 7), date(2024, 3, 8)}),
            calendar_code="NYSE",
        )
        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 3, 6), date(2024, 3, 7)],
                "close_price": [100.0, 100.0],
                "volume": [10, 10],
            }
        )
        events = _dividend_event(
            event_date=date(2024, 3, 1),  # record_date와 ex_date가 크게 벌어져 파생 미호출 증명
            cash_amount=1.0,
            event_subtype="Quarterly",
            ex_dividend_date=date(2024, 3, 8),
            currency_code="USD",
        )

        result = adjust_prices(df, events, as_of_date=date(2024, 4, 1), calendar=cal)

        deflator = 1 - 1.0 / 100.0
        assert result["close_price"].iloc[0] == pytest.approx(100.0 * deflator)
        assert result["close_price"].iloc[1] == pytest.approx(100.0 * deflator)


class TestCurrencyInvarianceDefensiveSkip:
    """REQ-AD-035/spec.md §4.3: 기대 통화와 다르면 조정 skip + 경고 로그."""

    def test_currency_mismatch_is_skipped_with_warning(self, caplog: pytest.LogCaptureFixture):
        cal = _calendar(frozenset({date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {"trade_date": [date(2024, 6, 5)], "close_price": [40000.0], "volume": [100]}
        )
        events = _dividend_event(currency_code="USD")  # KRX 시장은 KRW를 기대

        with caplog.at_level("WARNING"):
            result = adjust_prices(df, events, as_of_date=date(2024, 6, 20), calendar=cal)

        pd.testing.assert_frame_equal(result, df)
        assert "통화 불일치" in caplog.text


class TestDividendAsOfCutoff:
    """REQ-AD-040: DIVIDEND는 event_date가 아니라 해소된 ex_date를 컷오프로 사용한다."""

    def test_ex_date_after_as_of_is_not_applied(self):
        cal = _calendar(frozenset({date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {"trade_date": [date(2024, 6, 5)], "close_price": [40000.0], "volume": [100]}
        )
        events = _dividend_event(ex_dividend_date=date(2024, 6, 10))

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 9), calendar=cal)

        pd.testing.assert_frame_equal(result, df)

    def test_ex_date_equals_as_of_date_is_applied(self):
        cal = _calendar(frozenset({date(2024, 6, 5), date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {
                "trade_date": [date(2024, 6, 5), date(2024, 6, 7)],
                "close_price": [40000.0, 50000.0],
                "volume": [100, 200],
            }
        )
        events = _dividend_event(cash_amount=500.0, ex_dividend_date=date(2024, 6, 10))

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 10), calendar=cal)

        deflator = 1 - 500.0 / 50000.0
        assert result.loc[result["trade_date"] == date(2024, 6, 5), "close_price"].iloc[
            0
        ] == pytest.approx(40000.0 * deflator)


class TestMissingPriorCloseDefensiveSkip:
    """전일(prev_trading_day) 종가가 `df`에 없으면 디플레이터를 계산할 수 없어 skip한다."""

    def test_missing_prev_trading_day_row_is_skipped(self):
        cal = _calendar(frozenset({date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {"trade_date": [date(2024, 6, 5)], "close_price": [40000.0], "volume": [100]}
        )
        events = _dividend_event(ex_dividend_date=date(2024, 6, 10))

        result = adjust_prices(df, events, as_of_date=date(2024, 6, 20), calendar=cal)

        pd.testing.assert_frame_equal(result, df)


class TestSkipBatchAggregation:
    """AC-ATO-010(REQ-ATO-017): 배당 스킵 경고는 건별 개별 로그 대신 25건마다
    1회 집계 로그로 남기며, 각 집계 경고는 대상 식별자를 나열한다."""

    def test_forty_currency_mismatch_skips_emit_two_batched_warnings(
        self, caplog: pytest.LogCaptureFixture
    ):
        cal = _calendar(frozenset({date(2024, 6, 7), date(2024, 6, 10)}))
        df = pd.DataFrame(
            {"trade_date": [date(2024, 6, 5)], "close_price": [40000.0], "volume": [100]}
        )
        events = pd.DataFrame(
            {
                "event_type": ["DIVIDEND"] * 40,
                "event_date": [date(2024, 6, 5)] * 40,
                "cash_amount": [500.0] * 40,
                "event_subtype": ["결산"] * 40,
                "ex_dividend_date": [date(2024, 6, 10)] * 40,
                "currency_code": ["USD"] * 40,  # KRX 시장은 KRW를 기대 — 전건 불일치
                "stock_code": [f"S{i:04d}" for i in range(40)],
            }
        )

        with caplog.at_level("WARNING"):
            adjust_prices(df, events, as_of_date=date(2024, 6, 20), calendar=cal)

        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 2  # 1~25건, 26~40건
        assert "S0000" in warning_records[0].getMessage()
        assert "S0024" in warning_records[0].getMessage()
        assert "S0025" in warning_records[1].getMessage()
        assert "S0039" in warning_records[1].getMessage()

    def test_zero_skips_emit_no_warning(self, caplog: pytest.LogCaptureFixture):
        """acceptance.md §B 경계 사례: 마지막 배치가 0건짜리 빈 로그를 남기지 않는다."""
        from analyzer.data.dividend_adjustment import _log_skip_batches

        with caplog.at_level("WARNING"):
            _log_skip_batches("통화 불일치", [])

        assert caplog.records == []

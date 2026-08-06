"""SPLIT+DIVIDEND 혼재 종단 간(end-to-end) 통합 테스트 (SPEC-ANALYZER-DATA-001 M6).

REQ-AD-040(as-of 컷오프에 따른 look-ahead 방지)을 `adjust_prices()` 전체
디스패치 경로(M1 레지스트리 → M3 SPLIT 핸들러 + M4 DIVIDEND 핸들러 동시 발화)
기준으로 검증한다. 두 이벤트 타입이 같은 종목·같은 기간에 혼재할 때, 서로 다른
컷오프 필드(SPLIT=`event_date`, DIVIDEND=`ex_date`)가 각자 독립적으로 올바르게
적용/미적용되는지가 이 파일의 핵심 관심사다 — M3/M4 단위 테스트는 각 핸들러를
단독으로만 검증했고, 이 파일이 처음으로 두 핸들러의 동시 발화를 검증한다.

`TestAaplSplitIntegration`은 `@pytest.mark.integration`으로 표시된 실 DB 접속
테스트다 — `repository.py`(M5)의 조회 함수를 `adjust_prices()`에 실제로 배선하는
첫 지점이다(plan.md M6 "통합 레이어 연결"). MYSQL_* 환경변수가 없으면
`pytest.mark.skipif`로 정상 skip한다(에러 아님).
"""

from datetime import date

import pandas as pd
import pytest

# 핸들러 등록(HANDLER_REGISTRY 배선) 부수효과를 위한 임포트 — 이 파일 단독
# 실행(pytest 특정 파일만 지정) 시에도 SPLIT/DIVIDEND 핸들러가 등록되어 있어야
# `adjust_prices` 디스패치가 동작한다.
import analyzer.data.dividend_adjustment  # noqa: F401
import analyzer.data.split  # noqa: F401
from analyzer.data.adjustment import adjust_prices
from analyzer.data.config import MissingConfigError, get_db_config
from analyzer.data.models import TradingCalendar


def _mixed_events() -> pd.DataFrame:
    """DIVIDEND(ex_date=2020-08-07) + SPLIT(event_date=2020-08-31, 4:1) 혼재 이벤트."""
    return pd.DataFrame(
        {
            "event_type": ["DIVIDEND", "SPLIT"],
            "event_date": [date(2020, 8, 3), date(2020, 8, 31)],
            "stock_rate": [None, 4.0],
            "cash_amount": [2.0, None],
            "event_subtype": ["Quarterly", None],
            "ex_dividend_date": [date(2020, 8, 7), None],
            "currency_code": ["USD", None],
        }
    )


def _mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [
                date(2020, 8, 5),
                date(2020, 8, 6),
                date(2020, 8, 7),
                date(2020, 8, 27),
                date(2020, 8, 28),
                date(2020, 8, 31),
            ],
            "close_price": [100.0, 102.0, 104.0, 500.0, 499.0, 129.0],
            "volume": [10, 20, 30, 40, 50, 60],
        }
    )


def _calendar() -> TradingCalendar:
    trading_days = frozenset(
        {
            date(2020, 8, 5),
            date(2020, 8, 6),
            date(2020, 8, 7),
            date(2020, 8, 27),
            date(2020, 8, 28),
            date(2020, 8, 31),
        }
    )
    return TradingCalendar(calendar_code="NYSE", trading_days=trading_days)


class TestMixedSplitDividendNoLookAhead:
    """REQ-AD-040: SPLIT/DIVIDEND 혼재 종목의 as-of 컷오프 독립성."""

    def test_as_of_before_both_events_leaves_prices_raw(self):
        df, events, cal = _mixed_df(), _mixed_events(), _calendar()

        result = adjust_prices(df, events, as_of_date=date(2020, 8, 6), calendar=cal)

        pd.testing.assert_frame_equal(result, df)

    def test_as_of_after_dividend_but_before_split_applies_only_dividend(self):
        """ex_date(08-07) <= as_of < event_date(08-31): DIVIDEND만 반영, SPLIT 미반영."""
        df, events, cal = _mixed_df(), _mixed_events(), _calendar()

        result = adjust_prices(df, events, as_of_date=date(2020, 8, 10), calendar=cal)

        deflator = 1 - 2.0 / 102.0
        assert result.loc[result["trade_date"] == date(2020, 8, 5), "close_price"].iloc[
            0
        ] == pytest.approx(100.0 * deflator)
        assert result.loc[result["trade_date"] == date(2020, 8, 6), "close_price"].iloc[
            0
        ] == pytest.approx(102.0 * deflator)
        # ex_date 당일(08-07)은 배당 디플레이터 미적용 대상(trade_date < ex_date만 반영)
        assert result.loc[result["trade_date"] == date(2020, 8, 7), "close_price"].iloc[
            0
        ] == pytest.approx(104.0)
        # SPLIT event_date(08-31)가 아직 as_of를 넘지 않았으므로 이후 행은 전부 원가 그대로
        assert result.loc[result["trade_date"] == date(2020, 8, 27), "close_price"].iloc[
            0
        ] == pytest.approx(500.0)
        assert result.loc[result["trade_date"] == date(2020, 8, 31), "close_price"].iloc[
            0
        ] == pytest.approx(129.0)
        # DIVIDEND는 거래량을 조정하지 않는다
        assert result.loc[result["trade_date"] == date(2020, 8, 5), "volume"].iloc[0] == 10

    def test_as_of_equals_split_event_date_applies_both_cumulatively(self):
        """as_of == SPLIT event_date(08-31): 두 이벤트 모두 반영, 컷오프 이전 행은 곱셈 누적."""
        df, events, cal = _mixed_df(), _mixed_events(), _calendar()

        result = adjust_prices(df, events, as_of_date=date(2020, 8, 31), calendar=cal)

        deflator = 1 - 2.0 / 102.0

        # 08-05/08-06: ex_date(08-07) 이전 + SPLIT event_date(08-31) 이전 → 배당+분할 둘 다 적용
        assert result.loc[result["trade_date"] == date(2020, 8, 5), "close_price"].iloc[
            0
        ] == pytest.approx(100.0 * deflator / 4.0)
        assert result.loc[result["trade_date"] == date(2020, 8, 6), "close_price"].iloc[
            0
        ] == pytest.approx(102.0 * deflator / 4.0)
        # 08-07/08-27/08-28: ex_date 이후(배당 미적용) + SPLIT event_date 이전(분할 적용)
        assert result.loc[result["trade_date"] == date(2020, 8, 7), "close_price"].iloc[
            0
        ] == pytest.approx(104.0 / 4.0)
        assert result.loc[result["trade_date"] == date(2020, 8, 27), "close_price"].iloc[
            0
        ] == pytest.approx(500.0 / 4.0)
        assert result.loc[result["trade_date"] == date(2020, 8, 28), "close_price"].iloc[
            0
        ] == pytest.approx(499.0 / 4.0)
        # 08-31: SPLIT 이벤트 당일 자체는 미반영(REQ-AD-020 이벤트 당일 불변)
        assert result.loc[result["trade_date"] == date(2020, 8, 31), "close_price"].iloc[
            0
        ] == pytest.approx(129.0)

        # 거래량은 SPLIT만 반영(DIVIDEND는 거래량 미조정) — 컷오프 이전 전 행에 ×4
        assert result.loc[result["trade_date"] == date(2020, 8, 5), "volume"].iloc[
            0
        ] == pytest.approx(10 * 4.0)
        assert result.loc[result["trade_date"] == date(2020, 8, 28), "volume"].iloc[
            0
        ] == pytest.approx(50 * 4.0)
        assert result.loc[result["trade_date"] == date(2020, 8, 31), "volume"].iloc[0] == 60


def _db_config_available() -> bool:
    """MYSQL_* 환경변수가 이미 프로세스 환경에 로드되어 있는지 확인한다(.env 파일은 읽지 않음)."""
    try:
        get_db_config()
    except MissingConfigError:
        return False
    return True


@pytest.mark.integration
@pytest.mark.skipif(
    not _db_config_available(),
    reason="MYSQL_* 환경변수 미설정 — 실 DB 접속 통합 테스트는 로컬/수동 실행 전용",
)
class TestAaplSplitIntegration:
    """실 DB 접속: `repository.py`(M5) 조회 함수를 `adjust_prices()`에 배선하는 첫 지점.

    AAPL 2020-08-31 4:1 분할(M3 단위 테스트와 동일 worked example)을 실제
    daily_ohlcv 원주가로 조회해, 조정 후 분할 경계에서의 가격 갭이 사라지는지
    검증한다.
    """

    def test_aapl_2020_split_gap_disappears_after_adjustment(self):
        from analyzer.data.repository import (
            build_engine,
            fetch_corporate_events,
            fetch_daily_ohlcv,
            fetch_market_calendar,
        )

        engine = build_engine(get_db_config())
        calendar = fetch_market_calendar(engine, "NYSE")
        df = fetch_daily_ohlcv(
            engine, "AAPL", start_date=date(2020, 8, 20), end_date=date(2020, 9, 5)
        )
        events = fetch_corporate_events(engine, "AAPL")

        if df.empty:
            pytest.skip("AAPL daily_ohlcv 데이터가 이 DB에 없음 — 로컬 시드 데이터 의존")

        result = adjust_prices(df, events, as_of_date=date(2020, 9, 5), calendar=calendar)

        before = result.loc[result["trade_date"] < date(2020, 8, 31), "close_price"]
        after = result.loc[result["trade_date"] >= date(2020, 8, 31), "close_price"]
        assert not before.empty
        assert not after.empty

        # 분할 조정 전에는 08-28→08-31 구간에 raw 가격이 약 4배 갭으로 끊긴다(AC-AD-001
        # worked example: 499.23 → 129.04). 조정 후에는 그 갭이 사실상 사라져야 한다 —
        # 조정 경계 좌우 종가 차이가 각 값 대비 15% 미만이어야 한다(정상적인 일간 변동폭
        # 이내, 4배 갭이 남아있지 않음을 검증하는 보수적 임계값).
        max_before = before.max()
        min_after = after.min()
        assert abs(max_before - min_after) / max(max_before, min_after) < 0.15

"""배당락일(ex_date) 파생 함수 — 순수 함수, DB 비의존.

SPEC-ANALYZER-DATA-001 §4.2/REQ-AD-031/REQ-AD-032: 국내(또는 `ex_dividend_date`가
NULL인 해외) 종목의 배당락일은 원천에서 제공되지 않으므로, 기준일(`record_date`
= `corporate_events.event_date`)과 거래일 캘린더(`TradingCalendar`)로부터
파생해야 한다. 결제주기 T+N은 하드코딩 상수가 아니라 config(`get_settlement_days`)
로 구동된다(REQ-AD-032) — N=2(T+2, 기본값)일 때는
`ex_date = prevTradingDay(lastTradingDayOnOrBefore(record_date))`이고,
N=1(T+1)일 때는 `ex_date = lastTradingDayOnOrBefore(record_date)`로 단순화된다.

REQ-AD-030(해외 종목 저장값 우선)·COALESCE 적용은 이 모듈의 책임이 아니다 —
호출자가 `corporate_events.ex_dividend_date`가 NULL인 경우에만 이 함수를
호출한다(§4.2 worked example이 이 함수 자체의 테스트 케이스).
"""

from datetime import date

from analyzer.data.config import get_settlement_days
from analyzer.data.models import TradingCalendar


def derive_ex_date(record_date: date, calendar: TradingCalendar, market: str = "KRX") -> date:
    """`record_date`와 `market`의 결제주기 N(REQ-AD-032, config)으로 배당락일을 파생한다.

    일반형: `lastTradingDayOnOrBefore(record_date)`에서 시작해 `prevTradingDay`를
    `N - 1`회 적용한다 — N=1(T+1)이면 `lastTradingDayOnOrBefore(record_date)` 그대로,
    N=2(T+2, 기본값)이면 그 결과에 `prevTradingDay`를 1회 적용한다.
    """
    settlement_days = get_settlement_days(market)

    ex_date = calendar.last_trading_day_on_or_before(record_date)
    for _ in range(settlement_days - 1):
        ex_date = calendar.prev_trading_day(ex_date)

    return ex_date

"""daily_ohlcv/corporate_events/market_calendar 읽기 전용 타입 인터페이스.

SPEC-ANALYZER-DATA-001 §4.8/plan.md M1: DB 접근 계층은 SQLAlchemy ORM 매핑이
아니라 **raw SQL + `pd.read_sql`** 을 사용한다 — 조정 엔진(`adjustment.py`)이
pandas DataFrame 파이프라인과 직결되어 동작하므로, ORM 엔티티를 거쳤다가 다시
DataFrame으로 변환하는 매핑 오버헤드가 이득 없이 추가된다. 이 결정에 따라 아래
클래스들은 SQLAlchemy 모델이 아닌 순수 `dataclass` 이며, raw SQL 결과 row 하나의
형태를 문서화하는 타입 인터페이스 역할만 한다 — 실제 쿼리 실행/DB 연결 배선은
M5(DB 읽기 계층 배선) 소관이다.

REQ-AD-010/011/012에 대응.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class DailyOhlcvRow:
    """`daily_ohlcv` 원주가(raw) 행 하나를 표현한다."""

    stock_code: str
    trade_date: date
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int


@dataclass(frozen=True, slots=True)
class CorporateEventRow:
    """`corporate_events` 행 하나를 표현한다.

    `event_type`에 따라 관련 필드만 채워진다(SPLIT은 stock_rate, DIVIDEND는
    cash_amount/event_subtype/ex_dividend_date). `stock_rate`는 DB 코멘트상
    "주식배당률(%)"이나 실제로는 조정 배율(multiplier)로 사용된다(spec.md §4.6).
    """

    stock_code: str
    event_type: str
    event_date: date
    stock_rate: float | None
    cash_amount: float | None
    event_subtype: str | None
    ex_dividend_date: date | None
    currency_code: str | None


@dataclass(frozen=True, slots=True)
class MarketCalendarRow:
    """`market_calendar` 행 하나를 표현한다(REQ-AD-012)."""

    calendar_code: str
    cal_date: date
    is_open: bool


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    """`market_calendar` 조회 결과를 감싸는 거래일 판정 컨테이너.

    M1은 타입 구조(calendar_code + 거래일 집합)만 확정한다 — `prevTradingDay`/
    `lastTradingDayOnOrBefore` 같은 거래일 판정 로직은 M2(배당락일 파생 함수)에서
    이 타입을 인자로 받는 순수 함수로 구현된다(§4.2, DB 비의존 단위 테스트 용이성
    확보 목적).
    """

    calendar_code: str
    trading_days: frozenset[date]

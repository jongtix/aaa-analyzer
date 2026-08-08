"""`daily_ohlcv`/`corporate_events`/`market_calendar` DB 읽기 계층(SELECT 전용).

SPEC-ANALYZER-DATA-001 REQ-AD-010/011/012/013(M5): raw SQL + `pd.read_sql`을
사용한다(M1이 결정한 패턴, `models.py` 참조 — ORM 매핑 없음). analyzer 계정은
`stocks`/`daily_ohlcv`/`corporate_events`/`market_calendar`에 SELECT 전용
권한만 가지므로(REQ-AD-011), 이 모듈은 SELECT 문만 발행한다(DDL/DML 없음).

`daily_ohlcv`/`corporate_events`는 `stock_id`(FK)로 `stocks`를 참조하며
`stocks.symbol`이 종목코드다 — `models.py`의 `DailyOhlcvRow.stock_code`/
`CorporateEventRow.stock_code` 계약에 맞춰 두 쿼리 모두 `stocks`를 JOIN해
`symbol AS stock_code`로 해소한다.

REQ-AD-013: 대용량 시장 전체 조회는 `iter_daily_ohlcv_by_stock`으로 종목 단위
청크(제너레이터, 지연 평가)로 읽는다 — 전체 시장을 단일 DataFrame으로 적재하지
않는다. `fetch_daily_ohlcv`의 `start_date`/`end_date`는 동일 목적의 명시적
date-range 청크 옵션이다.
"""

from collections.abc import Iterator, Sequence
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from analyzer.data.config import DbConfig
from analyzer.data.models import TradingCalendar

_ANALYZER_DB_USER = "analyzer"

_MARKET_CALENDAR_QUERY = text(
    "SELECT calendar_code, cal_date, is_open "
    "FROM market_calendar "
    "WHERE calendar_code = :calendar_code"
)

_DAILY_OHLCV_QUERY_BASE = (
    "SELECT s.symbol AS stock_code, d.trade_date, d.open_price, d.high_price, "
    "d.low_price, d.close_price, d.volume "
    "FROM daily_ohlcv d "
    "JOIN stocks s ON s.id = d.stock_id "
    "WHERE s.symbol = :stock_code"
)

_CORPORATE_EVENTS_QUERY = text(
    "SELECT s.symbol AS stock_code, e.event_type, e.event_date, e.stock_rate, "
    "e.cash_amount, e.event_subtype, e.ex_dividend_date, e.currency_code "
    "FROM corporate_events e "
    "JOIN stocks s ON s.id = e.stock_id "
    "WHERE s.symbol = :stock_code "
    "ORDER BY e.event_date"
)

_INVESTOR_TREND_QUERY = text(
    "SELECT s.symbol AS stock_code, i.trade_date, i.foreign_net_value, "
    "i.institution_net_value, i.individual_net_value, i.total_trading_value "
    "FROM investor_trend i "
    "JOIN stocks s ON s.id = i.stock_id "
    "WHERE s.symbol = :stock_code "
    "ORDER BY i.trade_date"
)


def build_engine(config: DbConfig) -> Engine:
    """`DbConfig`로부터 커넥션 풀링을 지원하는 SQLAlchemy 엔진을 구성한다(REQ-AD-010)."""
    url = f"mysql+pymysql://{_ANALYZER_DB_USER}:{config.password}@{config.host}:{config.port}/{config.database}"
    return create_engine(url, pool_pre_ping=True)


def fetch_market_calendar(engine: Engine, calendar_code: str) -> TradingCalendar:
    """`market_calendar`에서 `calendar_code`의 거래일 집합을 읽는다(REQ-AD-012)."""
    df = pd.read_sql(_MARKET_CALENDAR_QUERY, engine, params={"calendar_code": calendar_code})
    trading_days = frozenset(df.loc[df["is_open"], "cal_date"])
    return TradingCalendar(calendar_code=calendar_code, trading_days=trading_days)


def fetch_daily_ohlcv(
    engine: Engine,
    stock_code: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """`daily_ohlcv`에서 1개 종목의 원주가(raw) 행을 읽는다(REQ-AD-011).

    `start_date`/`end_date`는 REQ-AD-013의 명시적 date-range 청크 옵션이다.
    """
    if not stock_code:
        raise ValueError("stock_code는 비어 있을 수 없다")

    sql = _DAILY_OHLCV_QUERY_BASE
    params: dict[str, object] = {"stock_code": stock_code}
    if start_date is not None:
        sql += " AND d.trade_date >= :start_date"
        params["start_date"] = start_date
    if end_date is not None:
        sql += " AND d.trade_date <= :end_date"
        params["end_date"] = end_date
    sql += " ORDER BY d.trade_date"

    return pd.read_sql(text(sql), engine, params=params)


def fetch_corporate_events(engine: Engine, stock_code: str) -> pd.DataFrame:
    """`corporate_events`에서 1개 종목의 이벤트 행을 읽는다(REQ-AD-011)."""
    if not stock_code:
        raise ValueError("stock_code는 비어 있을 수 없다")

    return pd.read_sql(_CORPORATE_EVENTS_QUERY, engine, params={"stock_code": stock_code})


def fetch_investor_trend(engine: Engine, stock_code: str) -> pd.DataFrame:
    """`investor_trend`에서 1개 종목의 수급 데이터를 읽는다(REQ-AF-010).

    `trade_date`만 SELECT/ORDER BY 대상으로 사용하고 `created_at`/`updated_at`은
    쿼리에 포함하지 않는다(REQ-AF-051 — DATE 전용 조인·정렬 키).
    """
    if not stock_code:
        raise ValueError("stock_code는 비어 있을 수 없다")

    return pd.read_sql(_INVESTOR_TREND_QUERY, engine, params={"stock_code": stock_code})


def iter_investor_trend_by_stock(
    engine: Engine,
    stock_codes: Sequence[str],
) -> Iterator[pd.DataFrame]:
    """시장 전체 수급 데이터 요청을 종목 단위 청크로 순회한다(REQ-AF-011, AC-AF-014).

    `iter_daily_ohlcv_by_stock`과 동일하게 제너레이터이므로 호출 시점에는
    아무 쿼리도 실행되지 않고, 각 청크가 소비될 때마다 `fetch_investor_trend`가
    정확히 1개 종목코드로 호출된다.
    """
    for stock_code in stock_codes:
        yield fetch_investor_trend(engine, stock_code)


def iter_daily_ohlcv_by_stock(
    engine: Engine,
    stock_codes: Sequence[str],
    start_date: date | None = None,
    end_date: date | None = None,
) -> Iterator[pd.DataFrame]:
    """시장 전체 조정 요청을 종목 단위 청크로 순회한다(REQ-AD-013, AC-AD-008).

    제너레이터이므로 호출 시점에는 아무 쿼리도 실행되지 않고, 각 청크가
    소비될 때마다 `fetch_daily_ohlcv`가 정확히 1개 종목코드로 호출된다 —
    전체 시장을 아우르는 단일 호출/단일 DataFrame이 만들어지지 않는다.
    """
    for stock_code in stock_codes:
        yield fetch_daily_ohlcv(engine, stock_code, start_date=start_date, end_date=end_date)

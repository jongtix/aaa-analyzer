"""`trade_date` 자체 산출 명세 테스트 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-021).

`stream:daily:complete` 이벤트는 `market`/`attempted`/`succeeded`/`skipped`
4개 필드만 담고 `trade_date`를 포함하지 않는다(collector 계약, 변경 불가).
따라서 analyzer는 시장별 `daily_ohlcv` 최신 거래일을 **DB 조회로** 산출해야
하며, KST 기준 오늘 날짜로 오산해서는 안 된다(해외 최신 일봉은 ET 기준
D-1 지연이 정상).
"""

from datetime import date

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from analyzer.inference.trade_date import (
    MARKET_TO_STOCKS_MARKET_CODES,
    resolve_trade_date,
)


def _make_engine() -> Engine:
    """`stocks`/`daily_ohlcv`의 최소 스키마를 담은 인메모리 SQLite 엔진."""
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE stocks ("
                "id INTEGER PRIMARY KEY, symbol TEXT, market TEXT, asset_type TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE daily_ohlcv ("
                "id INTEGER PRIMARY KEY, stock_id INTEGER, trade_date DATE)"
            )
        )
    return engine


def _insert_stock(engine: Engine, stock_id: int, market: str, asset_type: str = "STOCK") -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO stocks (id, symbol, market, asset_type) "
                "VALUES (:id, :symbol, :market, :asset_type)"
            ),
            {"id": stock_id, "symbol": f"S{stock_id}", "market": market, "asset_type": asset_type},
        )


def _insert_ohlcv(engine: Engine, stock_id: int, trade_date: date) -> None:
    # SQLite 기본 date 어댑터는 Python 3.12부터 deprecated이므로 ISO 문자열로
    # 직접 바인딩한다(조회 측은 `Date` 결과 타입이 다시 `date`로 되돌린다).
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO daily_ohlcv (stock_id, trade_date) VALUES (:stock_id, :trade_date)"),
            {"stock_id": stock_id, "trade_date": trade_date.isoformat()},
        )


class TestMarketTokenMapping:
    def test_maps_analyzer_market_tokens_to_exchange_codes(self):
        """`training/train.py`가 확립한 매핑을 그대로 계승한다(재정의 금지)."""
        assert MARKET_TO_STOCKS_MARKET_CODES == {
            "domestic": ("KOSPI", "KOSDAQ"),
            "overseas": ("NYSE", "NASDAQ", "AMEX"),
        }

    def test_unknown_market_token_is_rejected(self):
        engine = _make_engine()

        with pytest.raises(ValueError):
            resolve_trade_date(engine, "galactic")


class TestResolveTradeDate:
    def test_returns_latest_trade_date_of_the_requested_market(self):
        engine = _make_engine()
        _insert_stock(engine, 1, "KOSPI")
        _insert_ohlcv(engine, 1, date(2026, 9, 2))
        _insert_ohlcv(engine, 1, date(2026, 9, 3))

        assert resolve_trade_date(engine, "domestic") == date(2026, 9, 3)

    def test_overseas_latest_bar_is_not_confused_with_domestic(self):
        """AC-AIF-003: 해외 최신 확정 일봉은 국내보다 뒤처지는 것이 정상이며,
        국내 일봉의 최신 거래일로 대체되어서는 안 된다."""
        engine = _make_engine()
        _insert_stock(engine, 1, "KOSPI")
        _insert_stock(engine, 2, "NASDAQ")
        _insert_ohlcv(engine, 1, date(2026, 9, 3))
        _insert_ohlcv(engine, 2, date(2026, 9, 1))

        assert resolve_trade_date(engine, "overseas") == date(2026, 9, 1)

    def test_index_rows_are_excluded_from_the_universe(self):
        engine = _make_engine()
        _insert_stock(engine, 1, "KOSPI", asset_type="STOCK")
        _insert_stock(engine, 2, "KOSPI", asset_type="INDEX")
        _insert_ohlcv(engine, 1, date(2026, 9, 2))
        _insert_ohlcv(engine, 2, date(2026, 9, 4))

        assert resolve_trade_date(engine, "domestic") == date(2026, 9, 2)

    def test_returns_none_when_market_has_no_bars(self):
        engine = _make_engine()
        _insert_stock(engine, 1, "KOSPI")

        assert resolve_trade_date(engine, "overseas") is None

    def test_does_not_derive_trade_date_from_the_wall_clock(self):
        """AC-AIF-003(shall not): 오늘 날짜(KST)로 산출하지 않는다 — 소스에
        `date.today()`/`datetime.now()` 호출이 존재하지 않아야 한다."""
        from pathlib import Path

        import analyzer.inference.trade_date as module

        source = Path(module.__file__).read_text(encoding="utf-8")

        assert "date.today()" not in source
        assert "datetime.now(" not in source

"""SPEC-ANALYZER-PIPELINE-001 M6: 합성 DB 스키마 기반 통합 테스트.

`trading_signals`/`signal_price_bands` INSERT-ONLY 경로(`inference/writer.py`)
가 실제 SQLAlchemy 엔진(SQLite 인메모리, MySQL 프로덕션 스키마와 동일한
컬럼·UNIQUE 제약)에 대해 예상대로 동작하는지 검증한다 — mock 엔진으로는
잡히지 않는 SQL 문법·제약 위반 회귀를 잡기 위함이다(plan.md M6).
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from analyzer.inference.writer import (
    InsertOutcome,
    PriceBandRow,
    TradingSignalRow,
    insert_signal_price_bands,
    insert_trading_signal,
)

pytestmark = pytest.mark.integration

_CREATE_TRADING_SIGNALS = """
CREATE TABLE trading_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    horizon VARCHAR(3) NOT NULL,
    score REAL NOT NULL,
    p10 REAL,
    p90 REAL,
    lgbm_score REAL,
    xgb_score REAL,
    signal_class VARCHAR(16) NOT NULL,
    confidence REAL NOT NULL,
    regime_suppressed BOOLEAN NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    UNIQUE (stock_id, trade_date, horizon)
)
"""

_CREATE_SIGNAL_PRICE_BANDS = """
CREATE TABLE signal_price_bands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER NOT NULL,
    trade_date DATE NOT NULL,
    horizon VARCHAR(3) NOT NULL,
    boundary_set VARCHAR(8) NOT NULL,
    band_seq INTEGER NOT NULL,
    price_low REAL NOT NULL,
    price_high REAL NOT NULL,
    signal_class VARCHAR(16) NOT NULL,
    regime_suppressed BOOLEAN NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    created_at DATETIME NOT NULL,
    UNIQUE (stock_id, trade_date, horizon, boundary_set, band_seq)
)
"""


@pytest.fixture
def sqlite_engine() -> Engine:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text(_CREATE_TRADING_SIGNALS))
        conn.execute(text(_CREATE_SIGNAL_PRICE_BANDS))
    return engine


def _signal_row(**overrides: object) -> TradingSignalRow:
    defaults: dict[str, object] = {
        "stock_id": 1,
        "trade_date": date(2026, 9, 10),
        "horizon": 20,
        "score": 0.42,
        "p10": 0.1,
        "p90": 0.9,
        "lgbm_score": 0.4,
        "xgb_score": 0.44,
        "signal_class": "BUY",
        "confidence": 0.8,
        "model_version": "domestic_20_ensemble_2026-08-29",
    }
    defaults.update(overrides)
    return TradingSignalRow(**defaults)  # type: ignore[arg-type]


class TestTradingSignalsRealInsert:
    """AC-APL-100/104: 실제 (시장,horizon) 조합·종목에 대한 INSERT가 합성
    스키마에서 실제로 반영되는지 확인한다."""

    def test_insert_persists_a_row(self, sqlite_engine: Engine):
        outcome = insert_trading_signal(sqlite_engine, _signal_row())

        assert outcome == InsertOutcome.INSERTED
        with sqlite_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM trading_signals")).scalar_one()
        assert count == 1

    def test_duplicate_key_is_skipped_not_raised(self, sqlite_engine: Engine):
        insert_trading_signal(sqlite_engine, _signal_row())

        outcome = insert_trading_signal(sqlite_engine, _signal_row())

        assert outcome == InsertOutcome.SKIPPED_DUPLICATE
        with sqlite_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM trading_signals")).scalar_one()
        assert count == 1


class TestSignalPriceBandsRealInsert:
    """AC-APL-105: 밴드 스윕 결과의 실제 INSERT — boundary_set 단위 사전
    존재 확인 후 일괄 INSERT."""

    def _band_rows(self, boundary_set: str = "PROMOTE") -> list[PriceBandRow]:
        return [
            PriceBandRow(
                stock_id=1,
                trade_date=date(2026, 9, 10),
                horizon=20,
                boundary_set=boundary_set,
                band_seq=seq,
                price_low=100.0 + seq,
                price_high=101.0 + seq,
                signal_class="HOLD",
                model_version="domestic_20_ensemble_2026-08-29",
            )
            for seq in range(3)
        ]

    def test_insert_persists_all_rows_in_one_boundary_set(self, sqlite_engine: Engine):
        outcome = insert_signal_price_bands(sqlite_engine, self._band_rows())

        assert outcome == InsertOutcome.INSERTED
        with sqlite_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM signal_price_bands")).scalar_one()
        assert count == 3

    def test_duplicate_boundary_set_is_skipped_entirely(self, sqlite_engine: Engine):
        insert_signal_price_bands(sqlite_engine, self._band_rows())

        outcome = insert_signal_price_bands(sqlite_engine, self._band_rows())

        assert outcome == InsertOutcome.SKIPPED_DUPLICATE
        with sqlite_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM signal_price_bands")).scalar_one()
        assert count == 3

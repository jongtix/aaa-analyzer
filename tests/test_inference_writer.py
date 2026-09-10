"""`trading_signals` INSERT-ONLY 기록 + model_version 포맷 테스트
(SPEC-ANALYZER-INFER-001 M5, REQ-AIF-100, AC-AIF-017).

실 DB 접속 없이 `Engine.begin()`을 모킹한 단위 테스트다 — 통합 테스트
(collector V48 마이그레이션 실측 반영 확인 완료, `@pytest.mark.integration`)는
이 SPEC 범위 밖 후속 작업이다.
"""

from datetime import date
from unittest.mock import MagicMock

from sqlalchemy.exc import IntegrityError

from analyzer.inference.writer import (
    InsertOutcome,
    TradingSignalRow,
    format_horizon_label,
    format_model_version,
    insert_trading_signal,
)


class TestFormatModelVersion:
    """REQ-AIF-100: model_version은 `{market}_{horizon}_{algo}_{trained_date}`
    형식이며 FK가 아닌 자유 텍스트다(`persistence.model_filename()`과 동일
    관례, 확장자만 없음)."""

    def test_ensemble_combo_uses_ensemble_tag(self):
        result = format_model_version("domestic", 20, "ensemble", date(2026, 8, 19))

        assert result == "domestic_20_ensemble_2026-08-19"

    def test_solo_algorithm_uses_algorithm_name(self):
        result = format_model_version("overseas", 60, "xgboost", date(2026, 8, 19))

        assert result == "overseas_60_xgboost_2026-08-19"


class TestFormatHorizonLabel:
    """`trading_signals.horizon`(VARCHAR(3))은 "D20"/"D60" 형식이다 —
    `model_version`의 순수 정수 표기(REQ-AIF-100)와 다른 별개 표현이다."""

    def test_d20(self):
        assert format_horizon_label(20) == "D20"

    def test_d60(self):
        assert format_horizon_label(60) == "D60"


def _row(**overrides) -> TradingSignalRow:
    defaults = {
        "stock_id": 1,
        "trade_date": date(2026, 9, 10),
        "horizon": 20,
        "score": 0.043,
        "p10": -0.02,
        "p90": 0.09,
        "lgbm_score": 0.04,
        "xgb_score": 0.046,
        "signal_class": "BUY",
        "confidence": 0.8,
        "model_version": "domestic_20_ensemble_2026-08-19",
    }
    defaults.update(overrides)
    return TradingSignalRow(**defaults)


def _mock_engine_and_conn() -> tuple[MagicMock, MagicMock]:
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__.return_value = conn
    return engine, conn


class TestInsertTradingSignal:
    """AC-AIF-017: 최초 삽입은 성공하고, UNIQUE 키 재삽입은 IntegrityError를
    "이미 처리됨" 신호로 캐치-스킵한다 — 프로세스는 크래시하지 않는다."""

    def test_first_insert_succeeds(self):
        engine, conn = _mock_engine_and_conn()

        outcome = insert_trading_signal(engine, _row())

        assert outcome == InsertOutcome.INSERTED
        assert conn.execute.call_count == 1

    def test_duplicate_insert_is_caught_and_skipped(self):
        engine, conn = _mock_engine_and_conn()
        conn.execute.side_effect = IntegrityError("stmt", {}, Exception("Duplicate entry"))

        outcome = insert_trading_signal(engine, _row())

        assert outcome == InsertOutcome.SKIPPED_DUPLICATE

    def test_regime_suppressed_always_false(self):
        """REQ-AIF-090: `regime_suppressed`는 체제 판정 로직 없이 항상
        FALSE로 고정 기록된다."""
        engine, conn = _mock_engine_and_conn()

        insert_trading_signal(engine, _row())

        params = conn.execute.call_args[0][1]
        assert params["regime_suppressed"] is False

    def test_solo_strategy_null_lgbm_score_is_passed_through(self):
        """REQ-AIF-041: 단독 xgboost 전략이면 lgbm_score는 NULL(None)로
        그대로 전달돼야 한다 — 앙상블 공식을 적용하지 않는다."""
        engine, conn = _mock_engine_and_conn()

        insert_trading_signal(engine, _row(lgbm_score=None, score=0.046))

        params = conn.execute.call_args[0][1]
        assert params["lgbm_score"] is None
        assert params["xgb_score"] == 0.046

    def test_model_version_and_horizon_label_passed_to_statement(self):
        engine, conn = _mock_engine_and_conn()

        insert_trading_signal(
            engine, _row(horizon=60, model_version="overseas_60_xgboost_2026-08-19")
        )

        params = conn.execute.call_args[0][1]
        assert params["horizon"] == "D60"
        assert params["model_version"] == "overseas_60_xgboost_2026-08-19"

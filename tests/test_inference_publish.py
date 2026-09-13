"""`stream:signal:{market}` 발행 테스트 (SPEC-ANALYZER-INFER-001 M8,
REQ-AIF-120, AC-AIF-020, design.md §6).

Redis는 `tests/test_inference_lock.py`와 동일하게 필요한 명령만 흉내내는
페이크로 대체한다(`fakeredis` 미도입 — 기존 관례 계승).
"""

from datetime import date
from typing import Any

import pytest

from analyzer.inference.publish import (
    SIGNAL_STREAM_FIELDS,
    SIGNAL_STREAM_MAXLEN,
    publish_trading_signal,
    signal_stream_key,
)


class _FakeRedis:
    """`XADD`만 흉내내는 페이크 — 호출 인자를 그대로 보관한다."""

    def __init__(self) -> None:
        self.xadd_calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def xadd(self, name, fields, **kwargs):
        self.xadd_calls.append((name, dict(fields), dict(kwargs)))
        return "1757740800000-0"


def _publish(client: _FakeRedis, **overrides: Any) -> None:
    kwargs: dict[str, Any] = {
        "market": "domestic",
        "symbol": "005930",
        "horizon": 20,
        "trade_date": date(2026, 9, 12),
        "trace_id": "8f14e45fceea167a5a36dedd4bea2543",
        "signal_class": "STRONG_BUY",
        "score": 0.0432,
        "confidence": 0.7125,
    }
    kwargs.update(overrides)
    publish_trading_signal(client, **kwargs)


class TestSignalStreamKey:
    def test_key_is_scoped_by_market(self):
        assert signal_stream_key("domestic") == "stream:signal:domestic"
        assert signal_stream_key("overseas") == "stream:signal:overseas"

    def test_distinct_markets_do_not_share_a_stream(self):
        assert signal_stream_key("domestic") != signal_stream_key("overseas")


class TestPublishTradingSignalFields:
    """AC-AIF-020 전반부: 메시지 필드는 정확히 7개여야 한다."""

    def test_publishes_to_the_market_scoped_stream(self):
        client = _FakeRedis()

        _publish(client, market="overseas")

        assert len(client.xadd_calls) == 1
        assert client.xadd_calls[0][0] == "stream:signal:overseas"

    def test_message_has_exactly_the_seven_contract_fields(self):
        client = _FakeRedis()

        _publish(client)

        fields = client.xadd_calls[0][1]
        assert set(fields) == {
            "symbol",
            "horizon",
            "trade_date",
            "trace_id",
            "signal_class",
            "score",
            "confidence",
        }
        assert len(fields) == 7

    def test_declared_field_contract_matches_the_published_keys(self):
        """`SIGNAL_STREAM_FIELDS`(계약 선언)와 실제 발행 키가 어긋나지 않아야
        한다 — 한쪽만 고치는 표류를 막는 가드."""
        client = _FakeRedis()

        _publish(client)

        assert tuple(client.xadd_calls[0][1]) == SIGNAL_STREAM_FIELDS
        assert len(SIGNAL_STREAM_FIELDS) == 7

    def test_every_field_value_is_string_encoded(self):
        """design.md §6: 전부 문자열 인코딩(Redis Streams 필드 값 관례,
        `redis_client.py`의 `decode_responses=True` 대칭)."""
        client = _FakeRedis()

        _publish(client)

        for key, value in client.xadd_calls[0][1].items():
            assert isinstance(value, str), f"{key}={value!r} is not str"

    def test_field_values_carry_the_expected_encoding(self):
        client = _FakeRedis()

        _publish(client)

        fields = client.xadd_calls[0][1]
        assert fields["symbol"] == "005930"
        assert fields["horizon"] == "D20"
        assert fields["trade_date"] == "2026-09-12"
        assert fields["trace_id"] == "8f14e45fceea167a5a36dedd4bea2543"
        assert fields["signal_class"] == "STRONG_BUY"
        assert float(fields["score"]) == pytest.approx(0.0432)
        assert float(fields["confidence"]) == pytest.approx(0.7125)

    def test_horizon_uses_the_db_label_form(self):
        """`trading_signals.horizon`/`signal_price_bands.horizon`와 동일한
        "D20"/"D60" 표기 — 소비자가 스트림 필드를 그대로 DB 조회 키로 쓸 수
        있어야 한다(`writer.format_horizon_label()` 재사용)."""
        client = _FakeRedis()

        _publish(client, horizon=60)

        assert client.xadd_calls[0][1]["horizon"] == "D60"


class TestPublishTradingSignalTrimming:
    """AC-AIF-020 후반부: XADD 호출에 `MAXLEN ~ 500`이 포함돼야 한다."""

    def test_maxlen_is_five_hundred(self):
        client = _FakeRedis()

        _publish(client)

        assert SIGNAL_STREAM_MAXLEN == 500
        assert client.xadd_calls[0][2]["maxlen"] == 500

    def test_trimming_is_approximate(self):
        """`~`(근사 트리밍) — 정확한 카운트를 강제하지 않는다."""
        client = _FakeRedis()

        _publish(client)

        assert client.xadd_calls[0][2]["approximate"] is True


class TestPublishTradingSignalIsSynchronous:
    """`redis_client.py` 관례: 동기 클라이언트를 쓰고 `asyncio.to_thread()`
    래핑은 호출자(`orchestration/consumer.py`) 책임 — 이 모듈은 코루틴을
    반환하지 않는다."""

    def test_returns_none_not_a_coroutine(self):
        import inspect

        client = _FakeRedis()

        result = publish_trading_signal(
            client,
            market="domestic",
            symbol="005930",
            horizon=20,
            trade_date=date(2026, 9, 12),
            trace_id="t",
            signal_class="HOLD",
            score=0.0,
            confidence=0.5,
        )

        assert result is None
        assert not inspect.iscoroutinefunction(publish_trading_signal)

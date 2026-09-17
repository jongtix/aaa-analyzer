"""Redis 소켓 타임아웃 정합 회귀 가드 (SPEC-ANALYZER-REDIS-TIMEOUT-001).

REQ-RT-010/020/060: `build_redis_client()`가 생성하는 Redis 클라이언트의
`socket_timeout`이 `StreamConsumer`의 XREADGROUP `BLOCK`(초 환산)보다 충분한
안전마진을 두고 크다는 불변식을 고정한다. 값을 하드코딩해 항상 PASS하는
테스트가 아니라, 회귀 상태(마진 0 또는 음수)를 시뮬레이션하면 반드시 FAIL하는
구조적 가드다(AC-004) — 실제 프로덕션 결함(마진 0으로 인한 오탐성
TimeoutError, DLQ 유실 18건, 2026-09-04~09-16)의 재발을 방지한다.
"""

from pathlib import Path

import pytest

from analyzer.inference import redis_client
from analyzer.inference.config import InferenceConfig


def _assert_margin_invariant(socket_timeout_seconds: float, block_milliseconds: int) -> None:
    """REQ-RT-010/020의 불변식: socket_timeout(초) > BLOCK(초 환산).

    `redis_client.py` 모듈 레벨 assert(plan.md B.3)와 동일한 조건을 pytest
    가드로도 고정한다 — `python -O` 실행 환경에서 bare assert가 제거되는
    경우의 별도 방어선(plan.md F 리스크 대응).
    """
    assert socket_timeout_seconds > block_milliseconds / 1000, (
        f"socket_timeout({socket_timeout_seconds}s)이 BLOCK"
        f"({block_milliseconds}ms={block_milliseconds / 1000}s) 대비 "
        "안전마진을 확보하지 못했다"
    )


class TestSocketTimeoutMarginInvariant:
    """REQ-RT-060: 회귀 방지 구조적 가드."""

    def test_current_module_values_satisfy_margin_invariant(self):
        """AC-001/AC-004(GREEN): 현재 값 조합(BLOCK=5000ms)에서는 통과한다."""
        _assert_margin_invariant(
            redis_client.SOCKET_TIMEOUT_SECONDS,
            redis_client.DEFAULT_BLOCK_MILLISECONDS,
        )

    def test_regression_state_with_equal_timeout_and_block_fails(self):
        """AC-004(RED 시뮬레이션): socket_timeout(5초)==BLOCK(5000ms) 조합은
        이 결함이 실제로 만들었던 마진 0 상태다 — 반드시 FAIL해야 한다."""
        with pytest.raises(AssertionError):
            _assert_margin_invariant(socket_timeout_seconds=5.0, block_milliseconds=5_000)

    def test_regression_state_with_timeout_below_block_fails(self):
        """socket_timeout이 BLOCK보다 짧은, 더 심한 가상 회귀 상태도 FAIL한다."""
        with pytest.raises(AssertionError):
            _assert_margin_invariant(socket_timeout_seconds=3.0, block_milliseconds=5_000)

    def test_current_margin_is_ten_seconds(self):
        """AC-001: 5초 BLOCK 대비 socket_timeout=15초, 마진 10초(plan.md B.2
        설계값)가 정확히 반영됐는지 확정값으로 고정한다."""
        assert redis_client.DEFAULT_BLOCK_MILLISECONDS == 5_000
        assert redis_client.SOCKET_TIMEOUT_SECONDS == pytest.approx(15.0)


class TestBuildRedisClientSocketTimeout:
    """AC-002: 실제로 생성된 클라이언트에 명시 계산값이 전달됐는지 확인한다."""

    def test_client_socket_timeout_is_not_library_default(self):
        client = redis_client.build_redis_client(
            InferenceConfig(
                redis_host="redis-host",
                redis_port=6379,
                redis_username="appuser",
                redis_password="redis-secret",
                stream_claim_idle_seconds=600,
                container_models_root=Path("/mnt/models"),
            )
        )

        socket_timeout = client.connection_pool.connection_kwargs["socket_timeout"]

        assert socket_timeout == redis_client.SOCKET_TIMEOUT_SECONDS
        assert socket_timeout != 5  # redis-py 8.1.0 라이브러리 자체 기본값이 아니다


class TestConsumerSingleSourceOfTruth:
    """AC-003/REQ-RT-040: consumer.py가 독립된 두 번째 리터럴을 선언하지 않고
    동일한 단일 상수를 재수입한다."""

    def test_consumer_default_block_milliseconds_is_the_same_object(self):
        from analyzer.orchestration import consumer as consumer_module

        assert consumer_module.DEFAULT_BLOCK_MILLISECONDS is redis_client.DEFAULT_BLOCK_MILLISECONDS

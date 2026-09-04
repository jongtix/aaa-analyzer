"""Redis Streams 컨슈머 명세 테스트 (SPEC-ANALYZER-INFER-001 M1).

REQ-AIF-020: `stream:daily:complete`를 그룹 `analyzer`/컨슈머
`analyzer-daily-complete`로 구독하고, idle 임계 초과 미확인 메시지를
XAUTOCLAIM으로 재소유하며, 재전달 3회 초과 메시지를 DLQ로 이관한다.
REQ-AIF-021: `trade_date`는 이벤트 필드가 아니라 자체 DB 조회로 산출한다.
REQ-AIF-060: 인-플라이트 락으로 동일 (market, trade_date) 중복 스폰을 막는다.
"""

import asyncio
from collections.abc import Callable
from datetime import date

import pytest
from redis.exceptions import ResponseError

from analyzer.inference.config import InferenceConfig
from analyzer.inference.lock import inflight_lock_key
from analyzer.orchestration.consumer import (
    CONSUMER_GROUP,
    CONSUMER_NAME,
    DAILY_COMPLETE_STREAM,
    DLQ_STREAM,
    MAX_DELIVERY_COUNT,
    StreamConsumer,
)

_TRADE_DATE = date(2026, 9, 3)


def _config(idle_seconds: int = 600) -> InferenceConfig:
    return InferenceConfig(
        redis_host="redis",
        redis_port=6379,
        redis_username="appuser",
        redis_password="redis-secret",
        stream_claim_idle_seconds=idle_seconds,
    )


def _event(market: str = "domestic", *, succeeded: str = "45") -> dict[str, str]:
    return {"market": market, "attempted": "48", "succeeded": succeeded, "skipped": "3"}


class _FakeRedis:
    """컨슈머가 실제로 호출하는 Redis 명령만 흉내내는 페이크."""

    def __init__(
        self,
        *,
        new_messages: list[tuple[str, dict[str, str]]] | None = None,
        claimable: list[tuple[str, dict[str, str]]] | None = None,
        pending: list[dict[str, object]] | None = None,
    ) -> None:
        self._new = list(new_messages or [])
        self._claimable = list(claimable or [])
        self._pending = list(pending or [])
        self.created_groups: list[tuple[str, str, str, bool]] = []
        self.group_create_error: Exception | None = None
        self.acked: list[str] = []
        self.added: list[tuple[str, dict[str, str]]] = []
        self.values: dict[str, str] = {}
        self.autoclaim_calls: list[dict[str, object]] = []

    def xgroup_create(self, name, groupname, id, mkstream=False):
        if self.group_create_error is not None:
            raise self.group_create_error
        self.created_groups.append((name, groupname, id, mkstream))

    def xpending_range(self, name, groupname, min, max, count, idle=None):
        return list(self._pending)

    def xautoclaim(self, name, groupname, consumername, min_idle_time, start_id="0-0", count=None):
        self.autoclaim_calls.append(
            {
                "name": name,
                "groupname": groupname,
                "consumername": consumername,
                "min_idle_time": min_idle_time,
            }
        )
        claimed = list(self._claimable)
        self._claimable = []
        return ["0-0", claimed, []]

    def xreadgroup(self, groupname, consumername, streams, count=None, block=None):
        if not self._new:
            return []
        batch = list(self._new)
        self._new = []
        return [[DAILY_COMPLETE_STREAM, batch]]

    def xack(self, name, groupname, *ids):
        self.acked.extend(ids)
        return len(ids)

    def xadd(self, name, fields, **kwargs):
        self.added.append((name, dict(fields)))
        return "9-9"

    def set(self, name, value, nx=False, ex=None):
        if nx and name in self.values:
            return None
        self.values[name] = value
        return True

    def eval(self, script, numkeys, *args):
        key, token = args[0], args[1]
        if self.values.get(key) == token:
            del self.values[key]
            return 1
        return 0


class _SpawnRecorder:
    def __init__(self, exit_code: int = 0) -> None:
        self.calls: list[str] = []
        self.exit_code = exit_code
        self.on_call: Callable[[], None] | None = None

    async def __call__(self, market: str, *, trace_id: str) -> int:
        self.calls.append(market)
        if self.on_call is not None:
            self.on_call()
        return self.exit_code


def _make_consumer(
    client: _FakeRedis,
    *,
    spawn: _SpawnRecorder | None = None,
    trade_date: date | None = _TRADE_DATE,
    idle_seconds: int = 600,
) -> tuple[StreamConsumer, _SpawnRecorder]:
    spawn_child = spawn or _SpawnRecorder()
    consumer = StreamConsumer(
        config=_config(idle_seconds),
        client=client,
        spawn_child=spawn_child,
        resolve_trade_date=lambda market: trade_date,
    )
    return consumer, spawn_child


class TestConsumerGroupRegistration:
    def test_creates_the_group_from_the_stream_tail_with_mkstream(self):
        """AC-AIF-002: `XGROUP CREATE stream:daily:complete analyzer $ MKSTREAM`."""
        client = _FakeRedis()
        consumer, _ = _make_consumer(client)

        asyncio.run(consumer.poll_once())

        assert client.created_groups == [(DAILY_COMPLETE_STREAM, CONSUMER_GROUP, "$", True)]

    def test_existing_group_is_ignored(self):
        client = _FakeRedis()
        client.group_create_error = ResponseError("BUSYGROUP Consumer Group name already exists")
        consumer, _ = _make_consumer(client)

        assert asyncio.run(consumer.poll_once()) == 0

    def test_group_is_created_only_once_per_consumer(self):
        client = _FakeRedis()
        consumer, _ = _make_consumer(client)

        asyncio.run(consumer.poll_once())
        asyncio.run(consumer.poll_once())

        assert len(client.created_groups) == 1

    def test_unexpected_response_error_propagates(self):
        client = _FakeRedis()
        client.group_create_error = ResponseError("NOPERM this user has no permissions")
        consumer, _ = _make_consumer(client)

        with pytest.raises(ResponseError):
            asyncio.run(consumer.poll_once())


class TestNewMessageHandling:
    def test_spawns_a_child_per_event_and_acks(self):
        client = _FakeRedis(new_messages=[("1-1", _event())])
        consumer, spawn = _make_consumer(client)

        handled = asyncio.run(consumer.poll_once())

        assert handled == 1
        assert spawn.calls == ["domestic"]
        assert client.acked == ["1-1"]

    def test_reads_with_the_declared_group_and_consumer_name(self):
        recorded: dict[str, object] = {}
        client = _FakeRedis(new_messages=[("1-1", _event())])
        original = client.xreadgroup

        def _spy(groupname, consumername, streams, count=None, block=None):
            recorded["groupname"] = groupname
            recorded["consumername"] = consumername
            recorded["streams"] = streams
            return original(groupname, consumername, streams, count=count, block=block)

        client.xreadgroup = _spy
        consumer, _ = _make_consumer(client)

        asyncio.run(consumer.poll_once())

        assert recorded["groupname"] == CONSUMER_GROUP
        assert recorded["consumername"] == CONSUMER_NAME
        assert recorded["streams"] == {DAILY_COMPLETE_STREAM: ">"}

    def test_completeness_is_not_decided_from_event_fields(self):
        """AC-AIF-003: 휴장일 이벤트(succeeded=0)여도 이벤트 필드만으로 추론을
        건너뛰지 않는다 — 자체 DB 조회 결과가 판정 근거다."""
        client = _FakeRedis(new_messages=[("1-1", _event(succeeded="0"))])
        consumer, spawn = _make_consumer(client)

        asyncio.run(consumer.poll_once())

        assert spawn.calls == ["domestic"]

    def test_missing_trade_date_acks_without_spawning(self):
        client = _FakeRedis(new_messages=[("1-1", _event())])
        consumer, spawn = _make_consumer(client, trade_date=None)

        asyncio.run(consumer.poll_once())

        assert spawn.calls == []
        assert client.acked == ["1-1"]

    def test_event_without_market_field_is_acked_without_spawning(self):
        client = _FakeRedis(new_messages=[("1-1", {"attempted": "0"})])
        consumer, spawn = _make_consumer(client)

        asyncio.run(consumer.poll_once())

        assert spawn.calls == []
        assert client.acked == ["1-1"]

    def test_nonzero_child_exit_code_still_acks(self):
        """종료코드 1/2는 '완주'를 뜻하므로 메시지는 확인응답된다."""
        client = _FakeRedis(new_messages=[("1-1", _event())])
        consumer, _ = _make_consumer(client, spawn=_SpawnRecorder(exit_code=2))

        asyncio.run(consumer.poll_once())

        assert client.acked == ["1-1"]


class TestStaleMessageClaiming:
    def test_stale_message_is_claimed_with_the_configured_idle_threshold(self):
        client = _FakeRedis(
            claimable=[("5-5", _event())],
            pending=[{"message_id": "5-5", "times_delivered": 2}],
        )
        consumer, spawn = _make_consumer(client, idle_seconds=600)

        handled = asyncio.run(consumer.poll_once())

        assert handled == 1
        assert spawn.calls == ["domestic"]
        assert client.autoclaim_calls[0]["min_idle_time"] == 600_000
        assert client.autoclaim_calls[0]["consumername"] == CONSUMER_NAME

    def test_message_over_the_delivery_limit_goes_to_the_dlq(self):
        """AC-AIF-002: 재전달 3회 초과 메시지는 DLQ로 이관되고 원본에서
        XACK된다 — 자식 프로세스를 다시 스폰하지 않는다."""
        client = _FakeRedis(
            claimable=[("5-5", _event())],
            pending=[{"message_id": "5-5", "times_delivered": MAX_DELIVERY_COUNT + 1}],
        )
        consumer, spawn = _make_consumer(client)

        asyncio.run(consumer.poll_once())

        assert spawn.calls == []
        assert client.acked == ["5-5"]
        assert len(client.added) == 1
        stream_name, fields = client.added[0]
        assert stream_name == DLQ_STREAM
        assert fields["market"] == "domestic"
        assert fields["original_id"] == "5-5"

    def test_delivery_limit_is_three(self):
        assert MAX_DELIVERY_COUNT == 3


class TestInflightLock:
    def test_lock_is_released_after_the_child_completes(self):
        client = _FakeRedis(new_messages=[("1-1", _event())])
        consumer, _ = _make_consumer(client)

        asyncio.run(consumer.poll_once())

        assert client.values == {}

    def test_duplicate_spawn_is_blocked_while_a_child_is_in_flight(self):
        """AC-AIF-011: 동일 (market, trade_date)의 락이 이미 잡혀 있으면 두 번째
        스폰은 취소되고, 메시지는 재소유 가능하도록 pending 상태로 남는다."""
        client = _FakeRedis(new_messages=[("1-1", _event())])
        client.values[inflight_lock_key("domestic", _TRADE_DATE)] = "other-owner"
        consumer, spawn = _make_consumer(client)

        handled = asyncio.run(consumer.poll_once())

        assert handled == 0
        assert spawn.calls == []
        assert client.acked == []
        assert client.values[inflight_lock_key("domestic", _TRADE_DATE)] == "other-owner"

    def test_lock_is_released_even_when_the_spawn_raises(self):
        client = _FakeRedis(new_messages=[("1-1", _event())])
        spawn = _SpawnRecorder()

        def _boom():
            raise OSError("자식 프로세스 생성 실패")

        spawn.on_call = _boom
        consumer, _ = _make_consumer(client, spawn=spawn)

        with pytest.raises(OSError):
            asyncio.run(consumer.poll_once())

        assert client.values == {}
        assert client.acked == []


class TestStartLoop:
    def test_start_keeps_polling_until_cancelled(self):
        client = _FakeRedis(new_messages=[("1-1", _event())])
        consumer, spawn = _make_consumer(client)

        async def _drive() -> None:
            task = asyncio.create_task(consumer.start())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_drive())

        assert spawn.calls == ["domestic"]

    def test_transient_errors_do_not_stop_the_loop(self, caplog: pytest.LogCaptureFixture):
        client = _FakeRedis(new_messages=[("1-1", _event())])
        consumer, spawn = _make_consumer(client)
        failures = {"remaining": 1}
        original = consumer.poll_once

        async def _flaky() -> int:
            if failures["remaining"]:
                failures["remaining"] -= 1
                raise ConnectionError("redis unreachable")
            return await original()

        consumer.poll_once = _flaky
        consumer.error_backoff_seconds = 0.01

        async def _drive() -> None:
            task = asyncio.create_task(consumer.start())
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with caplog.at_level("ERROR"):
            asyncio.run(_drive())

        assert spawn.calls == ["domestic"]
        assert any(
            "redis unreachable" in record.message or record.exc_info for record in caplog.records
        )


class TestLazyDependencies:
    """주입 없이 생성된 컨슈머가 config/클라이언트/거래일 해석기를 지연
    구성하는 경로 — 프로덕션 기동 시 실제로 타는 경로다."""

    def test_config_is_loaded_lazily_and_only_once(self, monkeypatch: pytest.MonkeyPatch):
        import analyzer.orchestration.consumer as consumer_module

        loads = {"count": 0}

        def _fake_get_inference_config() -> InferenceConfig:
            loads["count"] += 1
            return _config()

        monkeypatch.setattr(consumer_module, "get_inference_config", _fake_get_inference_config)
        consumer = StreamConsumer(
            client=_FakeRedis(), resolve_trade_date=lambda market: _TRADE_DATE
        )

        asyncio.run(consumer.poll_once())
        asyncio.run(consumer.poll_once())

        assert loads["count"] == 1

    def test_client_is_built_lazily_and_only_once(self, monkeypatch: pytest.MonkeyPatch):
        import analyzer.orchestration.consumer as consumer_module

        client = _FakeRedis()
        built: list[object] = []

        def _fake_build_redis_client(config: object) -> _FakeRedis:
            built.append(config)
            return client

        monkeypatch.setattr(consumer_module, "build_redis_client", _fake_build_redis_client)
        consumer = StreamConsumer(config=_config(), resolve_trade_date=lambda market: _TRADE_DATE)

        asyncio.run(consumer.poll_once())
        asyncio.run(consumer.poll_once())

        assert len(built) == 1

    def test_default_trade_date_resolver_builds_the_engine_once(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import analyzer.orchestration.consumer as consumer_module

        engines: list[object] = []

        def _fake_build_engine(config: object) -> str:
            engines.append(config)
            return "engine"

        monkeypatch.setattr(consumer_module, "get_db_config", lambda: "db-config")
        monkeypatch.setattr(consumer_module, "build_engine", _fake_build_engine)
        monkeypatch.setattr(
            consumer_module, "resolve_trade_date", lambda engine, market: _TRADE_DATE
        )

        resolve = consumer_module._build_default_trade_date_resolver()

        assert resolve("domestic") == _TRADE_DATE
        assert resolve("overseas") == _TRADE_DATE
        assert engines == ["db-config"]

"""`observability:analyzer:last-cycle:{market}` Redis 영속화 + warm-start
테스트 (SPEC-OBSV-ANALYZER-DEADMAN-001 M1/M2, REQ-DMR-001~005).

`aaa-collector`의 `BatchLastLoadRepository`/`BatchMetricsWarmStarter` 패턴을
계승한다 — write(자식 프로세스 사이클 완료)와 read(부모 프로세스 부팅
warm-start)가 동일 Redis 키 파생을 공유하고, 양쪽 모두 Redis 실패를
흡수해(fail-open) 예외를 전파하지 않는다.

Redis는 `tests/test_inference_publish.py`와 동일하게 필요한 명령만
흉내내는 페이크로 대체한다(`fakeredis` 미도입 — 기존 관례 계승).
"""

from __future__ import annotations

import redis.exceptions
from prometheus_client import CollectorRegistry

from analyzer.inference.last_cycle import (
    ANALYZER_LAST_CYCLE_KEY_PREFIX,
    AnalyzerLastCycleRepository,
    record_cycle_completion,
    warm_start_last_cycle,
)
from analyzer.inference.metrics import INFERENCE_LAST_CYCLE_NAME, InferenceMetrics


class _FakeRedis:
    """`get`/`set`만 흉내내는 페이크 — 저장소는 호출부가 직접 들여다볼 수
    있는 평범한 dict다."""

    def __init__(self, store: dict[str, str] | None = None) -> None:
        self.store: dict[str, str] = store if store is not None else {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str) -> None:
        self.store[key] = value


class _FailingRedis:
    """모든 명령이 `RedisError`(의 서브클래스인 `ConnectionError`)를 던지는
    페이크 — Redis 두절/장애를 시뮬레이션한다."""

    def get(self, key: str) -> str | None:
        raise redis.exceptions.ConnectionError("연결 불가(시뮬레이션)")

    def set(self, key: str, value: str) -> None:
        raise redis.exceptions.ConnectionError("연결 불가(시뮬레이션)")


class TestAnalyzerLastCycleKeyScheme:
    """AC-DMR-002: Redis 키는 `observability:analyzer:last-cycle:{market}`
    형태이며 TTL 없이 저장된다(TTL은 `_FakeRedis`가 지원하지 않으므로 —
    실제 `redis.Redis.set()`은 `ex`/`px` 인자를 주지 않으면 TTL 없이
    저장한다 — 이 계약은 `save()`가 `ex`/`px`를 넘기지 않는다는 사실로
    검증한다)."""

    def test_key_prefix_matches_the_contract(self):
        assert ANALYZER_LAST_CYCLE_KEY_PREFIX == "observability:analyzer:last-cycle:"

    def test_save_writes_under_the_market_scoped_key(self):
        repo = AnalyzerLastCycleRepository(_FakeRedis())

        repo.save("domestic", 1_758_000_000.0)

        assert "observability:analyzer:last-cycle:domestic" in repo._redis_client.store  # noqa: SLF001

    def test_save_does_not_set_a_ttl(self):
        """`set()`이 `ex`/`px` 키워드 없이 호출됐는지 직접 확인한다 — TTL
        없는 저장이 회귀로 깨지면 여기서 잡힌다."""
        calls: list[tuple[tuple, dict]] = []

        class _RecordingRedis:
            def set(self, *args, **kwargs) -> None:
                calls.append((args, kwargs))

            def get(self, key: str) -> str | None:
                return None

        repo = AnalyzerLastCycleRepository(_RecordingRedis())
        repo.save("domestic", 1.0)

        assert len(calls) == 1
        _, kwargs = calls[0]
        assert "ex" not in kwargs
        assert "px" not in kwargs


class TestAnalyzerLastCycleRepositorySaveFind:
    """AC-DMR-002 왕복(round-trip): save 후 find는 동일한 UTC epoch 초를
    반환한다."""

    def test_save_and_find_round_trip(self):
        repo = AnalyzerLastCycleRepository(_FakeRedis())

        repo.save("domestic", 1_758_000_000.0)

        assert repo.find("domestic") == 1_758_000_000.0

    def test_markets_do_not_share_a_key(self):
        repo = AnalyzerLastCycleRepository(_FakeRedis())

        repo.save("domestic", 1.0)
        repo.save("overseas", 2.0)

        assert repo.find("domestic") == 1.0
        assert repo.find("overseas") == 2.0

    def test_find_returns_none_when_key_absent(self):
        repo = AnalyzerLastCycleRepository(_FakeRedis())

        assert repo.find("domestic") is None

    def test_stored_value_is_a_plain_epoch_second_string(self):
        """Java `BatchLastLoadRepository.save()`의 `Long.toString(epochSeconds)`
        관례를 계승 — 부동소수점 잔재(`.0`) 없는 정수 문자열로 저장한다."""
        store: dict[str, str] = {}
        repo = AnalyzerLastCycleRepository(_FakeRedis(store))

        repo.save("domestic", 1_758_000_000.0)

        assert store["observability:analyzer:last-cycle:domestic"] == "1758000000"


class TestAnalyzerLastCycleRepositoryCorruptedValueHandling:
    """Edge Cases(acceptance.md): 비숫자·범위 밖 손상값은 `Optional.empty()`에
    대응하는 `None`으로 흡수한다 — 예외를 전파해 warm-start/기동을 막지
    않는다(collector `BatchLastLoadRepository.find()`의
    `NumberFormatException`/`DateTimeException` 처리 관례의 Python 등가물)."""

    def test_non_numeric_value_is_absorbed_as_none(self):
        store = {"observability:analyzer:last-cycle:domestic": "not-a-number"}
        repo = AnalyzerLastCycleRepository(_FakeRedis(store))

        assert repo.find("domestic") is None

    def test_astronomically_large_value_is_absorbed_as_none(self):
        """Python `int`는 임의 정밀도라 `ValueError`를 내지 않지만,
        `float()` 변환에서 `OverflowError`가 발생하는 값 — 이 역시 손상값의
        한 형태로 흡수한다."""
        store = {"observability:analyzer:last-cycle:domestic": str(10**400)}
        repo = AnalyzerLastCycleRepository(_FakeRedis(store))

        assert repo.find("domestic") is None

    def test_empty_string_value_is_absorbed_as_none(self):
        store = {"observability:analyzer:last-cycle:domestic": ""}
        repo = AnalyzerLastCycleRepository(_FakeRedis(store))

        assert repo.find("domestic") is None


class TestRecordCycleCompletion:
    """REQ-DMR-002/004: 사이클 완료 시 게이지 갱신 + Redis 영속화를 함께
    수행하며, Redis 실패는 흡수하고 게이지 갱신은 흡수하지 않는다."""

    def test_sets_gauge_and_persists_to_redis(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        store: dict[str, str] = {}
        repo = AnalyzerLastCycleRepository(_FakeRedis(store))

        record_cycle_completion(metrics, repo, market="domestic", epoch_seconds=1_758_000_000.0)

        value = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        assert value == 1_758_000_000.0
        assert store["observability:analyzer:last-cycle:domestic"] == "1758000000"

    def test_gauge_is_set_even_when_redis_save_fails(self):
        """REQ-DMR-004 fail-open: Redis 저장 실패가 게이지 갱신(이미 완료된
        인메모리 연산)을 되돌리지 않는다."""
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        repo = AnalyzerLastCycleRepository(_FailingRedis())

        record_cycle_completion(metrics, repo, market="domestic", epoch_seconds=1_758_000_000.0)

        value = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        assert value == 1_758_000_000.0

    def test_redis_save_failure_does_not_propagate(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        repo = AnalyzerLastCycleRepository(_FailingRedis())

        # 예외를 던지면 이 호출 자체가 테스트를 실패시킨다(암묵적 단언).
        record_cycle_completion(metrics, repo, market="domestic", epoch_seconds=1.0)


class TestWarmStartLastCycle:
    """REQ-DMR-003: 부모 프로세스 기동 시 Redis 영속 완료 시각으로 게이지를
    warm-start한다 — Redis 조회 실패/키 부재는 기동을 막지 않고 게이지를
    부재 상태로 남긴다."""

    def test_warms_gauge_from_stored_value(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        store = {"observability:analyzer:last-cycle:domestic": "1758000000"}
        repo = AnalyzerLastCycleRepository(_FakeRedis(store))

        warm_start_last_cycle(metrics, repo, market="domestic")

        value = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        assert value == 1_758_000_000.0

    def test_absent_key_leaves_gauge_unset(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        repo = AnalyzerLastCycleRepository(_FakeRedis())

        warm_start_last_cycle(metrics, repo, market="domestic")

        value = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        assert value is None

    def test_redis_lookup_failure_does_not_propagate(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        repo = AnalyzerLastCycleRepository(_FailingRedis())

        # 예외를 던지면 이 호출 자체가 테스트를 실패시킨다(암묵적 단언).
        warm_start_last_cycle(metrics, repo, market="domestic")

    def test_redis_lookup_failure_leaves_gauge_unset(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        repo = AnalyzerLastCycleRepository(_FailingRedis())

        warm_start_last_cycle(metrics, repo, market="domestic")

        value = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        assert value is None

    def test_corrupted_value_leaves_gauge_unset(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        store = {"observability:analyzer:last-cycle:domestic": "not-a-number"}
        repo = AnalyzerLastCycleRepository(_FakeRedis(store))

        warm_start_last_cycle(metrics, repo, market="domestic")

        value = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        assert value is None

    def test_markets_warm_independently(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        store = {
            "observability:analyzer:last-cycle:domestic": "1758000000",
            "observability:analyzer:last-cycle:overseas": "1758003600",
        }
        repo = AnalyzerLastCycleRepository(_FakeRedis(store))

        warm_start_last_cycle(metrics, repo, market="domestic")
        warm_start_last_cycle(metrics, repo, market="overseas")

        domestic = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        overseas = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "overseas"})
        assert domestic == 1_758_000_000.0
        assert overseas == 1_758_003_600.0

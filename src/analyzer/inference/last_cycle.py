"""`aaa_analyzer_inference_last_cycle_seconds` 게이지의 Redis 영속화 +
warm-start 배선 (SPEC-OBSV-ANALYZER-DEADMAN-001 M1/M2, REQ-DMR-001~005).

`aaa-collector`의 `BatchLastLoadRepository`/`BatchMetricsWarmStarter`
패턴(SPEC-COLLECTOR-WARMSTART-REDIS-001)을 analyzer 측에 그대로 계승한다:

- **write(자식 프로세스, `record_cycle_completion`)** — 시장별 추론 사이클이
  성공적으로 완료되면 완료 시각(UTC epoch 초)을 게이지에 기록하는 동시에
  Redis에 TTL 없이 영속화한다(REQ-DMR-002).
- **read(부모 프로세스, `warm_start_last_cycle`)** — 상주 부모가 기동하면
  각 시장의 Redis 영속 완료 시각을 조회해 게이지를 warm-start한다
  (REQ-DMR-003). `PROMETHEUS_MULTIPROC_DIR`가 재기동마다 초기화되므로
  이 warm-start가 없으면 재기동 직후 게이지가 완전히 소실된다(spec.md §1.1).

두 경로 모두 Redis 실패(연결 불가, 손상값)를 흡수하고 예외를 전파하지
않는다(REQ-DMR-003/004 fail-open) — collector `BatchLastLoadRepository`의
`Optional.empty()` 폴백을 Python에서는 `None` 반환 + 상위 호출부의
무조건 계속 진행으로 재현한다.
"""

from __future__ import annotations

from typing import Any

import redis.exceptions

from analyzer.common.logging import get_logger
from analyzer.inference.metrics import InferenceMetrics

logger = get_logger(__name__)

ANALYZER_LAST_CYCLE_KEY_PREFIX = "observability:analyzer:last-cycle:"
"""collector `BatchLastLoadRepository.KEY_PREFIX`(`observability:collector:
last-load:`)와 동일 관례 — 네임스페이스만 `analyzer`/`last-cycle`로 교체."""


class AnalyzerLastCycleRepository:
    """시장별 마지막 성공 완료 시각을 Redis에 영속화하는 레포지토리.

    write(`save`)와 read(`find`)는 동일한 `_key()` 파생 로직을 공유한다 —
    키가 어긋나면 기록은 되나 복원되지 않는 은묵 결함이 생기므로 단일
    파생 소스로 캡슐화한다(collector REQ-WSR-008 동일 원칙).
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis_client = redis_client

    def save(self, market: str, epoch_seconds: float) -> None:
        """시장의 마지막 성공 epoch를 TTL 없이(영속) UTC epoch 초 문자열로
        저장한다(REQ-DMR-002). `int()`로 절삭해 부동소수점 잔재 없는 정수
        문자열을 만든다 — Java `Long.toString(epochSeconds)`와 동일 형태."""
        self._redis_client.set(self._key(market), str(int(epoch_seconds)))

    def find(self, market: str) -> float | None:
        """시장의 마지막 성공 시각을 조회한다(REQ-DMR-003 read-side).

        저장값이 손상되면 예외를 전파하지 않고 `None`을 반환한다 — (1)
        비숫자라 파싱에 실패하는 경우(`ValueError`), (2) 파싱은 되나
        `float` 변환에서 범위를 벗어나는 경우(`OverflowError`) 둘 다
        해당한다(collector `BatchLastLoadRepository.find()`의
        `NumberFormatException`/`DateTimeException` 처리 관례 계승).
        """
        raw = self._redis_client.get(self._key(market))
        if raw is None:
            return None
        try:
            return float(int(raw))
        except ValueError, OverflowError:
            logger.warning(
                "AnalyzerLastCycle 손상값 무시 market=%s raw=%r (warm-start 폴백)",
                market,
                raw,
            )
            return None

    def _key(self, market: str) -> str:
        return f"{ANALYZER_LAST_CYCLE_KEY_PREFIX}{market}"


def record_cycle_completion(
    metrics: InferenceMetrics,
    repository: AnalyzerLastCycleRepository,
    *,
    market: str,
    epoch_seconds: float,
) -> None:
    """REQ-DMR-002: 시장별 추론 사이클 완료를 게이지(즉시)와 Redis(영속)
    양쪽에 동시에 기록한다.

    게이지 갱신은 순수 인메모리(mmap) 연산이라 실패하지 않는다고 간주한다
    — Redis 저장 실패만 흡수 대상이다(REQ-DMR-004 fail-open 원칙 확장
    적용). Redis 실패는 게이지가 이미 갱신된 뒤에 발생하므로, 저장에
    실패해도 이번 사이클 값 자체는 `/metrics`에서 즉시 관측 가능하다 —
    다음 재기동 전까지는 Redis 없이도 정확하고, 재기동 이후의 정확성만
    Redis 영속화에 의존한다.
    """
    metrics.record_last_cycle(market=market, epoch_seconds=epoch_seconds)
    try:
        repository.save(market, epoch_seconds)
    except redis.exceptions.RedisError:
        logger.warning(
            "AnalyzerLastCycle Redis 저장 실패 — 이번 사이클 값은 게이지에 "
            "반영됐으나 재기동 시 복원되지 않을 수 있음 market=%s",
            market,
            exc_info=True,
        )


def warm_start_last_cycle(
    metrics: InferenceMetrics,
    repository: AnalyzerLastCycleRepository,
    *,
    market: str,
) -> None:
    """REQ-DMR-003: 부모 프로세스 기동 시 Redis 영속 완료 시각으로 게이지를
    warm-start한다.

    Redis 조회 실패 또는 키 부재는 기동을 막지 않고 게이지를 부재(unset)
    상태로 남긴다(`BatchMetricsWarmStarter`의 비차단 관례 계승) — 이
    함수를 호출하는 쪽(`api/main.py`)은 이 함수가 예외를 던지지 않는다는
    보장에 기대어 순차 호출만 하면 된다.
    """
    try:
        epoch_seconds = repository.find(market)
    except redis.exceptions.RedisError:
        logger.warning(
            "AnalyzerLastCycle Redis 조회 실패 — warm-start 건너뜀(게이지 "
            "부재 상태 유지) market=%s",
            market,
            exc_info=True,
        )
        return
    if epoch_seconds is None:
        return
    metrics.record_last_cycle(market=market, epoch_seconds=epoch_seconds)

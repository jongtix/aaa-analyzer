"""Redis 클라이언트 팩토리 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-140).

공식 `redis` 동기 클라이언트를 사용한다 — 상주 부모는 asyncio 프로세스이지만
스트림 명령은 `asyncio.to_thread()`로 감싸 호출하므로 이벤트 루프를 막지
않는다(동기 SQLAlchemy + PyMySQL을 쓰는 DB 접근 관례와 동일한 선택).

SPEC-ANALYZER-REDIS-TIMEOUT-001(REQ-RT-010/020/030): `socket_timeout`을
`DEFAULT_BLOCK_MILLISECONDS`(단일 출처, `StreamConsumer`가 재수입)로부터
파생 계산한다. 과거에는 `socket_timeout`을 명시하지 않아 redis-py
라이브러리 자체 기본값(5초)이 XREADGROUP `BLOCK`(5000ms=5초)과 정확히
일치해 안전마진이 0이었다 — 도커 브릿지 네트워크 지연 등 아주 작은 추가
지연만으로도 클라이언트 로컬 소켓 read가 서버 응답보다 먼저 타임아웃되어
메시지가 PEL에 고아로 남는 결함(DLQ 유실 18건, 2026-09-04~09-16)을 냈다.
"""

from typing import Any

from redis import Redis

from analyzer.inference.config import InferenceConfig

DEFAULT_BLOCK_MILLISECONDS = 5_000
"""`StreamConsumer`의 XREADGROUP `BLOCK` 인자 기본값(밀리초)의 단일 출처
(SSOT). `orchestration/consumer.py`는 이 값을 재수입할 뿐 자체 리터럴을
선언하지 않는다(REQ-RT-040) — 반대 방향(consumer.py가 소유)은 순환
임포트를 만들기 때문에 하위 계층인 이 모듈이 소유한다."""

_SOCKET_TIMEOUT_MARGIN_SECONDS = 10.0
"""도커 브릿지 네트워크 지연·GC 정지·`asyncio.to_thread` 스케줄링 지연을
흡수하기 위한 안전마진(초). BLOCK의 3배 배수 — XREADGROUP 응답 대기 자체를
늘리지 않으면서(BLOCK은 불변) 오탐성 클라이언트측 TimeoutError를 없앤다."""

SOCKET_TIMEOUT_SECONDS = DEFAULT_BLOCK_MILLISECONDS / 1000 + _SOCKET_TIMEOUT_MARGIN_SECONDS
"""REQ-RT-020: `DEFAULT_BLOCK_MILLISECONDS`로부터 파생된 계산값 — 향후
BLOCK 기본값이 바뀌어도 안전마진 관계가 구조적으로 자동 유지된다."""

assert SOCKET_TIMEOUT_SECONDS > DEFAULT_BLOCK_MILLISECONDS / 1000, (
    "SOCKET_TIMEOUT_SECONDS가 DEFAULT_BLOCK_MILLISECONDS(초 환산)보다 커야 한다"
    " — 안전마진 불변식(REQ-RT-010/020) 위반"
)
"""모듈 임포트 시점에 불변식을 단언한다 — 파생식이 있는 한 항상 참이지만,
누군가 나중에 `SOCKET_TIMEOUT_SECONDS`를 독립 리터럴로 바꿔치기하는 회귀를
즉시 잡아낸다. `python -O` 실행 시 제거되는 것에 대비해
`tests/test_inference_redis_client.py`가 별도 방어선을 둔다(REQ-RT-060)."""


def build_redis_client(config: InferenceConfig) -> Any:
    """`InferenceConfig`로부터 Redis 클라이언트를 구성한다.

    `decode_responses=True`로 스트림 필드를 `str`로 받는다 — collector의
    `StringRedisTemplate` 발행 계약(전 필드 문자열)과 대칭이며, 컨슈머 코드가
    bytes/str 분기를 갖지 않게 한다.

    `socket_timeout=SOCKET_TIMEOUT_SECONDS`(REQ-RT-030)로 클라이언트측 로컬
    소켓 read 타임아웃이 XREADGROUP `BLOCK`보다 충분히 크도록 명시한다.
    """
    return Redis(
        host=config.redis_host,
        port=config.redis_port,
        username=config.redis_username,
        password=config.redis_password,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT_SECONDS,
    )

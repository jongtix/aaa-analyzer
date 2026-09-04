"""Redis 클라이언트 팩토리 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-140).

공식 `redis` 동기 클라이언트를 사용한다 — 상주 부모는 asyncio 프로세스이지만
스트림 명령은 `asyncio.to_thread()`로 감싸 호출하므로 이벤트 루프를 막지
않는다(동기 SQLAlchemy + PyMySQL을 쓰는 DB 접근 관례와 동일한 선택).
"""

from typing import Any

from redis import Redis

from analyzer.inference.config import InferenceConfig


def build_redis_client(config: InferenceConfig) -> Any:
    """`InferenceConfig`로부터 Redis 클라이언트를 구성한다.

    `decode_responses=True`로 스트림 필드를 `str`로 받는다 — collector의
    `StringRedisTemplate` 발행 계약(전 필드 문자열)과 대칭이며, 컨슈머 코드가
    bytes/str 분기를 갖지 않게 한다.
    """
    return Redis(
        host=config.redis_host,
        port=config.redis_port,
        username=config.redis_username,
        password=config.redis_password,
        decode_responses=True,
    )

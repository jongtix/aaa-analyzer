"""Redis Streams 컨슈머의 구조적 자리 표시자.

REQ-ANALYZER-FOUNDATION-010: 부모 프로세스는 스트림 컨슈머에 대해 구조적
자리 표시자(모듈 경계)만 제공한다 — 실제 구독 로직은 본 SPEC 범위 밖이며
후속 SPEC(INFER-001: `stream:daily:complete` 구독)에서 구현된다.
"""


class StreamConsumer:
    """향후 도입될 `stream:daily:complete` Redis Streams 컨슈머의 자리 표시자.

    @MX:NOTE: [AUTO] 구조적 자리만 제공.
    @MX:REASON: 실제 구독 로직은 후속 SPEC(INFER-001) 소관(REQ-ANALYZER-FOUNDATION-010).
    """

    async def start(self) -> None: ...

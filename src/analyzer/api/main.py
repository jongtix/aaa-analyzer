"""상주 부모 프로세스를 위한 asyncio 진입점.

FastAPI 앱을 스트림 컨슈머·스케줄러 구조적 자리 표시자(REQ-ANALYZER-FOUNDATION-007/010)와
연결하고, uvicorn으로 앱을 서빙한다. 실제 스트림 구독이나 스케줄러 잡 로직은
여기에 구현되어 있지 않다 — 후속 SPEC이 채울 모듈 경계는
`analyzer.orchestration.consumer` / `analyzer.orchestration.scheduler`를 참고한다.
"""

import asyncio

import uvicorn

from analyzer.api.app import create_app
from analyzer.common.logging import get_logger
from analyzer.orchestration.consumer import StreamConsumer
from analyzer.orchestration.scheduler import SchedulerRegistry

logger = get_logger(__name__)


async def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """상주 부모 프로세스를 시작한다: FastAPI 앱 + 오케스트레이션 자리 표시자."""
    consumer = StreamConsumer()
    scheduler = SchedulerRegistry()

    # 구조적 배선만 수행 — 실제 구독/잡 로직 없음(REQ-ANALYZER-FOUNDATION-010).
    await consumer.start()
    logger.info(
        "orchestration placeholders wired (jobs=%d)",
        len(scheduler.registered_jobs()),
    )

    app = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

"""상주 부모 프로세스를 위한 FastAPI 애플리케이션 팩토리.

REQ-ANALYZER-FOUNDATION-007/008/009: 부모 프로세스는 단일 asyncio FastAPI 앱을
호스팅하며 GET /health(헬스 페이로드)와 GET /metrics(prometheus_client
exposition 포맷)를 노출한다.
"""

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


def create_app() -> FastAPI:
    """FastAPI 애플리케이션 인스턴스를 구성해 반환한다."""
    app = FastAPI(title="aaa-analyzer")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app

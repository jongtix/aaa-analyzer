"""상주 부모 프로세스를 위한 FastAPI 애플리케이션 팩토리.

REQ-ANALYZER-FOUNDATION-007/008/009: 부모 프로세스는 단일 asyncio FastAPI 앱을
호스팅하며 GET /health(헬스 페이로드)와 GET /metrics(prometheus_client
exposition 포맷)를 노출한다.

SPEC-ANALYZER-PIPELINE-001 REQ-APL-131: `PROMETHEUS_MULTIPROC_DIR`이 설정돼
있으면 자식 프로세스가 남긴 pid별 메트릭 덤프 파일을 `multiprocess.
MultiProcessCollector`로 합산해 노출한다 — 설정돼 있지 않으면(로컬 개발/
테스트) 기존 `generate_latest(REGISTRY)` 동작을 그대로 유지한다(로컬 개발이
인프라 의존성을 강제로 요구하지 않는다).
"""

import os

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess


def create_app() -> FastAPI:
    """FastAPI 애플리케이션 인스턴스를 구성해 반환한다."""
    app = FastAPI(title="aaa-analyzer")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app

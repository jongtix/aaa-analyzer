"""FastAPI 부모 프로세스 골격에 대한 명세 테스트.

REQ-ANALYZER-FOUNDATION-007/008/009/010: 상주 부모 프로세스는 단일 asyncio
FastAPI 앱을 호스팅한다; GET /health는 헬스 페이로드를 반환한다; GET /metrics는
prometheus_client exposition 포맷 메트릭을 반환한다; orchestration/은 스트림
컨슈머와 스케줄러에 대해 구조적 자리 표시자(구독/잡 로직 없음)만 제공한다.
"""

import asyncio

from prometheus_client import CONTENT_TYPE_LATEST
from starlette.testclient import TestClient

from analyzer.api.app import create_app
from analyzer.orchestration.consumer import StreamConsumer
from analyzer.orchestration.scheduler import SchedulerRegistry


class TestHealthEndpoint:
    def test_health_returns_200_ok_status(self):
        client = TestClient(create_app())

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        client = TestClient(create_app())

        response = client.get("/metrics")

        assert response.status_code == 200

    def test_metrics_uses_prometheus_content_type(self):
        client = TestClient(create_app())

        response = client.get("/metrics")

        assert response.headers["content-type"] == CONTENT_TYPE_LATEST


class TestMainEntrypoint:
    def test_run_wires_placeholders_then_serves(self, monkeypatch):
        from analyzer.api import main

        served = {"called": False}

        async def fake_serve(self):
            served["called"] = True

        monkeypatch.setattr("uvicorn.Server.serve", fake_serve)

        asyncio.run(main.run(host="127.0.0.1", port=8001))

        assert served["called"] is True


class TestOrchestrationPlaceholders:
    def test_stream_consumer_start_is_a_no_op_placeholder(self):
        consumer = StreamConsumer()

        # 아직 구독/소비 로직은 없다(INFER-001 범위) —
        # start() 호출은 예외 없이 즉시 완료되어야 한다.
        result = asyncio.run(consumer.start())

        assert result is None

    def test_scheduler_registry_starts_with_no_registered_jobs(self):
        registry = SchedulerRegistry()

        assert registry.registered_jobs() == []

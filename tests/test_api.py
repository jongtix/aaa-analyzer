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
    def test_run_wires_placeholders_then_serves(self, monkeypatch, tmp_path):
        """SPEC-ANALYZER-TRAIN-GATE-001 M5: run()은 이제 기동 시 주간 재학습
        cron 잡을 배선한다(G-1 fail-fast 포함) — 이 테스트는 설정 로딩과
        스케줄러 기동을 모킹해 uvicorn.serve() 호출까지 도달함을 확인한다."""
        from analyzer.api import main
        from analyzer.orchestration.config import AutomationConfig

        served = {"called": False}

        async def fake_serve(self):
            served["called"] = True

        monkeypatch.setattr("uvicorn.Server.serve", fake_serve)

        fake_config = AutomationConfig(
            target_mac_address="AA:BB:CC:DD:EE:FF",
            ssh_host="macbook.local",
            ssh_port=22,
            ssh_username="dispatch",
            ssh_private_key_path=tmp_path / "dispatch_key",
            known_hosts_path=tmp_path / "known_hosts",
            db_tunnel_host="nas-host",
            db_tunnel_port=22,
            db_tunnel_username="db_tunnel",
            db_tunnel_private_key_path=tmp_path / "db_tunnel_key",
            db_tunnel_local_port=3306,
            db_tunnel_remote_port=3306,
            weekly_timeout_seconds=14400,
            monthly_timeout_seconds=129600,
            staleness_threshold_days=28,
            staging_models_root=tmp_path / "staging",
            active_models_root=tmp_path / "models",
            container_models_root=tmp_path / "container-models",
            cache_dir=tmp_path / "cache",
            calendar_code="KRX",
            feature_code_version="v1",
            mount_script_path=tmp_path / "mount-nas-hdd1.sh",
            python_executable_path=tmp_path / ".venv" / "bin" / "python",
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=tmp_path / "logs" / "aaa-analyzer",
            monthly_optuna_storage_dir=tmp_path / "optuna" / "monthly",
            monthly_summary_report_path=tmp_path / "reports" / "monthly-campaign-summary.json",
        )
        # 전역 Prometheus 레지스트리 오염 방지(테스트 격리) — TrainingMetrics()가
        # 기본 인자로 REGISTRY를 사용하면 다른 테스트의 레지스트리 격리 검증과
        # 충돌한다(test_orchestration_metrics.py::test_uses_injected_registry_not_default).
        from prometheus_client import CollectorRegistry

        from analyzer.orchestration.metrics import TrainingMetrics

        shutdown_calls: list[bool] = []

        monkeypatch.setattr(main, "get_automation_config", lambda: fake_config)
        monkeypatch.setattr(
            main, "TrainingMetrics", lambda: TrainingMetrics(registry=CollectorRegistry())
        )
        monkeypatch.setattr(main.SchedulerRegistry, "start", lambda self: None)
        monkeypatch.setattr(
            main.SchedulerRegistry, "shutdown", lambda self, wait=True: shutdown_calls.append(True)
        )

        asyncio.run(main.run(host="127.0.0.1", port=8001))

        assert served["called"] is True
        # REQ-ATG-001: 프로세스 종료 경로에서 shutdown() 훅이 호출되어야 한다.
        assert shutdown_calls == [True]


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

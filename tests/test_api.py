"""FastAPI 부모 프로세스 골격에 대한 명세 테스트.

REQ-ANALYZER-FOUNDATION-007/008/009: 상주 부모 프로세스는 단일 asyncio
FastAPI 앱을 호스팅한다; GET /health는 헬스 페이로드를 반환한다; GET /metrics는
prometheus_client exposition 포맷 메트릭을 반환한다.

SPEC-ANALYZER-INFER-001 M1: FOUNDATION-001의 `StreamConsumer` 자리 표시자
검증은 실제 구독 배선 검증으로 대체됐다 — `run()`은 컨슈머를 백그라운드
asyncio 태스크로 기동하고 종료 경로에서 취소한다(컨슈머 자체의 동작은
`tests/test_orchestration_consumer.py` 소관).
"""

import asyncio
from pathlib import Path

import pytest
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry
from prometheus_client import values as prometheus_values
from starlette.testclient import TestClient

from analyzer.api.app import create_app
from analyzer.inference.config import InferenceConfig
from analyzer.inference.last_cycle import AnalyzerLastCycleRepository, warm_start_last_cycle
from analyzer.inference.metrics import (
    INFERENCE_LAST_CYCLE_NAME,
    INFERENCE_SKIP_TOTAL_NAME,
    InferenceMetrics,
)
from analyzer.inference.resolution import SkipReason
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


class TestMetricsMultiprocessMode:
    """SPEC-ANALYZER-PIPELINE-001 REQ-APL-131/AC-APL-131: `PROMETHEUS_
    MULTIPROC_DIR`이 설정되면 `/metrics`는 `multiprocess.MultiProcessCollector`
    로 여러 pid의 덤프 파일을 합산해 노출한다 — 미설정 시(로컬/CI 기본 경로)
    기존 `generate_latest(REGISTRY)` 단일-프로세스 동작을 그대로 유지한다."""

    def _write_pid_dump(self, tmp_path: Path, pid: int, skip_count: int) -> None:
        from prometheus_client import CollectorRegistry

        original_value_class = prometheus_values.ValueClass
        try:
            prometheus_values.ValueClass = prometheus_values.MultiProcessValue(
                process_identifier=lambda: pid
            )
            registry = CollectorRegistry()
            metrics = InferenceMetrics(registry=registry)
            for _ in range(skip_count):
                metrics.record_skip(market="domestic", horizon=20, reason=SkipReason.NO_MANIFEST)
        finally:
            prometheus_values.ValueClass = original_value_class

    def test_aggregates_multiple_pid_dumps_when_env_var_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
        self._write_pid_dump(tmp_path, pid=111, skip_count=3)
        self._write_pid_dump(tmp_path, pid=222, skip_count=2)

        client = TestClient(create_app())
        response = client.get("/metrics")

        assert response.status_code == 200
        body = response.text
        assert (
            f'{INFERENCE_SKIP_TOTAL_NAME}{{horizon="20",market="domestic",'
            'reason="no_manifest"} 5.0' in body
        )

    def test_falls_back_to_default_registry_when_env_var_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
        client = TestClient(create_app())

        response = client.get("/metrics")

        assert response.status_code == 200


class TestInferenceLastCycleRestartSimulation:
    """SPEC-OBSV-ANALYZER-DEADMAN-001 AC-DMR-006(REQ-DMR-005): 재기동으로
    `PROMETHEUS_MULTIPROC_DIR`의 잔존 pid db 파일이 전부 삭제된 뒤에도,
    (a) Redis 복원 기준선(warm-start)과 (b) 그 이후 실제 완료된 새 사이클
    중 항상 더 최근 값(=`multiprocess_mode="max"` 병합 결과)이 `/metrics`에
    노출된다 — 재기동 자체가 이 신호를 완전히 소실시키지 않는다."""

    def _with_pid_identity(self, pid: int, fn) -> None:  # noqa: ANN001
        original_value_class = prometheus_values.ValueClass
        try:
            prometheus_values.ValueClass = prometheus_values.MultiProcessValue(
                process_identifier=lambda: pid
            )
            fn()
        finally:
            prometheus_values.ValueClass = original_value_class

    def _write_last_cycle_as_pid(self, pid: int, market: str, epoch_seconds: float) -> None:
        """자식 프로세스(pid)가 사이클 완료를 게이지에 직접 기록하는 것을
        시뮬레이션한다."""

        def _write() -> None:
            registry = CollectorRegistry()
            metrics = InferenceMetrics(registry=registry)
            metrics.record_last_cycle(market=market, epoch_seconds=epoch_seconds)

        self._with_pid_identity(pid, _write)

    def _warm_start_as_pid(self, pid: int, market: str, repository) -> None:  # noqa: ANN001
        """상주 부모 프로세스(pid)가 자기 자신의 pid로 warm-start하는 것을
        시뮬레이션한다."""

        def _warm() -> None:
            registry = CollectorRegistry()
            metrics = InferenceMetrics(registry=registry)
            warm_start_last_cycle(metrics, repository, market=market)

        self._with_pid_identity(pid, _warm)

    def _last_cycle_value(self, body: str, market: str) -> float | None:
        """`/metrics` 텍스트 응답을 파싱해 `market` 레이블의 last-cycle
        게이지 값을 반환한다 — Prometheus exposition 포맷은 큰 float를
        지수 표기(`1.758e+09`)로 렌더링하므로 원본 텍스트를 그대로
        문자열 비교하면 오탐(false negative)이 난다. 파서를 거쳐 실제
        숫자값으로 비교한다."""
        from prometheus_client.parser import text_string_to_metric_families

        for family in text_string_to_metric_families(body):
            if family.name != INFERENCE_LAST_CYCLE_NAME:
                continue
            for sample in family.samples:
                if sample.labels.get("market") == market:
                    return sample.value
        return None

    def test_restart_survives_via_warm_start_then_yields_to_newer_cycle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        from analyzer.api.main import _bootstrap_prometheus_multiproc_dir

        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
        client = TestClient(create_app())

        # 1) 재기동 이전: 자식 프로세스(pid=111)가 사이클을 완료해 게이지에
        #    기록한다.
        pre_restart_epoch = 1_758_000_000.0
        self._write_last_cycle_as_pid(pid=111, market="domestic", epoch_seconds=pre_restart_epoch)

        before_restart = client.get("/metrics")
        assert before_restart.status_code == 200
        assert self._last_cycle_value(before_restart.text, "domestic") == pre_restart_epoch

        # 2) 재기동 시뮬레이션: 부트스트랩이 잔존 pid db 파일을 전부 삭제한다
        #    — warm-start 전이므로 게이지가 완전히 소실된다(구 결함 재현 확인).
        _bootstrap_prometheus_multiproc_dir()

        after_bootstrap = client.get("/metrics")
        assert after_bootstrap.status_code == 200
        assert self._last_cycle_value(after_bootstrap.text, "domestic") is None

        # 3) warm-start: 상주 부모(pid=999)가 Redis 복원값으로 게이지를
        #    채운다 — 재기동 이후에도 신호가 완전히 소실되지 않는다.
        store = {"observability:analyzer:last-cycle:domestic": str(int(pre_restart_epoch))}

        class _FakeRedis:
            def get(self, key: str) -> str | None:
                return store.get(key)

        repository = AnalyzerLastCycleRepository(_FakeRedis())
        self._warm_start_as_pid(pid=999, market="domestic", repository=repository)

        after_warm_start = client.get("/metrics")
        assert after_warm_start.status_code == 200
        assert self._last_cycle_value(after_warm_start.text, "domestic") == pre_restart_epoch

        # 4) 재기동 이후 실제 새 사이클 완료: 자식(pid=222)이 더 최근 값을
        #    기록한다 — REQ-DMR-005: (a) warm-start 기준선과 (b) 그 이후
        #    실제 완료 시각 중 더 최근 값(max)이 노출돼야 한다.
        newer_epoch = pre_restart_epoch + 3600.0
        self._write_last_cycle_as_pid(pid=222, market="domestic", epoch_seconds=newer_epoch)

        after_new_cycle = client.get("/metrics")
        assert after_new_cycle.status_code == 200
        # 병합 결과는 최댓값(max)이어야 한다 — 낡은 warm-start 기준선이 아니라
        # 방금 완료된 새 사이클의 값이 노출된다.
        assert self._last_cycle_value(after_new_cycle.text, "domestic") == newer_epoch

    def test_markets_do_not_cross_contaminate_during_restart(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Edge Cases(acceptance.md): 한 시장의 정체가 다른 시장의 정상
        상태를 오염시키지 않는다 — `market` 레이블로 완전히 분리된다."""
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
        client = TestClient(create_app())

        self._write_last_cycle_as_pid(pid=111, market="domestic", epoch_seconds=1_758_000_000.0)
        self._write_last_cycle_as_pid(pid=222, market="overseas", epoch_seconds=1_758_003_600.0)

        response = client.get("/metrics")

        assert response.status_code == 200
        assert self._last_cycle_value(response.text, "domestic") == 1_758_000_000.0
        assert self._last_cycle_value(response.text, "overseas") == 1_758_003_600.0


class TestMainEntrypoint:
    def test_run_wires_jobs_and_consumer_then_serves(self, monkeypatch, tmp_path):
        """SPEC-ANALYZER-TRAIN-GATE-001 M5 + INFER-001 M1: run()은 기동 시 주간
        재학습 cron 잡을 배선하고(G-1 fail-fast 포함) 스트림 컨슈머를 백그라운드
        태스크로 기동한 뒤 uvicorn.serve()에 도달한다. 종료 경로에서는 컨슈머
        태스크가 취소되고 스케줄러가 shutdown된다."""
        from analyzer.api import main
        from analyzer.inference.config import InferenceConfig
        from analyzer.orchestration.config import AutomationConfig

        served = {"called": False}

        async def fake_serve(self):
            # 실제 uvicorn serve()는 I/O를 await하므로 다른 태스크가 스케줄된다 —
            # 컨슈머 태스크가 최소 1회 실행되도록 제어권을 넘긴다.
            await asyncio.sleep(0)
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
        consumer_events: list[str] = []

        class _FakeConsumer:
            def __init__(self, **kwargs):
                consumer_events.append("constructed")

            async def start(self) -> None:
                consumer_events.append("started")
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    consumer_events.append("cancelled")
                    raise

        fake_inference_config = InferenceConfig(
            redis_host="redis",
            redis_port=6379,
            redis_username="appuser",
            redis_password="redis-secret",
            stream_claim_idle_seconds=600,
            container_models_root=Path("/mnt/models"),
        )

        # SPEC-OBSV-ANALYZER-DEADMAN-001 REQ-DMR-003: run()이 이제 부팅 시
        # last-cycle 게이지 warm-start도 수행하므로, 이 스모크 테스트에서는
        # 실제 Redis 접속을 시도하지 않도록 InferenceMetrics(전역 레지스트리
        # 오염 방지, TrainingMetrics와 동일 원칙)와 build_redis_client(네트워크
        # I/O 회피)를 격리 대역으로 교체한다.
        from analyzer.inference.metrics import InferenceMetrics

        class _FakeWarmStartRedis:
            def get(self, key: str) -> str | None:
                return None

            def close(self) -> None:
                pass

        monkeypatch.setattr(main, "get_automation_config", lambda: fake_config)
        monkeypatch.setattr(main, "get_inference_config", lambda: fake_inference_config)
        monkeypatch.setattr(main, "StreamConsumer", _FakeConsumer)
        monkeypatch.setattr(
            main, "TrainingMetrics", lambda: TrainingMetrics(registry=CollectorRegistry())
        )
        monkeypatch.setattr(
            main, "InferenceMetrics", lambda: InferenceMetrics(registry=CollectorRegistry())
        )
        monkeypatch.setattr(main, "build_redis_client", lambda *_a, **_k: _FakeWarmStartRedis())
        monkeypatch.setattr(main.SchedulerRegistry, "start", lambda self: None)
        monkeypatch.setattr(
            main.SchedulerRegistry, "shutdown", lambda self, wait=True: shutdown_calls.append(True)
        )

        asyncio.run(main.run(host="127.0.0.1", port=8001))

        assert served["called"] is True
        # REQ-ATG-001: 프로세스 종료 경로에서 shutdown() 훅이 호출되어야 한다.
        assert shutdown_calls == [True]
        # REQ-AIF-020: 컨슈머는 백그라운드 태스크로 기동되고 종료 경로에서 취소된다.
        assert consumer_events == ["constructed", "started", "cancelled"]


class TestWarmStartInferenceLastCycle:
    """SPEC-OBSV-ANALYZER-DEADMAN-001 REQ-DMR-003/004: 상주 부모 프로세스
    기동 시 두 시장 각각의 `aaa_analyzer_inference_last_cycle_seconds`
    게이지를 Redis 영속 완료 시각으로 warm-start한다. Redis 값 존재/부재/
    조회실패 3가지 경로 모두 예외를 전파하지 않는다(fail-open) — 복원
    시뮬레이션 3케이스."""

    @staticmethod
    def _fake_inference_config() -> InferenceConfig:
        return InferenceConfig(
            redis_host="redis",
            redis_port=6379,
            redis_username="appuser",
            redis_password="redis-secret",
            stream_claim_idle_seconds=1800,
            container_models_root=Path("/mnt/models"),
        )

    def test_case_1_redis_value_present_warms_both_markets(self, monkeypatch):
        """복원 시뮬레이션 케이스 (a): Redis에 두 시장 모두 값이 있으면
        게이지가 그 값으로 warm-start된다."""
        from analyzer.api import main
        from analyzer.inference.metrics import INFERENCE_LAST_CYCLE_NAME, InferenceMetrics

        registry = CollectorRegistry()
        fixed_metrics = InferenceMetrics(registry=registry)

        class _FakeRedis:
            def __init__(self, store: dict[str, str]) -> None:
                self.store = store
                self.closed = False

            def get(self, key: str) -> str | None:
                return self.store.get(key)

            def close(self) -> None:
                self.closed = True

        fake_redis = _FakeRedis(
            {
                "observability:analyzer:last-cycle:domestic": "1758000000",
                "observability:analyzer:last-cycle:overseas": "1758003600",
            }
        )

        monkeypatch.setattr(main, "InferenceMetrics", lambda: fixed_metrics)
        monkeypatch.setattr(main, "build_redis_client", lambda *_a, **_k: fake_redis)

        main._warm_start_inference_last_cycle(self._fake_inference_config())

        domestic = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        overseas = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "overseas"})
        assert domestic == 1_758_000_000.0
        assert overseas == 1_758_003_600.0
        # 리소스 정리: warm-start 전용 redis 클라이언트는 사용 후 close된다.
        assert fake_redis.closed is True

    def test_case_2_redis_value_absent_leaves_gauge_unset(self, monkeypatch):
        """복원 시뮬레이션 케이스 (b): Redis 키가 없으면(최초 기동 등)
        게이지는 부재 상태로 남고 기동은 예외 없이 계속된다."""
        from analyzer.api import main
        from analyzer.inference.metrics import INFERENCE_LAST_CYCLE_NAME, InferenceMetrics

        registry = CollectorRegistry()
        fixed_metrics = InferenceMetrics(registry=registry)

        class _EmptyRedis:
            def get(self, key: str) -> str | None:
                return None

            def close(self) -> None:
                pass

        monkeypatch.setattr(main, "InferenceMetrics", lambda: fixed_metrics)
        monkeypatch.setattr(main, "build_redis_client", lambda *_a, **_k: _EmptyRedis())

        main._warm_start_inference_last_cycle(self._fake_inference_config())

        domestic = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        overseas = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "overseas"})
        assert domestic is None
        assert overseas is None

    def test_case_3_redis_lookup_failure_fails_open(self, monkeypatch):
        """복원 시뮬레이션 케이스 (c): Redis 조회 자체가 실패(연결 불가)해도
        `_warm_start_inference_last_cycle()`은 예외를 전파하지 않는다
        (REQ-DMR-003 비차단 관례)."""
        import redis.exceptions

        from analyzer.api import main
        from analyzer.inference.metrics import InferenceMetrics

        registry = CollectorRegistry()
        fixed_metrics = InferenceMetrics(registry=registry)

        class _FailingRedis:
            def get(self, key: str) -> str | None:
                raise redis.exceptions.ConnectionError("연결 불가(시뮬레이션)")

            def close(self) -> None:
                pass

        monkeypatch.setattr(main, "InferenceMetrics", lambda: fixed_metrics)
        monkeypatch.setattr(main, "build_redis_client", lambda *_a, **_k: _FailingRedis())

        # 예외를 던지면 이 호출 자체가 테스트를 실패시킨다(암묵적 단언).
        main._warm_start_inference_last_cycle(self._fake_inference_config())

    def test_case_3_metrics_endpoint_still_returns_200_after_redis_failure(self, monkeypatch):
        """AC-DMR-005: warm-start 도중 Redis 조회가 실패해도 `/metrics`는
        여전히 HTTP 200을 반환한다(fail-open이 `/metrics` 서빙 가능성 자체를
        해치지 않음을 종단간으로 재확인)."""
        import redis.exceptions

        from analyzer.api import main
        from analyzer.inference.metrics import InferenceMetrics

        class _FailingRedis:
            def get(self, key: str) -> str | None:
                raise redis.exceptions.ConnectionError("연결 불가(시뮬레이션)")

            def close(self) -> None:
                pass

        monkeypatch.setattr(
            main, "InferenceMetrics", lambda: InferenceMetrics(registry=CollectorRegistry())
        )
        monkeypatch.setattr(main, "build_redis_client", lambda *_a, **_k: _FailingRedis())

        main._warm_start_inference_last_cycle(self._fake_inference_config())

        client = TestClient(create_app())
        response = client.get("/metrics")

        assert response.status_code == 200


class TestOrchestrationPlaceholders:
    def test_scheduler_registry_starts_with_no_registered_jobs(self):
        registry = SchedulerRegistry()

        assert registry.registered_jobs() == []

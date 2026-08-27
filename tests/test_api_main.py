"""프로덕션 기동 배선 테스트 (SPEC-ANALYZER-TRAIN-GATE-001 M5,
REQ-ATG-001/002/003/004/005/006).

`wire_weekly_retrain_job()` — 주간 재학습 cron 잡 하나만 등록하고
스케줄러를 기동한다. `register_default_jobs()`(월간/일일 포함)는 호출하지
않는다. `TrainingMetrics`는 프로세스당 1회만 생성되어 콜백 클로저에
주입된다. data_as_of는 발화 시점에 1회 계산되어 run_training()과 게이트
CLI 양쪽에 동일하게 주입된다.
"""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from prometheus_client import CollectorRegistry

from analyzer.api import main as main_module
from analyzer.orchestration import manual_run as manual_run_module
from analyzer.orchestration import runner as runner_module
from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.runner import RunOutcome
from analyzer.orchestration.scheduler import (
    DAILY_STALENESS_CHECK_JOB_ID,
    MONTHLY_OPTUNA_TUNING_JOB_ID,
    WEEKLY_FULL_RETRAIN_JOB_ID,
    SchedulerRegistry,
)
from analyzer.orchestration.ssh_dispatch import CommandResult
from analyzer.orchestration.staleness import ModelStalenessInfo


def _make_config(tmp_path: Path) -> AutomationConfig:
    return AutomationConfig(
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
    )


class _FakeWolSender:
    """실 UDP 브로드캐스트를 대체하는 페이크(항상 즉시 성공)."""

    def send(self, mac_address: str) -> bool:
        return True


class _FakeConnection:
    """실 SSH 연결을 대체하는 페이크 — 실행된 명령을 그대로 기록한다."""

    def __init__(self, *, exec_results: list[CommandResult] | None = None) -> None:
        self._exec_results = list(exec_results or [CommandResult(exit_code=0)])
        self.executed_commands: list[str] = []

    def connect(self) -> None:
        pass

    def exec_command(
        self,
        command: str,
        timeout_seconds: float,
        *,
        on_output_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        self.executed_commands.append(command)
        if self._exec_results:
            return self._exec_results.pop(0)
        return CommandResult(exit_code=0)

    def close(self) -> None:
        pass


class TestWireWeeklyRetrainJobRegistration:
    """AC-ATG-002: 주간 잡만 등록되고 스케줄러가 기동한다."""

    def test_registers_exactly_one_weekly_job(self, tmp_path: Path):
        registry = SchedulerRegistry(scheduler=MagicMock())
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)

        assert registry.registered_jobs() == [WEEKLY_FULL_RETRAIN_JOB_ID]
        assert MONTHLY_OPTUNA_TUNING_JOB_ID not in registry.registered_jobs()
        assert DAILY_STALENESS_CHECK_JOB_ID not in registry.registered_jobs()

    def test_does_not_call_register_default_jobs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        registry = SchedulerRegistry(scheduler=MagicMock())
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        called = {"invoked": False}

        def _fake_register_default_jobs(*args, **kwargs):
            called["invoked"] = True
            return []

        import analyzer.orchestration.scheduler as scheduler_module

        monkeypatch.setattr(scheduler_module, "register_default_jobs", _fake_register_default_jobs)

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)

        assert called["invoked"] is False

    def test_starts_scheduler_exactly_once(self, tmp_path: Path):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)

        mock_scheduler.start.assert_called_once()

    def test_add_job_receives_h1_safety_kwargs(self, tmp_path: Path):
        """H-1(REQ-ATG-003): max_instances=1/coalesce/misfire_grace_time이
        register_cron_job() 경유로 명시 전달되어야 한다."""
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)

        _, kwargs = mock_scheduler.add_job.call_args
        assert kwargs["max_instances"] == 1
        assert "coalesce" in kwargs
        assert "misfire_grace_time" in kwargs


class TestTrainingMetricsSingleton:
    """AC-ATG-004: TrainingMetrics는 프로세스당 1회만 생성되어 콜백 클로저에
    주입된다 — 콜백을 2회 이상 연속 발화시켜도 재생성되지 않는다."""

    def test_repeated_callback_invocation_does_not_recreate_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        # 격리 레지스트리에 1회만 생성 — wire_weekly_retrain_job()은 이미
        # 생성된 metrics를 인자로 받을 뿐 내부에서 재생성하지 않는다(구조적
        # 보장). 콜백을 2회 호출해도 "Duplicated timeseries" 예외가 없어야 한다.
        metrics = TrainingMetrics(registry=CollectorRegistry())

        monkeypatch.setattr(
            main_module, "run_training", lambda **_: RunOutcome(success=True, promoted=True)
        )
        monkeypatch.setattr(main_module, "build_gate_promotion_fn", lambda **_: lambda promoted: {})

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)

        _, kwargs = mock_scheduler.add_job.call_args
        weekly_callback = kwargs.get("func") or mock_scheduler.add_job.call_args.args[0]

        weekly_callback()
        weekly_callback()  # 2회 연속 발화 — 예외 없이 완료되어야 한다.


class TestWeeklyJobCallback:
    """AC-ATG-005/006: data_as_of 산출 + run_training() 전-조합 센티널 전달."""

    def test_callback_computes_data_as_of_and_forwards_sentinel_combo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        captured_run_training: dict = {}

        def fake_run_training(**kwargs):
            captured_run_training.update(kwargs)
            return RunOutcome(success=True, promoted=True)

        captured_gate_fn_kwargs: dict = {}

        def fake_build_gate_promotion_fn(**kwargs):
            captured_gate_fn_kwargs.update(kwargs)
            return lambda promoted: {}

        monkeypatch.setattr(main_module, "run_training", fake_run_training)
        monkeypatch.setattr(main_module, "build_gate_promotion_fn", fake_build_gate_promotion_fn)

        class _FixedDatetime:
            @staticmethod
            def now(tz):
                import datetime as _datetime_module

                return _datetime_module.datetime(2026, 8, 23, 1, 0, tzinfo=tz)

        monkeypatch.setattr(main_module, "datetime", _FixedDatetime)

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)

        # register_cron_job()에 전달된 콜백을 직접 호출해 발화를 시뮬레이션한다.
        _, kwargs = mock_scheduler.add_job.call_args
        weekly_callback = kwargs.get("func") or mock_scheduler.add_job.call_args.args[0]
        weekly_callback()

        assert captured_run_training["market"] == "all"
        assert captured_run_training["horizon"] == 0
        assert captured_run_training["algorithm"] == "all"
        assert captured_run_training["data_as_of"] == date(2026, 8, 22)
        assert captured_gate_fn_kwargs["data_as_of"] == date(2026, 8, 22)
        assert captured_run_training["promotion_gate_fn"] is not None

    def test_callback_raises_on_run_training_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """§B 리스크 3: 최상위 예외를 로그 후 재발생 — 실행 성공 위장 금지."""
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        monkeypatch.setattr(
            main_module,
            "run_training",
            lambda **_: RunOutcome(success=False, failure=None),
        )
        monkeypatch.setattr(main_module, "build_gate_promotion_fn", lambda **_: lambda promoted: {})

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)

        _, kwargs = mock_scheduler.add_job.call_args
        weekly_callback = kwargs.get("func") or mock_scheduler.add_job.call_args.args[0]

        with pytest.raises(RuntimeError):
            weekly_callback()

    def test_gate_failure_produces_structured_log_with_run_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """Critical-2 수정(REQ-ATG-012 §B 리스크 3): 게이트 실패
        (`GatePromotionFailure`)는 `runner.py`의 `promotion_gate_fn` 호출부에
        try/except가 없어 `run_training()` 밖으로 그대로 전파된다 — 이
        경로가 프로젝트 구조화 로거(`run_id` 포함)로 기록된 뒤 재발생하는지
        확인한다(이전에는 APScheduler 내부 로거로만 흘러가 trace_id 상관관계가
        끊겼다)."""
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        def _raise_run_training(**_kwargs):
            from analyzer.orchestration.gate_adapter import GatePromotionFailure

            raise GatePromotionFailure("게이트 stdout JSON 파싱 실패")

        monkeypatch.setattr(main_module, "run_training", _raise_run_training)
        monkeypatch.setattr(main_module, "build_gate_promotion_fn", lambda **_: lambda promoted: {})

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)

        _, kwargs = mock_scheduler.add_job.call_args
        weekly_callback = kwargs.get("func") or mock_scheduler.add_job.call_args.args[0]

        with caplog.at_level("ERROR"):
            from analyzer.orchestration.gate_adapter import GatePromotionFailure

            with pytest.raises(GatePromotionFailure):
                weekly_callback()

        matching = [r for r in caplog.records if "weekly training run raised" in r.message]
        assert len(matching) == 1
        assert "run_id=" in matching[0].message
        assert matching[0].exc_info is not None


class TestWeeklyJobRealChainParamsFromActiveMeta:
    """Critical-1 수정 재검증(AC-ATG-011): `main.py`의 실제 배선 →
    `run_training()` → `execute_scheduled_training_run()` → 원격 디스패치
    명령 문자열 체인 전체(단위 함수 단독 호출이 아니라 실제 호출 경로)를
    통해 `--params-from-active-meta`가 주간 프로덕션 경로에 실제로
    포함되는지 검증한다 — 기존 단위 테스트는 각 함수를 개별 호출·인자를
    수동 주입해 검증했기 때문에 배선 누락(호출자가 인자를 전달하지 않는
    결함)을 잡지 못했다."""

    def test_weekly_production_path_includes_params_from_active_meta_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        fake_connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )

        # run_training()(manual_run.py)이 내부에서 직접 생성하는 실 WoL/SSH
        # 구현체만 페이크로 치환한다 — run_training()/execute_scheduled_training_run()
        # 자체는 monkeypatch하지 않고 실제 호출 경로를 그대로 태운다.
        monkeypatch.setattr(manual_run_module, "UdpBroadcastWolSender", _FakeWolSender)
        monkeypatch.setattr(
            manual_run_module, "ParamikoSshConnection", lambda **_kwargs: fake_connection
        )
        # REQ-ATA-020의 실 30초 대기(sleep_fn 기본값 time.sleep)를 테스트에서
        # 우회한다 — manual_run.run_training()은 sleep_fn을 노출하지 않으므로
        # execute_scheduled_training_run()의 keyword-only 기본값을 직접 치환한다.
        kwdefaults = runner_module.execute_scheduled_training_run.__kwdefaults__
        assert kwdefaults is not None
        monkeypatch.setitem(kwdefaults, "sleep_fn", lambda _seconds: None)
        # 게이트 CLI 자체(별도 SSH 경로)는 이 테스트의 관심사가 아니므로
        # 빈 verdict를 반환하는 페이크로 대체한다.
        monkeypatch.setattr(main_module, "build_gate_promotion_fn", lambda **_: lambda promoted: {})

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)

        _, kwargs = mock_scheduler.add_job.call_args
        weekly_callback = kwargs.get("func") or mock_scheduler.add_job.call_args.args[0]
        weekly_callback()

        assert len(fake_connection.executed_commands) >= 1
        dispatch_command = fake_connection.executed_commands[0]
        assert "--params-from-active-meta" in dispatch_command
        assert str(config.active_models_root) in dispatch_command


class TestWireDailyStalenessCheckJobRegistration:
    """SPEC-ANALYZER-TRAIN-STALENESS-001 M3(REQ-ATD-005): 일일 정체 감지
    잡은 개별 `register_cron_job()` 호출로 등록되어야 하며, `register_default_jobs()`
    (월간 잡 포함)는 호출되어서는 안 된다. 주간 잡과 함께 등록하면
    `registered_jobs()`는 정확히 `["weekly-full-retrain", "daily-staleness-check"]`
    2건이어야 한다."""

    def test_registers_exactly_weekly_and_daily_jobs(self, tmp_path: Path):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        main_module.wire_weekly_retrain_job(registry, config=config, metrics=metrics)
        main_module.wire_daily_staleness_check_job(registry, config=config, metrics=metrics)

        assert registry.registered_jobs() == [
            WEEKLY_FULL_RETRAIN_JOB_ID,
            DAILY_STALENESS_CHECK_JOB_ID,
        ]
        assert MONTHLY_OPTUNA_TUNING_JOB_ID not in registry.registered_jobs()

    def test_does_not_call_register_default_jobs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        called = {"invoked": False}

        def _fake_register_default_jobs(*args, **kwargs):
            called["invoked"] = True
            return []

        import analyzer.orchestration.scheduler as scheduler_module

        monkeypatch.setattr(scheduler_module, "register_default_jobs", _fake_register_default_jobs)

        main_module.wire_daily_staleness_check_job(registry, config=config, metrics=metrics)

        assert called["invoked"] is False

    def test_does_not_call_scheduler_start(self, tmp_path: Path):
        """중복 `start()` 방지 — 스케줄러 기동은 `wire_weekly_retrain_job()`이
        전담하므로, 일일 잡 등록 함수는 잡 등록만 담당하고 `start()`를
        호출해서는 안 된다(이미 기동된 스케줄러에 `start()`를 재호출하면
        `SchedulerAlreadyRunningError`가 발생한다)."""
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        main_module.wire_daily_staleness_check_job(registry, config=config, metrics=metrics)

        mock_scheduler.start.assert_not_called()


class TestDailyStalenessCheckCallback:
    """REQ-ATD-007/010: 콜백은 `detect_stale_models(models_root=config.container_models_root,
    threshold_days=config.staleness_threshold_days)`를 호출해야 하며, 성공 시
    `metrics.record_staleness_batch()`로, 실패 시 `metrics.record_failure(stage="staleness_scan")`
    직접 호출 후 재발생으로 관측 가능해야 한다."""

    def test_callback_calls_detect_stale_models_with_config_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        captured: dict = {}

        def fake_detect_stale_models(models_root, *, threshold_days, as_of=None):
            captured["models_root"] = models_root
            captured["threshold_days"] = threshold_days
            return []

        monkeypatch.setattr(main_module, "detect_stale_models", fake_detect_stale_models)

        main_module.wire_daily_staleness_check_job(registry, config=config, metrics=metrics)

        _, kwargs = mock_scheduler.add_job.call_args
        daily_callback = kwargs.get("func") or mock_scheduler.add_job.call_args.args[0]
        daily_callback()

        assert captured["models_root"] == config.container_models_root
        assert captured["threshold_days"] == config.staleness_threshold_days

    def test_callback_records_staleness_batch_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        fake_results = [
            ModelStalenessInfo(
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                most_recent_trained_date=date(2026, 1, 1),
                is_stale=True,
            )
        ]
        monkeypatch.setattr(main_module, "detect_stale_models", lambda *_a, **_k: fake_results)

        recorded_batch: list = []
        monkeypatch.setattr(
            metrics, "record_staleness_batch", lambda results: recorded_batch.extend(results)
        )

        main_module.wire_daily_staleness_check_job(registry, config=config, metrics=metrics)

        _, kwargs = mock_scheduler.add_job.call_args
        daily_callback = kwargs.get("func") or mock_scheduler.add_job.call_args.args[0]
        daily_callback()

        assert recorded_batch == fake_results

    def test_callback_records_failure_and_reraises_on_scan_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """REQ-ATD-010: 스캔 실패(마운트 부재/권한 거부/기타 I/O)는
        `handle_training_run_failure()`를 경유하지 않고
        `metrics.record_failure(stage="staleness_scan")`를 직접 호출해 기록한
        뒤 예외를 재발생시킨다 — 예외를 삼켜 스케줄러 스레드를 조용히
        죽게 하지 않는다."""
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        config = _make_config(tmp_path)
        metrics = TrainingMetrics(registry=CollectorRegistry())

        def _raise_permission_error(*_a, **_k):
            raise PermissionError("mount not readable")

        monkeypatch.setattr(main_module, "detect_stale_models", _raise_permission_error)

        record_failure_mock = MagicMock()
        monkeypatch.setattr(metrics, "record_failure", record_failure_mock)
        record_staleness_batch_mock = MagicMock()
        monkeypatch.setattr(metrics, "record_staleness_batch", record_staleness_batch_mock)

        main_module.wire_daily_staleness_check_job(registry, config=config, metrics=metrics)

        _, kwargs = mock_scheduler.add_job.call_args
        daily_callback = kwargs.get("func") or mock_scheduler.add_job.call_args.args[0]

        with caplog.at_level("ERROR"):
            with pytest.raises(PermissionError):
                daily_callback()

        record_failure_mock.assert_called_once_with(stage="staleness_scan")
        record_staleness_batch_mock.assert_not_called()


class TestRunEntrypointFailFast:
    """AC-ATG-001: 필수 설정 누락 시 컨테이너가 기동에 실패한다(G-1)."""

    def test_run_propagates_missing_config_error(self, monkeypatch: pytest.MonkeyPatch):
        from analyzer.data.config import MissingConfigError

        def _raise_missing_config():
            raise MissingConfigError("필수 환경변수 누락: TRAIN_AUTOMATION_TARGET_MAC")

        monkeypatch.setattr(main_module, "get_automation_config", _raise_missing_config)

        import asyncio

        with pytest.raises(MissingConfigError):
            asyncio.run(main_module.run(host="127.0.0.1", port=8002))

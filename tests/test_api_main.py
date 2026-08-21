"""프로덕션 기동 배선 테스트 (SPEC-ANALYZER-TRAIN-GATE-001 M5,
REQ-ATG-001/002/003/004/005/006).

`wire_weekly_retrain_job()` — 주간 재학습 cron 잡 하나만 등록하고
스케줄러를 기동한다. `register_default_jobs()`(월간/일일 포함)는 호출하지
않는다. `TrainingMetrics`는 프로세스당 1회만 생성되어 콜백 클로저에
주입된다. data_as_of는 발화 시점에 1회 계산되어 run_training()과 게이트
CLI 양쪽에 동일하게 주입된다.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from prometheus_client import CollectorRegistry

from analyzer.api import main as main_module
from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.runner import RunOutcome
from analyzer.orchestration.scheduler import (
    DAILY_STALENESS_CHECK_JOB_ID,
    MONTHLY_OPTUNA_TUNING_JOB_ID,
    WEEKLY_FULL_RETRAIN_JOB_ID,
    SchedulerRegistry,
)


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
        cache_dir=tmp_path / "cache",
        calendar_code="KRX",
        feature_code_version="v1",
        mount_script_path=tmp_path / "mount-nas-hdd1.sh",
        python_executable_path=tmp_path / ".venv" / "bin" / "python",
        mysql_database="aaa",
        mysql_trainer_password="trainer-secret",
        trainer_log_base_dir=tmp_path / "logs" / "aaa-analyzer",
    )


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

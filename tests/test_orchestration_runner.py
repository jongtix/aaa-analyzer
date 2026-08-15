"""학습 실행 오케스트레이션 통합 테스트 (AC-ATA-001~006, AC-ATA-012).

WoL→SSH→터널+디스패치→완료감지→프로모션/실패처리 전체 흐름을 `WolSender`/
`SshConnection` 페이크로 조립해 검증한다. 실 네트워크를 사용하지 않으므로
`@pytest.mark.integration`이 아닌 일반 단위 테스트다.
"""

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

from prometheus_client import CollectorRegistry

from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.metrics import (
    LAST_SUCCESS_TIMESTAMP_NAME,
    TRAINING_RUN_TOTAL_NAME,
    TrainingMetrics,
)
from analyzer.orchestration.runner import _make_stage_marker_relay, execute_scheduled_training_run
from analyzer.orchestration.ssh_dispatch import CommandResult


class _FakeWolSender:
    def __init__(self, outcomes: list[bool]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def send(self, mac_address: str) -> bool:
        self.calls += 1
        return self._outcomes.pop(0)


class _FakeConnection:
    """`SshConnection` 프로토콜을 만족하는 단위 테스트 전용 페이크."""

    def __init__(
        self,
        *,
        connect_should_fail_times: int = 0,
        exec_results: list[CommandResult] | None = None,
    ) -> None:
        self._connect_failures_remaining = connect_should_fail_times
        self._exec_results = list(exec_results or [CommandResult(exit_code=0)])
        self.connect_attempts = 0
        self.executed_commands: list[str] = []
        self.close_calls = 0

    def connect(self) -> None:
        self.connect_attempts += 1
        if self._connect_failures_remaining > 0:
            self._connect_failures_remaining -= 1
            raise ConnectionError("connect failed (fake)")

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
        self.close_calls += 1


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


class TestMakeStageMarkerRelay:
    """AC-ATO-003(REQ-ATO-002/007, D-NEW-1): stage_marker:true JSON 레코드만
    릴레이하고, 상세 로그(필드 없음/false)나 비-JSON 라인은 릴레이하지 않는다."""

    def test_relays_stage_marker_true_records(self):
        logger = logging.getLogger("test-relay")
        relayed: list[str] = []
        logger.info = lambda msg, *args: relayed.append(msg % args if args else msg)  # type: ignore[method-assign]
        relay = _make_stage_marker_relay(logger)

        relay('{"stage_marker": true, "message": "market start"}')

        assert len(relayed) == 1
        assert "market start" in relayed[0]

    def test_does_not_relay_records_without_stage_marker_field(self):
        logger = logging.getLogger("test-relay-2")
        relayed: list[str] = []
        logger.info = lambda *a, **k: relayed.append(a)  # type: ignore[method-assign]
        relay = _make_stage_marker_relay(logger)

        relay('{"message": "detailed progress line"}')

        assert relayed == []

    def test_does_not_relay_stage_marker_false_records(self):
        logger = logging.getLogger("test-relay-3")
        relayed: list[str] = []
        logger.info = lambda *a, **k: relayed.append(a)  # type: ignore[method-assign]
        relay = _make_stage_marker_relay(logger)

        relay('{"stage_marker": false, "message": "detailed"}')

        assert relayed == []

    def test_ignores_non_json_lines_without_raising(self):
        logger = logging.getLogger("test-relay-4")
        relayed: list[str] = []
        logger.info = lambda *a, **k: relayed.append(a)  # type: ignore[method-assign]
        relay = _make_stage_marker_relay(logger)

        relay("plain text traceback line, not JSON")

        assert relayed == []


class TestExecuteScheduledTrainingRunSuccess:
    """AC-ATA-001: WoL 성공 → SSH 연결 성공 → 디스패치 → 성공 메트릭 발행."""

    def test_happy_path_promotes_and_records_success(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)
        sleeps: list[float] = []

        outcome = execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-1",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=sleeps.append,
            time_fn=lambda: 1234.0,
        )

        assert outcome.success is True
        assert outcome.promoted is True
        assert sleeps[0] == 30.0  # REQ-ATA-020 — WoL 이후 30초 대기
        assert connection.close_calls == 1
        # 디스패치 명령 1회 + 프로모션 명령 1회
        assert len(connection.executed_commands) == 2
        success_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "success"}
        )
        timestamp_value = registry.get_sample_value(
            LAST_SUCCESS_TIMESTAMP_NAME,
            {"market": "domestic", "horizon": "5", "algorithm": "lightgbm"},
        )
        assert success_value == 1.0
        assert timestamp_value == 1234.0

    def test_dispatch_command_targets_staging_models_root_not_active(self, tmp_path: Path):
        """plan.md §B.5(D6): 학습 스크립트는 항상 스테이징 경로에만 기록해야 한다."""
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        metrics = TrainingMetrics(registry=CollectorRegistry())

        execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-1",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
        )

        dispatch_command = connection.executed_commands[0]
        assert str(config.staging_models_root / "run-1") in dispatch_command
        assert str(config.active_models_root) not in dispatch_command


class TestExecuteScheduledTrainingRunWolFailure:
    """AC-ATA-002: WoL 3회 재시도 후 최종 실패 → 통합 실패 처리, 활성 모델 불변."""

    def test_routes_to_unified_failure_handler(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([False, False, False])
        connection = _FakeConnection()
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        outcome = execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-2",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
        )

        assert outcome.success is False
        assert outcome.failure is not None
        assert outcome.failure.stage == "wol"
        assert wol.calls == 3
        # SSH 연결을 아예 시도하지 않아야 한다 — 활성 모델 불변(REQ-ATA-062)
        assert connection.connect_attempts == 0
        assert connection.executed_commands == []
        failure_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "wol", "outcome": "failure"}
        )
        assert failure_value == 1.0


class TestExecuteScheduledTrainingRunSshFailure:
    """AC-ATA-003: SSH 연결 6회 재시도 후 최종 실패 → 통합 실패 처리, 활성 모델 불변."""

    def test_routes_to_unified_failure_handler(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(connect_should_fail_times=10)
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        outcome = execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-3",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
        )

        assert outcome.success is False
        assert outcome.failure is not None
        assert outcome.failure.stage == "ssh"
        assert connection.connect_attempts == 6
        assert connection.executed_commands == []  # 디스패치 명령이 실행되지 않아야 한다
        failure_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "ssh", "outcome": "failure"}
        )
        assert failure_value == 1.0


class TestExecuteScheduledTrainingRunTimeout:
    """AC-ATA-004: 학습 타임아웃 초과 → SSH 세션 강제 종료 + 통합 실패 처리,
    부분 저장된 모델이 있어도 활성 모델을 교체하지 않는다."""

    def test_does_not_promote_on_timeout(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=-1, timed_out=True)])
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        outcome = execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-4",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
        )

        assert outcome.success is False
        assert outcome.promoted is False
        assert outcome.failure is not None
        assert outcome.failure.stage == "timeout"
        # 디스패치 명령만 실행되고, 프로모션(mv/cp) 명령은 실행되지 않아야 한다
        # (REQ-ATA-062 — 부분 저장 모델이 있어도 활성 경로 불변)
        assert len(connection.executed_commands) == 1
        assert connection.close_calls == 1  # SSH 세션은 항상 정리한다
        failure_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "timeout", "outcome": "failure"}
        )
        assert failure_value == 1.0


class TestExecuteScheduledTrainingRunScriptFailure:
    """AC-ATA-005: 학습 스크립트 비정상 종료(종료코드 비0) → 통합 실패 처리,
    실행 전 존재하던 활성 모델 파일을 변경하지 않는다."""

    def test_does_not_promote_on_nonzero_exit_code(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=1)])
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        outcome = execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-5",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
        )

        assert outcome.success is False
        assert outcome.promoted is False
        assert outcome.failure is not None
        assert outcome.failure.stage == "training"
        assert len(connection.executed_commands) == 1  # 프로모션 명령 없음
        failure_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "training", "outcome": "failure"}
        )
        assert failure_value == 1.0


class TestExecuteScheduledTrainingRunTunnelTeardown:
    """AC-ATA-006: 학습 실행 종료 시(성공·실패·타임아웃 무관) SSH 세션을 정리한다."""

    def test_closes_connection_on_success(self, tmp_path: Path):
        config = _make_config(tmp_path)
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        metrics = TrainingMetrics(registry=CollectorRegistry())

        execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-6",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=_FakeWolSender([True]),
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
        )

        assert connection.close_calls == 1

    def test_closes_connection_on_training_script_failure(self, tmp_path: Path):
        config = _make_config(tmp_path)
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=1)])
        metrics = TrainingMetrics(registry=CollectorRegistry())

        execute_scheduled_training_run(
            run_kind="monthly",
            run_id="run-7",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=_FakeWolSender([True]),
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
        )

        assert connection.close_calls == 1

    def test_uses_monthly_timeout_for_monthly_run_kind(self, tmp_path: Path):
        config = _make_config(tmp_path)
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        metrics = TrainingMetrics(registry=CollectorRegistry())
        captured_timeouts: list[float] = []
        original_exec = connection.exec_command

        def _capturing_exec(
            command: str,
            timeout_seconds: float,
            *,
            on_output_line: Callable[[str], None] | None = None,
        ) -> CommandResult:
            captured_timeouts.append(timeout_seconds)
            return original_exec(command, timeout_seconds, on_output_line=on_output_line)

        connection.exec_command = _capturing_exec  # type: ignore[method-assign]

        execute_scheduled_training_run(
            run_kind="monthly",
            run_id="run-8",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=_FakeWolSender([True]),
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
        )

        assert captured_timeouts[0] == config.monthly_timeout_seconds


class TestExecuteScheduledTrainingRunIdempotentWol:
    """AC-ATA-012(REQ-ATA-014): 이미 깨어있는 MacBook에 도달해도 정상 SSH 연결
    절차가 진행되며 부가 효과가 발생하지 않는다."""

    def test_already_awake_target_proceeds_normally(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])  # 이미 깨어있어도 send()는 성공(True) 반환
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        metrics = TrainingMetrics(registry=CollectorRegistry())

        outcome = execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-9",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
        )

        assert outcome.success is True
        assert wol.calls == 1  # 재시도 없이 1회 만에 정상 진행(부가 효과 없음)

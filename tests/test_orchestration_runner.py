"""학습 실행 오케스트레이션 통합 테스트 (AC-ATA-001~006, AC-ATA-012).

WoL→SSH→터널+디스패치→완료감지→프로모션/실패처리 전체 흐름을 `WolSender`/
`SshConnection` 페이크로 조립해 검증한다. 실 네트워크를 사용하지 않으므로
`@pytest.mark.integration`이 아닌 일반 단위 테스트다.
"""

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from analyzer.common.trace import get_trace_id
from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.metrics import (
    LAST_SUCCESS_TIMESTAMP_NAME,
    RANK_IC_NAME,
    TRAINING_RUN_TOTAL_NAME,
    TrainingMetrics,
)
from analyzer.orchestration.promotion_gate import PromotionVerdict
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

    def test_params_from_active_meta_forwarded_to_dispatch_command(self, tmp_path: Path):
        """Critical-1 수정: `params_from_active_meta`가 지정되면
        `--params-from-active-meta <경로>`가 원격 디스패치 명령에 포함되어야
        한다 — 주간 원격 학습이 게이트 챌린저와 동일한 동결 하이퍼파라미터로
        학습하게 하는 유일한 배선 지점(AC-ATG-011)."""
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
            params_from_active_meta=config.active_models_root,
        )

        dispatch_command = connection.executed_commands[0]
        assert "--params-from-active-meta" in dispatch_command
        assert str(config.active_models_root) in dispatch_command

    def test_params_from_active_meta_omitted_when_not_provided(self, tmp_path: Path):
        """하위 호환: 미지정 시(기본값 None) 기존 동작이 유지되어야 한다."""
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
        assert "--params-from-active-meta" not in dispatch_command


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


class TestExecuteScheduledTrainingRunPromotionGateWiring:
    """M6(REQ-ATE-055/064/065): `promotion_gate_fn`이 주어지면 조합별 승격/
    보류 판정에 따라 `record_success(outcome=...)`가 개별 호출되어야 한다 —
    실행당 1회가 아니라 저장된 조합 수만큼(AC-ATE-049)."""

    def test_records_success_and_held_back_per_combo_independently(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        verdicts = {
            ("domestic", 20, "lightgbm"): PromotionVerdict(
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                promoted=True,
                challenger_rank_ic=0.05,
                champion_rank_ic=0.01,
                challenger_trained_date=date(2026, 8, 17),
            ),
            ("domestic", 20, "xgboost"): PromotionVerdict(
                market="domestic",
                horizon=20,
                algorithm="xgboost",
                promoted=False,
                challenger_rank_ic=0.0,
                champion_rank_ic=0.02,
                challenger_trained_date=date(2026, 8, 17),
            ),
        }
        promotion_gate_calls: list[bool] = []

        def _fake_promotion_gate_fn(merged: bool):
            promotion_gate_calls.append(merged)
            return verdicts

        outcome = execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-promo-1",
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 17),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
            time_fn=lambda: 5000.0,
            promotion_gate_fn=_fake_promotion_gate_fn,
        )

        assert outcome.success is True
        assert promotion_gate_calls == [True]  # promote_staging_to_active() 결과가 전달됨

        success_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "success"}
        )
        held_back_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "held-back"}
        )
        assert success_value == 1.0
        assert held_back_value == 1.0

        promoted_gauge = registry.get_sample_value(
            LAST_SUCCESS_TIMESTAMP_NAME,
            {"market": "domestic", "horizon": "20", "algorithm": "lightgbm"},
        )
        held_back_gauge = registry.get_sample_value(
            LAST_SUCCESS_TIMESTAMP_NAME,
            {"market": "domestic", "horizon": "20", "algorithm": "xgboost"},
        )
        assert promoted_gauge == 5000.0
        assert held_back_gauge is None

        rank_ic_promoted = registry.get_sample_value(
            RANK_IC_NAME, {"market": "domestic", "horizon": "20", "algorithm": "lightgbm"}
        )
        rank_ic_held_back = registry.get_sample_value(
            RANK_IC_NAME, {"market": "domestic", "horizon": "20", "algorithm": "xgboost"}
        )
        assert rank_ic_promoted == 0.05
        assert rank_ic_held_back == 0.0

    def test_promotion_gate_fn_none_falls_back_to_single_combo_success(self, tmp_path: Path):
        """`promotion_gate_fn`이 None이면(1차 배포 이전) 기존 단일-호출
        동작으로 하위 호환된다."""
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        outcome = execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-promo-2",
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 17),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            sleep_fn=lambda _s: None,
            time_fn=lambda: 42.0,
        )

        assert outcome.success is True
        success_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "success"}
        )
        assert success_value == 1.0


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


class TestExecuteScheduledTrainingRunStageTransitionLogs:
    """AC-ATO-013(REQ-ATO-021): 정상 성공 경로 완료 시 WoL 송신 결과, SSH
    연결 성공(+시도 횟수), 디스패치 시작(run_id+타임아웃값), 원격 종료코드
    수신, 프로모션 결과 각각 1줄 이상의 단계 전이 로그가 남아야 한다."""

    def test_success_path_emits_all_five_stage_transition_logs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        metrics = TrainingMetrics(registry=CollectorRegistry())

        with (
            caplog.at_level("INFO", logger="analyzer.orchestration.runner"),
            caplog.at_level("INFO", logger="analyzer.orchestration.ssh_dispatch"),
        ):
            execute_scheduled_training_run(
                run_kind="weekly",
                run_id="run-10",
                market="domestic",
                horizon=5,
                algorithm="lightgbm",
                data_as_of=date(2026, 8, 11),
                config=config,
                wol_sender=wol,
                connection_factory=lambda: connection,
                metrics=metrics,
                sleep_fn=lambda _s: None,
                time_fn=lambda: 1234.0,
            )

        assert "wol send result" in caplog.text
        assert "dispatch start" in caplog.text
        assert "run_id=run-10" in caplog.text
        assert "remote exit code received" in caplog.text
        assert "promotion result" in caplog.text


class TestExecuteScheduledTrainingRunTraceIdPropagation:
    """AC-ATO-008(REQ-ATO-012/013/014): run_id가 오케스트레이터 단계 전이
    로그(릴레이 대상)의 trace_id 필드에 반영되어야 한다. `execute_scheduled_
    training_run()`이 `set_trace_id(run_id)`를 호출하지 않으면 이 컨텍스트변수는
    None으로 남아 회귀를 즉시 드러낸다."""

    def test_stage_transition_logs_carry_run_id_as_trace_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import analyzer.orchestration.runner as runner_module

        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        metrics = TrainingMetrics(registry=CollectorRegistry())
        observed_trace_ids: list[str | None] = []
        original_info = runner_module._logger.info

        def _capturing_info(*args, **kwargs):
            observed_trace_ids.append(get_trace_id())
            return original_info(*args, **kwargs)

        monkeypatch.setattr(runner_module._logger, "info", _capturing_info)

        execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-trace-xyz789",
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

        # AC-ATO-013(REQ-ATO-021)이 요구하는 5개 단계 전이 로그 중 4개는
        # `runner._logger`가 남긴다(wol/dispatch start/exit code/promotion) —
        # 나머지 1개(SSH 연결 성공)는 `ssh_dispatch._logger`가 남기며 이 SPEC의
        # F4 수정 범위 밖이다(별도 로거 인스턴스).
        assert len(observed_trace_ids) == 4
        assert all(trace_id == "run-trace-xyz789" for trace_id in observed_trace_ids)

    def test_trace_id_context_does_not_leak_after_run_completes(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(
            exec_results=[CommandResult(exit_code=0), CommandResult(exit_code=0)]
        )
        metrics = TrainingMetrics(registry=CollectorRegistry())

        assert get_trace_id() is None

        execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-trace-leak-check",
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

        assert get_trace_id() is None

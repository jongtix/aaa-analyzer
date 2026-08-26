"""NAS 측 게이트 어댑터 테스트 (SPEC-ANALYZER-TRAIN-GATE-001 M4, REQ-ATG-007/012).

`promotion_gate_fn` 어댑터 팩토리 — SSH로 게이트 CLI를 원격 실행하고 stdout
verdict JSON을 역직렬화해 runner.py의 훅 계약을 만족하는 클로저를 반환한다.
E-1(REQ-ATG-012): 실패 5종 각각에서 `metrics.record_failure(stage="promotion_gate")`
직접 호출 + 예외 재발생을 확인한다(handle_training_run_failure() 미경유).
"""

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.gate_adapter import (
    DATA_AS_OF_OFFSET_DAYS,
    GATE_TIMEOUT_ENV_VAR,
    GATE_TIMEOUT_SECONDS_DEFAULT,
    GatePromotionFailure,
    build_gate_promotion_fn,
    build_gate_remote_command,
    compute_data_as_of,
    resolve_gate_timeout_seconds,
)
from analyzer.orchestration.promotion_gate import PromotionVerdict
from analyzer.orchestration.ssh_dispatch import CommandResult
from analyzer.training.gate import serialize_verdicts


class _FakeConnection:
    """`SshConnection` 프로토콜 페이크 — `on_output_line`에 사전 지정된 라인을
    실제로 전달한다(test_orchestration_runner.py의 페이크와 달리 콜백을 호출)."""

    def __init__(
        self,
        *,
        connect_should_fail: bool = False,
        output_lines: list[str] | None = None,
        exec_result: CommandResult | None = None,
    ) -> None:
        self._connect_should_fail = connect_should_fail
        self._output_lines = list(output_lines or [])
        self._exec_result = exec_result if exec_result is not None else CommandResult(exit_code=0)
        self.connect_attempts = 0
        self.executed_commands: list[str] = []
        self.close_calls = 0

    def connect(self) -> None:
        self.connect_attempts += 1
        if self._connect_should_fail:
            raise ConnectionError("connect failed (fake)")

    def exec_command(
        self,
        command: str,
        timeout_seconds: float,
        *,
        on_output_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        self.executed_commands.append(command)
        if on_output_line is not None:
            for line in self._output_lines:
                on_output_line(line)
        return self._exec_result

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


class TestComputeDataAsOf:
    """AC-ATG-005: data_as_of는 발화일 전일(달력일 -1) 고정 — 거래 캘린더
    조회를 수행하지 않는다."""

    def test_returns_fire_date_minus_one_calendar_day(self):
        fire_time = date(2026, 8, 23)  # 일요일 01:00 KST 발화

        assert compute_data_as_of(fire_time) == date(2026, 8, 22)  # 토요일

    def test_offset_constant_is_one_day(self):
        assert DATA_AS_OF_OFFSET_DAYS == 1

    def test_pure_function_uses_named_constant_offset(self):
        fire_time = date(2026, 3, 1)

        assert compute_data_as_of(fire_time) == fire_time - timedelta(days=DATA_AS_OF_OFFSET_DAYS)


class TestResolveGateTimeoutSeconds:
    def test_returns_default_when_env_var_absent(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(GATE_TIMEOUT_ENV_VAR, raising=False)

        assert resolve_gate_timeout_seconds() == GATE_TIMEOUT_SECONDS_DEFAULT
        assert GATE_TIMEOUT_SECONDS_DEFAULT == 2 * 60 * 60

    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(GATE_TIMEOUT_ENV_VAR, "3600")

        assert resolve_gate_timeout_seconds() == 3600.0


class TestBuildGateRemoteCommand:
    def test_includes_gate_module_invocation_and_flags(self, tmp_path: Path):
        command = build_gate_remote_command(
            active_models_root=tmp_path / "models",
            cache_dir=tmp_path / "cache",
            data_as_of=date(2026, 8, 22),
            feature_code_version="v1",
            calendar_code="KRX",
            merged_to_active=True,
            mount_script_path=tmp_path / "mount.sh",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=tmp_path / "db_tunnel_key",
            python_executable_path=tmp_path / "python",
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
        )

        assert "-m analyzer.training.gate" in command
        assert "--merged-to-active" in command
        assert "--no-merged-to-active" not in command
        assert "db_tunnel@nas-host" in command
        assert str(tmp_path / "models") in command

    def test_merged_to_active_false_uses_negative_flag(self, tmp_path: Path):
        command = build_gate_remote_command(
            active_models_root=tmp_path / "models",
            cache_dir=tmp_path / "cache",
            data_as_of=date(2026, 8, 22),
            feature_code_version="v1",
            calendar_code="KRX",
            merged_to_active=False,
            mount_script_path=tmp_path / "mount.sh",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=tmp_path / "db_tunnel_key",
            python_executable_path=tmp_path / "python",
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
        )

        assert "--no-merged-to-active" in command


class TestBuildGatePromotionFnSuccess:
    def test_returns_deserialized_verdicts_on_success(self, tmp_path: Path):
        config = _make_config(tmp_path)
        metrics = MagicMock()
        verdict = PromotionVerdict(
            market="domestic",
            horizon=60,
            algorithm="xgboost",
            promoted=True,
            challenger_rank_ic=0.05,
            champion_rank_ic=0.01,
            challenger_trained_date=date(2026, 8, 22),
        )
        raw_json = serialize_verdicts({("domestic", 60, "xgboost"): verdict})
        connection = _FakeConnection(output_lines=[raw_json])

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            sleep_fn=lambda _seconds: None,
        )

        verdicts = promotion_gate_fn(True)

        assert verdicts == {("domestic", 60, "xgboost"): verdict}
        metrics.record_failure.assert_not_called()
        assert connection.close_calls == 1

    def test_extracts_last_valid_json_line_when_logs_are_interleaved(self, tmp_path: Path):
        """§B 리스크 2: stdout/stderr 혼입 관용 파싱 — 로그 라인이 섞여도
        마지막 유효 JSON 라인을 verdict로 채택한다."""
        config = _make_config(tmp_path)
        metrics = MagicMock()
        raw_json = serialize_verdicts({})
        connection = _FakeConnection(
            output_lines=[
                '{"level": "info", "message": "market start"}',
                "not json at all",
                raw_json,
            ]
        )

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            sleep_fn=lambda _seconds: None,
        )

        verdicts = promotion_gate_fn(True)

        assert verdicts == {}
        metrics.record_failure.assert_not_called()


class TestBuildGatePromotionFnFailures:
    """AC-ATG-012(E-1): 실패 5종 각각 record_failure(stage="promotion_gate")
    정확히 1회 + 예외 재발생."""

    def test_ssh_connect_failure(self, tmp_path: Path):
        config = _make_config(tmp_path)
        metrics = MagicMock()
        connection = _FakeConnection(connect_should_fail=True)

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            ssh_max_retries=1,
            ssh_retry_interval_seconds=0.0,
            sleep_fn=lambda _seconds: None,
        )

        with pytest.raises(GatePromotionFailure):
            promotion_gate_fn(True)

        metrics.record_failure.assert_called_once_with(stage="promotion_gate")
        metrics.record_success.assert_not_called()

    def test_nonzero_exit_code(self, tmp_path: Path):
        config = _make_config(tmp_path)
        metrics = MagicMock()
        connection = _FakeConnection(exec_result=CommandResult(exit_code=1))

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            sleep_fn=lambda _seconds: None,
        )

        with pytest.raises(GatePromotionFailure):
            promotion_gate_fn(True)

        metrics.record_failure.assert_called_once_with(stage="promotion_gate")
        metrics.record_success.assert_not_called()
        assert connection.close_calls == 1

    def test_timeout(self, tmp_path: Path):
        config = _make_config(tmp_path)
        metrics = MagicMock()
        connection = _FakeConnection(exec_result=CommandResult(exit_code=-1, timed_out=True))

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            sleep_fn=lambda _seconds: None,
        )

        with pytest.raises(GatePromotionFailure):
            promotion_gate_fn(True)

        metrics.record_failure.assert_called_once_with(stage="promotion_gate")
        metrics.record_success.assert_not_called()

    def test_stdout_json_parse_failure(self, tmp_path: Path):
        config = _make_config(tmp_path)
        metrics = MagicMock()
        connection = _FakeConnection(output_lines=["this is not json", "still not json"])

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            sleep_fn=lambda _seconds: None,
        )

        with pytest.raises(GatePromotionFailure):
            promotion_gate_fn(True)

        metrics.record_failure.assert_called_once_with(stage="promotion_gate")
        metrics.record_success.assert_not_called()

    def test_verdict_deserialize_failure(self, tmp_path: Path):
        config = _make_config(tmp_path)
        metrics = MagicMock()
        # 유효한 JSON이지만 PromotionVerdict 필드 계약을 위반(마켓 필드 누락).
        connection = _FakeConnection(output_lines=['[{"market": "domestic"}]'])

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            sleep_fn=lambda _seconds: None,
        )

        with pytest.raises(GatePromotionFailure):
            promotion_gate_fn(True)

        metrics.record_failure.assert_called_once_with(stage="promotion_gate")
        metrics.record_success.assert_not_called()

    def test_failure_does_not_call_handle_training_run_failure(self, tmp_path: Path):
        """D1: handle_training_run_failure() 미경유 — FailureStage Literal이
        "promotion_gate"를 포함하지 않는다(failure.py 무수정)."""
        from analyzer.orchestration import gate_adapter as gate_adapter_module

        config = _make_config(tmp_path)
        metrics = MagicMock()
        connection = _FakeConnection(connect_should_fail=True)

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            ssh_max_retries=1,
            ssh_retry_interval_seconds=0.0,
            sleep_fn=lambda _seconds: None,
        )

        assert not hasattr(gate_adapter_module, "handle_training_run_failure")

        with pytest.raises(GatePromotionFailure):
            promotion_gate_fn(True)


class TestBuildGatePromotionFnFailureLogging:
    """High-3 수정: SSH 연결 실패/타임아웃/비정상 종료코드 3종 분기가
    diagnostic 정보(캡처된 stdout/stderr 또는 연결 대상)와 함께
    logger.error()를 호출해야 한다 — 기존에는 예외 재발생만 하고
    로그를 남기지 않아 원인 진단이 불가능했다."""

    def test_ssh_connect_failure_logs_error(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        config = _make_config(tmp_path)
        metrics = MagicMock()
        connection = _FakeConnection(connect_should_fail=True)

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            ssh_max_retries=1,
            ssh_retry_interval_seconds=0.0,
            sleep_fn=lambda _seconds: None,
        )

        with caplog.at_level("ERROR"):
            with pytest.raises(GatePromotionFailure):
                promotion_gate_fn(True)

        assert any("ssh connect failed" in record.message for record in caplog.records)
        assert any(config.ssh_host in record.message for record in caplog.records)

    def test_timeout_logs_error_with_captured_lines(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        config = _make_config(tmp_path)
        metrics = MagicMock()
        connection = _FakeConnection(
            output_lines=["remote traceback line 1", "remote traceback line 2"],
            exec_result=CommandResult(exit_code=-1, timed_out=True),
        )

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            sleep_fn=lambda _seconds: None,
        )

        with caplog.at_level("ERROR"):
            with pytest.raises(GatePromotionFailure):
                promotion_gate_fn(True)

        assert any("timed out" in record.message for record in caplog.records)
        assert any("remote traceback line 1" in record.message for record in caplog.records)

    def test_nonzero_exit_code_logs_error_with_captured_lines(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        config = _make_config(tmp_path)
        metrics = MagicMock()
        connection = _FakeConnection(
            output_lines=["Traceback (most recent call last):", "RuntimeError: boom"],
            exec_result=CommandResult(exit_code=1),
        )

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=date(2026, 8, 22),
            connection_factory=lambda: connection,
            sleep_fn=lambda _seconds: None,
        )

        with caplog.at_level("ERROR"):
            with pytest.raises(GatePromotionFailure):
                promotion_gate_fn(True)

        assert any("exited non-zero" in record.message for record in caplog.records)
        assert any("RuntimeError: boom" in record.message for record in caplog.records)

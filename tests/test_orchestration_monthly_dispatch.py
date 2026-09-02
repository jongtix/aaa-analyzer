"""월간 원격 캠페인 후처리 훅 통합 테스트 (SPEC-ANALYZER-TRAIN-TUNING-001 M4).

AC-ATT-013/014/015/016 — 성공/실패(비정상 종료코드)/타임아웃/SSH 연결 실패
4케이스 + `execute_scheduled_training_run()`/`handle_training_run_failure()`
미호출 확인(grep 가드, REQ-ATT-013/014).
"""

import hashlib
import inspect
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from analyzer.orchestration import monthly_dispatch
from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.metrics import TRAINING_RUN_TOTAL_NAME, TrainingMetrics
from analyzer.orchestration.monthly_dispatch import (
    MonthlyCampaignRunError,
    execute_monthly_campaign_run,
)
from analyzer.orchestration.ssh_dispatch import CommandResult
from analyzer.training.campaign import POINT_COMBOS
from analyzer.training.persistence import model_dir, model_filename


def _write_version(
    models_root: Path,
    market: str,
    horizon: int,
    algorithm: str,
    trained_date: date,
) -> None:
    """실제 학습 없이 파일명 관례에 맞는 모델 파일 + 사이드카 쌍을 기록한다
    (test_training_persistence.py `_write_version`과 동일한 경량 픽스처 패턴)."""
    target_dir = model_dir(models_root, market, horizon, algorithm)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_filename(market, horizon, algorithm, trained_date)
    payload = trained_date.isoformat().encode("utf-8")
    model_path.write_bytes(payload)
    sidecar_path = model_path.with_suffix(model_path.suffix + ".sha256")
    sidecar_path.write_text(hashlib.sha256(payload).hexdigest(), encoding="utf-8")


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
        monthly_timeout_seconds=50400,
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


class TestExecuteMonthlyCampaignRunSuccess:
    """AC-ATT-013/015/016: 종료코드 0 → execute_scheduled_training_run 미호출 +
    전-조합 센티널로 record_success 1회 호출."""

    def test_success_records_sentinel_and_does_not_call_execute_scheduled_training_run(
        self, tmp_path: Path
    ):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=0)])
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)
        sleeps: list[float] = []

        execute_monthly_campaign_run(
            run_id="monthly-run-1",
            data_as_of=date(2026, 9, 1),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            combos=(),
            sleep_fn=sleeps.append,
            time_fn=lambda: 5555.0,
        )

        assert connection.close_calls == 1
        assert len(connection.executed_commands) == 1
        assert "analyzer.training.campaign" in connection.executed_commands[0]
        success_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "success"}
        )
        assert success_value == 1.0

    def test_success_applies_retention_for_each_combo(self, tmp_path: Path):
        """REQ-ATT-017: 성공 후 전달된 각 조합에 대해 보존 정책이 실제로 적용된다."""
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=0)])
        metrics = TrainingMetrics(registry=CollectorRegistry())

        combo = ("domestic", 5, "lightgbm")
        for i in range(40):
            _write_version(
                config.active_models_root,
                *combo,
                trained_date=date(2020, 1, 1) + timedelta(days=i),
            )

        execute_monthly_campaign_run(
            run_id="monthly-run-2",
            data_as_of=date(2026, 9, 1),
            config=config,
            wol_sender=wol,
            connection_factory=lambda: connection,
            metrics=metrics,
            combos=(combo,),
        )

        remaining = list(model_dir(config.active_models_root, *combo).glob("*.txt"))
        assert len(remaining) == 36

    def test_default_combos_is_point_combos(self, tmp_path: Path):
        """REQ-ATT-022: 스킵리스트를 도입하지 않는다 — 기본 조합 목록은 캠페인
        자신의 POINT_COMBOS(8개 전체)와 동일해야 한다."""
        assert (
            inspect.signature(execute_monthly_campaign_run).parameters["combos"].default
            == POINT_COMBOS
        )


class TestExecuteMonthlyCampaignRunRetentionFailure:
    """review finding W1: 성공 기록(record_success) 이후
    apply_retention_for_combos()가 예외를 던지면, run_id를 포함한 구조화
    로그를 기록한 뒤 그대로 재발생해야 한다(metrics.record_failure()는
    호출하지 않는다 — 캠페인 실행 자체는 성공했으므로, M2/M4 기존 설계
    전제 유지)."""

    def test_retention_failure_logs_run_id_and_reraises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=0)])
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        def _raise_retention_failure(*_args, **_kwargs):
            raise ValueError("아카이브 무결성 검증 실패")

        monkeypatch.setattr(
            monthly_dispatch, "apply_retention_for_combos", _raise_retention_failure
        )

        with caplog.at_level("ERROR"):
            with pytest.raises(ValueError, match="아카이브 무결성 검증 실패"):
                execute_monthly_campaign_run(
                    run_id="monthly-run-retention-fail",
                    data_as_of=date(2026, 9, 1),
                    config=config,
                    wol_sender=wol,
                    connection_factory=lambda: connection,
                    metrics=metrics,
                    combos=(("domestic", 5, "lightgbm"),),
                    sleep_fn=lambda _seconds: None,
                )

        matching = [r for r in caplog.records if "retention" in r.message]
        assert len(matching) == 1
        assert "monthly-run-retention-fail" in matching[0].message
        assert matching[0].exc_info is not None

        success_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "success"}
        )
        assert success_value == 1.0
        failure_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "monthly_tuning", "outcome": "failure"}
        )
        assert failure_value is None


class TestExecuteMonthlyCampaignRunFailure:
    """AC-ATT-014: 비정상 종료코드 → record_failure(stage="monthly_tuning") 1회 +
    handle_training_run_failure 미경유 + 예외 재발생(삼키지 않음)."""

    def test_nonzero_exit_records_failure_and_raises(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=1)])
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        try:
            execute_monthly_campaign_run(
                run_id="monthly-run-3",
                data_as_of=date(2026, 9, 1),
                config=config,
                wol_sender=wol,
                connection_factory=lambda: connection,
                metrics=metrics,
                combos=(),
                sleep_fn=lambda _seconds: None,
            )
            raised = False
        except MonthlyCampaignRunError:
            raised = True

        assert raised is True
        failure_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "monthly_tuning", "outcome": "failure"}
        )
        assert failure_value == 1.0
        success_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "success"}
        )
        assert success_value is None
        assert connection.close_calls == 1


class TestExecuteMonthlyCampaignRunTimeout:
    """AC-ATT-014: 타임아웃 → record_failure(stage="monthly_tuning") + 재발생."""

    def test_timeout_records_failure_and_raises(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=-1, timed_out=True)])
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        try:
            execute_monthly_campaign_run(
                run_id="monthly-run-4",
                data_as_of=date(2026, 9, 1),
                config=config,
                wol_sender=wol,
                connection_factory=lambda: connection,
                metrics=metrics,
                combos=(),
                sleep_fn=lambda _seconds: None,
            )
            raised = False
        except MonthlyCampaignRunError:
            raised = True

        assert raised is True
        failure_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "monthly_tuning", "outcome": "failure"}
        )
        assert failure_value == 1.0


class TestExecuteMonthlyCampaignRunSshConnectionFailure:
    """AC-ATT-015(REQ-ATT-014): SSH 연결 자체 실패 → record_failure 호출."""

    def test_ssh_connect_exhausted_records_failure_and_raises(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([True])
        connection = _FakeConnection(connect_should_fail_times=6, exec_results=[])
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        try:
            execute_monthly_campaign_run(
                run_id="monthly-run-5",
                data_as_of=date(2026, 9, 1),
                config=config,
                wol_sender=wol,
                connection_factory=lambda: connection,
                metrics=metrics,
                combos=(),
                sleep_fn=lambda _seconds: None,
            )
            raised = False
        except MonthlyCampaignRunError:
            raised = True

        assert raised is True
        failure_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "monthly_tuning", "outcome": "failure"}
        )
        assert failure_value == 1.0
        assert connection.executed_commands == []


class TestExecuteMonthlyCampaignRunWolFailure:
    """WoL 송신 실패도 동일한 monthly_tuning 실패 경로로 기록된다(부가 케이스)."""

    def test_wol_failure_records_failure_and_raises(self, tmp_path: Path):
        config = _make_config(tmp_path)
        wol = _FakeWolSender([False, False, False])
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        try:
            execute_monthly_campaign_run(
                run_id="monthly-run-6",
                data_as_of=date(2026, 9, 1),
                config=config,
                wol_sender=wol,
                connection_factory=lambda: _FakeConnection(),
                metrics=metrics,
                combos=(),
                sleep_fn=lambda _seconds: None,
            )
            raised = False
        except MonthlyCampaignRunError:
            raised = True

        assert raised is True
        failure_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "monthly_tuning", "outcome": "failure"}
        )
        assert failure_value == 1.0


class TestSubagentBoundaryGuard:
    """REQ-ATT-013/014: 월간 콜백 모듈은 execute_scheduled_training_run()과
    handle_training_run_failure()를 임포트하거나 호출하지 않는다 — docstring
    설명 산문(REQ-ATT-013/014 근거 인용)은 검사 대상에서 제외한다(코드 레벨
    참조만 판정)."""

    def test_does_not_import_or_call_execute_scheduled_training_run_or_handle_training_run_failure(
        self,
    ):
        import ast

        tree = ast.parse(inspect.getsource(monthly_dispatch))
        forbidden = {"execute_scheduled_training_run", "handle_training_run_failure"}
        referenced: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # docstring 리터럴은 건너뛴다
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    referenced.add(alias.asname or alias.name)
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            if isinstance(node, ast.Attribute):
                referenced.add(node.attr)

        assert not (referenced & forbidden)

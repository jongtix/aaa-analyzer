"""수동 학습 실행 진입점 테스트 (SPEC-ANALYZER-TRAIN-AUTOMATION-001).

cron 자동 등록 전 수동 실행 CLI — `run_training()`이 CLI와 향후 cron 콜백이
공유하는 배선 지점이므로, 이 함수가 `execute_scheduled_training_run()`에 올바른
인자를 그대로 전달하는지가 핵심 검증 대상이다. 실제 WoL/SSH 네트워크 I/O는
`execute_scheduled_training_run` 자체(runner.py)의 테스트 범위이므로 여기서는
그 호출부를 페이크로 대체한다.
"""

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from analyzer.orchestration import manual_run
from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.failure import TrainingRunFailure
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.runner import RunOutcome


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
    )


class TestParseArgs:
    def test_parses_all_required_arguments(self):
        args = manual_run.parse_args(
            [
                "--run-kind",
                "weekly",
                "--market",
                "domestic",
                "--horizon",
                "5",
                "--algorithm",
                "lightgbm",
                "--data-as-of",
                "2026-08-11",
            ]
        )

        assert args.run_kind == "weekly"
        assert args.market == "domestic"
        assert args.horizon == 5
        assert args.algorithm == "lightgbm"
        assert args.data_as_of == date(2026, 8, 11)
        assert args.run_id is None

    def test_rejects_invalid_run_kind(self):
        with pytest.raises(SystemExit):
            manual_run.parse_args(
                [
                    "--run-kind",
                    "daily",  # weekly|monthly만 허용
                    "--market",
                    "domestic",
                    "--horizon",
                    "5",
                    "--algorithm",
                    "lightgbm",
                    "--data-as-of",
                    "2026-08-11",
                ]
            )

    def test_accepts_explicit_run_id(self):
        args = manual_run.parse_args(
            [
                "--run-kind",
                "monthly",
                "--market",
                "overseas",
                "--horizon",
                "20",
                "--algorithm",
                "xgboost",
                "--data-as-of",
                "2026-08-01",
                "--run-id",
                "manual-run-001",
            ]
        )

        assert args.run_id == "manual-run-001"


class TestRunTraining:
    """cron 콜백과 공유될 배선 지점 — execute_scheduled_training_run()에 인자가
    그대로 전달되는지가 핵심이다."""

    def test_forwards_all_arguments_to_execute_scheduled_training_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config = _make_config(tmp_path)
        captured: dict = {}

        def fake_execute(**kwargs):
            captured.update(kwargs)
            return RunOutcome(success=True, promoted=True)

        monkeypatch.setattr(manual_run, "execute_scheduled_training_run", fake_execute)

        metrics = TrainingMetrics(registry=CollectorRegistry())

        outcome = manual_run.run_training(
            run_kind="weekly",
            run_id="run-1",
            market="domestic",
            horizon=5,
            algorithm="lightgbm",
            data_as_of=date(2026, 8, 11),
            config=config,
            metrics=metrics,
        )

        assert outcome.success is True
        assert captured["run_kind"] == "weekly"
        assert captured["run_id"] == "run-1"
        assert captured["market"] == "domestic"
        assert captured["horizon"] == 5
        assert captured["algorithm"] == "lightgbm"
        assert captured["data_as_of"] == date(2026, 8, 11)
        assert captured["config"] is config
        assert captured["metrics"] is metrics
        assert callable(captured["connection_factory"])
        assert hasattr(captured["wol_sender"], "send")

    def test_returns_failure_outcome_unmodified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config = _make_config(tmp_path)
        failure = TrainingRunFailure(stage="ssh", message="연결 실패", run_id="run-2")

        monkeypatch.setattr(
            manual_run,
            "execute_scheduled_training_run",
            lambda **_: RunOutcome(success=False, failure=failure),
        )

        outcome = manual_run.run_training(
            run_kind="monthly",
            run_id="run-2",
            market="overseas",
            horizon=20,
            algorithm="xgboost",
            data_as_of=date(2026, 8, 1),
            config=config,
            metrics=TrainingMetrics(registry=CollectorRegistry()),
        )

        assert outcome.success is False
        assert outcome.failure is failure


class TestMainFunction:
    def test_returns_zero_on_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config = _make_config(tmp_path)
        monkeypatch.setattr(manual_run, "get_automation_config", lambda: config)
        monkeypatch.setattr(
            manual_run, "TrainingMetrics", lambda: TrainingMetrics(registry=CollectorRegistry())
        )
        monkeypatch.setattr(
            manual_run,
            "run_training",
            lambda **_: RunOutcome(success=True, promoted=True),
        )

        exit_code = manual_run.main(
            [
                "--run-kind",
                "weekly",
                "--market",
                "domestic",
                "--horizon",
                "5",
                "--algorithm",
                "lightgbm",
                "--data-as-of",
                "2026-08-11",
            ]
        )

        assert exit_code == 0

    def test_returns_one_on_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config = _make_config(tmp_path)
        failure = TrainingRunFailure(stage="training", message="종료코드 1", run_id="run-x")
        monkeypatch.setattr(manual_run, "get_automation_config", lambda: config)
        monkeypatch.setattr(
            manual_run, "TrainingMetrics", lambda: TrainingMetrics(registry=CollectorRegistry())
        )
        monkeypatch.setattr(
            manual_run,
            "run_training",
            lambda **_: RunOutcome(success=False, failure=failure),
        )

        exit_code = manual_run.main(
            [
                "--run-kind",
                "weekly",
                "--market",
                "domestic",
                "--horizon",
                "5",
                "--algorithm",
                "lightgbm",
                "--data-as-of",
                "2026-08-11",
            ]
        )

        assert exit_code == 1

    def test_generates_run_id_when_not_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config = _make_config(tmp_path)
        captured: dict = {}

        def fake_run_training(**kwargs):
            captured.update(kwargs)
            return RunOutcome(success=True)

        monkeypatch.setattr(manual_run, "get_automation_config", lambda: config)
        monkeypatch.setattr(
            manual_run, "TrainingMetrics", lambda: TrainingMetrics(registry=CollectorRegistry())
        )
        monkeypatch.setattr(manual_run, "run_training", fake_run_training)

        manual_run.main(
            [
                "--run-kind",
                "weekly",
                "--market",
                "domestic",
                "--horizon",
                "5",
                "--algorithm",
                "lightgbm",
                "--data-as-of",
                "2026-08-11",
            ]
        )

        assert captured["run_id"]  # 자동 생성된 non-empty 문자열

    def test_uses_explicit_run_id_when_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        config = _make_config(tmp_path)
        captured: dict = {}

        def fake_run_training(**kwargs):
            captured.update(kwargs)
            return RunOutcome(success=True)

        monkeypatch.setattr(manual_run, "get_automation_config", lambda: config)
        monkeypatch.setattr(
            manual_run, "TrainingMetrics", lambda: TrainingMetrics(registry=CollectorRegistry())
        )
        monkeypatch.setattr(manual_run, "run_training", fake_run_training)

        manual_run.main(
            [
                "--run-kind",
                "weekly",
                "--market",
                "domestic",
                "--horizon",
                "5",
                "--algorithm",
                "lightgbm",
                "--data-as-of",
                "2026-08-11",
                "--run-id",
                "explicit-id",
            ]
        )

        assert captured["run_id"] == "explicit-id"


class TestSubprocessInvocation:
    def test_module_invocation_exits_nonzero_without_required_args(self):
        """argparse 단계에서 실패 — 실제 설정 로딩/네트워크 I/O에 도달하지 않는다."""
        result = subprocess.run(
            [sys.executable, "-m", "analyzer.orchestration.manual_run"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0

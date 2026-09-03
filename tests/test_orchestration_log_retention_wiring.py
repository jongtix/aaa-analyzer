"""트레이너 로그 보존 sweep 배선 테스트 (SPEC-OBSV-LOGS-003 M2).

AC-003(REQ-002/006): 두 SSH 디스패치 호출부(정기 주간 학습 `runner.py` /
월간 캠페인 `monthly_dispatch.py`)가 디스패치 완료 직후 **동일한 단일**
sweep 함수를 호출한다(중복 구현 없음).

`ssh_dispatch.py`의 `build_remote_*_dispatch_command()`는 원격 셸 명령
문자열만 조립하는 순수 함수이고 디스패치를 실행하지 않으므로, "디스패치
완료 직후"(spec.md §4 DP-1)에 해당하는 지점은 그 두 빌더를 호출해
`connection.exec_command()`를 수행하는 위 두 모듈이다.
"""

from datetime import date
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry
from test_orchestration_runner import _FakeConnection, _FakeWolSender, _make_config

from analyzer.orchestration import log_retention, monthly_dispatch, runner
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.monthly_dispatch import (
    MonthlyCampaignRunError,
    execute_monthly_campaign_run,
)
from analyzer.orchestration.runner import execute_scheduled_training_run
from analyzer.orchestration.ssh_dispatch import CommandResult


class _SweepSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, *, current_run_id: str) -> None:
        self.calls.append(current_run_id)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _SweepSpy:
    sweep = _SweepSpy()
    monkeypatch.setattr(runner, "sweep_trainer_logs", sweep)
    monkeypatch.setattr(monthly_dispatch, "sweep_trainer_logs", sweep)
    return sweep


class TestSingleImplementation:
    def test_두_호출부가_동일_함수_참조를_임포트한다(self) -> None:
        """중복 구현 금지(REQ-006) — 두 모듈이 같은 객체를 가리켜야 한다."""
        assert runner.sweep_trainer_logs is log_retention.sweep_trainer_logs
        assert monthly_dispatch.sweep_trainer_logs is log_retention.sweep_trainer_logs


class TestWeeklyDispatchWiring:
    def test_주간_디스패치_완료_후_sweep이_호출된다(self, tmp_path: Path, spy: _SweepSpy) -> None:
        config = _make_config(tmp_path)
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=0)])
        metrics = TrainingMetrics(registry=CollectorRegistry())

        execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-1",
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

        assert spy.calls == ["run-1"]

    def test_디스패치_실패에도_sweep이_호출된다(self, tmp_path: Path, spy: _SweepSpy) -> None:
        """DP-1: 트리거는 성공/실패 무관 디스패치 완료 직후."""
        config = _make_config(tmp_path)
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=1)])
        metrics = TrainingMetrics(registry=CollectorRegistry())

        outcome = execute_scheduled_training_run(
            run_kind="weekly",
            run_id="run-fail",
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

        assert outcome.success is False
        assert spy.calls == ["run-fail"]


class TestMonthlyDispatchWiring:
    def test_월간_캠페인_완료_후_sweep이_호출된다(self, tmp_path: Path, spy: _SweepSpy) -> None:
        config = _make_config(tmp_path)
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=0)])
        metrics = TrainingMetrics(registry=CollectorRegistry())

        execute_monthly_campaign_run(
            run_id="monthly-run-1",
            data_as_of=date(2026, 9, 1),
            config=config,
            wol_sender=_FakeWolSender([True]),
            connection_factory=lambda: connection,
            metrics=metrics,
            combos=(),
            sleep_fn=lambda _s: None,
        )

        assert spy.calls == ["monthly-run-1"]

    def test_캠페인_실패에도_sweep이_호출된다(self, tmp_path: Path, spy: _SweepSpy) -> None:
        config = _make_config(tmp_path)
        connection = _FakeConnection(exec_results=[CommandResult(exit_code=7)])
        metrics = TrainingMetrics(registry=CollectorRegistry())

        with pytest.raises(MonthlyCampaignRunError):
            execute_monthly_campaign_run(
                run_id="monthly-run-fail",
                data_as_of=date(2026, 9, 1),
                config=config,
                wol_sender=_FakeWolSender([True]),
                connection_factory=lambda: connection,
                metrics=metrics,
                combos=(),
                sleep_fn=lambda _s: None,
            )

        assert spy.calls == ["monthly-run-fail"]

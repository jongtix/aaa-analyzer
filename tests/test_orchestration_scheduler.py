"""`SchedulerRegistry` cron 잡 등록 테스트 (REQ-ATA-080/081/082/083, AC-ATA-008).

REQ-ANALYZER-FOUNDATION-010이 도입한 빈 레지스트리(`_job_ids: list[str] = []`)를
확장한다 — 병렬 스케줄링 메커니즘을 신규 도입하지 않고(REQ-ATA-082), 기존
`SchedulerRegistry`를 통해서만 cron 잡을 등록한다. `interval` 트리거는 사용하지
않는다(REQ-ATA-080).
"""

from unittest.mock import MagicMock

from apscheduler.triggers.cron import CronTrigger

from analyzer.orchestration.scheduler import (
    DAILY_STALENESS_CHECK_JOB_ID,
    MONTHLY_OPTUNA_TUNING_JOB_ID,
    WEEKLY_FULL_RETRAIN_JOB_ID,
    SchedulerRegistry,
    daily_staleness_check_trigger,
    monthly_optuna_tuning_trigger,
    register_default_jobs,
    weekly_full_retrain_trigger,
)


class TestSchedulerRegistryBackwardCompatibility:
    """REQ-ANALYZER-FOUNDATION-010: 기존 빈 레지스트리 계약을 보존한다."""

    def test_starts_empty(self):
        registry = SchedulerRegistry(scheduler=MagicMock())

        assert registry.registered_jobs() == []

    def test_registered_jobs_returns_a_copy(self):
        registry = SchedulerRegistry(scheduler=MagicMock())
        registry.register_cron_job("job-1", CronTrigger(hour=1), lambda: None)

        jobs = registry.registered_jobs()
        jobs.append("mutated")

        assert registry.registered_jobs() == ["job-1"]


class TestRegisterCronJob:
    """REQ-ATA-082: 기존 SchedulerRegistry를 통해서만 cron 잡을 등록한다."""

    def test_delegates_to_underlying_scheduler_add_job(self):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        trigger = CronTrigger(hour=1)

        def _job() -> None:
            return None

        registry.register_cron_job("job-1", trigger, _job)

        mock_scheduler.add_job.assert_called_once_with(
            _job, trigger=trigger, id="job-1", replace_existing=True
        )

    def test_appends_job_id_to_registry(self):
        registry = SchedulerRegistry(scheduler=MagicMock())

        registry.register_cron_job("job-1", CronTrigger(hour=1), lambda: None)

        assert registry.registered_jobs() == ["job-1"]

    def test_does_not_duplicate_job_id_on_replace(self):
        registry = SchedulerRegistry(scheduler=MagicMock())

        registry.register_cron_job("job-1", CronTrigger(hour=1), lambda: None)
        registry.register_cron_job("job-1", CronTrigger(hour=2), lambda: None)

        assert registry.registered_jobs() == ["job-1"]


class TestCronTriggerFactories:
    """REQ-ATA-081: 모든 스케줄 시각은 IANA 타임존(zoneinfo) 기반 KST로 해석된다."""

    def test_weekly_full_retrain_is_on_weekend(self):
        trigger = weekly_full_retrain_trigger()

        fields = {f.name: str(f) for f in trigger.fields}
        assert "sat" in fields["day_of_week"]
        assert str(trigger.timezone) == "Asia/Seoul"

    def test_monthly_optuna_tuning_is_once_per_month(self):
        trigger = monthly_optuna_tuning_trigger()

        fields = {f.name: str(f) for f in trigger.fields}
        assert fields["day"] == "1"
        assert str(trigger.timezone) == "Asia/Seoul"

    def test_daily_staleness_check_is_daily_kst(self):
        trigger = daily_staleness_check_trigger()

        fields = {f.name: str(f) for f in trigger.fields}
        # day/day_of_week/month가 모두 와일드카드(*)면 "매일" 실행이다.
        assert fields["day"] == "*"
        assert fields["day_of_week"] == "*"
        assert fields["month"] == "*"
        assert str(trigger.timezone) == "Asia/Seoul"

    def test_daily_staleness_check_is_independent_schedule_from_training_jobs(self):
        """REQ-ATA-083: 정체 감지 잡은 주간/월간 학습 잡과 독립적인 별도 스케줄이다
        (학습 잡 실패 여부와 무관하게 매일 실행)."""
        weekly = weekly_full_retrain_trigger()
        monthly = monthly_optuna_tuning_trigger()
        daily = daily_staleness_check_trigger()

        assert daily is not weekly
        assert daily is not monthly


class TestRegisterDefaultJobs:
    """AC-ATA-008: 주간/월간/일일 정체감지 3개 cron 잡이 모두 등록된다."""

    def test_registers_all_three_jobs(self):
        registry = SchedulerRegistry(scheduler=MagicMock())

        job_ids = register_default_jobs(
            registry,
            weekly_full_retrain_func=lambda: None,
            monthly_optuna_tuning_func=lambda: None,
            daily_staleness_check_func=lambda: None,
        )

        assert set(job_ids) == {
            WEEKLY_FULL_RETRAIN_JOB_ID,
            MONTHLY_OPTUNA_TUNING_JOB_ID,
            DAILY_STALENESS_CHECK_JOB_ID,
        }
        assert set(registry.registered_jobs()) == set(job_ids)

    def test_registered_functions_are_wired_correctly(self):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)
        weekly_func = MagicMock()
        monthly_func = MagicMock()
        daily_func = MagicMock()

        register_default_jobs(
            registry,
            weekly_full_retrain_func=weekly_func,
            monthly_optuna_tuning_func=monthly_func,
            daily_staleness_check_func=daily_func,
        )

        called_funcs = {call.args[0] for call in mock_scheduler.add_job.call_args_list}
        assert called_funcs == {weekly_func, monthly_func, daily_func}


class TestSchedulerRegistryLifecycle:
    def test_start_delegates_to_underlying_scheduler(self):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)

        registry.start()

        mock_scheduler.start.assert_called_once()

    def test_shutdown_delegates_to_underlying_scheduler(self):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)

        registry.shutdown()

        mock_scheduler.shutdown.assert_called_once_with(wait=True)

    def test_default_scheduler_is_background_scheduler_when_none_injected(self):
        from apscheduler.schedulers.background import BackgroundScheduler

        registry = SchedulerRegistry()

        assert isinstance(registry._scheduler, BackgroundScheduler)

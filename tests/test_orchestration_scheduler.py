"""`SchedulerRegistry` cron 잡 등록 테스트 (REQ-ATA-080/081/082/083, AC-ATA-008).

REQ-ANALYZER-FOUNDATION-010이 도입한 빈 레지스트리(`_job_ids: list[str] = []`)를
확장한다 — 병렬 스케줄링 메커니즘을 신규 도입하지 않고(REQ-ATA-082), 기존
`SchedulerRegistry`를 통해서만 cron 잡을 등록한다. `interval` 트리거는 사용하지
않는다(REQ-ATA-080).
"""

from unittest.mock import MagicMock

from apscheduler.triggers.cron import CronTrigger

from analyzer.orchestration.scheduler import (
    COALESCE_DEFAULT,
    DAILY_STALENESS_CHECK_JOB_ID,
    MAX_INSTANCES_DEFAULT,
    MISFIRE_GRACE_TIME_SECONDS_DEFAULT,
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
            _job,
            trigger=trigger,
            id="job-1",
            replace_existing=True,
            max_instances=MAX_INSTANCES_DEFAULT,
            coalesce=COALESCE_DEFAULT,
            misfire_grace_time=MISFIRE_GRACE_TIME_SECONDS_DEFAULT,
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


class TestRegisterCronJobSafetyArgs:
    """REQ-ATG-003(H-1): max_instances=1/coalesce/misfire_grace_time이
    명시 전달되어야 한다(AC-ATG-003)."""

    def test_add_job_receives_max_instances_coalesce_misfire_grace_time(self):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)

        registry.register_cron_job("job-1", CronTrigger(hour=1), lambda: None)

        _, kwargs = mock_scheduler.add_job.call_args
        assert kwargs["max_instances"] == 1
        assert kwargs["coalesce"] is True
        assert kwargs["misfire_grace_time"] == MISFIRE_GRACE_TIME_SECONDS_DEFAULT

    def test_caller_can_override_safety_args(self):
        mock_scheduler = MagicMock()
        registry = SchedulerRegistry(scheduler=mock_scheduler)

        registry.register_cron_job(
            "job-1",
            CronTrigger(hour=1),
            lambda: None,
            max_instances=2,
            coalesce=False,
            misfire_grace_time=60,
        )

        _, kwargs = mock_scheduler.add_job.call_args
        assert kwargs["max_instances"] == 2
        assert kwargs["coalesce"] is False
        assert kwargs["misfire_grace_time"] == 60


class TestCronTriggerFactories:
    """REQ-ATA-081: 모든 스케줄 시각은 IANA 타임존(zoneinfo) 기반 KST로 해석된다."""

    def test_weekly_full_retrain_is_on_weekend(self):
        trigger = weekly_full_retrain_trigger()

        fields = {f.name: str(f) for f in trigger.fields}
        assert "sun" in fields["day_of_week"]
        assert str(trigger.timezone) == "Asia/Seoul"

    def test_monthly_optuna_tuning_is_once_per_month(self):
        trigger = monthly_optuna_tuning_trigger()

        fields = {f.name: str(f) for f in trigger.fields}
        assert fields["day"] == "1"
        assert str(trigger.timezone) == "Asia/Seoul"

    def test_monthly_optuna_tuning_fires_at_06_00_kst(self):
        """AC-ATT-004(REQ-ATT-003): 발화 시각이 07:00 KST에서 06:00 KST로
        변경됐다 — day="1"/minute=0/timezone=Asia/Seoul은 무수정."""
        trigger = monthly_optuna_tuning_trigger()

        fields = {f.name: str(f) for f in trigger.fields}
        assert fields["day"] == "1"
        assert fields["hour"] == "6"
        assert fields["minute"] == "0"
        assert str(trigger.timezone) == "Asia/Seoul"

    def test_daily_staleness_check_is_daily_kst(self):
        trigger = daily_staleness_check_trigger()

        fields = {f.name: str(f) for f in trigger.fields}
        # day/day_of_week/month가 모두 와일드카드(*)면 "매일" 실행이다.
        assert fields["day"] == "*"
        assert fields["day_of_week"] == "*"
        assert fields["month"] == "*"
        assert str(trigger.timezone) == "Asia/Seoul"

    def test_daily_staleness_check_fires_at_04_00_kst(self):
        """REQ-ATD-006: 발화 시각이 07:00 KST에서 04:00 KST로 변경됐다."""
        trigger = daily_staleness_check_trigger()

        fields = {f.name: str(f) for f in trigger.fields}
        assert fields["hour"] == "4"
        assert fields["minute"] == "0"

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

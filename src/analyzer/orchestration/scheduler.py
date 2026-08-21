"""APScheduler cron 잡 레지스트리 (SPEC-ANALYZER-TRAIN-AUTOMATION-001 §2.8, REQ-ATA-080~083).

REQ-ANALYZER-FOUNDATION-010이 도입한 빈 레지스트리 자리 표시자(`_job_ids:
list[str] = []`)를 실제 cron 잡 등록 레지스트리로 확장한다 — 이 SPEC은 병렬
스케줄링 메커니즘을 신규 도입하지 않고(REQ-ATA-082, shall not) 기존
`SchedulerRegistry`를 통해서만 잡을 등록한다.

- cron 트리거 전용, `interval` 트리거 미사용(REQ-ATA-080).
- 모든 스케줄 시각은 `zoneinfo.ZoneInfo`(IANA 타임존 DB) 기반 KST로 해석한다
  (REQ-ATA-081, D8 — `pytz` 미사용, plan.md §D).
- 주간 전체 재학습(주말), 월간 Optuna 튜닝(매월 1회), 일일 모델 정체 감지
  (매일 1회, 학습 잡과 독립된 별도 스케줄) 3개 cron 잡을 등록한다(REQ-ATA-081/083).

@MX:ANCHOR: [AUTO] SchedulerRegistry — 이 SPEC과 후속 SPEC(INFER-001)이 공유하는
유일한 스케줄링 진입점(REQ-ATA-082 "병렬 스케줄링 메커니즘 신규 도입 금지").
@MX:REASON: fan_in >= 3 예상 — 주간/월간/일일 3개 잡 등록 경로가 모두 이
레지스트리를 통과하며, 후속 SPEC도 동일 레지스트리를 재사용할 가능성이 있다
(plan.md §B.3).
"""

from collections.abc import Callable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.triggers.cron import CronTrigger

_KST = ZoneInfo("Asia/Seoul")

WEEKLY_FULL_RETRAIN_JOB_ID = "weekly-full-retrain"
MONTHLY_OPTUNA_TUNING_JOB_ID = "monthly-optuna-tuning"
DAILY_STALENESS_CHECK_JOB_ID = "daily-staleness-check"

MAX_INSTANCES_DEFAULT = 1
"""REQ-ATG-003(H-1): 중복 발화 방지 — 동시에 실행 중인 잡 인스턴스를 1개로
제한한다(명시 전달, 기본값 의존 금지)."""

COALESCE_DEFAULT = True
"""REQ-ATG-003(H-1): 놓친 발화가 여러 번 누적돼도 재기동 시 1회만 실행한다."""

MISFIRE_GRACE_TIME_SECONDS_DEFAULT = 300
"""REQ-ATG-003(H-1): misfire_grace_time — 놓친 발화 스킵 의미론(APScheduler
기본 의미론 유지). 이름 있는 상수(REVISABLE, 초기값 5분)."""


class SchedulerRegistry:
    """APScheduler cron 잡 레지스트리 — 실제 잡 등록/조회를 담당한다.

    `scheduler` 인자로 `BaseScheduler` 구현체를 주입할 수 있다(테스트에서는
    `MagicMock()`을 주입해 실 스레드 기동 없이 위임 호출만 검증한다). 미주입 시
    `BackgroundScheduler()`를 기본으로 사용한다.
    """

    def __init__(self, scheduler: BaseScheduler | None = None) -> None:
        self._job_ids: list[str] = []
        self._scheduler: BaseScheduler = (
            scheduler if scheduler is not None else BackgroundScheduler()
        )

    def registered_jobs(self) -> list[str]:
        """현재 등록된 잡 식별자 목록을 반환한다."""
        return list(self._job_ids)

    def register_cron_job(
        self,
        job_id: str,
        trigger: CronTrigger,
        func: Callable[[], None],
        *,
        max_instances: int = MAX_INSTANCES_DEFAULT,
        coalesce: bool = COALESCE_DEFAULT,
        misfire_grace_time: int = MISFIRE_GRACE_TIME_SECONDS_DEFAULT,
    ) -> str:
        """REQ-ATA-080/082 + REQ-ATG-003(H-1): cron 트리거 전용으로 잡을
        등록한다 — `max_instances`/`coalesce`/`misfire_grace_time`을
        기저 스케줄러 `add_job()`에 **명시적으로** 전달한다(기본값 의존 금지).

        동일 `job_id`로 재등록하면 기존 잡을 대체한다(`replace_existing=True`) —
        `_job_ids`에는 중복 추가하지 않는다.
        """
        self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            max_instances=max_instances,
            coalesce=coalesce,
            misfire_grace_time=misfire_grace_time,
        )
        if job_id not in self._job_ids:
            self._job_ids.append(job_id)
        return job_id

    def start(self) -> None:
        """기저 스케줄러를 시작한다."""
        self._scheduler.start()

    def shutdown(self, wait: bool = True) -> None:
        """기저 스케줄러를 종료한다."""
        self._scheduler.shutdown(wait=wait)


def weekly_full_retrain_trigger() -> CronTrigger:
    """REQ-ATA-081: 주간 전체 재학습 — 주말(일요일) 1회, KST.

    미장(美 정규장) 마감 이후 안전 여유를 두기 위해 일요일 01:00 KST로 고정한다
    — 토요일 01:00 KST는 ET 기준 금요일 정규장 진행 중 시각이라 부적절하다.
    """
    return CronTrigger(day_of_week="sun", hour=1, minute=0, timezone=_KST)


def monthly_optuna_tuning_trigger() -> CronTrigger:
    """REQ-ATA-081: 월간 Optuna 튜닝 — 매월 1일 1회, KST.

    `daily_staleness_check_trigger()`(REQ-ATA-083)와 동일한 마감 이후 안전
    여유 관례를 따라 07:00 KST로 고정한다 — 매월 1일 02:00 KST는 ET 기준 전일
    정오 무렵으로 미장 정규장 마감 훨씬 이전이라 부적절하다.
    """
    return CronTrigger(day="1", hour=7, minute=0, timezone=_KST)


def daily_staleness_check_trigger() -> CronTrigger:
    """REQ-ATA-083: 모델 정체 감지 — 매일 1회, 학습 잡과 독립적인 별도 스케줄, KST."""
    return CronTrigger(hour=7, minute=0, timezone=_KST)


def register_default_jobs(
    registry: SchedulerRegistry,
    *,
    weekly_full_retrain_func: Callable[[], None],
    monthly_optuna_tuning_func: Callable[[], None],
    daily_staleness_check_func: Callable[[], None],
) -> list[str]:
    """AC-ATA-008: 주간/월간/일일 정체감지 3개 cron 잡을 모두 등록한다(REQ-ATA-082/083)."""
    registry.register_cron_job(
        WEEKLY_FULL_RETRAIN_JOB_ID, weekly_full_retrain_trigger(), weekly_full_retrain_func
    )
    registry.register_cron_job(
        MONTHLY_OPTUNA_TUNING_JOB_ID, monthly_optuna_tuning_trigger(), monthly_optuna_tuning_func
    )
    registry.register_cron_job(
        DAILY_STALENESS_CHECK_JOB_ID, daily_staleness_check_trigger(), daily_staleness_check_func
    )
    return registry.registered_jobs()

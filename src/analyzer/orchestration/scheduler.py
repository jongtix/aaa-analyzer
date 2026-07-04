"""APScheduler 잡 레지스트리의 구조적 자리 표시자.

REQ-ANALYZER-FOUNDATION-010: 부모 프로세스는 APScheduler 기반 스케줄러에 대해
구조적 자리 표시자(모듈 경계)만 제공한다 — 실제 cron 잡 등록은 본 SPEC 범위
밖이며 후속 SPEC에서 구현된다.

@MX:NOTE: [AUTO] 구조적 자리만 제공 — 실제 잡 등록 로직은 후속 SPEC(INFER-001/TRAIN-001) 소관.
"""


class SchedulerRegistry:
    """향후 도입될 APScheduler cron 잡 등록을 위한 빈 레지스트리 자리 표시자."""

    def __init__(self) -> None:
        self._job_ids: list[str] = []

    def registered_jobs(self) -> list[str]:
        """현재 등록된 잡 식별자 목록을 반환한다(본 SPEC에서는 비어 있음)."""
        return list(self._job_ids)

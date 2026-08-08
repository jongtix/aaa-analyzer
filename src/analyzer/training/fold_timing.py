"""WFV 폴드별 학습 소요 시간 측정 훅 (SPEC-ANALYZER-TRAIN-001 M3).

REQ-AT-073: fold 수는 실측된 학습 소요 시간으로부터 경험적으로 결정해야
하며(shall), 4시간 학습 타임아웃 내에 들어와야 한다(shall). 이 SPEC은
그 타임아웃 자체를 강제하는 cron/워치독 로직을 구현하지 않는다(shall
not, §3) — 이 모듈은 실측 데이터를 수집하는 측정 훅(인터페이스/콜백
지점)만 제공한다. 실제 fold 수 확정 로직(4시간 타임아웃 내로 맞추는
계산)은 plan.md §F M6 소관이다.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FoldTiming:
    """단일 폴드 1회의 실측 학습 소요 시간."""

    fold_index: int
    duration_seconds: float


@dataclass(slots=True)
class FoldDurationTracker:
    """`expanding_window_folds()`가 산출한 폴드를 순회하며 소요 시간을 기록한다.

    fold 수 자체를 결정하지 않는다 — 수집된 `timings`를 소비해 실제 fold
    수를 확정하는 것은 M6(plan.md §F)의 책임이다.
    """

    _timings: list[FoldTiming] = field(default_factory=list)

    @contextmanager
    def measure(self, fold_index: int) -> Iterator[None]:
        """`fold_index` 폴드의 학습 블록을 감싸 소요 시간을 기록한다.

        블록 안에서 예외가 발생해도 그 시점까지의 소요 시간을 기록한 뒤
        예외를 그대로 재전파한다.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._timings.append(FoldTiming(fold_index=fold_index, duration_seconds=elapsed))

    @property
    def timings(self) -> tuple[FoldTiming, ...]:
        """지금까지 기록된 폴드별 소요 시간의 불변 스냅샷을 반환한다."""
        return tuple(self._timings)

    def total_duration_seconds(self) -> float:
        """지금까지 기록된 모든 폴드 소요 시간의 합을 반환한다."""
        return sum(t.duration_seconds for t in self._timings)

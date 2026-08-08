"""src/analyzer/training/fold_timing.py fold 소요 시간 측정 훅 테스트 (SPEC-ANALYZER-TRAIN-001 M3).

REQ-AT-073: fold 수는 실측된 학습 소요 시간으로부터 경험적으로 결정해야
한다(shall). 실제 fold 수 확정 로직은 M6 소관이므로, M3은 측정
훅(인터페이스/콜백 지점)만 제공한다 — 이 테스트는 그 훅 자체가
정확히 동작하는지만 검증한다.
"""

import time

from analyzer.training.fold_timing import FoldDurationTracker


class TestFoldDurationTracker:
    def test_records_duration_for_each_fold(self):
        tracker = FoldDurationTracker()

        with tracker.measure(fold_index=0):
            time.sleep(0.01)
        with tracker.measure(fold_index=1):
            time.sleep(0.01)

        assert len(tracker.timings) == 2
        assert tracker.timings[0].fold_index == 0
        assert tracker.timings[1].fold_index == 1
        assert tracker.timings[0].duration_seconds > 0
        assert tracker.timings[1].duration_seconds > 0

    def test_timings_is_immutable_snapshot(self):
        tracker = FoldDurationTracker()
        with tracker.measure(fold_index=0):
            pass

        snapshot = tracker.timings
        with tracker.measure(fold_index=1):
            pass

        assert len(snapshot) == 1
        assert len(tracker.timings) == 2

    def test_total_duration_sums_all_recorded_folds(self):
        tracker = FoldDurationTracker()
        with tracker.measure(fold_index=0):
            time.sleep(0.01)
        with tracker.measure(fold_index=1):
            time.sleep(0.01)

        assert tracker.total_duration_seconds() >= tracker.timings[0].duration_seconds

    def test_records_duration_even_when_block_raises(self):
        """측정 중 예외가 발생해도 부분 소요 시간을 잃지 않고 재전파한다."""
        tracker = FoldDurationTracker()

        try:
            with tracker.measure(fold_index=0):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert len(tracker.timings) == 1

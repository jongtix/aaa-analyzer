"""src/analyzer/training/split.py Purged Walk-Forward 분할기 테스트 (SPEC-ANALYZER-TRAIN-001 M3).

REQ-AT-070(expanding window 자체 구현)/REQ-AT-071(PURGE_GAP_TRADING_DAYS
단일 소스 소비)/REQ-AT-072(70/15/15 외부화 파라미터)를 검증한다.
AC-AT-005의 worked example(horizon=20, 합성 시계열 200개 거래일, 70/15/15
분할 → 학습/검증 경계에 정확히 20영업일 purge gap)을 그대로 구현한다.
"""

import pytest

from analyzer.labels.config import PURGE_GAP_TRADING_DAYS
from analyzer.training.split import expanding_window_folds, purged_walk_forward_split


class TestPurgedWalkForwardSplit:
    """AC-AT-005: 학습/검증/테스트 경계에서 정확히 horizon 길이만큼 purge gap."""

    def test_ac_at_005_exact_20_trading_day_purge_gap(self):
        n_samples = 200
        horizon = 20

        bounds = purged_walk_forward_split(n_samples, horizon)

        gap = PURGE_GAP_TRADING_DAYS[horizon]
        assert gap == 20

        # 학습 구간 마지막 인덱스와 검증 구간 시작 인덱스 사이에 정확히
        # horizon 길이(20)만큼의 인덱스가 존재해야 한다(그 구간은 어디에도
        # 포함되지 않음).
        assert bounds.val_start - bounds.train_end == gap
        assert bounds.test_start - bounds.val_end == gap

    def test_default_ratios_are_70_15_15(self):
        bounds = purged_walk_forward_split(200, horizon=20)
        gap = PURGE_GAP_TRADING_DAYS[20]
        remaining = 200 - 2 * gap  # 160

        train_count = bounds.train_end - bounds.train_start
        val_count = bounds.val_end - bounds.val_start
        test_count = bounds.test_end - bounds.test_start

        assert train_count == round(remaining * 0.70)
        assert val_count == round(remaining * 0.15)
        assert train_count + val_count + test_count == remaining

    def test_ratios_are_externalized_not_hardcoded(self):
        """REQ-AT-072: 분할 비율은 코드에 하드코딩되지 않고 파라미터로 조정 가능해야 한다."""
        bounds_default = purged_walk_forward_split(200, horizon=20)
        bounds_custom = purged_walk_forward_split(
            200, horizon=20, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
        )

        default_train_count = bounds_default.train_end - bounds_default.train_start
        custom_train_count = bounds_custom.train_end - bounds_custom.train_start

        assert default_train_count != custom_train_count

    def test_no_index_appears_in_both_train_and_val(self):
        """데이터 리크 부재 가드: 학습/검증 구간이 겹치지 않고 purge gap이 제외된다."""
        bounds = purged_walk_forward_split(200, horizon=20)

        train_indices = set(range(bounds.train_start, bounds.train_end))
        purge1_indices = set(range(bounds.train_end, bounds.val_start))
        val_indices = set(range(bounds.val_start, bounds.val_end))
        purge2_indices = set(range(bounds.val_end, bounds.test_start))
        test_indices = set(range(bounds.test_start, bounds.test_end))

        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)
        assert train_indices.isdisjoint(purge1_indices)
        assert val_indices.isdisjoint(purge1_indices)
        assert val_indices.isdisjoint(purge2_indices)
        assert test_indices.isdisjoint(purge2_indices)
        assert len(purge1_indices) == PURGE_GAP_TRADING_DAYS[20]
        assert len(purge2_indices) == PURGE_GAP_TRADING_DAYS[20]

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="합"):
            purged_walk_forward_split(
                200, horizon=20, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5
            )

    def test_rejects_unsupported_horizon(self):
        with pytest.raises(KeyError):
            purged_walk_forward_split(200, horizon=99)

    def test_raises_when_insufficient_samples_for_purge_gaps(self):
        with pytest.raises(ValueError, match="샘플"):
            purged_walk_forward_split(10, horizon=20)


class TestExpandingWindowFolds:
    """REQ-AT-070: expanding window 방식 — 학습 구간이 확장되고 검증 구간이 앞으로 이동."""

    def test_train_window_expands_across_folds(self):
        folds = expanding_window_folds(n_samples=300, horizon=20, n_folds=3, val_size=20)

        assert len(folds) == 3
        train_sizes = [f.train_end - f.train_start for f in folds]
        assert train_sizes == sorted(train_sizes)
        assert train_sizes[0] < train_sizes[1] < train_sizes[2]

    def test_val_window_moves_forward_and_purge_gap_holds(self):
        folds = expanding_window_folds(n_samples=300, horizon=20, n_folds=3, val_size=20)
        gap = PURGE_GAP_TRADING_DAYS[20]

        for i in range(1, len(folds)):
            assert folds[i].val_start > folds[i - 1].val_start
            assert folds[i].train_end == folds[i - 1].val_end

        for fold in folds:
            assert fold.val_start - fold.train_end == gap

    def test_all_folds_share_the_same_held_out_test_region(self):
        """테스트 구간은 모든 폴드가 공유하는 고정 홀드아웃이며 학습/검증에 사용되지 않는다."""
        folds = expanding_window_folds(n_samples=300, horizon=20, n_folds=3, val_size=20)

        test_regions = {(f.test_start, f.test_end) for f in folds}
        assert len(test_regions) == 1

        for fold in folds:
            train_indices = set(range(fold.train_start, fold.train_end))
            val_indices = set(range(fold.val_start, fold.val_end))
            test_indices = set(range(fold.test_start, fold.test_end))
            assert train_indices.isdisjoint(test_indices)
            assert val_indices.isdisjoint(test_indices)

    def test_no_fold_data_leaks_into_test_region_via_purge_gap(self):
        gap = PURGE_GAP_TRADING_DAYS[20]
        folds = expanding_window_folds(n_samples=300, horizon=20, n_folds=3, val_size=20)

        last_val_end = max(f.val_end for f in folds)
        test_start = folds[0].test_start
        assert test_start - last_val_end == gap

    def test_raises_when_insufficient_samples_for_requested_folds(self):
        with pytest.raises(ValueError):
            expanding_window_folds(n_samples=50, horizon=20, n_folds=5, val_size=20)

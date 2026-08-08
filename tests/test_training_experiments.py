"""src/analyzer/training/experiments.py 단조 제약 비교 실험 테스트 (SPEC-ANALYZER-TRAIN-001 M7).

REQ-AT-110(단조 제약 적용/미적용 비교)/REQ-AT-111(초기 기본값 미적용,
자동 채택 금지)을 검증한다.
"""

import numpy as np
import pytest

from analyzer.training.experiments import (
    MonotoneConstraintExperimentResult,
    build_monotone_constraints,
    compare_monotone_constraints,
)


class TestBuildMonotoneConstraints:
    def test_price_derived_features_get_positive_constraint(self):
        constraints = build_monotone_constraints(["KMID", "ROC_5"])

        assert constraints == [1, 1]

    def test_frozen_features_get_no_constraint(self):
        constraints = build_monotone_constraints(["foreign_net_ratio"])

        assert constraints == [0]

    def test_mixed_features(self):
        constraints = build_monotone_constraints(["KMID", "foreign_net_ratio", "MA_20"])

        assert constraints == [1, 0, 1]

    def test_unregistered_feature_raises_value_error(self):
        with pytest.raises(ValueError):
            build_monotone_constraints(["not_a_real_feature"])


class TestCompareMonotoneConstraints:
    def _synthetic_data(self, seed: int, n: int = 150) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 2))
        y = x[:, 0] * 0.02 + x[:, 1] * -0.01 + rng.normal(scale=0.01, size=n)
        return x, y

    def test_returns_both_strategy_mae(self):
        x, y = self._synthetic_data(seed=1)

        result = compare_monotone_constraints(
            x, y, feature_names=["KMID", "foreign_net_ratio"], random_state=42
        )

        assert isinstance(result, MonotoneConstraintExperimentResult)
        assert result.unconstrained_val_mae >= 0
        assert result.constrained_val_mae >= 0

    def test_reproducible_with_fixed_random_state(self):
        x, y = self._synthetic_data(seed=1)

        result1 = compare_monotone_constraints(
            x, y, feature_names=["KMID", "foreign_net_ratio"], random_state=42
        )
        result2 = compare_monotone_constraints(
            x, y, feature_names=["KMID", "foreign_net_ratio"], random_state=42
        )

        assert result1 == result2

    def test_constrained_is_better_is_a_comparison_signal_only(self):
        """REQ-AT-111: 이 프로퍼티는 참고 신호일 뿐 자동 채택을 트리거하지 않는다."""
        x, y = self._synthetic_data(seed=2)

        result = compare_monotone_constraints(
            x, y, feature_names=["KMID", "foreign_net_ratio"], random_state=7
        )

        assert isinstance(result.constrained_is_better, bool)

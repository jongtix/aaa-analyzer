"""src/analyzer/training/thick_tail.py Huber vs 윈저라이징 비교 하네스 테스트
(SPEC-ANALYZER-TRAIN-001 M4).

REQ-AT-050(Huber loss vs 타깃 윈저라이징 백테스트 비교)/REQ-AT-051
(#141 이상치 클래스 함께 평가)/REQ-AT-052(SMOTE 미채택, shall not)를
검증한다. AC-AT-012의 worked example(이상치 포함 합성 레이블, 동일 시드
2회 실행 → 완전히 동일한 성능 지표, SMOTE/ADASYN 키워드 부재)을 그대로
구현한다.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from analyzer.training.thick_tail import (
    ThickTailComparisonResult,
    compare_thick_tail_strategies,
    winsorize_targets,
)


def _synthetic_features_and_targets_with_outliers(
    n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """REQ-AT-051: aaa-infra#141이 유발하는 허위 극단 음수 수익률 이상치 클래스를 모사한다."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n, 3))
    y = X @ np.array([0.02, -0.01, 0.015]) + rng.normal(loc=0.0, scale=0.02, size=n)

    # 마지막 5%는 허위 극단 음수 수익률 이상치(해외 배당 커버리지 결함 모사).
    n_outliers = max(1, n // 20)
    y[-n_outliers:] = -1.0 + rng.normal(loc=0.0, scale=0.05, size=n_outliers)

    return X, y


class TestWinsorizeTargets:
    def test_clips_values_outside_quantile_range(self):
        y = np.array([-100.0, -1.0, 0.0, 1.0, 2.0, 100.0])

        clipped = winsorize_targets(y, lower_quantile=0.1, upper_quantile=0.9)

        assert clipped.min() > y.min()
        assert clipped.max() < y.max()

    def test_does_not_mutate_input_array(self):
        y = np.array([-100.0, -1.0, 0.0, 1.0, 2.0, 100.0])
        original = y.copy()

        winsorize_targets(y, lower_quantile=0.1, upper_quantile=0.9)

        np.testing.assert_array_equal(y, original)


class TestCompareThickTailStrategies:
    """AC-AT-012: 이상치 포함 합성 레이블, 동일 시드 2회 실행 → 완전히 동일한 결과."""

    def test_ac_at_012_reproducible_across_identical_runs(self):
        X, y = _synthetic_features_and_targets_with_outliers(n=200, seed=7)

        result1 = compare_thick_tail_strategies(X, y, random_state=42)
        result2 = compare_thick_tail_strategies(X, y, random_state=42)

        assert result1 == result2

    def test_returns_both_strategy_mae_values(self):
        X, y = _synthetic_features_and_targets_with_outliers(n=200, seed=7)

        result = compare_thick_tail_strategies(X, y, random_state=42)

        assert isinstance(result, ThickTailComparisonResult)
        assert result.huber_val_mae >= 0
        assert result.winsorized_val_mae >= 0

    def test_different_random_state_may_yield_different_result(self):
        X, y = _synthetic_features_and_targets_with_outliers(n=200, seed=7)

        result1 = compare_thick_tail_strategies(X, y, random_state=1)
        result2 = compare_thick_tail_strategies(X, y, random_state=2)

        assert result1 != result2


class TestSmoteGuard:
    """REQ-AT-052: SMOTE류 합성 샘플링 기법을 채택하지 않는다(shall not)."""

    def test_ac_at_012_no_smote_or_adasyn_keywords_in_training_package(self):
        training_dir = Path(__file__).resolve().parent.parent / "src" / "analyzer" / "training"

        result = subprocess.run(
            [
                "grep",
                "-rliE",
                "--include=*.py",
                "smote|adasyn|synthetic.*sampl",
                str(training_dir),
            ],
            capture_output=True,
            text=True,
        )

        # grep 매치 0건 → exit code 1. 매치가 있으면(exit 0) 실패해야 한다.
        assert result.returncode == 1, f"금지된 합성 샘플링 키워드 발견:\n{result.stdout}"

    def test_no_smote_import_in_process(self):
        script = (
            "import analyzer.training.thick_tail\n"
            "import sys\n"
            "assert not any(\n"
            "    'imblearn' in name or 'smote' in name.lower() for name in sys.modules\n"
            ")\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "OK"


if __name__ == "__main__":
    pytest.main([__file__])

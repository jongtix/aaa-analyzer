"""src/analyzer/training/backtest.py 초기 백테스트 지표 테스트 (SPEC-ANALYZER-TRAIN-001 M7).

REQ-AT-120(6개 지표: Hit Rate/IC(Rank IC)/Precision/Sharpe/MDD/confidence
캘리브레이션)/REQ-AT-121(낮은 R²는 결함이 아닌 인지된 한계)을 검증한다.
AC-AT-013의 worked example(방향이 60% 일치하는 합성 예측/실현 수익률
시퀀스)을 그대로 구현한다.
"""

import numpy as np
import pytest
from scipy import stats

from analyzer.training.backtest import BacktestMetrics, compute_backtest_metrics


def _synthetic_60pct_direction_match(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """방향이 60% 일치하도록 구성된 합성 score/실현수익률 시퀀스."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n)
    correct_direction = rng.random(n) < 0.60
    magnitude = np.abs(rng.normal(scale=0.02, size=n))
    returns = np.where(correct_direction, np.sign(scores) * magnitude, -np.sign(scores) * magnitude)
    return scores, returns


class TestComputeBacktestMetrics:
    """AC-AT-013: 6개 지표 모두 NaN 없이 반환."""

    def test_ac_at_013_hit_rate_approximately_060(self):
        scores, returns = _synthetic_60pct_direction_match()
        confidences = np.full_like(scores, 0.7)

        metrics = compute_backtest_metrics(scores, returns, confidences)

        assert metrics.hit_rate == pytest.approx(0.60, abs=0.05)

    def test_ac_at_013_ic_matches_spearman_definition(self):
        scores, returns = _synthetic_60pct_direction_match()
        confidences = np.full_like(scores, 0.7)

        metrics = compute_backtest_metrics(scores, returns, confidences)

        expected_ic, _pvalue = stats.spearmanr(scores, returns)
        assert metrics.ic == pytest.approx(expected_ic, abs=1e-9)

    def test_ac_at_013_all_6_metrics_present_and_not_nan(self):
        scores, returns = _synthetic_60pct_direction_match()
        confidences = np.full_like(scores, 0.7)

        metrics = compute_backtest_metrics(scores, returns, confidences)

        assert isinstance(metrics, BacktestMetrics)
        values = [
            metrics.hit_rate,
            metrics.ic,
            metrics.precision,
            metrics.sharpe_ratio,
            metrics.max_drawdown,
            metrics.confidence_calibration,
        ]
        assert len(values) == 6
        for value in values:
            assert not np.isnan(value)

    def test_hit_rate_is_1_when_all_directions_match(self):
        scores = np.array([1.0, -1.0, 2.0, -3.0])
        returns = np.array([0.01, -0.01, 0.02, -0.03])
        confidences = np.full_like(scores, 0.9)

        metrics = compute_backtest_metrics(scores, returns, confidences)

        assert metrics.hit_rate == 1.0

    def test_hit_rate_is_0_when_all_directions_mismatch(self):
        scores = np.array([1.0, -1.0, 2.0, -3.0])
        returns = np.array([-0.01, 0.01, -0.02, 0.03])
        confidences = np.full_like(scores, 0.9)

        metrics = compute_backtest_metrics(scores, returns, confidences)

        assert metrics.hit_rate == 0.0

    def test_max_drawdown_is_non_negative(self):
        scores, returns = _synthetic_60pct_direction_match()
        confidences = np.full_like(scores, 0.7)

        metrics = compute_backtest_metrics(scores, returns, confidences)

        assert metrics.max_drawdown >= 0

    def test_precision_restricted_to_directional_calls(self):
        """score==0인 표본(방향 없음)은 precision 분모에서 제외된다."""
        scores = np.array([1.0, -1.0, 0.0, 0.0])
        returns = np.array([0.01, -0.01, 0.05, -0.05])
        confidences = np.full_like(scores, 0.9)

        metrics = compute_backtest_metrics(scores, returns, confidences)

        assert metrics.precision == 1.0

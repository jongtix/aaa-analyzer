"""src/analyzer/training/ensemble.py 앙상블 score + confidence 테스트 (SPEC-ANALYZER-TRAIN-001 M5).

REQ-AT-080/081(앙상블 score + 개별 score 보존)/REQ-AT-082~084(confidence +
score_ensemble=0 + 크로싱 보정)를 검증한다. AC-AT-006/AC-AT-007의
worked example을 그대로 구현한다.
"""

import math

import pytest

from analyzer.training.ensemble import (
    EnsembleResult,
    compute_confidence,
    compute_ensemble_score,
    resolve_quantile_crossing,
)


class TestComputeEnsembleScore:
    """AC-AT-006: 부호 일치 시 sign × min(|lgbm|, |xgb|), 불일치 시 0."""

    def test_ac_at_006_case_a_matching_positive_signs(self):
        result = compute_ensemble_score(lgbm_score=0.05, xgb_score=0.03)

        assert result.score_ensemble == pytest.approx(0.03, abs=0.0001)

    def test_ac_at_006_case_b_mismatched_signs(self):
        result = compute_ensemble_score(lgbm_score=0.05, xgb_score=-0.02)

        assert result.score_ensemble == pytest.approx(0.0, abs=0.0001)

    def test_matching_negative_signs(self):
        result = compute_ensemble_score(lgbm_score=-0.05, xgb_score=-0.03)

        assert result.score_ensemble == pytest.approx(-0.03, abs=0.0001)

    def test_zero_score_is_treated_as_mismatch(self):
        """부호가 없는 0은 어느 쪽과도 "일치"로 볼 수 없다 — 앙상블 score는 0."""
        result = compute_ensemble_score(lgbm_score=0.0, xgb_score=0.05)

        assert result.score_ensemble == pytest.approx(0.0, abs=0.0001)

    def test_preserves_individual_scores(self):
        """REQ-AT-081: 개별 lgbm_score/xgb_score를 앙상블 score와 함께 보존."""
        result = compute_ensemble_score(lgbm_score=0.05, xgb_score=0.03)

        assert isinstance(result, EnsembleResult)
        assert result.lgbm_score == 0.05
        assert result.xgb_score == 0.03


class TestResolveQuantileCrossing:
    """REQ-AT-084: p10 > p90(크로싱) 발생 시 정렬 스왑으로 보정."""

    def test_no_crossing_returns_unchanged(self):
        p10, p90 = resolve_quantile_crossing(p10=-0.02, p90=0.06)

        assert (p10, p90) == (-0.02, 0.06)

    def test_crossing_is_corrected_by_swap(self):
        p10, p90 = resolve_quantile_crossing(p10=0.05, p90=0.03)

        assert p10 <= p90
        assert (p10, p90) == (0.03, 0.05)


class TestComputeConfidence:
    """AC-AT-007: confidence = Φ(|score_ensemble|/σ), σ=(p90-p10)/2.563."""

    def test_ac_at_007_worked_example(self):
        confidence = compute_confidence(score_ensemble=0.04, p10=-0.02, p90=0.06)

        assert confidence == pytest.approx(0.90, abs=0.01)

    def test_ac_at_007_score_ensemble_zero_returns_half(self):
        confidence = compute_confidence(score_ensemble=0.0, p10=-0.02, p90=0.06)

        assert confidence == 0.5

    def test_ac_at_007_crossing_is_corrected_before_confidence_calc(self):
        """p10=0.05 > p90=0.03(크로싱) → 보정 후 계산이 성립해야 한다."""
        confidence = compute_confidence(score_ensemble=0.04, p10=0.05, p90=0.03)

        assert 0.0 <= confidence <= 1.0

    def test_confidence_matches_standard_normal_cdf_directly(self):
        score_ensemble = 0.04
        p10, p90 = -0.02, 0.06
        sigma = (p90 - p10) / 2.563
        expected = 0.5 * (1.0 + math.erf((abs(score_ensemble) / sigma) / math.sqrt(2.0)))

        confidence = compute_confidence(score_ensemble=score_ensemble, p10=p10, p90=p90)

        assert confidence == pytest.approx(expected, abs=1e-9)

    def test_negative_score_ensemble_uses_absolute_value(self):
        positive_confidence = compute_confidence(score_ensemble=0.04, p10=-0.02, p90=0.06)
        negative_confidence = compute_confidence(score_ensemble=-0.04, p10=-0.02, p90=0.06)

        assert positive_confidence == pytest.approx(negative_confidence, abs=1e-9)

    def test_degenerate_sigma_raises_value_error(self):
        """p10 == p90(축퇴 분포)이면 0-나눗셈 대신 명시적으로 실패해야 한다."""
        with pytest.raises(ValueError):
            compute_confidence(score_ensemble=0.04, p10=0.03, p90=0.03)

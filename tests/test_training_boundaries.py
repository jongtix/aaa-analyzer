"""src/analyzer/training/boundaries.py 등급 경계 분위수 역산 테스트 (SPEC-ANALYZER-TRAIN-001 M4).

REQ-AT-040을 검증한다. AC-AT-011의 worked example(합성 실현 수익률 1000개
샘플, 목표 비율은 테스트 코드 내부 파라미터로만 사용 — SPEC 문서에는
어떤 확정 수치도 기록하지 않는다, REQ-AT-041)을 그대로 구현한다.
"""

import numpy as np
import pytest

from analyzer.training.boundaries import (
    GRADE_ORDER,
    classify_by_boundaries,
    infer_grade_boundaries,
    infer_grade_boundaries_all_combinations,
)


def _synthetic_returns(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.1, size=n)


class TestInferGradeBoundaries:
    """AC-AT-011: 목표 등급 비율에 대응하는 경계값을 실현 수익률 분포로부터 역산."""

    def test_ac_at_011_boundaries_are_monotonically_increasing(self):
        returns = _synthetic_returns(1000, seed=42)
        # 테스트 전용 목표 비율(구현 파라미터) — SPEC 문서에는 기록하지 않는다.
        target_ratios = {
            "STRONG_SELL": 0.10,
            "SELL": 0.20,
            "HOLD": 0.40,
            "BUY": 0.20,
            "STRONG_BUY": 0.10,
        }

        boundaries = infer_grade_boundaries(returns, target_ratios)

        values = [boundaries[k] for k in boundaries]
        assert values == sorted(values)
        assert len(set(values)) == len(values)  # 엄격한 단조 증가(중복 없음)

    def test_ac_at_011_class_fractions_match_target_within_tolerance(self):
        returns = _synthetic_returns(1000, seed=42)
        target_ratios = {
            "STRONG_SELL": 0.10,
            "SELL": 0.20,
            "HOLD": 0.40,
            "BUY": 0.20,
            "STRONG_BUY": 0.10,
        }

        boundaries = infer_grade_boundaries(returns, target_ratios)
        classified = classify_by_boundaries(returns, boundaries)

        for grade, target_ratio in target_ratios.items():
            actual_ratio = float(np.mean(classified == grade))
            assert actual_ratio == pytest.approx(target_ratio, abs=0.03)

    def test_returns_four_boundaries_for_five_grade_classes(self):
        """5개 등급을 나누는 데 필요한 분할점은 수학적으로 4개다."""
        returns = _synthetic_returns(1000, seed=1)
        target_ratios = dict.fromkeys(GRADE_ORDER, 0.2)

        boundaries = infer_grade_boundaries(returns, target_ratios)

        assert len(boundaries) == 4

    def test_boundary_keys_are_adjacent_grade_pairs(self):
        returns = _synthetic_returns(500, seed=2)
        target_ratios = dict.fromkeys(GRADE_ORDER, 0.2)

        boundaries = infer_grade_boundaries(returns, target_ratios)

        assert list(boundaries.keys()) == [
            "STRONG_SELL_SELL",
            "SELL_HOLD",
            "HOLD_BUY",
            "BUY_STRONG_BUY",
        ]

    def test_rejects_ratios_not_summing_to_one(self):
        returns = _synthetic_returns(100, seed=3)
        with pytest.raises(ValueError, match="합"):
            infer_grade_boundaries(
                returns,
                {
                    "STRONG_SELL": 0.10,
                    "SELL": 0.20,
                    "HOLD": 0.40,
                    "BUY": 0.20,
                    "STRONG_BUY": 0.20,
                },
            )

    def test_rejects_non_positive_ratio(self):
        returns = _synthetic_returns(100, seed=4)
        with pytest.raises(ValueError):
            infer_grade_boundaries(
                returns,
                {
                    "STRONG_SELL": 0.0,
                    "SELL": 0.30,
                    "HOLD": 0.40,
                    "BUY": 0.20,
                    "STRONG_BUY": 0.10,
                },
            )


class TestInferGradeBoundariesAllCombinations:
    """REQ-AT-040: 시장(국내/해외) × horizon(D20/D60) 4개 조합 각각에 대해 독립 역산."""

    def test_computes_independent_boundaries_per_combination(self):
        target_ratios = dict.fromkeys(GRADE_ORDER, 0.2)
        returns_by_combo = {
            ("domestic", 20): _synthetic_returns(500, seed=10),
            ("domestic", 60): _synthetic_returns(500, seed=11),
            ("overseas", 20): _synthetic_returns(500, seed=12),
            ("overseas", 60): _synthetic_returns(500, seed=13),
        }

        result = infer_grade_boundaries_all_combinations(returns_by_combo, target_ratios)

        assert set(result.keys()) == set(returns_by_combo.keys())
        for _combo, boundaries in result.items():
            assert len(boundaries) == 4
            values = list(boundaries.values())
            assert values == sorted(values)

        # 서로 다른 시드의 분포이므로 조합 간 경계값이 달라야 한다(독립 역산 확인).
        assert result[("domestic", 20)]["HOLD_BUY"] != result[("overseas", 60)]["HOLD_BUY"]

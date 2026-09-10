"""경계값 저장소 로더·히스테리시스 시프트·등급 분류 배선 테스트
(SPEC-ANALYZER-INFER-001 M4, REQ-AIF-081/111, AC-AIF-015).

모든 수치는 합성 입력이다 — 실측 경계값은 저장소 산출물(JSON)에만 존재하며
테스트 리터럴로 복제하지 않는다(REQ-AIF-080 하드코딩 금지 원칙).
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from analyzer.inference import boundaries_store
from analyzer.inference.boundaries_store import (
    BoundarySet,
    GradeBoundariesArtifact,
    classify_grades,
    default_artifact_path,
    derive_grade_margin_delta,
    load_grade_boundaries,
    shift_boundaries,
)
from analyzer.training import boundaries as boundaries_module
from analyzer.training.boundaries import GRADE_ORDER

_SYNTHETIC_BOUNDARIES = {
    "STRONG_SELL_SELL": -0.10,
    "SELL_HOLD": -0.03,
    "HOLD_BUY": 0.03,
    "BUY_STRONG_BUY": 0.10,
}
_SYNTHETIC_RATIOS = {
    "STRONG_SELL": 0.05,
    "SELL": 0.20,
    "HOLD": 0.50,
    "BUY": 0.20,
    "STRONG_BUY": 0.05,
}


def _write_synthetic_artifact(path: Path, *, delta: float = 0.006) -> Path:
    payload = {
        "schema_version": 1,
        "generated_at": "2026-01-01T00:00:00+09:00",
        "data_as_of": "2025-12-31",
        "target_ratios": _SYNTHETIC_RATIOS,
        "combinations": [
            {
                "market": market,
                "horizon": horizon,
                "row_count": 1000 + horizon,
                "boundaries": {k: v * (horizon / 20) for k, v in _SYNTHETIC_BOUNDARIES.items()},
            }
            for market in ("domestic", "overseas")
            for horizon in (20, 60)
        ],
        "grade_margin_delta": {
            "value": delta,
            "provisional": True,
            "derivation": "synthetic",
            "min_adjacent_gap": delta * 10,
            "replace_by": "SPEC-NOTIFIER-FILTER-001",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadGradeBoundaries:
    def test_loads_four_combinations_with_metadata(self, tmp_path: Path):
        path = _write_synthetic_artifact(tmp_path / "gb.json")

        artifact = load_grade_boundaries(path)

        assert isinstance(artifact, GradeBoundariesArtifact)
        assert set(artifact.boundaries_by_combination) == {
            ("domestic", 20),
            ("domestic", 60),
            ("overseas", 20),
            ("overseas", 60),
        }
        assert artifact.boundaries_by_combination[("domestic", 20)] == _SYNTHETIC_BOUNDARIES
        assert artifact.grade_margin_delta == pytest.approx(0.006)
        assert artifact.grade_margin_delta_provisional is True
        assert artifact.target_ratios == _SYNTHETIC_RATIOS
        assert artifact.data_as_of == date(2025, 12, 31)
        assert artifact.row_counts[("overseas", 60)] == 1060

    def test_boundary_keys_follow_grade_order(self, tmp_path: Path):
        path = _write_synthetic_artifact(tmp_path / "gb.json")

        artifact = load_grade_boundaries(path)

        expected_keys = [f"{GRADE_ORDER[i]}_{GRADE_ORDER[i + 1]}" for i in range(4)]
        for boundaries in artifact.boundaries_by_combination.values():
            assert list(boundaries) == expected_keys

    def test_ac_aif_015_second_load_never_reinvokes_quantile_inversion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Arrange — 분위수 역산 함수 전부를 mock으로 교체(호출 카운트 0 검증).
        infer_single = MagicMock(name="infer_grade_boundaries")
        infer_all = MagicMock(name="infer_grade_boundaries_all_combinations")
        monkeypatch.setattr(boundaries_module, "infer_grade_boundaries", infer_single)
        monkeypatch.setattr(boundaries_module, "infer_grade_boundaries_all_combinations", infer_all)
        path = _write_synthetic_artifact(tmp_path / "gb.json")

        # Act — 추론 프로세스 2회 연속 기동을 로더 2회 호출로 모사한다.
        first = load_grade_boundaries(path)
        second = load_grade_boundaries(path)

        # Assert
        assert infer_single.call_count == 0
        assert infer_all.call_count == 0
        assert first == second

    def test_rejects_non_monotonic_boundaries(self, tmp_path: Path):
        path = _write_synthetic_artifact(tmp_path / "gb.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["combinations"][0]["boundaries"]["HOLD_BUY"] = -0.5
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="단조"):
            load_grade_boundaries(path)

    def test_rejects_missing_combination(self, tmp_path: Path):
        path = _write_synthetic_artifact(tmp_path / "gb.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["combinations"].pop()
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="조합"):
            load_grade_boundaries(path)

    def test_rejects_non_positive_delta(self, tmp_path: Path):
        path = _write_synthetic_artifact(tmp_path / "gb.json", delta=0.0)

        with pytest.raises(ValueError, match="grade_margin_delta"):
            load_grade_boundaries(path)


class TestShiftBoundaries:
    def test_promote_moves_every_boundary_away_from_hold(self):
        shifted = shift_boundaries(
            _SYNTHETIC_BOUNDARIES, delta=0.01, boundary_set=BoundarySet.PROMOTE
        )

        assert shifted["STRONG_SELL_SELL"] == pytest.approx(-0.11)
        assert shifted["SELL_HOLD"] == pytest.approx(-0.04)
        assert shifted["HOLD_BUY"] == pytest.approx(0.04)
        assert shifted["BUY_STRONG_BUY"] == pytest.approx(0.11)

    def test_demote_moves_every_boundary_toward_hold(self):
        shifted = shift_boundaries(
            _SYNTHETIC_BOUNDARIES, delta=0.01, boundary_set=BoundarySet.DEMOTE
        )

        assert shifted["STRONG_SELL_SELL"] == pytest.approx(-0.09)
        assert shifted["SELL_HOLD"] == pytest.approx(-0.02)
        assert shifted["HOLD_BUY"] == pytest.approx(0.02)
        assert shifted["BUY_STRONG_BUY"] == pytest.approx(0.09)

    def test_base_set_returns_boundaries_unchanged(self):
        shifted = shift_boundaries(_SYNTHETIC_BOUNDARIES, delta=0.01, boundary_set=BoundarySet.BASE)

        assert shifted == _SYNTHETIC_BOUNDARIES

    def test_promote_and_demote_form_a_dead_zone_of_two_delta(self):
        promote = shift_boundaries(
            _SYNTHETIC_BOUNDARIES, delta=0.01, boundary_set=BoundarySet.PROMOTE
        )
        demote = shift_boundaries(
            _SYNTHETIC_BOUNDARIES, delta=0.01, boundary_set=BoundarySet.DEMOTE
        )

        for key in _SYNTHETIC_BOUNDARIES:
            assert abs(promote[key] - demote[key]) == pytest.approx(0.02)

    def test_score_inside_dead_zone_disagrees_between_sets(self):
        promote = shift_boundaries(
            _SYNTHETIC_BOUNDARIES, delta=0.01, boundary_set=BoundarySet.PROMOTE
        )
        demote = shift_boundaries(
            _SYNTHETIC_BOUNDARIES, delta=0.01, boundary_set=BoundarySet.DEMOTE
        )

        # HOLD_BUY 경계(0.03) 바로 위 — 승급(PROMOTE)은 아직 HOLD, 강등(DEMOTE)은 BUY.
        assert classify_grades([0.035], promote)[0] == "HOLD"
        assert classify_grades([0.035], demote)[0] == "BUY"
        # 데드존 밖 — 두 세트가 일치한다.
        assert classify_grades([0.05], promote)[0] == classify_grades([0.05], demote)[0] == "BUY"

    def test_rejects_delta_that_breaks_monotonicity(self):
        with pytest.raises(ValueError, match="단조"):
            shift_boundaries(_SYNTHETIC_BOUNDARIES, delta=0.05, boundary_set=BoundarySet.DEMOTE)

    def test_rejects_negative_delta(self):
        with pytest.raises(ValueError, match="delta"):
            shift_boundaries(_SYNTHETIC_BOUNDARIES, delta=-0.01, boundary_set=BoundarySet.PROMOTE)


class TestDeriveGradeMarginDelta:
    def test_uses_ten_percent_of_minimum_adjacent_gap_across_all_combinations(self):
        by_combination = {
            ("domestic", 20): {"a": -0.10, "b": -0.03, "c": 0.03, "d": 0.10},
            ("overseas", 60): {"a": -0.20, "b": -0.05, "c": 0.02, "d": 0.30},
        }

        delta, min_gap = derive_grade_margin_delta(by_combination)

        assert min_gap == pytest.approx(0.06)
        assert delta == pytest.approx(0.006)

    def test_custom_ratio_is_applied(self):
        by_combination = {("domestic", 20): {"a": 0.0, "b": 0.5, "c": 1.0, "d": 1.5}}

        delta, _ = derive_grade_margin_delta(by_combination, ratio=0.2)

        assert delta == pytest.approx(0.1)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="조합"):
            derive_grade_margin_delta({})


class TestClassifyGrades:
    def test_delegates_to_training_classify_by_boundaries(self, monkeypatch: pytest.MonkeyPatch):
        spy = MagicMock(wraps=boundaries_module.classify_by_boundaries)
        monkeypatch.setattr(boundaries_store, "classify_by_boundaries", spy)

        result = classify_grades([-0.5, 0.0, 0.5], _SYNTHETIC_BOUNDARIES)

        assert spy.call_count == 1
        assert list(result) == ["STRONG_SELL", "HOLD", "STRONG_BUY"]

    def test_artifact_classifies_for_a_combination_and_boundary_set(self, tmp_path: Path):
        artifact = load_grade_boundaries(
            _write_synthetic_artifact(tmp_path / "gb.json", delta=0.01)
        )

        base = artifact.classify("domestic", 20, np.array([0.035]))
        promote = artifact.classify("domestic", 20, np.array([0.035]), BoundarySet.PROMOTE)

        assert base[0] == "BUY"
        assert promote[0] == "HOLD"

    def test_artifact_rejects_unknown_combination(self, tmp_path: Path):
        artifact = load_grade_boundaries(_write_synthetic_artifact(tmp_path / "gb.json"))

        with pytest.raises(KeyError):
            artifact.classify("domestic", 120, [0.0])


class TestPackagedArtifact:
    """패키지 동봉 실측 산출물(`grade_boundaries.json`) — 구조만 검증하고 수치는 단언하지 않는다."""

    def test_default_path_points_inside_inference_package(self):
        path = default_artifact_path()

        assert path.name == "grade_boundaries.json"
        assert path.parent.name == "inference"
        assert path.is_file()

    def test_packaged_artifact_loads_with_four_combinations_and_provisional_delta(self):
        artifact = load_grade_boundaries()

        assert len(artifact.boundaries_by_combination) == 4
        assert artifact.grade_margin_delta > 0
        assert artifact.grade_margin_delta_provisional is True
        assert artifact.target_ratios == _SYNTHETIC_RATIOS  # 사용자 확정 목표 비율(입력값)
        assert all(n > 0 for n in artifact.row_counts.values())
        assert artifact.data_as_of <= date.today()

    def test_packaged_delta_keeps_all_shifted_sets_monotonic(self):
        artifact = load_grade_boundaries()

        for market, horizon in artifact.boundaries_by_combination:
            for boundary_set in BoundarySet:
                shifted = artifact.boundaries_for(market, horizon, boundary_set)
                values = list(shifted.values())
                assert values == sorted(values)

    def test_ac_aif_015_packaged_artifact_two_startups_never_recompute(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        infer_single = MagicMock(name="infer_grade_boundaries")
        infer_all = MagicMock(name="infer_grade_boundaries_all_combinations")
        monkeypatch.setattr(boundaries_module, "infer_grade_boundaries", infer_single)
        monkeypatch.setattr(boundaries_module, "infer_grade_boundaries_all_combinations", infer_all)

        first = load_grade_boundaries()
        second = load_grade_boundaries()

        assert infer_single.call_count == 0
        assert infer_all.call_count == 0
        assert first == second

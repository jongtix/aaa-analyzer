"""등급 경계값 저장소 — 실측 경계표 + 마진 δ 로더와 히스테리시스 경계 파생
(SPEC-ANALYZER-INFER-001 M4, REQ-AIF-080/081/111, design.md §3).

REQ-AIF-081: 추론 시스템은 매 프로세스 기동마다 경계값을 재계산하지 않고
이 모듈이 읽는 저장소(패키지 동봉 JSON `grade_boundaries.json`)에서만
로드한다. 이 모듈은 `training/boundaries.py`의 분위수 역산 함수
(`infer_grade_boundaries*`)를 **절대 호출하지 않는다**(AC-AIF-015) — 역산은
`training/boundaries_cli.py`가 반기 재검토 시점에 1회 실행해 JSON을
갱신하는 별도 경로다.

REQ-AIF-111: 밴드 스윕이 사용할 이원 경계(PROMOTE/DEMOTE) 두 세트는 기본
경계에 마진 δ(`grade_margin_delta`, 저장소에 함께 저장)를 적용해 파생한다
(TECHSPEC §6.6 4단계 공식). δ는 착수 시점 잠정값(인접 경계 최소 간격의
10%)이며 NOTIFIER-FILTER-001과의 합동 결정값으로 **JSON 값만 교체**하면
전환된다 — 이 모듈의 구조는 그 교체에 영향받지 않는다.

`classify_by_boundaries()`(`training/boundaries.py`, PRESERVE)를 그대로
재사용하며 `np.digitize` 분류를 재구현하지 않는다(plan.md §D).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from importlib import resources
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from analyzer.training.boundaries import GRADE_ORDER, classify_by_boundaries

GRADE_BOUNDARIES_RESOURCE = "grade_boundaries.json"
"""패키지(`analyzer.inference`) 동봉 경계값 저장소 파일명(REQ-AIF-081)."""

BOUNDARY_KEYS: tuple[str, ...] = tuple(
    f"{GRADE_ORDER[i]}_{GRADE_ORDER[i + 1]}" for i in range(len(GRADE_ORDER) - 1)
)
"""`infer_grade_boundaries()`가 산출하는 4개 경계 키(인접 등급 쌍, 오름차순)."""

_HOLD_INDEX = GRADE_ORDER.index("HOLD")

BOUNDARY_DIRECTIONS: tuple[int, ...] = tuple(
    -1 if i < _HOLD_INDEX else +1 for i in range(len(BOUNDARY_KEYS))
)
"""각 경계의 '방향'(REQ-AIF-111 공식의 δ·방향 항): HOLD 아래쪽 경계
(STRONG_SELL_SELL, SELL_HOLD)는 −1, HOLD 위쪽 경계(HOLD_BUY, BUY_STRONG_BUY)는
+1. 방향은 "HOLD에서 멀어지는 쪽"이며 PROMOTE 세트는 극단 등급으로의 승급을
어렵게(경계를 HOLD 반대편으로), DEMOTE 세트는 강등을 어렵게(경계를 HOLD
쪽으로) 이동시킨다 — 두 세트 사이 폭 2δ의 데드존이 히스테리시스 전부다
(TECHSPEC §6.6 소비 규칙)."""

DEFAULT_GRADE_MARGIN_DELTA_RATIO = 0.10
"""δ 잠정값 유도 비율(plan.md M4): 4개 조합 전체의 인접 경계 최소 간격 × 10%."""

_EXPECTED_COMBINATIONS: frozenset[tuple[str, int]] = frozenset(
    (market, horizon) for market in ("domestic", "overseas") for horizon in (20, 60)
)


class BoundarySet(StrEnum):
    """등급 분류에 사용할 경계 세트(REQ-AIF-111, `signal_price_bands.boundary_set`)."""

    BASE = "BASE"
    PROMOTE = "PROMOTE"
    DEMOTE = "DEMOTE"


def _ensure_monotonic(boundaries: Mapping[str, float], *, context: str) -> None:
    values = [boundaries[key] for key in BOUNDARY_KEYS]
    if any(b <= a for a, b in pairwise(values)):
        raise ValueError(f"{context}: 경계값이 단조 증가하지 않는다 ({values})")


def shift_boundaries(
    boundaries: Mapping[str, float],
    *,
    delta: float,
    boundary_set: BoundarySet,
) -> dict[str, float]:
    """기본 경계에 마진 δ를 방향 인식으로 적용해 PROMOTE/DEMOTE 경계를 파생한다(REQ-AIF-111).

    - PROMOTE = 기본 경계 + δ·방향 (승급 경계 — HOLD에서 멀어짐)
    - DEMOTE  = 기본 경계 − δ·방향 (강등 경계 — HOLD 쪽으로 접근)
    - BASE    = 기본 경계 그대로

    `boundaries`의 키는 `BOUNDARY_KEYS` 4개여야 하며, 시프트 결과가 단조
    증가를 깨면(δ가 인접 경계 간격의 절반 이상) `ValueError`를 던진다 —
    δ 잠정값(최소 간격의 10%)에서는 발생하지 않는다.
    """
    if delta < 0:
        raise ValueError(f"delta는 0 이상이어야 한다 (입력값: {delta})")
    sign = {BoundarySet.BASE: 0, BoundarySet.PROMOTE: +1, BoundarySet.DEMOTE: -1}[boundary_set]
    shifted = {
        key: float(boundaries[key]) + sign * delta * direction
        for key, direction in zip(BOUNDARY_KEYS, BOUNDARY_DIRECTIONS, strict=True)
    }
    _ensure_monotonic(shifted, context=f"{boundary_set.value} 경계(delta={delta})")
    return shifted


def derive_grade_margin_delta(
    boundaries_by_combination: Mapping[tuple[str, int], Mapping[str, float]],
    *,
    ratio: float = DEFAULT_GRADE_MARGIN_DELTA_RATIO,
) -> tuple[float, float]:
    """모든 조합·인접 경계 쌍의 최소 간격에 `ratio`를 곱해 δ 잠정값을 유도한다(plan.md M4).

    반환값은 `(delta, min_adjacent_gap)` — 저장소 메타데이터에 유도 근거로
    함께 기록된다. 경계표가 먼저 확정돼야 호출 가능한 순서 의존 서브 Task다.
    """
    gaps: list[float] = []
    for boundaries in boundaries_by_combination.values():
        values = [float(v) for v in boundaries.values()]
        gaps.extend(b - a for a, b in pairwise(values))
    if not gaps:
        raise ValueError("δ를 유도할 조합이 없다")
    min_gap = min(gaps)
    return ratio * min_gap, min_gap


def classify_grades(
    scores: Sequence[float] | np.ndarray | pd.Series,
    boundaries: Mapping[str, float],
) -> np.ndarray:
    """앙상블 score를 등급 5클래스로 이산화한다 — `classify_by_boundaries()` 위임(REQ-AIF-081)."""
    return classify_by_boundaries(scores, boundaries)


# @MX:ANCHOR: [AUTO] 추론 경계값 저장소 계약 — M5+(추론 배치)와 M6(밴드 스윕)이
# 등급 산출 시 이 아티팩트를 공유 소비한다.
# @MX:REASON: REQ-AIF-081(기동 시 재계산 금지)·REQ-AIF-111(PROMOTE/DEMOTE 파생)의
# 단일 진입점 — 필드 추가는 허용되나 `classify()`/`boundaries_for()` 시그니처와
# JSON 스키마(`schema_version`)는 하위 소비자와 함께 바꿔야 한다.
@dataclass(frozen=True, slots=True)
class GradeBoundariesArtifact:
    """저장소 JSON을 파싱한 결과 — 4개 조합 경계표 + δ + 생성 메타데이터."""

    boundaries_by_combination: dict[tuple[str, int], dict[str, float]]
    grade_margin_delta: float
    grade_margin_delta_provisional: bool
    target_ratios: dict[str, float]
    data_as_of: date
    generated_at: str
    row_counts: dict[tuple[str, int], int]

    def boundaries_for(
        self, market: str, horizon: int, boundary_set: BoundarySet = BoundarySet.BASE
    ) -> dict[str, float]:
        """(시장, horizon) 조합의 경계를 요청한 세트로 파생해 반환한다."""
        base = self.boundaries_by_combination[(market, horizon)]
        return shift_boundaries(base, delta=self.grade_margin_delta, boundary_set=boundary_set)

    def classify(
        self,
        market: str,
        horizon: int,
        scores: Sequence[float] | np.ndarray | pd.Series,
        boundary_set: BoundarySet = BoundarySet.BASE,
    ) -> np.ndarray:
        """(시장, horizon) 조합·경계 세트로 score를 등급으로 분류한다."""
        return classify_grades(scores, self.boundaries_for(market, horizon, boundary_set))


def default_artifact_path() -> Path:
    """패키지 동봉 저장소 파일 경로(`analyzer/inference/grade_boundaries.json`)."""
    return Path(str(resources.files("analyzer.inference").joinpath(GRADE_BOUNDARIES_RESOURCE)))


def load_grade_boundaries(path: Path | None = None) -> GradeBoundariesArtifact:
    """저장소 JSON을 읽어 검증한다(REQ-AIF-081, AC-AIF-015).

    분위수 역산은 수행하지 않는다 — 파일에 기록된 값을 읽고 구조(4개 조합,
    4개 단조 증가 경계, 양수 δ)만 검증한다. `path`가 `None`이면 패키지 동봉
    파일을 사용한다.
    """
    artifact_path = default_artifact_path() if path is None else path
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    boundaries_by_combination: dict[tuple[str, int], dict[str, float]] = {}
    row_counts: dict[tuple[str, int], int] = {}
    for entry in payload["combinations"]:
        combination = (str(entry["market"]), int(entry["horizon"]))
        raw = entry["boundaries"]
        missing = [key for key in BOUNDARY_KEYS if key not in raw]
        if missing:
            raise ValueError(f"{combination}: 경계 키 누락 {missing}")
        boundaries = {key: float(raw[key]) for key in BOUNDARY_KEYS}
        _ensure_monotonic(boundaries, context=str(combination))
        boundaries_by_combination[combination] = boundaries
        row_counts[combination] = int(entry["row_count"])

    if set(boundaries_by_combination) != _EXPECTED_COMBINATIONS:
        raise ValueError(
            f"조합 집합 불일치: 기대 {sorted(_EXPECTED_COMBINATIONS)}, "
            f"실제 {sorted(boundaries_by_combination)}"
        )

    delta_entry = payload["grade_margin_delta"]
    delta = float(delta_entry["value"])
    if delta <= 0:
        raise ValueError(f"grade_margin_delta는 양수여야 한다 (입력값: {delta})")

    return GradeBoundariesArtifact(
        boundaries_by_combination=boundaries_by_combination,
        grade_margin_delta=delta,
        grade_margin_delta_provisional=bool(delta_entry.get("provisional", True)),
        target_ratios={str(k): float(v) for k, v in payload["target_ratios"].items()},
        data_as_of=date.fromisoformat(payload["data_as_of"]),
        generated_at=str(payload["generated_at"]),
        row_counts=row_counts,
    )

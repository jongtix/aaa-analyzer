"""등급 경계 분위수 역산 (SPEC-ANALYZER-TRAIN-001 M4).

REQ-AT-040: 시장(국내/해외) × horizon(D20/D60) 4개 조합 각각에 대해,
조립된 실현 수익률 분포(`dataset.assemble_dataset()` 산출물)에서 목표
등급 비율 분위수를 역산해 STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL 5클래스
경계값을 산출한다(TECHSPEC §6.1 절차의 실제 수행자, LABEL-001 REQ-AL-080
이 명시적으로 이관).

REQ-AT-041: 목표 등급 비율의 구체적 수치와 산출된 경계값 숫자는 이
SPEC(spec.md 본문)에 확정치로 기록하지 않는다 — 이 모듈은 어떤 목표
비율도 하드코딩하지 않으며, 호출자가 `target_ratios` 인자로 항상 명시
주입한다. 산출물(경계값 자체)은 TECHSPEC §6.1 개정안이라는 별도
문서/커밋으로 다뤄지며, 이 코드 모듈 자체에 담기지 않는다.

REQ-AT-042: 경계값은 재학습을 요구하지 않는 사후 이산화 파라미터다
(ADR-033) — 이 모듈은 모델 학습(§2.6)과 독립적이며 어느 학습 함수도
호출하지 않는다.
"""

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

GRADE_ORDER: tuple[str, ...] = ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")
"""등급 5클래스를 실현 수익률 오름차순으로 나열한 순서(REQ-AT-040)."""


def infer_grade_boundaries(
    returns: Sequence[float] | np.ndarray | pd.Series,
    target_ratios: Mapping[str, float],
) -> dict[str, float]:
    """실현 수익률 분포에서 목표 등급 비율에 대응하는 경계값을 역산한다(REQ-AT-040).

    `target_ratios`는 `GRADE_ORDER`의 5개 클래스 각각에 대한 목표 비율
    (합계 1.0, 각 값은 0보다 커야 함)이다 — 구체적 수치는 호출자가 주입하며
    이 함수는 어떤 값도 하드코딩하지 않는다(REQ-AT-041).

    반환값은 4개의 단조 증가(오름차순) 경계값이다 — 5개 클래스를 나누는
    분할점은 수학적으로 4개뿐이다. 키는 인접 등급 쌍(예: `"HOLD_BUY"` =
    HOLD와 BUY 사이 경계)이며 `GRADE_ORDER` 순서를 그대로 따른다.
    """
    ratios = [target_ratios[grade] for grade in GRADE_ORDER]
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("target_ratios의 합은 1.0이어야 한다")
    if any(r <= 0 for r in ratios):
        raise ValueError("target_ratios의 각 값은 0보다 커야 한다")

    arr = np.asarray(returns, dtype=float)
    cumulative_probs = np.cumsum(ratios)[:-1]
    boundary_values = np.quantile(arr, cumulative_probs)

    boundary_keys = [f"{GRADE_ORDER[i]}_{GRADE_ORDER[i + 1]}" for i in range(len(GRADE_ORDER) - 1)]
    return dict(zip(boundary_keys, (float(v) for v in boundary_values), strict=True))


def classify_by_boundaries(
    returns: Sequence[float] | np.ndarray | pd.Series,
    boundaries: Mapping[str, float],
) -> np.ndarray:
    """`infer_grade_boundaries()`가 산출한 경계값으로 수익률을 5클래스로 분류한다.

    검증(목표 비율과의 대조) 및 후속 소비(등급 산출) 양쪽에서 재사용되는
    보조 함수다 — `boundaries`의 4개 값을 오름차순 경계로 사용해
    `np.digitize`로 분류한다.
    """
    arr = np.asarray(returns, dtype=float)
    boundary_values = np.array(list(boundaries.values()), dtype=float)
    bin_indices = np.digitize(arr, boundary_values)
    return np.asarray(GRADE_ORDER)[bin_indices]


def infer_grade_boundaries_all_combinations[K: Hashable](
    returns_by_combination: Mapping[K, Sequence[float] | np.ndarray | pd.Series],
    target_ratios: Mapping[str, float],
) -> dict[K, dict[str, float]]:
    """시장×horizon 4개 조합 각각에 대해 독립적으로 경계값을 역산한다(REQ-AT-040).

    각 조합(예: `("domestic", 20)`)은 자신의 실현 수익률 분포만을 사용해
    독립적으로 분위수 역산을 수행한다 — 조합 간 분포를 합치거나 공유하지
    않는다.
    """
    return {
        combination: infer_grade_boundaries(returns, target_ratios)
        for combination, returns in returns_by_combination.items()
    }

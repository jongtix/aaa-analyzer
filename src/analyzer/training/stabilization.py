"""롤링 집계 안정화 게이트 + 챔피언 선정 (SPEC-ANALYZER-TRAIN-EVAL-001 M5, design.md §2C).

REQ-ATE-039: 안정화 판정은 어떤 경우에도 단일 폴드의 원값에 의존해서는
안 된다 — 반드시 롤링 집계(트레일링 윈도우에 대한 평균 또는 ICIR)에
기반해야 한다. 이 모듈의 모든 게이트 함수는 폴드별 Rank IC 시계열
전체(`Sequence[float]`)를 입력받아 내부에서 트레일링 윈도우를 슬라이싱한
뒤에만 집계값을 pass/fail 조건에 사용한다 — 단일 스칼라를 직접
pass/fail에 사용하는 경로는 존재하지 않는다(코드 리뷰 가드, REQ-ATE-039).

REQ-ATE-040/044: (시장,horizon,algorithm) 조합 단위로 GATE-1/2/3을 모두
적용해 독립적으로 안정화 여부를 판정한다 — 캠페인 전체를 단일 판정으로
묶지 않는다.

REQ-ATE-045/046/047(F1): (시장,horizon) 조합의 스코어링 전략 자격은
조합 단위 안정화 판정을 (m,h) 수준으로 집계하는 규칙이다 — LightGBM
단독/XGBoost 단독/앙상블 세 전략의 자격 조건이 서로 다르며, 앙상블은
LightGBM과 XGBoost 둘 다 안정화된 경우에만 자격을 얻는다(부분 안정화
시 단독 전략만 자격을 얻고 앙상블은 자격을 얻지 못한다 — F1 핵심
수정 사항). 자격 있는 전략이 하나도 없으면 그 (m,h) 조합은 배포
전면 금지(REQ-ATE-047)이며, 실패 원인 진단 정보를 반환한다.

이 모듈은 순수 함수만 제공한다(design.md line 28) — DataFrame/배열
입력과 판정 결과 출력만 다루며, 파일 I/O를 전혀 수행하지 않는다.
활성화 매니페스트 갱신(파일 쓰기)은 M6(`activation.py`)의 책임이다.
"""

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

CAMPAIGN_GATE1_ROLLING_WEEKS: int = 12
"""GATE-1 트레일링 롤링 평균 Rank IC 윈도우 길이(주, REQ-ATE-041)."""

CAMPAIGN_GATE3_ROLLING_WEEKS: int = 52
"""GATE-3 트레일링 롤링 ICIR 윈도우 길이(주, REQ-ATE-043)."""

CAMPAIGN_GATE3_ICIR_FLOOR: float = 0.10
"""GATE-3 롤링 ICIR 임계값(REQ-ATE-043) — Qlib 벤치마크는 참고 맥락일 뿐
직접 합격선 동등 비교로 취급하지 않는다."""

_GATE1_NAME = "GATE-1"
_GATE2_NAME = "GATE-2"
_GATE3_NAME = "GATE-3"


def _rolling_mean(values: Sequence[float], window: int) -> float | None:
    """values의 마지막 `window`개에 대한 트레일링 평균 — 관측치가 window
    미만이면 아직 롤링 윈도우가 충족되지 않은 것으로 보고 None을 반환한다
    (REQ-ATE-039의 롤링 집계 원칙 — 미충족 구간은 단일 폴드 값으로
    대체 계산하지 않는다)."""
    if len(values) < window:
        return None
    tail = values[-window:]
    return statistics.fmean(tail)


def _rolling_icir(values: Sequence[float], window: int) -> float | None:
    """values의 마지막 `window`개에 대한 트레일링 ICIR(평균 ÷ 표준편차).

    관측치가 window 미만이면 None. 표준편차가 0이면(윈도우 내 값 전부
    동일) ICIR을 0.0으로 취급한다(0으로 나누기 회피, campaign_metrics.py
    `compute_aggregate_metrics`와 동일한 관례).
    """
    if len(values) < window:
        return None
    tail = values[-window:]
    mean = statistics.fmean(tail)
    stddev = statistics.pstdev(tail) if len(tail) > 1 else 0.0
    return mean / stddev if stddev else 0.0


def gate1_passed(
    rank_ic_values: Sequence[float], *, rolling_weeks: int = CAMPAIGN_GATE1_ROLLING_WEEKS
) -> bool:
    """GATE-1(REQ-ATE-041): 트레일링 롤링 평균 Rank IC > 0.

    윈도우가 아직 충족되지 않은 초반 구간(관측치 < rolling_weeks)은
    실패로 판정한다(집계 불가 상태를 통과로 취급하지 않는다).
    """
    rolling_mean = _rolling_mean(rank_ic_values, rolling_weeks)
    return rolling_mean is not None and rolling_mean > 0


def gate2_passed(rank_ic_values: Sequence[float]) -> bool:
    """GATE-2(REQ-ATE-042): 캠페인 전체 폴드 평균 Rank IC > 0 — 롤링
    윈도우가 아닌 캠페인 전체 기간을 대상으로 한다."""
    if not rank_ic_values:
        return False
    return statistics.fmean(rank_ic_values) > 0


def gate3_passed(
    rank_ic_values: Sequence[float],
    *,
    rolling_weeks: int = CAMPAIGN_GATE3_ROLLING_WEEKS,
    icir_floor: float = CAMPAIGN_GATE3_ICIR_FLOOR,
) -> bool:
    """GATE-3(REQ-ATE-043): 트레일링 롤링 ICIR > icir_floor."""
    rolling_icir = _rolling_icir(rank_ic_values, rolling_weeks)
    return rolling_icir is not None and rolling_icir > icir_floor


@dataclass(frozen=True, slots=True)
class ComboStabilizationVerdict:
    """(시장,horizon,algorithm) 조합 1개의 안정화 판정 결과(REQ-ATE-040/044).

    관측 롤링/전체 집계 지표값을 함께 보존해 진단 정보(REQ-ATE-047)로
    재사용한다.
    """

    market: str
    horizon: int
    algorithm: str
    stabilized: bool
    gate1_passed: bool
    gate2_passed: bool
    gate3_passed: bool
    gate1_rolling_mean_rank_ic: float | None
    gate2_mean_rank_ic: float | None
    gate3_rolling_icir: float | None


def evaluate_combo_stabilization(
    market: str, horizon: int, algorithm: str, rank_ic_values: Sequence[float]
) -> ComboStabilizationVerdict:
    """(시장,horizon,algorithm) 조합에 GATE-1/2/3을 모두 적용해 안정화
    여부를 판정한다(REQ-ATE-040/044) — 세 게이트 중 하나라도 실패하면
    안정화되지 않음으로 판정한다."""
    g1 = gate1_passed(rank_ic_values)
    g2 = gate2_passed(rank_ic_values)
    g3 = gate3_passed(rank_ic_values)
    return ComboStabilizationVerdict(
        market=market,
        horizon=horizon,
        algorithm=algorithm,
        stabilized=g1 and g2 and g3,
        gate1_passed=g1,
        gate2_passed=g2,
        gate3_passed=g3,
        gate1_rolling_mean_rank_ic=_rolling_mean(rank_ic_values, CAMPAIGN_GATE1_ROLLING_WEEKS),
        gate2_mean_rank_ic=statistics.fmean(rank_ic_values) if rank_ic_values else None,
        gate3_rolling_icir=_rolling_icir(rank_ic_values, CAMPAIGN_GATE3_ROLLING_WEEKS),
    )


def _combo_failure_diagnostics(verdict: ComboStabilizationVerdict) -> dict[str, Any]:
    """조합 1개의 게이트 실패 진단 정보(REQ-ATE-047) — 어느 게이트에서
    실패했는지 + 관측된 롤링/전체 집계 지표값."""
    failed_gates: list[str] = []
    if not verdict.gate1_passed:
        failed_gates.append(_GATE1_NAME)
    if not verdict.gate2_passed:
        failed_gates.append(_GATE2_NAME)
    if not verdict.gate3_passed:
        failed_gates.append(_GATE3_NAME)
    return {
        "algorithm": verdict.algorithm,
        "stabilized": verdict.stabilized,
        "failed_gates": tuple(failed_gates),
        "gate1_rolling_mean_rank_ic": verdict.gate1_rolling_mean_rank_ic,
        "gate2_mean_rank_ic": verdict.gate2_mean_rank_ic,
        "gate3_rolling_icir": verdict.gate3_rolling_icir,
    }


@dataclass(frozen=True, slots=True)
class ChampionSelection:
    """(시장,horizon) 조합의 챔피언 스코어링 전략 선정 결과(REQ-ATE-045/046/047).

    - `champion_algorithm`: 챔피언으로 선정된 전략(`"lightgbm"` /
      `"xgboost"` / `"ensemble"`) — 자격 있는 전략이 하나도 없으면 None.
    - `eligible_strategies`: 자격을 얻은 전략 전체(챔피언 포함, 1~3개).
    - `excluded_artifact_algorithms`: 챔피언이 단독 전략이고 반대편
      알고리즘이 미안정화 상태일 때 활성화 매니페스트 대상에서 제외되는
      알고리즘(REQ-ATE-046) — 챔피언이 앙상블이거나 반대편도 안정화된
      경우 비어 있다.
    - `deployment_prohibited`: 자격 있는 전략이 하나도 없어 배포가 전면
      금지되는 경우(REQ-ATE-047) True.
    - `diagnostics`: `deployment_prohibited`일 때만 채워지는 실패 원인
      진단 정보 — `{"lightgbm": {...}, "xgboost": {...}}`.
    """

    market: str
    horizon: int
    champion_algorithm: str | None
    eligible_strategies: tuple[str, ...]
    excluded_artifact_algorithms: tuple[str, ...]
    deployment_prohibited: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def select_champion_strategy(
    market: str,
    horizon: int,
    lightgbm_verdict: ComboStabilizationVerdict,
    xgboost_verdict: ComboStabilizationVerdict,
    strategy_rolling_mean_rank_ic: Mapping[str, float],
) -> ChampionSelection:
    """(시장,horizon) 조합의 챔피언 스코어링 전략을 선정한다(REQ-ATE-045).

    `strategy_rolling_mean_rank_ic`는 GATE-1과 동일한 롤링 평균 Rank IC
    정의로 산출된 전략별(`"lightgbm"`/`"xgboost"`/`"ensemble"`) 지표값
    딕셔너리다 — 앙상블 값은 앙상블 자체의 롤링 평균 Rank IC이며(호출자가
    campaign_metrics.py의 앙상블 의사조합 JSONL 시계열로부터 `_rolling_mean()`
    을 적용해 산출), 두 알고리즘 지표의 단순 평균이 아니다.

    자격 규칙(REQ-ATE-045, F1 핵심 수정): LightGBM 단독 전략은 LightGBM이
    안정화된 경우에만, XGBoost 단독 전략은 XGBoost가 안정화된 경우에만,
    앙상블 전략은 LightGBM과 XGBoost 둘 다 안정화된 경우에만 각각 자격을
    얻는다 — 부분 안정화(하나만 안정화)에서는 앙상블이 자격을 얻지 않는다.
    """
    lgbm_ok = lightgbm_verdict.stabilized
    xgb_ok = xgboost_verdict.stabilized

    eligible: list[str] = []
    if lgbm_ok:
        eligible.append("lightgbm")
    if xgb_ok:
        eligible.append("xgboost")
    if lgbm_ok and xgb_ok:
        eligible.append("ensemble")

    if not eligible:
        return ChampionSelection(
            market=market,
            horizon=horizon,
            champion_algorithm=None,
            eligible_strategies=(),
            excluded_artifact_algorithms=(),
            deployment_prohibited=True,
            diagnostics={
                "lightgbm": _combo_failure_diagnostics(lightgbm_verdict),
                "xgboost": _combo_failure_diagnostics(xgboost_verdict),
            },
        )

    champion = max(eligible, key=lambda strategy: strategy_rolling_mean_rank_ic[strategy])

    excluded: tuple[str, ...] = ()
    if champion == "lightgbm" and not xgb_ok:
        excluded = ("xgboost",)
    elif champion == "xgboost" and not lgbm_ok:
        excluded = ("lightgbm",)

    return ChampionSelection(
        market=market,
        horizon=horizon,
        champion_algorithm=champion,
        eligible_strategies=tuple(eligible),
        excluded_artifact_algorithms=excluded,
        deployment_prohibited=False,
        diagnostics={},
    )

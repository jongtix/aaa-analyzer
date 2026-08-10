"""초기 백테스트 지표 산출 (SPEC-ANALYZER-TRAIN-001 M7).

REQ-AT-120: 7개 지표를 계산한다 — Hit Rate(방향 적중률), Pearson IC(score-
실현수익률 피어슨 상관), Rank IC(score-실현수익률 스피어만 순위상관,
TECHSPEC §6.4가 명시하는 회귀 신호 평가의 1차 축 — ADR-033), Precision
by direction(방향성 예측 중 실제 방향이 맞은 비율), Sharpe ratio, Max
Drawdown, confidence 캘리브레이션 정확도(예측 confidence 구간별 실측
방향 적중률 대조). Pearson IC/Rank IC를 별도 필드로 분리하는 것은 업계
표준 관행(Qlib이 `ic`/`rank_ic`를 별도 컬럼으로 산출)과 일치하며, 두 값의
괴리 자체가 진단 신호다(Pearson만 높으면 소수 이상치가 상관을 견인 중,
Rank만 높으면 비선형 관계가 존재).

REQ-AT-121: 회귀 R²이 구조적으로 낮게 나오는 것은 결함이 아니라 인지된
한계로 취급한다(ADR-033 — IC/Rank IC가 실질적 평가축) — 이 모듈은 R²을
지표로 계산하지 않는다(의도적 누락, 위 6개 지표만 산출).
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

_N_CALIBRATION_BINS = 5
"""confidence 캘리브레이션 계산 시 사용하는 quantile bin 개수."""


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """AC-AT-013이 요구하는 7개 백테스트 지표(REQ-AT-120)."""

    hit_rate: float
    pearson_ic: float
    rank_ic: float
    precision: float
    sharpe_ratio: float
    max_drawdown: float
    confidence_calibration: float


def _hit_rate(scores: np.ndarray, returns: np.ndarray) -> float:
    """예측 방향(sign(score))과 실현 방향(sign(return))이 일치하는 비율."""
    return float(np.mean(np.sign(scores) == np.sign(returns)))


def _pearson_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    """score와 실현수익률의 피어슨 상관(REQ-AT-120 "Pearson IC")."""
    # scipy 1.18 `PearsonRResult`의 pyright stub이 필드를 제네릭 `_T_co`로만
    # 선언해 float() 변환 인자 타입이 좁혀지지 않는다(런타임은 정상 float64).
    return float(stats.pearsonr(scores, returns)[0])  # pyright: ignore[reportArgumentType]


def _rank_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    """score와 실현수익률의 스피어만 순위상관(REQ-AT-120 "Rank IC")."""
    correlation, _pvalue = stats.spearmanr(scores, returns)
    return float(correlation)


def _precision(scores: np.ndarray, returns: np.ndarray) -> float:
    """방향성 예측(score != 0)만 대상으로, 실제 방향이 맞은 비율(TP/(TP+FP))."""
    directional = scores != 0
    if not np.any(directional):
        return float("nan")
    matches = np.sign(scores[directional]) == np.sign(returns[directional])
    return float(np.mean(matches))


def _strategy_returns(scores: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """score 부호를 방향으로 삼는 단순 모델 기반 전략의 기간별 수익률."""
    return np.sign(scores) * returns


def _sharpe_ratio(strategy_returns: np.ndarray) -> float:
    """전략 수익률의 Sharpe ratio(무위험 이자율 0 가정, 연율화 없음)."""
    std = np.std(strategy_returns)
    if std == 0:
        return 0.0
    return float(np.mean(strategy_returns) / std)


def _max_drawdown(strategy_returns: np.ndarray) -> float:
    """전략 누적 수익 곡선의 최대 낙폭(음수 없는 양수 비율로 반환)."""
    equity_curve = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return float(abs(np.min(drawdown)))


def _confidence_calibration(
    scores: np.ndarray, returns: np.ndarray, confidences: np.ndarray
) -> float:
    """confidence 구간별 실측 방향 적중률과 평균 confidence의 상관관계.

    `confidences`를 `_N_CALIBRATION_BINS`개 quantile bin으로 나누고,
    bin별 평균 confidence와 bin별 hit rate 사이의 피어슨 상관계수를
    반환한다. 값이 1에 가까울수록 confidence가 높을수록 실제 적중률도
    높다는 뜻(좋은 캘리브레이션)이다. bin이 1개뿐이거나(모든 confidence
    값이 동일) 상관계수를 계산할 수 없으면 `0.0`을 반환한다(방향 정보
    없음과 동일하게 취급 — NaN을 반환하지 않는다, AC-AT-013).
    """
    n = len(confidences)
    n_bins = min(_N_CALIBRATION_BINS, n)
    if n_bins < 2:
        return 0.0

    order = np.argsort(confidences)
    bin_edges = np.array_split(order, n_bins)

    bin_mean_confidence: list[float] = []
    bin_hit_rate: list[float] = []
    for indices in bin_edges:
        if len(indices) == 0:
            continue
        bin_mean_confidence.append(float(np.mean(confidences[indices])))
        bin_hit_rate.append(_hit_rate(scores[indices], returns[indices]))

    if len(bin_mean_confidence) < 2 or np.std(bin_mean_confidence) == 0:
        return 0.0

    correlation = np.corrcoef(bin_mean_confidence, bin_hit_rate)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


def compute_backtest_metrics(
    scores: np.ndarray,
    returns: np.ndarray,
    confidences: np.ndarray,
) -> BacktestMetrics:
    """7개 백테스트 지표를 계산한다(REQ-AT-120, AC-AT-013).

    `scores`/`returns`/`confidences`는 동일 길이의 병렬 배열이다 —
    각 인덱스가 동일 (종목, 거래일) 예측·실현·confidence 삼중항을
    나타낸다. 7개 지표 모두 NaN 없이 반환한다(AC-AT-013).
    """
    strategy_returns = _strategy_returns(scores, returns)

    return BacktestMetrics(
        hit_rate=_hit_rate(scores, returns),
        pearson_ic=_pearson_ic(scores, returns),
        rank_ic=_rank_ic(scores, returns),
        precision=_precision(scores, returns),
        sharpe_ratio=_sharpe_ratio(strategy_returns),
        max_drawdown=_max_drawdown(strategy_returns),
        confidence_calibration=_confidence_calibration(scores, returns, confidences),
    )

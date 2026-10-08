"""밴드 스윕 + signal_price_bands INSERT + 학습 잡 레이스 방어
(SPEC-ANALYZER-INFER-001 M6/M7, REQ-AIF-110/111/060, design.md §5/§7).

가격 그리드(국내 ±30%/0.5%, 해외 ±21.5%/0.5% — 둘 다 M7 실측으로 확정,
`OVERSEAS_GRID_RANGE_PCT`/`MERGE_THRESHOLD_PCT` 참조)로 가상 종가를
구성해, PRICE_DERIVED 피처만 그리드별로 재계산하고 FROZEN 피처는 실제
마지막 행의 값으로 동결한다(TECHSPEC:1192 "나머지 조건은 전일과 동일") —
`inference/features.py`가 이미 산출하는 실제 조립 결과를 그대로 재사용해
수급 피처를 그리드마다 다시 계산하지 않는다.

booster는 그리드 크기만큼 반복 로드하지 않고 `predict.
predict_point_models_batch()`로 한 번만 로드한다. score/등급 파생은
`resolution.compute_score_columns()`/`boundaries_store.classify_grades()`
(둘 다 PRESERVE 대상 순수 함수 위임)를 그대로 재사용하며 별도 변형을
만들지 않는다(REQ-AIF-111, plan.md §D).

1% 조각 병합 뒤 살아남은 내부 경계는 격자 칸 안에서 이분 탐색으로 정련해,
인접 밴드가 경계 가격 하나를 공유하는 연속 파티션으로 저장한다
(SPEC-ANALYZER-INFER-002 REQ-AIR-001~008). 정련 평가는 격자 평가와 같은
피처 조립·예측·score 파생 경로를 쓴다.

격자·정련 평가(마지막 모델 로드)를 모두 마친 뒤 `resolution.
detect_manifest_race()`로 학습 잡과의 레이스를 재확인한다(design.md §7,
REQ-AIR-007) — 레이스가 감지되면 이번 스윕 결과 전체를 폐기하고
`SkipReason.MANIFEST_RACE`로 라우팅한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from analyzer.data.adjustment import adjust_prices
from analyzer.data.models import TradingCalendar
from analyzer.data.repository import fetch_corporate_events, fetch_daily_ohlcv, fetch_investor_trend
from analyzer.features.classification import FeatureClass, classify_feature
from analyzer.features.supply_demand import compute_supply_demand_features
from analyzer.features.technical import compute_technical_features
from analyzer.inference.boundaries_store import (
    BoundarySet,
    GradeBoundariesArtifact,
    classify_grades,
)
from analyzer.inference.features import LOOKBACK_TRADING_DAYS, resolve_feature_columns
from analyzer.inference.predict import predict_point_models_batch
from analyzer.inference.resolution import (
    ServingPlan,
    SkipReason,
    compute_score_columns,
    detect_manifest_race,
)
from analyzer.inference.writer import PriceBandRow, insert_signal_price_bands

DOMESTIC_GRID_RANGE_PCT = 0.30
"""REQ-AIF-110 확정값 — 국내 가격제한폭(상한가/하한가) ±30%(TECHSPEC:1174)."""

OVERSEAS_GRID_RANGE_PCT = 0.215
"""REQ-AIF-110 확정값(M7 실측, 2026-09-12) — 해외는 가격제한폭이 없어
과거 일중 등락 분포 실측으로 범위를 확정한다(TECHSPEC:1175).

방법: NAS 프로덕션 DB `daily_ohlcv` 해외 유니버스(`stocks.market IN
('NYSE','NASDAQ','AMEX') AND asset_type='STOCK'`, 76종목, 278,064행,
2007-08-20~2026-09-10) 전체에 대해 `LAG(close_price) OVER (PARTITION BY
stock_id ORDER BY trade_date)`로 종목별 일간 등락률을 계산했다.
`corporate_events`의 확정 SPLIT 이벤트(18건) 전후 ±3일 구간을 제외해도
p99.9 값은 21.39% → 21.25%로만 소폭 변동해 SPLIT 아티팩트가 이 분위수에
실질적 영향을 주지 않음을 확인했다(=해외 유니버스 최댓값 187.1%,
SERV 2024-07-19은 SPLIT이 아닌 실제 급등).

설계 의도(§ 밴드 스윕 알고리즘 docstring, plan.md M7)는 99.9% 과거
커버리지를 목표로 하므로, SPLIT 제외 p99.9(21.25%)를 `GRID_STEP_PCT`(0.5%)
단위로 올림해 커버리지 보장을 유지했다: 21.25% → **21.5%**."""

GRID_STEP_PCT = 0.005
"""REQ-AIF-110 확정값 — 그리드 간격 0.5%(TECHSPEC:1176)."""

MERGE_THRESHOLD_PCT = 0.01
"""REQ-AIF-111 확정값(M7 실측, 2026-09-12) — 조각 병합 폭 임계 1%
(TECHSPEC:1180). 착수 시점 초안값과 동일하게 확정됐다(측정이 초안을
검증한 사례 — 값 자체는 변경 없음).

방법: NAS 프로덕션 DB의 실제 챔피언 모델(domestic D60 xgboost, overseas
D20 xgboost)로 표본 종목(각 12종목)에 대해 실제 밴드 스윕을 실행하고,
후보 임계값 {0.5%, 1%, 1.5%, 2%, 3%}마다 두 지표를 측정했다: (a) 병합
전(run-length) 원시 밴드 중 "슬리버"(그리드 1칸 폭, `GRID_STEP_PCT` ≈
0.5%인 노이즈 조각)가 병합되는 비율, (b) 슬리버가 아닌 "정상" 전환
밴드(2칸 이상 폭)가 잘못 병합되는 비율.

domestic D60(표본 12종목, 원시 평균 10.08밴드): 0.5%에서는 슬리버
24/41(59%)만 병합돼 불충분, **1%에서 슬리버 41/41(100%) 병합되면서
정상 밴드 병합은 0/80(0%)** — 슬리버를 전부 흡수하는 가장 작은 임계값이자
정상 전환을 훼손하지 않는 경계. 1.5%부터 정상 밴드 병합이 시작된다
(13/80, 16%), 2%는 18/80(22.5%), 3%는 25/80(31%) — 임계값을 더 키울수록
실제 등급 전환 정보가 점점 더 손실된다. overseas D20(표본 12종목)은 원시
밴드 수 자체가 매우 적어(평균 1.33밴드) 임계값 민감도를 독립적으로
검증하기엔 정보량이 부족했으나 domestic 결론과 모순되는 신호는 없었다.
overseas D60은 별개로 발견된 프로덕션 결함(챔피언 모델에 `.meta.json`
사이드카 부재 → FEATURE_REGISTRY 전체 폴백 → 해외 수급 데이터 부재와
충돌, `.moai/specs/SPEC-ANALYZER-INFER-001/progress.md` M7 항목 참조)으로
인해 실측 불가 — Gap으로 보고."""

REFINE_MAX_ITERATIONS = 8
"""REQ-AIR-005 확정값(SPEC-ANALYZER-INFER-002 plan.md PD-2) — 경계 하나당 이분
탐색 최대 반복 수. 격자 칸이 전일 종가의 0.5%로 고정이므로 최종 탐색 구간
폭은 0.5%/2⁸ ≈ 전일 종가의 0.00195%다."""

PRICE_DECIMALS = 4
"""정련 중간 가격 양자화 자릿수 — `signal_price_bands.price_low/price_high`
`DECIMAL(18,4)`와 같아, 평가한 가격이 DB 반올림 뒤에도 그대로 저장된다."""

_GRID_RANGE_PCT_BY_MARKET: dict[str, float] = {
    "domestic": DOMESTIC_GRID_RANGE_PCT,
    "overseas": OVERSEAS_GRID_RANGE_PCT,
}


def build_price_grid(prev_close: float, market: str) -> np.ndarray:
    """`prev_close` 대비 대칭 그리드를 생성한다(design.md §5, AC-AIF-018).

    정수 스텝 개수로 오프셋을 산출해(부동소수점 누적 오차 방지) 그리드
    양끝이 `prev_close * (1 ± 범위)`와 정확히 일치하도록 한다. `market`이
    `_GRID_RANGE_PCT_BY_MARKET`에 없으면 `KeyError`를 던진다.
    """
    range_pct = _GRID_RANGE_PCT_BY_MARKET[market]
    n_steps = round(range_pct / GRID_STEP_PCT)
    offsets = np.arange(-n_steps, n_steps + 1) * GRID_STEP_PCT
    return prev_close * (1.0 + offsets)


@dataclass(frozen=True, slots=True)
class PriceBand:
    """등급 분류 결과를 병합한 가격 구간 하나(design.md §5, AC-AIF-018/019)."""

    price_low: float
    price_high: float
    signal_class: str


def _run_length_bands(grid: np.ndarray, grades: np.ndarray) -> list[PriceBand]:
    bands: list[PriceBand] = []
    start = 0
    n = len(grades)
    for i in range(1, n + 1):
        if i == n or grades[i] != grades[start]:
            bands.append(
                PriceBand(
                    price_low=float(grid[start]),
                    price_high=float(grid[i - 1]),
                    signal_class=str(grades[start]),
                )
            )
            start = i
    return bands


def merge_adjacent_bands(
    grid: np.ndarray | Sequence[float],
    grades: np.ndarray | Sequence[str],
    *,
    prev_close: float,
    threshold: float = MERGE_THRESHOLD_PCT,
) -> list[PriceBand]:
    """연속 동일 등급 그리드 포인트를 하나의 밴드로 접고(run-length), 폭이
    `prev_close` 대비 `threshold` 미만인 조각은 인접 밴드에 병합한다
    (REQ-AIF-111, TECHSPEC:1180). 병합 대상 조각은 뒤쪽 이웃에 흡수시키되,
    조각이 마지막 밴드이면 앞쪽 이웃에 흡수시킨다(순서 무관 나머지는
    TECHSPEC에 명시되지 않은 자유도 — 이 함수의 규칙으로 확정).
    """
    grid = np.asarray(grid, dtype=float)
    grades = np.asarray(grades)
    bands = _run_length_bands(grid, grades)

    changed = True
    while changed and len(bands) > 1:
        changed = False
        for i, band in enumerate(bands):
            width_pct = (band.price_high - band.price_low) / prev_close
            if width_pct < threshold:
                if i + 1 < len(bands):
                    neighbor = bands[i + 1]
                    merged = PriceBand(
                        price_low=band.price_low,
                        price_high=neighbor.price_high,
                        signal_class=neighbor.signal_class,
                    )
                    bands = bands[:i] + [merged] + bands[i + 2 :]
                else:
                    neighbor = bands[i - 1]
                    merged = PriceBand(
                        price_low=neighbor.price_low,
                        price_high=band.price_high,
                        signal_class=neighbor.signal_class,
                    )
                    bands = bands[: i - 1] + [merged]
                changed = True
                break
    return bands


def _price_derived_columns(feature_columns: Sequence[str]) -> list[str]:
    return [c for c in feature_columns if classify_feature(c) == FeatureClass.PRICE_DERIVED]


def _frozen_columns(feature_columns: Sequence[str]) -> list[str]:
    return [c for c in feature_columns if classify_feature(c) == FeatureClass.FROZEN]


def _freeze_last_row_at_price(adjusted: pd.DataFrame, price: float) -> pd.DataFrame:
    """`adjusted`의 마지막 행 `close_price`만 `price`로 치환한 사본을
    반환한다 — open/high/low/volume은 그대로 유지된다(TECHSPEC:1192
    "가상 종가로 간주하고 나머지 조건은 전일과 동일")."""
    virtual = adjusted.copy()
    virtual.iloc[-1, virtual.columns.get_loc("close_price")] = price
    return virtual


@dataclass(frozen=True, slots=True, eq=False)
class SweepContext:
    """(종목, 시장) 1건의 밴드 스윕 평가 입력 — 수정 주가 이력과 동결(FROZEN)
    피처 값을 한 번만 준비해 격자 평가와 정련 평가가 함께 쓴다(REQ-AIR-006)."""

    grid: np.ndarray
    prev_close: float
    adjusted: pd.DataFrame
    feature_columns: tuple[str, ...]
    price_derived_columns: tuple[str, ...]
    frozen_values: Mapping[str, object]


def prepare_sweep_context(
    engine: Engine,
    calendar: TradingCalendar,
    stock_code: str,
    as_of_date: date,
    market: str,
    feature_columns: Sequence[str],
) -> SweepContext | SkipReason:
    """밴드 스윕 평가 입력을 준비한다(design.md §5, AC-AIF-018).

    FROZEN 컬럼(수급 피처)은 실제 마지막 행 값을 1회만 계산해 둔다. 이력이
    `LOOKBACK_TRADING_DAYS` 미만이거나 FROZEN 컬럼이 필요한데 수급 데이터가
    없으면 `SkipReason.FEATURE_INSUFFICIENT`를 반환한다(예외를 던지지 않는다,
    AC-AIF-011).
    """
    raw = fetch_daily_ohlcv(engine, stock_code, end_date=as_of_date)
    if len(raw) < LOOKBACK_TRADING_DAYS:
        return SkipReason.FEATURE_INSUFFICIENT

    events = fetch_corporate_events(engine, stock_code)
    adjusted = adjust_prices(raw, events, as_of_date=as_of_date, calendar=calendar)
    prev_close = float(adjusted["close_price"].iloc[-1])
    grid = build_price_grid(prev_close, market)

    price_derived_cols = _price_derived_columns(feature_columns)
    frozen_cols = _frozen_columns(feature_columns)

    frozen_values: dict[str, object] = {}
    if frozen_cols:
        trend = fetch_investor_trend(engine, stock_code)
        if trend.empty:
            return SkipReason.FEATURE_INSUFFICIENT
        supply_demand = compute_supply_demand_features(trend)
        real_features = compute_technical_features(adjusted)
        new_columns = [c for c in supply_demand.columns if c not in real_features.columns]
        merged = real_features.merge(
            supply_demand[["trade_date", *new_columns]], on="trade_date", how="left"
        )
        frozen_row = merged.tail(1).reset_index(drop=True).iloc[0]
        missing = [c for c in frozen_cols if c not in frozen_row.index]
        if missing:
            raise ValueError(f"밴드 스윕에 필요한 FROZEN 피처 컬럼이 누락되었다: {missing}")
        frozen_values = {col: frozen_row[col] for col in frozen_cols}

    return SweepContext(
        grid=grid,
        prev_close=prev_close,
        adjusted=adjusted,
        feature_columns=tuple(feature_columns),
        price_derived_columns=tuple(price_derived_cols),
        frozen_values=frozen_values,
    )


def feature_matrix_at_prices(context: SweepContext, prices: np.ndarray) -> pd.DataFrame:
    """가격마다 PRICE_DERIVED 컬럼을 `compute_technical_features()` 재호출로
    재계산하고(전체 이력을 다시 통과시켜야 롤링 윈도가 올바르게 갱신된다),
    FROZEN 컬럼은 동결 값으로 채운 피처 행렬을 만든다 — 격자 가격과 정련
    가격이 같은 경로를 쓴다(REQ-AIR-006). 컬럼 순서는 `feature_columns`
    그대로다(예측 시 `predict.py`의 컬럼 선택과 정합)."""
    price_derived_cols = list(context.price_derived_columns)
    price_derived_rows: list[pd.Series] = []
    for price in prices:
        virtual = _freeze_last_row_at_price(context.adjusted, float(price))
        recomputed = compute_technical_features(virtual)
        last = recomputed.tail(1).reset_index(drop=True).iloc[0]
        missing = [c for c in price_derived_cols if c not in last.index]
        if missing:
            raise ValueError(f"밴드 스윕에 필요한 PRICE_DERIVED 피처 컬럼이 누락되었다: {missing}")
        price_derived_rows.append(last[price_derived_cols])

    matrix = pd.DataFrame(price_derived_rows).reset_index(drop=True)
    for col, value in context.frozen_values.items():
        matrix[col] = value
    return matrix.loc[:, list(context.feature_columns)]


def assemble_sweep_feature_matrix(
    engine: Engine,
    calendar: TradingCalendar,
    stock_code: str,
    as_of_date: date,
    market: str,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, pd.DataFrame, float] | SkipReason:
    """그리드 전체에 대한 피처 행렬을 조립한다(design.md §5, AC-AIF-018).

    반환값은 `(grid, matrix, prev_close)` — 스킵 조건은
    `prepare_sweep_context()`, 행 조립 규칙은 `feature_matrix_at_prices()`와 같다.
    """
    context = prepare_sweep_context(
        engine, calendar, stock_code, as_of_date, market, feature_columns
    )
    if isinstance(context, SkipReason):
        return context
    return context.grid, feature_matrix_at_prices(context, context.grid), context.prev_close


def score_prices(serving_plan: ServingPlan, feature_matrix: pd.DataFrame) -> np.ndarray:
    """피처 행렬 전체를 booster 1회 로드로 예측하고 행마다 score를 파생한다
    (`compute_score_columns()` 위임 — 격자·정련 평가 공용, REQ-AIR-006)."""
    predictions_by_algo = predict_point_models_batch(serving_plan, feature_matrix)
    return np.array(
        [
            compute_score_columns(
                serving_plan.active_strategy,
                {algo: float(values[i]) for algo, values in predictions_by_algo.items()},
            ).score
            for i in range(len(feature_matrix))
        ],
        dtype=float,
    )


@dataclass(slots=True)
class _Bracket:
    """내부 경계 하나의 이분 탐색 구간 — `low`는 앞 밴드 등급이 유지되는 가격,
    `high`는 앞 밴드 등급이 아닌 가격(REQ-AIR-003)."""

    prior_class: str
    low: float
    high: float


def _refine_internal_boundaries(
    merged_by_set: Mapping[BoundarySet, list[PriceBand]],
    boundaries_by_set: Mapping[BoundarySet, Mapping[str, float]],
    evaluate: Callable[[np.ndarray], np.ndarray],
) -> dict[BoundarySet, list[float]]:
    # @MX:NOTE: [AUTO] 판정 조건은 "앞 밴드 등급이 유지되는가"(plan.md PD-5) — 뒤 밴드 첫
    # 격자점은 병합된 조각일 수 있어 "뒤 밴드 등급인가"로 판정하면 안 된다. 모든 세트·경계를
    # 반복 단위로 묶어 반복당 평가 1회(같은 가격은 1번만)로 모델 로드를 제한한다(PD-3).
    brackets = {
        boundary_set: [
            _Bracket(prior_class=prior.signal_class, low=prior.price_high, high=nxt.price_low)
            for prior, nxt in pairwise(bands)
        ]
        for boundary_set, bands in merged_by_set.items()
    }
    searching = [
        (boundary_set, bracket) for boundary_set, items in brackets.items() for bracket in items
    ]
    for _ in range(REFINE_MAX_ITERATIONS):
        pending: list[tuple[BoundarySet, _Bracket, float]] = []
        for boundary_set, bracket in searching:
            mid = round((bracket.low + bracket.high) / 2, PRICE_DECIMALS)
            # 양자화한 중간 가격이 구간 끝과 같으면 더 좁힐 수 없다 — 이 경계의 탐색 종료.
            if mid not in (bracket.low, bracket.high):
                pending.append((boundary_set, bracket, mid))
        if not pending:
            break
        searching = [(boundary_set, bracket) for boundary_set, bracket, _ in pending]

        prices = np.unique(np.array([mid for _, _, mid in pending], dtype=float))
        scores = np.asarray(evaluate(prices), dtype=float)
        grade_at = {
            boundary_set: dict(
                zip(prices.tolist(), classify_grades(scores, boundaries).tolist(), strict=True)
            )
            for boundary_set, boundaries in boundaries_by_set.items()
        }
        for boundary_set, bracket, mid in pending:
            if grade_at[boundary_set][mid] == bracket.prior_class:
                bracket.low = mid
            else:
                bracket.high = mid
    return {
        boundary_set: [bracket.high for bracket in items]
        for boundary_set, items in brackets.items()
    }


# @MX:ANCHOR: [AUTO] 밴드 파티션 산출 계약 — 격자 등급 → run-length → 1% 조각 병합 → 내부 경계
# 정련(PD-1) → 인접 밴드가 경계를 공유하는 연속 파티션(REQ-AIR-001).
# @MX:REASON: notifier `BandPartition.lookup`의 반개구간 해석(REQ-AIR-002)과 T4 재생 검증이
# 이 출력 형태에 의존한다 — 밴드 개수·등급 순서를 바꾸는 변경은 AC-AIR-005 회귀다.
def sweep_price_partitions(
    grid: np.ndarray,
    boundaries_by_set: Mapping[BoundarySet, Mapping[str, float]],
    *,
    prev_close: float,
    evaluate: Callable[[np.ndarray], np.ndarray],
) -> dict[BoundarySet, list[PriceBand]]:
    """경계 세트별 연속 가격 파티션을 산출한다(REQ-AIR-001~005·008).

    `evaluate`는 가격 배열을 받아 같은 길이의 score 배열을 돌려주는 함수다 —
    격자 평가에 한 번, 정련 반복마다 한 번 호출된다(최대 1 +
    `REFINE_MAX_ITERATIONS`회). 각 내부 경계는 `(앞 밴드 마지막 격자점, 뒤 밴드
    첫 격자점]` 안에서 이분 탐색하며, 저장 경계는 최종 구간의 위 끝(앞 밴드
    등급이 아니라고 평가된 가격)이다. 밴드 개수와 등급 순서는 병합 결과
    그대로다. 마지막 밴드를 제외한 밴드는 `[price_low, price_high)`, 마지막
    밴드는 `[price_low, price_high]`로 읽는다.
    """
    grid = np.asarray(grid, dtype=float)
    grid_scores = np.asarray(evaluate(grid), dtype=float)
    merged_by_set = {
        boundary_set: merge_adjacent_bands(
            grid, classify_grades(grid_scores, boundaries), prev_close=prev_close
        )
        for boundary_set, boundaries in boundaries_by_set.items()
    }
    refined = _refine_internal_boundaries(merged_by_set, boundaries_by_set, evaluate)

    partitions: dict[BoundarySet, list[PriceBand]] = {}
    for boundary_set, bands in merged_by_set.items():
        edges = [bands[0].price_low, *refined[boundary_set], bands[-1].price_high]
        partitions[boundary_set] = [
            PriceBand(price_low=low, price_high=high, signal_class=band.signal_class)
            for band, (low, high) in zip(bands, pairwise(edges), strict=True)
        ]
    return partitions


def _union_feature_columns(serving_plan: ServingPlan) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for model_path in serving_plan.model_paths.values():
        for col in resolve_feature_columns(model_path):
            if col not in seen:
                seen.add(col)
                columns.append(col)
    return columns


def sweep_and_write_price_bands(
    *,
    engine: Engine,
    calendar: TradingCalendar,
    boundaries_artifact: GradeBoundariesArtifact,
    serving_plan: ServingPlan,
    models_root: Path,
    stock_id: int,
    stock_code: str,
    market: str,
    horizon: int,
    trade_date: date,
    model_version: str,
) -> dict[str, str] | SkipReason:
    """(종목, 시장, horizon) 1건에 대해 밴드 스윕을 실행하고
    `signal_price_bands`에 PROMOTE/DEMOTE 두 파티션을 기록한다
    (design.md §5, REQ-AIF-110/111/060).

    피처 조립 실패(이력 부족)는 `SkipReason.FEATURE_INSUFFICIENT`로,
    격자·정련 평가를 모두 마친 뒤 감지된 학습 잡 레이스는
    `SkipReason.MANIFEST_RACE`로 라우팅한다(design.md §7, REQ-AIR-007) — 두
    경우 모두 `signal_price_bands`에 어떤 행도 기록하지 않는다. 성공 시
    boundary_set별 INSERT 결과를 담은 매핑(`{"PROMOTE": ..., "DEMOTE": ...}`)을
    반환한다.
    """
    feature_columns = _union_feature_columns(serving_plan)
    context = prepare_sweep_context(
        engine, calendar, stock_code, trade_date, market, feature_columns
    )
    if isinstance(context, SkipReason):
        return context

    def evaluate(prices: np.ndarray) -> np.ndarray:
        return score_prices(serving_plan, feature_matrix_at_prices(context, prices))

    boundaries_by_set = {
        boundary_set: boundaries_artifact.boundaries_for(market, horizon, boundary_set)
        for boundary_set in (BoundarySet.PROMOTE, BoundarySet.DEMOTE)
    }
    partitions = sweep_price_partitions(
        context.grid, boundaries_by_set, prev_close=context.prev_close, evaluate=evaluate
    )

    # design.md §7 / REQ-AIR-007: 마지막 모델 로드(정련 평가 포함) 뒤에 매니페스트를 재확인한다.
    if detect_manifest_race(models_root, serving_plan):
        return SkipReason.MANIFEST_RACE

    outcomes: dict[str, str] = {}
    for boundary_set, bands in partitions.items():
        rows = [
            PriceBandRow(
                stock_id=stock_id,
                trade_date=trade_date,
                horizon=horizon,
                boundary_set=boundary_set.value,
                band_seq=seq,
                price_low=band.price_low,
                price_high=band.price_high,
                signal_class=band.signal_class,
                model_version=model_version,
            )
            for seq, band in enumerate(bands)
        ]
        outcomes[boundary_set.value] = insert_signal_price_bands(engine, rows)
    return outcomes

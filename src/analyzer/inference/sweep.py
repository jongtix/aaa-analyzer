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

모델을 실제로 로드(=예측)한 직후 `resolution.detect_manifest_race()`로
학습 잡과의 레이스를 재확인한다(design.md §7) — 레이스가 감지되면 이번
스윕 결과 전체를 폐기하고 `SkipReason.MANIFEST_RACE`로 라우팅한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
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


def assemble_sweep_feature_matrix(
    engine: Engine,
    calendar: TradingCalendar,
    stock_code: str,
    as_of_date: date,
    market: str,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, pd.DataFrame, float] | SkipReason:
    """그리드 전체에 대한 피처 행렬을 조립한다(design.md §5, AC-AIF-018).

    PRICE_DERIVED 컬럼은 그리드 가격마다 `compute_technical_features()`를
    재호출해 재계산하고(전체 이력을 다시 통과시켜야 롤링 윈도가 올바르게
    갱신된다), FROZEN 컬럼(수급 피처)은 실제 마지막 행 값을 1회만 계산해
    그리드 전체에 동일하게 채운다. 이력이 `LOOKBACK_TRADING_DAYS` 미만이거나
    FROZEN 컬럼이 필요한데 수급 데이터가 없으면 `SkipReason.
    FEATURE_INSUFFICIENT`를 반환한다(예외를 던지지 않는다, AC-AIF-011).

    반환값은 `(grid, matrix, prev_close)` — `matrix`의 컬럼 순서는
    `feature_columns` 그대로다(예측 시 `predict.py`의 컬럼 선택과 정합).
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

    price_derived_rows: list[pd.Series] = []
    for price in grid:
        virtual = _freeze_last_row_at_price(adjusted, float(price))
        recomputed = compute_technical_features(virtual)
        last = recomputed.tail(1).reset_index(drop=True).iloc[0]
        missing = [c for c in price_derived_cols if c not in last.index]
        if missing:
            raise ValueError(f"밴드 스윕에 필요한 PRICE_DERIVED 피처 컬럼이 누락되었다: {missing}")
        price_derived_rows.append(last[price_derived_cols])

    matrix = pd.DataFrame(price_derived_rows).reset_index(drop=True)
    for col, value in frozen_values.items():
        matrix[col] = value
    matrix = matrix.loc[:, list(feature_columns)]
    return grid, matrix, prev_close


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
    모델 로드 직후 감지된 학습 잡 레이스는 `SkipReason.MANIFEST_RACE`로
    라우팅한다(design.md §7) — 두 경우 모두 `signal_price_bands`에 어떤
    행도 기록하지 않는다. 성공 시 boundary_set별 INSERT 결과를 담은
    매핑(`{"PROMOTE": ..., "DEMOTE": ...}`)을 반환한다.
    """
    feature_columns = _union_feature_columns(serving_plan)
    result = assemble_sweep_feature_matrix(
        engine, calendar, stock_code, trade_date, market, feature_columns
    )
    if isinstance(result, SkipReason):
        return result
    grid, matrix, prev_close = result

    predictions_by_algo = predict_point_models_batch(serving_plan, matrix)

    # design.md §7: 모델을 실제로 로드(=예측)한 직후 매니페스트를 재확인한다.
    if detect_manifest_race(models_root, serving_plan):
        return SkipReason.MANIFEST_RACE

    scores = np.array(
        [
            compute_score_columns(
                serving_plan.active_strategy,
                {algo: float(values[i]) for algo, values in predictions_by_algo.items()},
            ).score
            for i in range(len(grid))
        ]
    )

    outcomes: dict[str, str] = {}
    for boundary_set in (BoundarySet.PROMOTE, BoundarySet.DEMOTE):
        shifted = boundaries_artifact.boundaries_for(market, horizon, boundary_set)
        grades = classify_grades(scores, shifted)
        bands = merge_adjacent_bands(grid, grades, prev_close=prev_close)
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

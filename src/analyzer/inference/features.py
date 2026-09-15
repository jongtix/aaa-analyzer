"""61거래일 룩백 피처 조립기 + feature_columns 해석 (SPEC-ANALYZER-INFER-001
M5, REQ-AIF-070/071, design.md §1).

`training/dataset.py`(TRAIN-001)와 동일한 원칙으로 `compute_technical_
features()`/`compute_supply_demand_features()`(FEATURE-001)를 무수정
재사용한다 — 레이블 계산(`compute_labels()`)은 추론에 불필요하므로
호출하지 않는다는 점만 다르다. 종목별 원주가 전체 이력(unbounded fetch)을
`adjust_prices()`/`compute_technical_features()`에 그대로 통과시켜 마지막
1행만 반환한다 — 학습 시점과 동일한 방식으로 계산해야 롤링 윈도(ROC_60 등)
값이 train/inference 사이에 어긋나지 않는다(윈도를 사전에 잘라내면 같은
결과가 나오지만, 전체 이력을 그대로 넘기는 쪽이 `dataset.assemble_dataset()`
과 코드 경로를 동일하게 유지해 추후 유지보수 시 두 경로가 갈라질 위험을
줄인다).

`analyzer.data.split`/`analyzer.data.dividend_adjustment`를 `adjustment.py`와
함께 명시적으로 임포트해 SPLIT/DIVIDEND 핸들러 레지스트리가 비어있지 않음을
보증한다(REQ-AIF-070, TRAIN-001 REQ-AT-021과 동일한 조용한 실패 함정 방어 —
AC-AIF-012).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

import analyzer.data.dividend_adjustment  # noqa: F401 — HANDLER_REGISTRY 등록 부수효과
import analyzer.data.split  # noqa: F401 — HANDLER_REGISTRY 등록 부수효과
from analyzer.data.adjustment import HANDLER_REGISTRY, adjust_prices
from analyzer.data.models import TradingCalendar
from analyzer.data.repository import (
    fetch_corporate_events,
    fetch_daily_ohlcv,
    fetch_investor_trend,
)
from analyzer.features.classification import FEATURE_REGISTRY, FeatureClass, classify_feature
from analyzer.features.supply_demand import compute_supply_demand_features
from analyzer.features.technical import compute_technical_features
from analyzer.inference.resolution import SkipReason
from analyzer.training.campaign_metrics import sidecar_path_for

# REQ-AIF-070 명시적 가드(AC-AIF-012): 위 import만으로도 SPLIT/DIVIDEND
# 핸들러가 등록되지만, "동등한 레지스트리 비어있음 방지 assertion"을 이
# 모듈 로드 시점에 직접 단언해 향후 import 순서 변경 등으로 가드가 조용히
# 무력화되는 것을 방지한다(fail-fast, TRAIN-001 dataset.py와 동일 패턴).
assert "SPLIT" in HANDLER_REGISTRY and "DIVIDEND" in HANDLER_REGISTRY, (
    "SPLIT/DIVIDEND 핸들러 레지스트리가 비어 있다 — "
    "analyzer.data.split/dividend_adjustment 임포트를 확인하라(REQ-AIF-070)"
)

LOOKBACK_TRADING_DAYS: int = 61
"""REQ-AIF-070: 60일 롤링 윈도(ROC_60/MA_60/STD_60/RANK_60/CORR_60)가 마지막
행에서 non-NaN이 되려면 `close_price.shift(60)`이 접근하는 인덱스가 존재해야
하므로 최소 61거래일의 원주가 이력이 필요하다. 이 미만이면 그 종목은
`FEATURE_INSUFFICIENT`로 스킵된다(REQ-AIF-060 전반부, AC-AIF-011)."""


def assemble_inference_features(
    engine: Engine,
    calendar: TradingCalendar,
    stock_code: str,
    as_of_date: date,
) -> pd.DataFrame | None:
    """`stock_code`의 `as_of_date` 기준 최신 거래일 1행에 대한 피처를 조립한다.

    이력이 `LOOKBACK_TRADING_DAYS` 미만이면 `None`을 반환한다(REQ-AIF-060
    feature_insufficient 스킵 경로 — 호출자가 `SkipReason.FEATURE_INSUFFICIENT`로
    라우팅한다, `assemble_inference_features_batch` 참조). 예외를 던지지
    않는다(AC-AIF-011).
    """
    raw = fetch_daily_ohlcv(engine, stock_code, end_date=as_of_date)
    if len(raw) < LOOKBACK_TRADING_DAYS:
        return None

    events = fetch_corporate_events(engine, stock_code)
    adjusted = adjust_prices(raw, events, as_of_date=as_of_date, calendar=calendar)
    features = compute_technical_features(adjusted)

    trend = fetch_investor_trend(engine, stock_code)
    if not trend.empty:
        supply_demand = compute_supply_demand_features(trend)
        new_columns = [c for c in supply_demand.columns if c not in trend.columns]
        features = features.merge(
            supply_demand[["trade_date", *new_columns]], on="trade_date", how="left"
        )

    return features.tail(1).reset_index(drop=True)


def assemble_inference_features_batch(
    engine: Engine,
    calendar: TradingCalendar,
    stock_codes: Sequence[str],
    as_of_date: date,
    *,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame | SkipReason]:
    """복수 종목에 대해 피처를 배치로 조립한다(REQ-AIF-060 전반부, AC-AIF-011).

    개별 종목의 이력 부족(< `LOOKBACK_TRADING_DAYS`)은 그 종목만
    `SkipReason.FEATURE_INSUFFICIENT`로 표시하고, 나머지 종목은 정상
    처리를 계속한다 — 한 종목의 실패가 (시장,horizon) 조합 전체를
    abort시키지 않는다.

    SPEC-ANALYZER-TRAIN-META-001 M7(REQ-TM-011) — `feature_columns`가
    명시적으로 제공되면, 각 종목의 `investor_trend`를 먼저 조회해
    `has_supply_demand_gap()`(M2)으로 조기 판별한다. 갭이 있으면(FROZEN
    수급 컬럼 요구 + investor_trend 결측) 실제 피처 조립을 시도하지 않고
    그 종목의 매핑 값을 `SkipReason.FEATURE_INSUFFICIENT`로 기록한다 —
    `predict.py::_select_feature_columns()`의 무방비 `ValueError` 호출부에
    도달하기 전에 스킵한다(`inference/sweep.py`의 기존 조기 스킵과 동일한
    방어 수준, AC-TM-010). `feature_columns`가 생략되면(기본값 `None`) 이
    가드는 완전히 비활성화되고 기존 동작을 그대로 유지한다 — 기존 호출자는
    이 milestone으로 영향받지 않는다(REQ-TM-006 방향의 회귀 최소화 원칙과
    동일 취지, AC-TM-010b).
    """
    results: dict[str, pd.DataFrame | SkipReason] = {}
    for stock_code in stock_codes:
        if feature_columns is not None:
            trend = fetch_investor_trend(engine, stock_code)
            if has_supply_demand_gap(feature_columns, trend):
                results[stock_code] = SkipReason.FEATURE_INSUFFICIENT
                continue
        features = assemble_inference_features(engine, calendar, stock_code, as_of_date)
        results[stock_code] = features if features is not None else SkipReason.FEATURE_INSUFFICIENT
    return results


def has_supply_demand_gap(feature_columns: Sequence[str], investor_trend: pd.DataFrame) -> bool:
    """단일종목 스코어링 경로 방어 심층화(SPEC-ANALYZER-TRAIN-META-001 M2,
    REQ-TM-007) — `feature_columns`(예: `resolve_feature_columns()`가 해석한
    목록)에 FROZEN 수급 피처가 포함되어 있는데 `investor_trend`가 비어
    있으면 `True`를 반환한다.

    `inference/sweep.py::assemble_sweep_feature_matrix()`의 기존 조기 스킵
    조건(FROZEN 컬럼 필요 + `trend.empty`)과 동일한 판별 로직이다 —
    호출자는 `True`일 때 `SkipReason.FEATURE_INSUFFICIENT`로 라우팅해
    `predict.py::_select_feature_columns()`의 무방비 `ValueError` 호출부에
    도달하기 전에 스킵해야 한다(research.md §3.4의 비대칭 해소).

    이 함수 자체는 스킵하지 않는다 — 판별만 하고 라우팅은 호출자 책임이다.
    `assemble_inference_features()`의 기존 반환 계약(`DataFrame | None`,
    INFER-001 M5)은 이 함수의 도입으로 변경되지 않는다(REQ-TM-006 방향의
    회귀 최소화 원칙과 동일 취지). `predict.py::_select_feature_columns()`
    의 기존 `ValueError` 데이터 무결성 가드는 이 함수와 무관하게 무수정
    유지된다(REQ-TM-008) — investor_trend 결측과 무관한 진짜 컬럼 누락은
    여전히 그 가드가 처리한다.
    """
    frozen_required = any(classify_feature(c) == FeatureClass.FROZEN for c in feature_columns)
    return frozen_required and investor_trend.empty


def resolve_feature_columns(model_path: Path) -> list[str]:
    """(REQ-AIF-071, AC-AIF-013) `model_path`의 `.meta.json` 사이드카에서
    `feature_columns`를 읽는다.

    사이드카 경로는 `campaign_metrics.sidecar_path_for()`(TRAIN-EVAL-001)와
    동일한 명명 관례(`model_path.suffix + ".meta.json"`)를 그대로 재사용한다
    — 이 관례는 캠페인 배포 챔피언에만 기록되므로(research.md §1), 상시
    게이트로 승격된 챔피언 등 사이드카가 없는 조합에서는 `FEATURE_REGISTRY`
    (40개 전체, FEATURE-001)로 폴백한다 — 예외로 실패하지 않는다.
    """
    sidecar_path = sidecar_path_for(model_path)
    if not sidecar_path.is_file():
        return list(FEATURE_REGISTRY)

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return list(payload["feature_columns"])

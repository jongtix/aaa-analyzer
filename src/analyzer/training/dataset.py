"""FEATURE-001 피처 + LABEL-001 레이블을 종목·거래일 기준으로 조인한 학습용 데이터셋 조립.

SPEC-ANALYZER-TRAIN-001 M1(REQ-AT-020/022/023/025): `assemble_dataset()`이
이 모듈의 유일한 공개 함수이며, 이 계약(입력·출력 스키마)은 이후 마일스톤
(M2 Parquet 캐시, M3 Walk-Forward 분할, M4 등급 경계 역산 등)이 그대로
소비한다(plan.md §F M1 — 가역성 높은 결정을 먼저 확정).

`compute_technical_features()`/`compute_supply_demand_features()`/
`compute_labels()`(모두 순수 함수, FEATURE-001/LABEL-001 소관)를 그대로
재사용하며 재구현하지 않는다. DB 조회(trainer 엔진 배선, M2 캐시)는 이
함수의 책임이 아니다 — 호출자가 이미 조회한 종목별 DataFrame을 인자로
전달한다(`compute_labels()`의 순수 함수 설계 원칙을 그대로 계승).
"""

from collections.abc import Mapping

import pandas as pd

# 핸들러 등록(HANDLER_REGISTRY 배선) 부수효과를 위한 임포트 — 이 모듈이
# `adjust_prices()`를 직접 호출하므로, `adjustment.py` 단독 임포트만으로는
# SPLIT/DIVIDEND 핸들러가 등록되지 않는 함정(REQ-AT-021, labels/core.py의
# 동일 패턴 계승)이 이 조립 경로에서도 재발하지 않도록 방어한다.
import analyzer.data.dividend_adjustment  # noqa: F401
import analyzer.data.split  # noqa: F401
from analyzer.data.adjustment import adjust_prices
from analyzer.data.models import TradingCalendar
from analyzer.features.supply_demand import compute_supply_demand_features
from analyzer.features.technical import compute_technical_features
from analyzer.labels.config import DEFAULT_START_DATES
from analyzer.labels.core import compute_labels

_EMPTY_EVENTS_COLUMNS = [
    "event_type",
    "event_date",
    "stock_rate",
    "cash_amount",
    "event_subtype",
    "ex_dividend_date",
    "currency_code",
]


def assemble_dataset(
    stocks: pd.DataFrame,
    ohlcv_by_stock: Mapping[str, pd.DataFrame],
    events_by_stock: Mapping[str, pd.DataFrame],
    investor_trend_by_stock: Mapping[str, pd.DataFrame],
    calendar: TradingCalendar,
    market: str,
) -> pd.DataFrame:
    """종목 유니버스·시장·데이터 범위를 입력받아 학습용 단일 DataFrame을 조립한다(REQ-AT-020).

    입력:
    - `stocks`: 후보 유니버스. `stock_code`/`grade`/`delisted_at` 컬럼을
      가져야 한다(`delisted_at`이 `None`/`NaT`가 아니면 상장폐지로 취급,
      REQ-AT-025).
    - `ohlcv_by_stock`/`events_by_stock`/`investor_trend_by_stock`: 종목코드
      → `repository.fetch_daily_ohlcv`/`fetch_corporate_events`/
      `fetch_investor_trend`가 반환하는 것과 동일한 스키마의 DataFrame.
      `investor_trend_by_stock`에 종목코드가 없으면(또는 빈 DataFrame이면)
      그 종목의 수급 피처는 생략된다(해외 수급 결측 등 알려진 한계,
      plan.md §H aaa-infra#141).
    - `calendar`: 시장의 `TradingCalendar`.
    - `market`: `DEFAULT_START_DATES`의 키(`"domestic"`/`"overseas"`) — 학습
      유니버스 시작일 필터(REQ-AT-023)에 사용된다.

    출력: 종목·거래일당 1행, `compute_technical_features()`/
    `compute_supply_demand_features()` 피처 열(+ 원본 `stock_code`/`trade_date`
    등 raw 열) + `label_D20`/`label_D60`/`label_D20_exclude_reason`/
    `label_D60_exclude_reason` 열을 포함한 단일 DataFrame.

    유니버스 필터: `grade in ('A', 'B')`(REQ-AT-022) 종목만, 그 종목의
    `trade_date >= DEFAULT_START_DATES[market]`(REQ-AT-023) 이후 행만
    포함한다. 유니버스에 없거나(등급 불일치) OHLCV 데이터가 없는 종목,
    필터 후 남는 행이 없는 종목은 결과에서 제외된다.
    """
    start_date = DEFAULT_START_DATES[market]
    universe = stocks.loc[stocks["grade"].isin(["A", "B"])]

    assembled: list[pd.DataFrame] = []
    for row in universe.itertuples():
        stock_code = row.stock_code
        raw = ohlcv_by_stock.get(stock_code)
        if raw is None or raw.empty:
            continue

        raw = raw.loc[raw["trade_date"] >= start_date]
        if raw.empty:
            continue

        events = events_by_stock.get(stock_code)
        if events is None:
            events = pd.DataFrame(columns=_EMPTY_EVENTS_COLUMNS)

        is_delisted = bool(pd.notna(row.delisted_at))

        labeled = compute_labels(raw, events, calendar, is_delisted=is_delisted)

        as_of_date = raw["trade_date"].max()
        adjusted = adjust_prices(raw, events, as_of_date=as_of_date, calendar=calendar)
        features = compute_technical_features(adjusted)

        trend = investor_trend_by_stock.get(stock_code)
        if trend is not None and not trend.empty:
            supply_demand = compute_supply_demand_features(trend)
            new_columns = [c for c in supply_demand.columns if c not in trend.columns]
            features = features.merge(
                supply_demand[["trade_date", *new_columns]], on="trade_date", how="left"
            )

        label_columns = [c for c in labeled.columns if c not in raw.columns]
        merged = features.merge(
            labeled[["trade_date", *label_columns]], on="trade_date", how="left"
        )
        assembled.append(merged)

    if not assembled:
        return pd.DataFrame()

    return pd.concat(assembled, ignore_index=True)

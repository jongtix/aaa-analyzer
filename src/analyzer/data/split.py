"""SPLIT 이벤트 조정 핸들러 — 배율(stock_rate) 기반 가격/거래량 소급 조정.

SPEC-ANALYZER-DATA-001 REQ-AD-020/021/022: `HANDLER_REGISTRY`에 `"SPLIT"`으로
등록되는 조정 핸들러다(REQ-AD-041 확장 지점을 통해 배선 — `adjustment.py`의
디스패치 로직 자체는 건드리지 않는다). 정배분할(`stock_rate >= 1`)과
병합/역병합(`stock_rate < 1`) 모두 **동일한 공식**을 방향과 무관하게 적용한다
(REQ-AD-022) — `price_adjusted = price_raw / stock_rate`,
`volume_adjusted = volume_raw * stock_rate`.

한 종목에 이벤트가 여러 건 존재하면(REQ-AD-021), 각 행(`trade_date`)보다
이후에 발생한(`event_date > trade_date`) 이벤트들의 `stock_rate`를 모두
누적곱해 그 행에 적용한다 — 곱셈은 교환법칙이 성립하므로 이벤트 순회 순서와
무관하게 결과가 동일하다.

as-of 컷오프(REQ-AD-040)는 이 핸들러 스스로 책임진다: `event_date`가
`as_of_date` 이하(포함)인 이벤트만 반영 대상으로 필터링한다.
"""

from datetime import date

import pandas as pd

from analyzer.data.adjustment import register_handler
from analyzer.data.models import TradingCalendar

_PRICE_COLUMN_SUFFIX = "_price"


@register_handler("SPLIT")
def adjust_split(
    df: pd.DataFrame,
    events: pd.DataFrame,
    as_of_date: date,
    calendar: TradingCalendar,  # noqa: ARG001 — EventHandler 시그니처 계약상 필요(SPLIT 조정은 캘린더 비의존)
) -> pd.DataFrame:
    """`events`(이미 `event_type == "SPLIT"`으로 필터링된 부분집합)를 `df`에 적용한다."""
    if events.empty:
        return df

    applicable = events.loc[events["event_date"] <= as_of_date]
    if applicable.empty:
        return df

    multiplier = pd.Series(1.0, index=df.index)
    for _, event in applicable.iterrows():
        affected = df["trade_date"] < event["event_date"]
        multiplier.loc[affected] *= event["stock_rate"]

    if (multiplier == 1.0).all():
        return df

    df = df.copy()
    price_columns = [c for c in df.columns if c.endswith(_PRICE_COLUMN_SUFFIX)]
    for col in price_columns:
        df[col] = df[col] / multiplier
    if "volume" in df.columns:
        df["volume"] = df["volume"] * multiplier

    return df

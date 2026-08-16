"""DIVIDEND 이벤트 조정 핸들러 — 디플레이터 기반 소급 조정.

SPEC-ANALYZER-DATA-001 REQ-AD-030/033/034/035/040(M4): `HANDLER_REGISTRY`에
`"DIVIDEND"`로 등록되는 조정 핸들러다(REQ-AD-041 확장 지점을 통해 배선 —
`adjustment.py`의 디스패치 로직 자체는 건드리지 않는다). `ex_date` 이전
가격에 디플레이터 `(1 - cash_amount / close_price_on_ex_date_prev)`를
누적곱으로 적용한다(REQ-AD-033, Total Return 방향 — TECHSPEC 6.1 개정과
정합). SPLIT과 달리 거래량(volume)은 배당으로 인한 조정 대상이 아니므로
건드리지 않는다.

ex_date는 REQ-AD-030에 따라 `COALESCE(corporate_events.ex_dividend_date,
derive_ex_date(...))`로 해소한다 — 저장값이 있으면 M2의 파생 함수
(`dividend.py`의 `derive_ex_date`)를 호출하지 않는다. `cash_amount == 0`이거나
`event_subtype`이 빈 문자열/알려지지 않은 값인 행은 REQ-AD-034에 따라 조정
계산에서 제외(skip, no-op)한다 — SPEC/plan에 유효 subtype 화이트리스트가
명시되어 있지 않으므로, "알려지지 않은 값"은 결측(None/NaN)으로 해석한다
(화이트리스트를 임의로 발명하지 않는다 — YAGNI).

통화 불변 전제(REQ-AD-035, spec.md §4.3): `currency_code`는 캘린더 시장별
기대 통화(KRX→KRW, 그 외→USD)와 동일하다고 가정하고 크로스 커런시 환산을
수행하지 않는다(shall not). 기대와 다른 통화가 관측되면 해당 이벤트를
조정에서 제외하고 경고 로그를 남기는 방어적 no-op으로 처리한다(스코프
확대가 아니다 — §4.3).

as-of 컷오프(REQ-AD-040)는 이 핸들러 스스로 책임진다: 해소된 `ex_date`가
`as_of_date` 이하(포함)인 이벤트만 반영 대상으로 필터링한다 — `event_date`를
컷오프로 사용해서는 안 된다.
"""

from datetime import date
from typing import cast

import pandas as pd

from analyzer.common.logging import get_logger
from analyzer.data.adjustment import register_handler
from analyzer.data.dividend import derive_ex_date
from analyzer.data.models import TradingCalendar

logger = get_logger(__name__)

_PRICE_COLUMN_SUFFIX = "_price"
_KRX_CALENDAR_CODES = frozenset({"KRX"})
_SKIP_LOG_BATCH_SIZE = 25
"""SPEC-ANALYZER-TRAIN-OBSV-001 REQ-ATO-017: 배당 스킵 경고는 건별 개별
로그 대신 25건마다 1회 집계 로그로 남긴다 — 사용자가 확정한 값이므로
코드 상수로 고정한다(환경변수 오버라이드 없음)."""


def _log_skip_batches(category: str, identifiers: list[object]) -> None:
    """REQ-ATO-017: 스킵된 대상 식별자를 25건 배치로 나눠 집계 경고 로그를 남긴다.

    빈 리스트면 아무 로그도 남기지 않는다(마지막 배치가 0건짜리 빈 로그를
    남기지 않아야 한다는 경계 조건, acceptance.md §B).
    """
    for start in range(0, len(identifiers), _SKIP_LOG_BATCH_SIZE):
        batch = identifiers[start : start + _SKIP_LOG_BATCH_SIZE]
        logger.warning("DIVIDEND 조정 %s(으)로 skip된 대상 배치: %s", category, batch)


def _settlement_market(calendar_code: str) -> str:
    """`calendar_code`를 `get_settlement_days`가 받는 시장 키(KRX/US)로 사상한다."""
    return "KRX" if calendar_code in _KRX_CALENDAR_CODES else "US"


def _expected_currency(calendar_code: str) -> str:
    """`calendar_code`에 대해 REQ-AD-035가 가정하는 기대 통화(KRW/USD)를 반환한다."""
    return "KRW" if calendar_code in _KRX_CALENDAR_CODES else "USD"


def _is_unknown_subtype(subtype: object) -> bool:
    """REQ-AD-034: 빈 문자열 또는 결측값을 "알려지지 않은 subtype"으로 판정한다."""
    if subtype is None or subtype == "":
        return True
    return isinstance(subtype, float) and pd.isna(subtype)


def _resolve_ex_date(event: pd.Series, calendar: TradingCalendar) -> date:
    """REQ-AD-030: 저장된 `ex_dividend_date`를 우선 사용하고, 없으면 파생한다."""
    stored = event.get("ex_dividend_date")
    if stored is not None and not (isinstance(stored, float) and pd.isna(stored)):
        return cast(date, stored)
    market = _settlement_market(calendar.calendar_code)
    return derive_ex_date(cast(date, event["event_date"]), calendar, market=market)


@register_handler("DIVIDEND")
def adjust_dividend(
    df: pd.DataFrame,
    events: pd.DataFrame,
    as_of_date: date,
    calendar: TradingCalendar,
) -> pd.DataFrame:
    """`events`(이미 `event_type == "DIVIDEND"`로 필터링된 부분집합)를 `df`에 적용한다."""
    if events.empty:
        return df

    valid = events.loc[
        (events["cash_amount"] != 0) & (~events["event_subtype"].apply(_is_unknown_subtype))
    ]
    if valid.empty:
        return df

    expected_currency = _expected_currency(calendar.calendar_code)

    multiplier = pd.Series(1.0, index=df.index)
    applied_any = False
    currency_mismatch_skips: list[object] = []
    missing_prev_close_skips: list[object] = []
    for _, event in valid.iterrows():
        currency_code = event.get("currency_code")
        if currency_code != expected_currency:
            currency_mismatch_skips.append(event.get("stock_code"))
            continue

        ex_date = _resolve_ex_date(event, calendar)
        if ex_date > as_of_date:
            continue

        prev_trading_day = calendar.prev_trading_day(ex_date)
        prev_close_rows = df.loc[df["trade_date"] == prev_trading_day, "close_price"]
        if prev_close_rows.empty:
            missing_prev_close_skips.append(event.get("stock_code"))
            continue

        close_price_on_ex_date_prev = prev_close_rows.iloc[0]
        deflator = 1 - (event["cash_amount"] / close_price_on_ex_date_prev)

        affected = df["trade_date"] < ex_date
        multiplier.loc[affected] *= deflator
        applied_any = True

    # REQ-ATO-017: 건별 개별 로그 대신 25건 배치 집계 로그로 남긴다.
    _log_skip_batches("통화 불일치", currency_mismatch_skips)
    _log_skip_batches("전일 종가 없음", missing_prev_close_skips)

    if not applied_any:
        return df

    df = df.copy()
    price_columns = [c for c in df.columns if c.endswith(_PRICE_COLUMN_SUFFIX)]
    for col in price_columns:
        df[col] = df[col] * multiplier

    return df

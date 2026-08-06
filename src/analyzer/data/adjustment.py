"""가격 조정 엔진의 공개 시그니처 + 이벤트 타입 확장 가능 디스패치 구조.

SPEC-ANALYZER-DATA-001 REQ-AD-041: 조정 엔진은 이벤트 타입(SPLIT/DIVIDEND)에
대해 확장 가능한 디스패치 구조로 설계해야 한다 — 향후 RIGHTS_ISSUE·스크립 배당
핸들러 추가 시 기존 SPLIT/DIVIDEND 로직 변경 없이 새 핸들러를 등록할 수 있어야
한다.

M1은 이 디스패치 메커니즘과 `adjust_prices()`의 공개 시그니처만 확정한다.
SPLIT 조정 수식(M3)과 DIVIDEND 디플레이터 계산(M4)은 `register_handler`로
등록되는 핸들러로 후속 마일스톤에서 추가되며, 이 파일은 그 핸들러들을 구현하지
않는다(YAGNI — plan.md §G).
"""

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from analyzer.data.models import TradingCalendar

EventHandler = Callable[[pd.DataFrame, pd.DataFrame, date, "TradingCalendar"], pd.DataFrame]

# event_type(예: "SPLIT", "DIVIDEND") → 해당 타입의 조정을 수행하는 핸들러.
# 모듈 레벨 가변 dict — `register_handler`로만 채워진다. 미등록 이벤트 타입은
# `adjust_prices`에서 방어적으로 skip된다(REQ-AD-050/051, RIGHTS_ISSUE 등).
HANDLER_REGISTRY: dict[str, EventHandler] = {}


def register_handler(event_type: str) -> Callable[[EventHandler], EventHandler]:
    """`event_type`에 대한 조정 핸들러를 `HANDLER_REGISTRY`에 등록하는 데코레이터.

    기존 SPLIT/DIVIDEND 디스패치 로직을 건드리지 않고 새 이벤트 타입 핸들러를
    추가할 수 있게 하는 REQ-AD-041의 확장 지점이다.
    """

    def _decorator(handler: EventHandler) -> EventHandler:
        HANDLER_REGISTRY[event_type] = handler
        return handler

    return _decorator


def adjust_prices(
    df: pd.DataFrame,
    events: pd.DataFrame,
    as_of_date: date,
    calendar: TradingCalendar,
) -> pd.DataFrame:
    """`events`에 등록된 이벤트 타입별 핸들러를 순차 적용해 조정 가격을 산출한다.

    `events`에 존재하는 각 `event_type`에 대해 `HANDLER_REGISTRY`에 등록된
    핸들러가 있으면 해당 이벤트 부분집합과 함께 호출하고, 없으면 조용히
    skip한다(확장 가능 디스패치 — REQ-AD-041). 각 핸들러 자신이 담당 이벤트
    타입의 as-of 컷오프 필드(REQ-AD-040: SPLIT은 event_date, DIVIDEND는
    해소된 ex_date)를 사용해 look-ahead 방지를 보증할 책임을 진다.
    """
    if events.empty:
        return df

    for event_type in events["event_type"].unique():
        handler = HANDLER_REGISTRY.get(event_type)
        if handler is None:
            continue
        event_subset: pd.DataFrame = events.loc[events["event_type"] == event_type]
        df = handler(df, event_subset, as_of_date, calendar)

    return df

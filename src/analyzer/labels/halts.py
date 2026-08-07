"""거래정지 구간 판정 — market_calendar와 daily_ohlcv 관측일의 차집합 기반.

SPEC-ANALYZER-LABEL-001 M3(REQ-AL-040, spec.md §2.4/§4.3): collector는 거래
정지를 별도 이벤트 테이블로 저장하지 않는다. 이 모듈은 신규 데이터 소스를
도입하지 않고, 시장 전체 개장일(`TradingCalendar`)과 종목별 실제 관측일
(`daily_ohlcv.trade_date`)의 **차집합**으로 정지 구간을 판정한다 — 시장은
개장했는데 해당 종목만 결측인 연속 구간이 정지 구간이다.

판정 범위는 종목의 첫 관측일(`first_obs`) 이후 ~ 마지막 관측일(`last_obs`)
이전으로 한정한다(plan.md §B 리스크 2) — 상장일 이전(애초에 데이터가 없는)
구간을 정지로 오판하지 않기 위함이다. `last_obs` 상한을 넘는 구간은
REQ-AL-050(데이터 가용범위 제한) 소관이며 이 모듈이 다루지 않는다 — 호출자
(core.py)는 `target <= last_obs`를 먼저 확인한 뒤에만 `analyze_halt`를
호출해야 한다.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from analyzer.data.models import TradingCalendar


@dataclass(frozen=True, slots=True)
class HaltInfo:
    """`analyze_halt`의 결과 — 정지 구간(있다면)과 정지 해제일(있다면)."""

    segment: tuple[date, ...]
    """`target`을 포함하는 연속 결측 거래일(정지 기간). `target`이 이미
    관측일이면(정지 아님) 빈 튜플이다."""

    resumed_at: date | None
    """정지 해제 후 첫 실제 거래일. 호출자가 `first_obs <= target <= last_obs`를
    보장하는 한 이 값은 이론상 항상 발견된다 — `last_obs` 자체가 관측일이므로
    forward walk가 늦어도 그곳에서 멈춘다. `target`이 이미 관측일이면
    `target` 자신이 반환된다."""


def _next_trading_day(calendar: TradingCalendar, d: date, upper_bound: date) -> date | None:
    """`d`보다 이후(과거 제외)의 첫 거래일을 반환한다. `upper_bound`를 넘으면 None."""
    current = d + timedelta(days=1)
    while current <= upper_bound:
        if calendar.is_trading_day(current):
            return current
        current += timedelta(days=1)
    return None


def analyze_halt(
    calendar: TradingCalendar,
    observed_dates: frozenset[date],
    target: date,
    first_obs: date,
    last_obs: date,
) -> HaltInfo:
    """`target`을 포함하는 연속 결측 거래일 구간과 정지 해제일을 판정한다(REQ-AL-040).

    `target`이 이미 관측일이면 `segment=()`, `resumed_at=target`을 반환한다.
    호출자는 `first_obs <= target <= last_obs`를 보장해야 한다 — 이 범위
    밖은 REQ-AL-050(데이터 가용범위 제한) 판정이 이 호출 이전에 완료되어
    있어야 한다(plan.md §B 리스크 2/3).
    """
    if target in observed_dates:
        return HaltInfo(segment=(), resumed_at=target)

    backward: list[date] = []
    cursor = calendar.prev_trading_day(target)
    while cursor >= first_obs and cursor not in observed_dates:
        backward.append(cursor)
        cursor = calendar.prev_trading_day(cursor)
    backward.reverse()

    forward: list[date] = []
    resumed_at: date | None = None
    cursor2 = _next_trading_day(calendar, target, last_obs)
    while cursor2 is not None:
        if cursor2 in observed_dates:
            resumed_at = cursor2
            break
        forward.append(cursor2)
        cursor2 = _next_trading_day(calendar, cursor2, last_obs)

    segment = (*backward, target, *forward)
    return HaltInfo(segment=segment, resumed_at=resumed_at)

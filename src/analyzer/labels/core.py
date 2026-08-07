"""영업일 기반 T+H 산출 + 레이블 핵심 계산 + 정지/가용범위 통합(공개 진입점).

SPEC-ANALYZER-LABEL-001 M1~M4(REQ-AL-020/021/022/030/031/040~043/050~052):
`compute_labels()`가 이 SPEC의 유일한 공개 함수다. DATA-001의
`adjust_prices()`를 `as_of_date=T+H`(정확히 고정 — plan.md §B 리스크 1
채택안, spec.md §4.1)로 재사용해 T·T+H 양쪽 가격을 조정하고, 거래정지
(halts.py)와 데이터 가용범위 제한(상장폐지/최근 미완결)을 단일 함수로
통합 처리한다. 순수 함수(REQ-AL-070) — DB 접근, 파일 I/O 없음.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

# 핸들러 등록(HANDLER_REGISTRY 배선) 부수효과를 위한 임포트 — SPLIT/DIVIDEND
# 핸들러는 `@register_handler` 데코레이터가 모듈 로드 시점에 실행되어야
# 등록된다(DATA-001 M3/M4). `adjustment.py`만 임포트하면 레지스트리가 비어
# `adjust_prices()`가 모든 이벤트를 조용히 skip한다 — REQ-AL-022(재사용,
# 재구현 금지)를 지키려면 이 모듈이 직접 두 핸들러를 등록시켜야 한다.
import analyzer.data.dividend_adjustment  # noqa: F401
import analyzer.data.split  # noqa: F401
from analyzer.data.adjustment import adjust_prices
from analyzer.data.models import TradingCalendar
from analyzer.labels.config import HORIZONS, ExcludeReason
from analyzer.labels.halts import analyze_halt


def nth_trading_day_on_or_after(calendar: TradingCalendar, start: date, n: int) -> date | None:
    """`start` 이상(on-or-after)인 거래일 중 n번째 거래일을 반환한다(REQ-AL-030).

    `start` 자신이 거래일이면 1번째로 카운트된다. `calendar`의 거래일 데이터
    범위를 벗어나 n번째를 찾지 못하면 None을 반환한다(REQ-AL-031).
    """
    if n < 1:
        raise ValueError("n은 1 이상이어야 한다")
    if not calendar.trading_days:
        return None

    max_day = max(calendar.trading_days)
    if start > max_day:
        return None

    current = start
    count = 0
    while current <= max_day:
        if calendar.is_trading_day(current):
            count += 1
            if count == n:
                return current
        current += timedelta(days=1)
    return None


def _target_date(calendar: TradingCalendar, t: date, horizon: int) -> date | None:
    """T로부터 정확히 `horizon`번째 이후(T 자신 제외) 개장일을 산출한다(REQ-AL-030).

    `nth_trading_day_on_or_after`를 `start=T+1일`로 호출해, T 자신은
    카운트에서 제외하고 T보다 엄격히 이후인 거래일부터 세도록 한다.
    """
    return nth_trading_day_on_or_after(calendar, t + timedelta(days=1), horizon)


@dataclass(slots=True)
class _LabelContext:
    """`compute_labels` 호출 1회에 대해 horizon 전체가 공유하는 불변 컨텍스트.

    `adjusted_cache`는 `as_of_date`(target 날짜)별로 `adjust_prices()` 결과를
    캐싱해, 같은 target을 여러 (T, horizon) 조합이 공유할 때 중복 계산을
    피한다 — 값 자체(REQ-AL-020)는 캐싱과 무관하게 항상 `as_of_date=target`
    고정 호출로 계산된다.
    """

    df: pd.DataFrame
    events: pd.DataFrame
    calendar: TradingCalendar
    observed_dates: frozenset[date]
    first_obs: date
    last_obs: date
    is_delisted: bool
    adjusted_cache: dict[date, pd.DataFrame] = field(default_factory=dict)


def _compute_single_label(
    ctx: _LabelContext, t: date, horizon: int
) -> tuple[float, ExcludeReason | None]:
    """단일 (T, horizon) 조합의 레이블 값과 exclude_reason을 계산한다."""
    target = _target_date(ctx.calendar, t, horizon)

    if target is None:
        # REQ-AL-031: market_calendar 데이터 범위 밖 — 아직 확정할 수 없는
        # 미래 구간이라는 점에서 REQ-AL-050의 "데이터 가용범위 제한"과 같은
        # 원인 계열로 분류한다(AC-AL-004는 exclude_reason 값을 검증하지
        # 않으나, 다운스트림 감사 편의를 위해 일관된 사유를 부여한다).
        return float("nan"), "insufficient_future_data"

    if target > ctx.last_obs:
        # REQ-AL-050: 상장폐지(종료) 또는 최근 미완결(아직 미도래) — 단일
        # NaN 규칙이며, 사유만 REQ-AL-051에 따라 구분 표시한다.
        reason: ExcludeReason = "delisted" if ctx.is_delisted else "insufficient_future_data"
        return float("nan"), reason

    if target in ctx.observed_dates:
        resolved_date = target
    else:
        halt_info = analyze_halt(
            ctx.calendar, ctx.observed_dates, target, ctx.first_obs, ctx.last_obs
        )
        if len(halt_info.segment) >= horizon:
            # REQ-AL-042/043: 정지 기간 >= horizon → NaN, 롤포워드하지 않는다.
            return float("nan"), "halted"
        if halt_info.resumed_at is None:
            # target <= last_obs가 보장된 상태에서는 이론상 도달하지 않는다
            # (last_obs 자체가 관측일이므로 forward walk가 늦어도 거기서
            # 멈춘다 — 방어적 fallback).
            reason = "delisted" if ctx.is_delisted else "insufficient_future_data"
            return float("nan"), reason
        # REQ-AL-041: 정지 기간 < horizon → 롤포워드(정지 해제 후 다음 실제
        # 거래일 가격을 T+H 가격으로 사용).
        resolved_date = halt_info.resumed_at

    adjusted = ctx.adjusted_cache.get(target)
    if adjusted is None:
        adjusted = adjust_prices(ctx.df, ctx.events, as_of_date=target, calendar=ctx.calendar)
        ctx.adjusted_cache[target] = adjusted

    start_price = adjusted.loc[adjusted["trade_date"] == t, "close_price"].iloc[0]
    end_price = adjusted.loc[adjusted["trade_date"] == resolved_date, "close_price"].iloc[0]
    label = float(end_price / start_price - 1)
    return label, None


def compute_labels(
    df: pd.DataFrame,
    events: pd.DataFrame,
    calendar: TradingCalendar,
    horizons: tuple[int, ...] = HORIZONS,
    is_delisted: bool = False,
) -> pd.DataFrame:
    """종목 1개의 `daily_ohlcv`(원주가) + `corporate_events` + `TradingCalendar`를
    입력받아, horizon별 실현 수익률 레이블을 계산해 반환한다(REQ-AL-020/021).

    입력 `df`는 `repository.fetch_daily_ohlcv`가 반환하는 스키마(`trade_date`,
    `close_price` 포함, 단일 종목의 원주가 행)를 가정한다. 출력은 `df`의
    원본 컬럼을 보존하고, horizon별로 `label_D{h}`(연속값 실현 수익률,
    조건 미충족 시 NaN)와 `label_D{h}_exclude_reason`(nullable 문자열 —
    `None`/`"halted"`/`"delisted"`/`"insufficient_future_data"`) 컬럼을
    추가한다(REQ-AL-060, M1 출력 스키마 계약).

    순수 함수(REQ-AL-070) — DB 접근, 파일 I/O 없음, 동일 입력에 항상 동일
    출력을 반환한다. 이 함수는 `repository.py`를 확장하지 않으며 DB 접근은
    상위 오케스트레이션(ANALYZER-TRAIN-001) 소관이다.

    `is_delisted`는 호출자가 `stocks.delisted_at IS NOT NULL` 여부를
    전달하는 선택적 파라미터다(REQ-AL-051) — 이 함수는 `stocks` 테이블을
    직접 조회하지 않는다(신규 repository 함수 추가 금지, plan.md §D).
    """
    result = df.copy()
    if result.empty:
        for horizon in horizons:
            result[f"label_D{horizon}"] = pd.Series(dtype="float64")
            result[f"label_D{horizon}_exclude_reason"] = pd.Series(dtype="object")
        return result

    observed_dates: frozenset[date] = frozenset(result["trade_date"])
    first_obs: date = min(observed_dates)
    last_obs: date = max(observed_dates)

    # `ctx.df`는 원본 `df`(레이블 컬럼이 아직 추가되지 않은 상태)를 참조한다
    # — `adjust_prices()`는 `_price` 접미사 컬럼만 조정하므로 이후 추가되는
    # `label_D{h}` 컬럼과는 간섭하지 않지만, 입력/출력 책임을 명확히
    # 분리하기 위해 원본을 그대로 사용한다.
    ctx = _LabelContext(
        df=df,
        events=events,
        calendar=calendar,
        observed_dates=observed_dates,
        first_obs=first_obs,
        last_obs=last_obs,
        is_delisted=is_delisted,
    )

    for horizon in horizons:
        labels: list[float] = []
        reasons: list[ExcludeReason | None] = []
        for t in result["trade_date"]:
            label, reason = _compute_single_label(ctx, t, horizon)
            labels.append(label)
            reasons.append(reason)
        # dtype을 명시해, reasons가 전부 None인 극단적 경우에도 object dtype이
        # 유지되어 None이 NaN으로 암묵 변환되지 않도록 한다.
        result[f"label_D{horizon}"] = pd.Series(labels, dtype="float64", index=result.index)
        result[f"label_D{horizon}_exclude_reason"] = pd.Series(
            reasons, dtype="object", index=result.index
        )

    return result

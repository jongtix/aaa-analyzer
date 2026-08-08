"""영업일 기반 T+H 산출 + 레이블 핵심 계산 + 정지/가용범위 통합(공개 진입점).

SPEC-ANALYZER-LABEL-001 M1~M4(REQ-AL-020/021/022/030/031/040~043/050~052):
`compute_labels()`가 이 SPEC의 유일한 공개 함수다. DATA-001의
`adjust_prices()`를 `as_of_date=T+H`(정확히 고정 — plan.md §B 리스크 1
채택안, spec.md §4.1)로 재사용해 T·T+H 양쪽 가격을 조정하고, 거래정지
(halts.py)와 데이터 가용범위 제한(상장폐지/최근 미완결)을 단일 함수로
통합 처리한다. 순수 함수(REQ-AL-070) — DB 접근, 파일 I/O 없음.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import cast

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


def nth_trading_day_on_or_after(
    calendar: TradingCalendar, start: date, n: int, max_day: date | None = None
) -> date | None:
    """`start` 이상(on-or-after)인 거래일 중 n번째 거래일을 반환한다(REQ-AL-030).

    `start` 자신이 거래일이면 1번째로 카운트된다. `calendar`의 거래일 데이터
    범위를 벗어나 n번째를 찾지 못하면 None을 반환한다(REQ-AL-031).

    `max_day`(성능 최적화, 리뷰 finding #3)는 `calendar.trading_days`의
    최댓값을 호출자가 이미 계산해둔 경우 재사용하기 위한 선택적 파라미터다.
    생략하면(기본 None) 이 함수가 직접 `max(calendar.trading_days)`를
    계산한다 — 공개 함수의 기존 3-인자 호출 형태(직접 단위 테스트 포함)는
    동작 변화 없이 그대로 유지된다. `compute_labels`의 내부 호출 경로는
    `_LabelContext.max_trading_day`에 1회만 계산해둔 값을 전달해, 같은
    calendar에 대해 (T, horizon) 조합마다 O(|trading_days|) 재계산이
    반복되던 문제를 제거한다.
    """
    if n < 1:
        raise ValueError("n은 1 이상이어야 한다")
    if not calendar.trading_days:
        return None

    if max_day is None:
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


def _target_date(
    calendar: TradingCalendar, t: date, horizon: int, max_day: date | None = None
) -> date | None:
    """T로부터 정확히 `horizon`번째 이후(T 자신 제외) 개장일을 산출한다(REQ-AL-030).

    `nth_trading_day_on_or_after`를 `start=T+1일`로 호출해, T 자신은
    카운트에서 제외하고 T보다 엄격히 이후인 거래일부터 세도록 한다.
    `max_day`는 `nth_trading_day_on_or_after`에 그대로 전달되는 선택적
    사전계산 캘린더 경계값이다.
    """
    return nth_trading_day_on_or_after(calendar, t + timedelta(days=1), horizon, max_day)


@dataclass(slots=True)
class _LabelContext:
    """`compute_labels` 호출 1회에 대해 horizon 전체가 공유하는 불변 컨텍스트.

    `max_trading_day`는 `calendar.trading_days`의 최댓값을 1회만 계산해둔
    값이다(리뷰 finding #3) — `nth_trading_day_on_or_after`가 (T, horizon)
    조합마다 매번 `max()`를 재계산하지 않도록 전달한다.
    """

    df: pd.DataFrame
    events: pd.DataFrame
    calendar: TradingCalendar
    observed_dates: frozenset[date]
    first_obs: date
    last_obs: date
    is_delisted: bool
    max_trading_day: date | None


@dataclass(slots=True)
class _PendingPriceLookup:
    """가격 조회가 아직 필요한 단일 (horizon, row) 계산 — target별로 묶어
    `adjust_prices()`를 target당 정확히 1회만 호출하기 위한 중간 표현
    (리뷰 finding #1/#2: `adjusted_cache`가 target별 전체 DataFrame을
    무기한 보관하며 O(rows²) 메모리를 쓰던 문제와, near-unique target으로
    캐시가 사실상 히트하지 않던 문제를 함께 해소한다).

    `horizon`은 계산된 레이블을 어느 horizon의 결과 버퍼로 되돌려야
    하는지 식별한다 — `pending_by_target`은 모든 horizon이 공유하는
    단일 딕셔너리이므로(동일 target을 서로 다른 horizon이 공유할 때도
    `adjust_prices()` 중복 호출을 피하기 위함), horizon 정보가 없으면
    결과를 어느 버퍼에 기록할지 알 수 없다.
    """

    horizon: int
    row_index: int
    t: date
    resolved_date: date


def _resolve_target_and_dates(
    ctx: _LabelContext, t: date, horizon: int
) -> tuple[date, date] | ExcludeReason:
    """target·정지·가용범위를 판정해 (target, resolved_date) 또는 확정된
    exclude_reason을 반환한다. 가격 조회(adjust_prices 호출)는 하지
    않는다 — 이 판정 로직(정지 판정 포함)은 리뷰 대상 3건에 포함되지
    않으며, 기존 `_compute_single_label`의 판정 흐름을 그대로 보존한다.
    """
    target = _target_date(ctx.calendar, t, horizon, ctx.max_trading_day)

    if target is None:
        # REQ-AL-031: market_calendar 데이터 범위 밖 — 아직 확정할 수 없는
        # 미래 구간이라는 점에서 REQ-AL-050의 "데이터 가용범위 제한"과 같은
        # 원인 계열로 분류한다(AC-AL-004는 exclude_reason 값을 검증하지
        # 않으나, 다운스트림 감사 편의를 위해 일관된 사유를 부여한다).
        return "insufficient_future_data"

    if target > ctx.last_obs:
        # REQ-AL-050: 상장폐지(종료) 또는 최근 미완결(아직 미도래) — 단일
        # NaN 규칙이며, 사유만 REQ-AL-051에 따라 구분 표시한다.
        return "delisted" if ctx.is_delisted else "insufficient_future_data"

    if target in ctx.observed_dates:
        return target, target

    halt_info = analyze_halt(ctx.calendar, ctx.observed_dates, target, ctx.first_obs, ctx.last_obs)
    if len(halt_info.segment) >= horizon:
        # REQ-AL-042/043: 정지 기간 >= horizon → NaN, 롤포워드하지 않는다.
        return "halted"
    if halt_info.resumed_at is None:
        # target <= last_obs가 보장된 상태에서는 이론상 도달하지 않는다
        # (last_obs 자체가 관측일이므로 forward walk가 늦어도 거기서
        # 멈춘다 — 방어적 fallback).
        return "delisted" if ctx.is_delisted else "insufficient_future_data"
    # REQ-AL-041: 정지 기간 < horizon → 롤포워드(정지 해제 후 다음 실제
    # 거래일 가격을 T+H 가격으로 사용).
    return target, halt_info.resumed_at


def _resolve_pending_prices(
    ctx: _LabelContext,
    pending_by_target: dict[date, list[_PendingPriceLookup]],
    labels_by_horizon: dict[int, dict[int, float]],
) -> None:
    """target별로 그룹핑된 가격 조회 요청을 일괄 처리한다(리뷰 finding #1/#2).

    target당 `adjust_prices()`를 정확히 1회만 호출하고(값 자체는 여전히
    항상 `as_of_date=target` 고정 호출로 계산되어 REQ-AL-020을 그대로
    지킨다), 조정 결과에서 이번 target에 필요한 종가만 인덱싱된
    `trade_date -> close_price` 매핑으로 즉시 추출한 뒤 버린다 — 전체
    DataFrame을 캐시에 보관하지 않으므로 peak 메모리가 O(rows)로
    유지된다. 가격 접근도 O(n) 불리언 마스크 2회(리뷰 Warning finding)
    대신 O(1) 인덱스 조회로 대체된다. `pending_by_target`은 모든
    horizon이 공유하므로, 서로 다른 horizon의 (T, horizon) 조합이 우연히
    같은 target을 참조하는 경우에도 `adjust_prices()`는 한 번만 호출된다.

    `adjust_prices()`는 적용 가능한 이벤트가 없으면(전체 미보유거나, 이
    target 시점까지 해당되는 이벤트가 아직 없는 경우) 입력 `df` 객체를
    그대로 반환한다(adjustment.py/split.py/dividend_adjustment.py의
    no-op 단축 경로). 이 경우 target마다 매번 동일한(미조정) 종가
    인덱스를 새로 만드는 것은 불필요한 O(rows) 반복이므로,
    `base_price_by_date`에 최초 1회만 구축해 재사용한다.
    """
    base_price_by_date: pd.Series | None = None
    for target, lookups in pending_by_target.items():
        adjusted = adjust_prices(ctx.df, ctx.events, as_of_date=target, calendar=ctx.calendar)
        if adjusted is ctx.df:
            if base_price_by_date is None:
                base_price_by_date = _indexed_close_prices(ctx.df)
            price_by_date = base_price_by_date
        else:
            price_by_date = _indexed_close_prices(adjusted)
        for lookup in lookups:
            start_price = float(price_by_date.at[lookup.t])
            end_price = float(price_by_date.at[lookup.resolved_date])
            label = float(end_price / start_price - 1)
            labels_by_horizon[lookup.horizon][lookup.row_index] = label


def _indexed_close_prices(adjusted: pd.DataFrame) -> pd.Series:
    """`trade_date -> close_price` 인덱스를 구축한다(리뷰 Warning finding).

    `drop_duplicates(keep="first")`는 기존 `.iloc[0]`(첫 매치 우선) 불리언
    마스크 조회와 동일한 선택 규칙을 보존하기 위한 방어적 조치다 — 정상
    입력(종목당 trade_date 유일)에서는 아무 행도 제거하지 않는다.
    """
    deduped = adjusted.drop_duplicates(subset="trade_date", keep="first")
    return cast(pd.Series, deduped.set_index("trade_date")["close_price"])


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
    # 리뷰 finding #3: calendar는 이 `compute_labels` 호출 전체에서 불변이므로
    # 최댓값을 여기서 1회만 계산해 ctx에 실어 보낸다 — 이전에는
    # `nth_trading_day_on_or_after`가 (T, horizon) 조합마다(row 수 × len(horizons)회)
    # `max(calendar.trading_days)`를 매번 재계산했다.
    max_trading_day: date | None = max(calendar.trading_days) if calendar.trading_days else None

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
        max_trading_day=max_trading_day,
    )

    trade_dates = list(result["trade_date"])
    n = len(trade_dates)

    # 1단계(가격 조회 없이 target·정지·가용범위만 판정): horizon마다 결과
    # 버퍼를 미리 NaN/None으로 채워두고, 실제 가격이 필요한 (row, target)
    # 조합만 target별로 그룹핑해 `pending_by_target`에 모은다. 이 단계는
    # `adjust_prices()`를 전혀 호출하지 않는다.
    labels_by_horizon: dict[int, dict[int, float]] = {}
    reasons_by_horizon: dict[int, list[ExcludeReason | None]] = {}
    pending_by_target: dict[date, list[_PendingPriceLookup]] = {}

    for horizon in horizons:
        labels_by_row: dict[int, float] = {}
        reasons: list[ExcludeReason | None] = [None] * n
        labels_by_horizon[horizon] = labels_by_row
        reasons_by_horizon[horizon] = reasons

        for row_index, t in enumerate(trade_dates):
            outcome = _resolve_target_and_dates(ctx, t, horizon)
            if isinstance(outcome, tuple):
                target, resolved_date = outcome
                pending_by_target.setdefault(target, []).append(
                    _PendingPriceLookup(
                        horizon=horizon, row_index=row_index, t=t, resolved_date=resolved_date
                    )
                )
            else:
                reasons[row_index] = outcome

    # 2단계(가격 일괄 조회, 리뷰 finding #1/#2): target별로 `adjust_prices()`를
    # 정확히 1회만 호출하고, 그 결과에서 이번 target에 필요한 스칼라 가격만
    # 즉시 추출한다. horizon 사이에 target이 우연히 겹치는 경우(target이
    # 같은 두 (T, horizon) 조합)에도 `pending_by_target`에서 이미 하나로
    # 묶여 있으므로 자연히 공유되어 중복 호출되지 않는다.
    _resolve_pending_prices(ctx, pending_by_target, labels_by_horizon)

    for horizon in horizons:
        labels: list[float] = [
            labels_by_horizon[horizon].get(row_index, float("nan")) for row_index in range(n)
        ]
        # dtype을 명시해, reasons가 전부 None인 극단적 경우에도 object dtype이
        # 유지되어 None이 NaN으로 암묵 변환되지 않도록 한다.
        result[f"label_D{horizon}"] = pd.Series(labels, dtype="float64", index=result.index)
        result[f"label_D{horizon}_exclude_reason"] = pd.Series(
            reasons_by_horizon[horizon], dtype="object", index=result.index
        )

    return result

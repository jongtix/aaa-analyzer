"""풀링 패널 날짜-경계 어댑터 + 신규 주간 스트라이드 폴드 생성기 (SPEC-ANALYZER-TRAIN-EVAL-001 M2).

REQ-ATE-017: `dataset.assemble_dataset()`의 출력은 종목별로 연속 배치된
풀링 패널(전역 시간순 정렬이 아님)이다 — 캠페인의 폴드 경계는 이 패널이
아니라, 패널에서 추출한 고유 정렬 거래일 축(전역 캘린더)을 기준으로
계산되어야 한다.

REQ-ATE-018: 폴드 `i`의 학습 종료 인덱스(`train_end[i]`)는 이전 폴드
대비 정확히 검증 윈도우 크기(`val_size`)만큼만 전진하는 신규 주간
스트라이드 산식을 구현한다 — `training.split.expanding_window_folds()`의
기존 스트라이드 공식(전진폭 = purge gap + 검증 윈도우 크기)은 재사용하지
않는다(design.md §2A — 기존 공식으로는 D60 기준 약 130.0년치 이력이
필요해 실제 가용 이력을 초과함). 각 폴드 자신의 검증 윈도우 시작
(`val_start[i]`)은 여전히 `train_end[i] + horizon`(purge gap,
`labels.config.PURGE_GAP_TRADING_DAYS` 단일 소스 재사용)이다.

REQ-ATE-019: 각 폴드의 `train_end`/`val_start`/`val_end` 정수 인덱스를
실제 `trade_date` 값으로 역매핑하고, 패널을 그 `trade_date` 값 비교로
필터링해 학습/검증 행 부분집합을 구성한다 — 행 위치가 아니라 값 비교로
분할한다(상장폐지/상장지연 종목도 자신이 존재하는 기간 내에서 올바른
폴드에 배정됨).

REQ-ATE-020: 이 모듈은 `training/split.py`가 아니라 신규 캠페인
오케스트레이션 모듈에 위치한다 — `split.py` 파일 자체는 무수정이다.

purge gap 경계 판정은 반개구간(half-open) `[train_end, val_start)`
관례를 그대로 재사용한다(plan.md §E F16, split.py의 기존 동결 계약과
동일) — `train_end`/`val_end`는 배타적 상한이다.
"""

from dataclasses import dataclass
from typing import cast

import pandas as pd

from analyzer.labels.config import PURGE_GAP_TRADING_DAYS


@dataclass(frozen=True, slots=True)
class PanelFoldIndexBounds:
    """정수 인덱스(0..len(T)-1) 축 위에서 표현된 폴드 경계(반개구간).

    `train_end`/`val_end`는 배타적 상한이며, `train_end`~`val_start`
    사이는 purge gap으로 어느 구간에도 포함되지 않는다(split.py
    `SplitBounds`와 동일한 관례). 폴드는 항상 인덱스 0부터 학습을
    시작한다(expanding — `train_start`는 암묵적으로 0).
    """

    train_end: int
    val_start: int
    val_end: int


@dataclass(frozen=True, slots=True)
class PanelFoldDateBounds:
    """실제 `trade_date` 값으로 역매핑된 폴드 경계(반개구간, 날짜값 비교용)."""

    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp | None
    """`val_end`가 None이면 검증 윈도우 상한이 없다는 뜻이다 — 이 폴드가
    거래일 축의 마지막 날짜를 포함해 그 이후 전부를 검증 구간으로
    삼는 마지막 폴드일 때(`T[val_end]`가 배열 범위를 벗어남) 발생한다."""


def extract_global_trade_date_axis(
    panel: pd.DataFrame, date_column: str = "trade_date"
) -> pd.DatetimeIndex:
    """조립된 풀링 패널에서 유니버스 전체를 아우르는 고유 거래일 축을 추출한다(REQ-ATE-017).

    반환 배열은 오름차순 정렬되며, 길이는 종목 수와 무관하게 그 시장의
    실제 거래일 수와 같다.
    """
    return pd.DatetimeIndex(sorted(panel[date_column].unique()))


def weekly_stride_fold_index_bounds(
    n_dates: int,
    horizon: int,
    initial_train_end: int,
    val_size: int,
    n_folds: int,
) -> list[PanelFoldIndexBounds]:
    """신규 주간 스트라이드 산식으로 정수 인덱스 폴드 경계를 생성한다(REQ-ATE-018, design.md §2A).

    `train_end[i] = initial_train_end + i * val_size`(전진폭 = val_size만,
    gap 제외) — `val_start[i] = train_end[i] + gap`, `val_end[i] =
    val_start[i] + val_size`. 이 산식에서 인접 폴드의 검증 윈도우는
    빈틈없이 맞닿는다(`val_start[i+1] - val_end[i] == 0`).

    `gap`은 `labels.config.PURGE_GAP_TRADING_DAYS`에서 조회한다(단일
    소스 재사용) — `training.split.expanding_window_folds()`는 이 계산에
    사용하지 않는다.
    """
    if n_folds < 1:
        raise ValueError("n_folds는 1 이상이어야 한다")
    if val_size < 1:
        raise ValueError("val_size는 1 이상이어야 한다")
    if initial_train_end < 1:
        raise ValueError("initial_train_end는 1 이상이어야 한다")

    gap = PURGE_GAP_TRADING_DAYS[horizon]

    bounds: list[PanelFoldIndexBounds] = []
    for i in range(n_folds):
        train_end = initial_train_end + i * val_size
        val_start = train_end + gap
        val_end = val_start + val_size
        if val_start > n_dates:
            raise ValueError(
                f"거래일 수({n_dates})가 n_folds={n_folds}개 폴드를 생성하기에 "
                f"부족하다(폴드 {i}의 val_start={val_start})"
            )
        bounds.append(
            PanelFoldIndexBounds(train_end=train_end, val_start=val_start, val_end=val_end)
        )

    return bounds


def map_index_bounds_to_dates(
    bounds: PanelFoldIndexBounds, trade_dates: pd.DatetimeIndex
) -> PanelFoldDateBounds:
    """정수 인덱스 경계를 실제 `trade_date` 값으로 역매핑한다(REQ-ATE-019)."""
    n_dates = len(trade_dates)
    if bounds.train_end >= n_dates or bounds.val_start >= n_dates:
        raise ValueError(
            f"거래일 수({n_dates})가 폴드 경계(train_end={bounds.train_end}, "
            f"val_start={bounds.val_start})를 수용하기에 부족하다"
        )

    val_end_date = (
        cast(pd.Timestamp, trade_dates[bounds.val_end]) if bounds.val_end < n_dates else None
    )

    return PanelFoldDateBounds(
        train_end=cast(pd.Timestamp, trade_dates[bounds.train_end]),
        val_start=cast(pd.Timestamp, trade_dates[bounds.val_start]),
        val_end=val_end_date,
    )


def slice_panel_by_date_bounds(
    panel: pd.DataFrame,
    bounds: PanelFoldDateBounds,
    date_column: str = "trade_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """패널을 `trade_date` 값 비교로 학습/검증 부분집합으로 필터링한다(REQ-ATE-019).

    행 위치가 아니라 값 비교이므로, 종목마다 실제 거래일이 전역
    캘린더와 다르더라도(상장폐지, 상장지연 등) 모든 종목이 동시에 같은
    날짜 경계로 분할된다.
    """
    dates = panel[date_column]
    train_df = panel.loc[dates < bounds.train_end]
    if bounds.val_end is None:
        val_df = panel.loc[dates >= bounds.val_start]
    else:
        val_df = panel.loc[(dates >= bounds.val_start) & (dates < bounds.val_end)]

    return train_df, val_df

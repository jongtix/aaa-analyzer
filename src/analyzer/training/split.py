"""자체 구현 Purged Expanding-Window Walk-Forward 분할기 (SPEC-ANALYZER-TRAIN-001 M3).

REQ-AT-070: scikit-learn `TimeSeriesSplit`은 purge/embargo를 지원하지 않고,
서드파티 `purgedcv`는 검증되지 않은 단일 관리자 라이브러리라 신규
의존성으로 추가하지 않는다(§5.7) — 이 모듈이 자체 구현을 제공한다.

REQ-AT-071: purge gap 길이는 `labels.config.PURGE_GAP_TRADING_DAYS`
(horizon과 동일)를 유일한 소스로 소비한다 — 값을 재정의하거나
하드코딩하지 않는다.

REQ-AT-072: 분할 비율(학습/검증/테스트, 기본 70/15/15)은 함수 인자로
외부화된 파라미터다 — 코드에 하드코딩하지 않는다.

TECHSPEC §6.3(analyzer 설계 [D-13], 2026-07-04 확정): fold 구성은
expanding window(학습 구간을 점진적으로 확장하며 검증 구간을 앞으로
이동)이며, 테스트 구간(기본 마지막 15%)은 모든 폴드가 공유하는 고정
홀드아웃이다. fold 수 자체의 실측 확정(REQ-AT-073)은 M6 소관이며, 이
모듈은 `n_folds`/`val_size`를 호출자가 외부에서 주입받는 파라미터로만
받는다(하드코딩하지 않음).

이 모듈은 정수 인덱스(0..n_samples-1, 이미 시간순 정렬된 샘플)에 대해
경계만 산출한다 — 실제 DataFrame 슬라이싱은 호출자(오케스트레이션,
M5/M6 소관) 책임이다.
"""

from dataclasses import dataclass

from analyzer.labels.config import PURGE_GAP_TRADING_DAYS


@dataclass(frozen=True, slots=True)
class SplitBounds:
    """[start, end) 반개구간(exclusive end)으로 표현된 학습/검증/테스트 경계.

    `train_end`와 `val_start` 사이, `val_end`와 `test_start` 사이의 인덱스는
    purge gap이며 어느 구간에도 포함되지 않는다.
    """

    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def _purge_gap(horizon: int) -> int:
    """`horizon`에 대응하는 purge gap 길이를 조회한다(REQ-AT-071).

    `PURGE_GAP_TRADING_DAYS`에 없는 horizon이면 `KeyError`를 발생시킨다 —
    임의의 gap 길이를 추정하거나 기본값으로 대체하지 않는다.
    """
    return PURGE_GAP_TRADING_DAYS[horizon]


def purged_walk_forward_split(
    n_samples: int,
    horizon: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitBounds:
    """시간순 샘플을 purge gap을 포함한 학습/검증/테스트로 1회 분할한다(REQ-AT-070/071/072).

    TECHSPEC §6.3의 macro 분할 비율(기본 70/15/15)을 그대로 구현한다 —
    학습 구간 마지막과 검증 구간 시작 사이, 검증 구간 마지막과 테스트
    구간 시작 사이에 각각 정확히 `horizon` 길이(영업일)만큼의 purge
    gap이 존재하며, 그 구간의 샘플은 어느 분할에도 포함되지 않는다
    (AC-AT-005).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio의 합은 1.0이어야 한다")

    gap = _purge_gap(horizon)
    remaining = n_samples - 2 * gap
    if remaining <= 0:
        raise ValueError(f"샘플 수({n_samples})가 purge gap 2개(각 {gap})를 수용하기에 부족하다")

    train_count = round(remaining * train_ratio)
    val_count = round(remaining * val_ratio)
    test_count = remaining - train_count - val_count
    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError("주어진 비율로는 각 구간에 최소 1개 샘플도 배정할 수 없다")

    train_start = 0
    train_end = train_start + train_count
    val_start = train_end + gap
    val_end = val_start + val_count
    test_start = val_end + gap
    test_end = test_start + test_count

    return SplitBounds(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )


def expanding_window_folds(
    n_samples: int,
    horizon: int,
    n_folds: int,
    val_size: int,
    test_ratio: float = 0.15,
) -> list[SplitBounds]:
    """purge-aware expanding-window WFV 폴드를 생성한다(REQ-AT-070, TECHSPEC §6.3).

    마지막 `test_ratio` 비율(기본 15%)은 모든 폴드가 공유하는 고정
    홀드아웃이며 어떤 폴드의 학습/검증에도 사용되지 않는다. 그 앞
    (학습+검증 후보) 구간 안에서 `n_folds`개의 폴드를 만든다 — 폴드마다
    학습 구간이 확장되고(이전 폴드의 검증 구간이 다음 폴드의 학습
    구간에 포함됨), 검증 구간(고정 크기 `val_size`)이 앞으로 이동한다.
    각 폴드의 학습-검증, (공유) 검증 종료-테스트 시작 경계에는 정확히
    `horizon` 길이의 purge gap이 있다.

    `n_folds`/`val_size`는 fold 수 실측 확정(REQ-AT-073, M6 소관)이
    이루어지기 전까지 호출자가 명시적으로 주입해야 하는 파라미터다 —
    이 함수는 기본값을 추정하거나 하드코딩하지 않는다.
    """
    if n_folds < 1:
        raise ValueError("n_folds는 1 이상이어야 한다")
    if val_size < 1:
        raise ValueError("val_size는 1 이상이어야 한다")

    gap = _purge_gap(horizon)
    test_size = round(n_samples * test_ratio)
    test_start = n_samples - test_size
    test_end = n_samples
    trainval_end = test_start - gap

    initial_train_end = trainval_end - n_folds * (gap + val_size)
    if initial_train_end <= 0:
        raise ValueError(
            f"샘플 수({n_samples})가 n_folds={n_folds}, val_size={val_size}, "
            f"gap={gap}, test_ratio={test_ratio} 조합을 수용하기에 부족하다"
        )

    folds: list[SplitBounds] = []
    for i in range(n_folds):
        train_end = initial_train_end + i * (gap + val_size)
        val_start = train_end + gap
        val_end = val_start + val_size
        folds.append(
            SplitBounds(
                train_start=0,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return folds

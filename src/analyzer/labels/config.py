"""레이블 계산 파라미터 — 예측 기간(horizon), 학습 시작일, purge gap 상수.

SPEC-ANALYZER-LABEL-001 M1(REQ-AL-010/011/061): `HORIZONS`(예측 기간, 영업일
단위)와 시장별 학습 데이터 시작일, horizon별 purge gap 길이를 모듈 상수로
노출한다. 계산 로직(core.py/halts.py)은 이 상수만을 단일 소스로 참조하며
값을 하드코딩하지 않는다.
"""

from datetime import date
from typing import Literal

HORIZONS: tuple[int, ...] = (20, 60)
"""단기(20영업일)/중기(60영업일) 예측 기간(REQ-AL-010)."""

DEFAULT_START_DATES: dict[str, date] = {
    "domestic": date(2005, 1, 1),
    "overseas": date(2007, 8, 20),
}
"""시장별 학습 데이터 시작일 기본값(REQ-AL-011, spec.md §4.5).

레이블 계산 함수(`compute_labels`) 자체는 이 값을 소비하지 않는다 — 학습
유니버스 필터링(어느 (종목, T) 쌍을 학습 샘플로 포함할지)은 이 SPEC이
구현하지 않으며(§2.8, REQ-AL-011 예외 사유), ANALYZER-TRAIN-001이 이
파라미터를 소비한다. 이 모듈은 그 경계 파라미터의 단일 소스만 호스팅한다.
"""

PURGE_GAP_TRADING_DAYS: dict[int, int] = {20: 20, 60: 60}
"""horizon별 purge gap 길이(영업일) — TECHSPEC §6.3 정의(purge gap =
horizon 길이)와 정확히 일치한다(REQ-AL-061). 실제 Walk-Forward 분할에서의
purge gap 적용 자체는 이 SPEC이 구현하지 않는다(§2.8)."""

ExcludeReason = Literal["halted", "delisted", "insufficient_future_data"]
"""`label_D{h}_exclude_reason` 컬럼 값 enum(REQ-AL-043/051).

`None`은 정상 케이스(레이블 값이 유효한 실현 수익률)를 의미하며 별도
리터럴로 표현하지 않는다 — 컬럼 값이 `None`(NaN 아님)이면 그 행의
`label_D{h}`는 유효한 값이다.
"""

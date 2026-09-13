"""`stream:signal:{market}` 신호 발행 (SPEC-ANALYZER-INFER-001 M8,
REQ-AIF-120, AC-AIF-020, design.md §6).

발행 순서 계약(design.md §6) — 발행은 `writer.insert_trading_signal()`이
정상 반환한 **이후에만** 수행한다. 그 함수는 `InsertOutcome.INSERTED`(신규
삽입) 또는 `InsertOutcome.SKIPPED_DUPLICATE`(UNIQUE 스킵 = "이미 존재")만
반환하고, 그 밖의 실패는 예외로 전파되므로 — 정상 반환 = 발행 허용이다.
이 모듈이 별도의 결과 판정 함수를 두지 않는 이유가 그것이다(두 반환값 모두
발행 대상이라 판정 함수가 항상 참을 반환하는 공허한 추상이 된다). 호출부는
`insert_trading_signal()` 반환 뒤에 이 함수를 호출하는 순서만 지키면 된다.

`signal_price_bands`는 발행하지 않는다(design.md §6) — 밴드는 DB 조회로만
소비된다.

동기/비동기 경계 — `redis_client.py` 관례대로 이 모듈은 동기 함수만 제공하고,
`asyncio.to_thread()` 래핑은 호출자(`orchestration/consumer.py`가 XADD/
XREADGROUP을 감싸는 방식과 동일)의 책임이다.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from analyzer.inference.writer import format_horizon_label

SIGNAL_STREAM_PREFIX = "stream:signal"

SIGNAL_STREAM_MAXLEN = 500
"""REQ-AIF-120 / TECHSPEC 확정값. `approximate=True`(XADD `MAXLEN ~`)와 함께
쓰여 정확한 카운트를 강제하지 않는다 — 근사 트리밍이 라딕스 트리 노드 경계에서
멈출 수 있게 해 XADD 지연을 상수로 유지한다. 외부 설정화는 이 SPEC 범위 밖."""

SIGNAL_STREAM_FIELDS: tuple[str, ...] = (
    "symbol",
    "horizon",
    "trade_date",
    "trace_id",
    "signal_class",
    "score",
    "confidence",
)
"""REQ-AIF-120이 확정한 7개 키(이 SPEC이 TECHSPEC의 "실제 키 문자열 미정"
상태를 해소). 전부 문자열로 인코딩된다 — `redis_client.py`의
`decode_responses=True`와 대칭이며, collector의 `StringRedisTemplate` 발행
계약(전 필드 문자열)과도 동일하다."""


def signal_stream_key(market: str) -> str:
    """시장별 신호 스트림 키를 만든다 — `stream:signal:{market}`."""
    return f"{SIGNAL_STREAM_PREFIX}:{market}"


def publish_trading_signal(
    client: Any,
    *,
    market: str,
    symbol: str,
    horizon: int,
    trade_date: date,
    trace_id: str,
    signal_class: str,
    score: float,
    confidence: float,
) -> None:
    """신호 1건을 `stream:signal:{market}`에 XADD한다(REQ-AIF-120).

    `horizon`은 `trading_signals`/`signal_price_bands`와 동일한 "D20"/"D60"
    라벨로 인코딩한다(`writer.format_horizon_label()` 재사용) — 소비자
    (NOTIFIER-FILTER-001 소관)가 스트림 필드를 변환 없이 그대로 밴드 조회
    키로 쓸 수 있게 하기 위함이다.
    """
    fields = {
        "symbol": symbol,
        "horizon": format_horizon_label(horizon),
        "trade_date": trade_date.isoformat(),
        "trace_id": trace_id,
        "signal_class": signal_class,
        "score": str(score),
        "confidence": str(confidence),
    }
    client.xadd(
        signal_stream_key(market),
        fields,
        maxlen=SIGNAL_STREAM_MAXLEN,
        approximate=True,
    )

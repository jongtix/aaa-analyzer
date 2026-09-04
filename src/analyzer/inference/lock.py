"""인-플라이트 락 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-060 확장).

XAUTOCLAIM 재소유는 추론 사이클이 idle 임계보다 길어지면 **아직 처리 중인
자기 자신의 메시지**를 되가져올 수 있다. idle 임계는 어떤 값을 고르든 초과될
여지가 남으므로(REQ-AIF-020), 동일 (market, trade_date)에 대한 중복 자식
스폰을 원천 차단하는 최종 방어선은 idle 임계와 독립적인 애플리케이션 레벨
락이다.

TTL은 `InferenceConfig.stream_claim_idle_seconds`와 동일 값을 사용한다 —
자식이 비정상 종료해 명시적 해제가 누락되어도 락이 자연 만료된다.
"""

from datetime import date
from typing import Any

INFLIGHT_LOCK_KEY_PREFIX = "lock:inference:inflight"

_RELEASE_IF_OWNER_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""
"""소유 토큰이 일치할 때만 삭제한다 — TTL 만료 후 다른 프로세스가 재획득한
락을 앞선 소유자가 지워버리는 것을 막는다."""


def inflight_lock_key(market: str, trade_date: date) -> str:
    """(market, trade_date) 단위 락 키를 만든다."""
    return f"{INFLIGHT_LOCK_KEY_PREFIX}:{market}:{trade_date.isoformat()}"


def acquire_inflight_lock(
    client: Any,
    *,
    market: str,
    trade_date: date,
    token: str,
    ttl_seconds: int,
) -> bool:
    """`SET NX EX`로 락을 획득한다. 이미 잡혀 있으면 `False`를 반환한다."""
    acquired = client.set(inflight_lock_key(market, trade_date), token, nx=True, ex=ttl_seconds)
    return bool(acquired)


def release_inflight_lock(
    client: Any,
    *,
    market: str,
    trade_date: date,
    token: str,
) -> None:
    """자신이 획득한 락만 해제한다(토큰 대조 삭제)."""
    client.eval(_RELEASE_IF_OWNER_SCRIPT, 1, inflight_lock_key(market, trade_date), token)

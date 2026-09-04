"""인-플라이트 락 명세 테스트 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-060 확장).

XAUTOCLAIM 자기재소유가 아직 처리 중인 자기 자신의 메시지를 되가져올 수
있으므로, idle 임계 튜닝만으로는 동일 (market, trade_date)에 대한 중복 자식
스폰을 원천 배제할 수 없다. 최종 방어선은 Redis `SET NX EX` 인-플라이트
락이다.
"""

from datetime import date

from analyzer.inference.lock import (
    INFLIGHT_LOCK_KEY_PREFIX,
    acquire_inflight_lock,
    inflight_lock_key,
    release_inflight_lock,
)


class _FakeRedis:
    """`SET NX EX` + 소유 토큰 대조 삭제(Lua eval)만 흉내내는 페이크."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    def set(self, name, value, nx=False, ex=None):  # noqa: A002 - redis-py 시그니처 계승
        if nx and name in self.values:
            return None
        self.values[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    def eval(self, script, numkeys, *args):
        key, token = args[0], args[1]
        if self.values.get(key) == token:
            del self.values[key]
            return 1
        return 0


class TestInflightLockKey:
    def test_key_is_scoped_by_market_and_trade_date(self):
        key = inflight_lock_key("domestic", date(2026, 9, 3))

        assert key == f"{INFLIGHT_LOCK_KEY_PREFIX}:domestic:2026-09-03"

    def test_distinct_markets_do_not_share_a_key(self):
        assert inflight_lock_key("domestic", date(2026, 9, 3)) != inflight_lock_key(
            "overseas", date(2026, 9, 3)
        )


class TestAcquireInflightLock:
    def test_first_acquisition_succeeds_and_sets_ttl(self):
        client = _FakeRedis()

        acquired = acquire_inflight_lock(
            client, market="domestic", trade_date=date(2026, 9, 3), token="tok-1", ttl_seconds=600
        )

        key = inflight_lock_key("domestic", date(2026, 9, 3))
        assert acquired is True
        assert client.values[key] == "tok-1"
        assert client.expirations[key] == 600

    def test_second_acquisition_of_the_same_key_fails(self):
        """AC-AIF-011 세 번째 시나리오: 동일 (market, trade_date)에 대한 두 번째
        스폰이 락 획득 실패로 차단된다."""
        client = _FakeRedis()
        acquire_inflight_lock(
            client, market="domestic", trade_date=date(2026, 9, 3), token="tok-1", ttl_seconds=600
        )

        acquired = acquire_inflight_lock(
            client, market="domestic", trade_date=date(2026, 9, 3), token="tok-2", ttl_seconds=600
        )

        assert acquired is False

    def test_a_different_trade_date_is_not_blocked(self):
        client = _FakeRedis()
        acquire_inflight_lock(
            client, market="domestic", trade_date=date(2026, 9, 3), token="tok-1", ttl_seconds=600
        )

        acquired = acquire_inflight_lock(
            client, market="domestic", trade_date=date(2026, 9, 4), token="tok-2", ttl_seconds=600
        )

        assert acquired is True


class TestReleaseInflightLock:
    def test_owner_releases_its_own_lock(self):
        client = _FakeRedis()
        acquire_inflight_lock(
            client, market="domestic", trade_date=date(2026, 9, 3), token="tok-1", ttl_seconds=600
        )

        release_inflight_lock(client, market="domestic", trade_date=date(2026, 9, 3), token="tok-1")

        assert client.values == {}

    def test_non_owner_cannot_release_another_holders_lock(self):
        """TTL 만료 후 다른 프로세스가 재획득한 락을 앞선 소유자가 지워버리는
        것을 막는다(토큰 대조 삭제)."""
        client = _FakeRedis()
        acquire_inflight_lock(
            client, market="domestic", trade_date=date(2026, 9, 3), token="tok-1", ttl_seconds=600
        )

        release_inflight_lock(
            client, market="domestic", trade_date=date(2026, 9, 3), token="other-token"
        )

        assert client.values[inflight_lock_key("domestic", date(2026, 9, 3))] == "tok-1"

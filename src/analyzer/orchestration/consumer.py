"""`stream:daily:complete` Redis Streams 컨슈머 (SPEC-ANALYZER-INFER-001 M1).

REQ-AIF-020: 그룹 `analyzer`/컨슈머 `analyzer-daily-complete`로 구독한다.
그룹 생성은 `XGROUP CREATE ... $ MKSTREAM`(존재 시 무시)이며, 확인응답 없이
`InferenceConfig.stream_claim_idle_seconds`를 초과한 메시지는 XAUTOCLAIM으로
재소유하고, 재전달 3회를 초과한 메시지는 DLQ로 이관한 뒤 원본에서 XACK한다.

REQ-AIF-021: 이벤트는 `trade_date`를 담지 않으므로 시장별 최신 거래일을 자체
DB 조회로 산출한다. `attempted`/`succeeded`/`skipped` 값은 완전성 판정 근거로
쓰이지 않는다 — 휴장일에도 이벤트가 발행되고 catch-up 재실행으로 하루 2회
도착할 수 있어, 최종 멱등 방어선은 DB INSERT 계층의 UNIQUE 키 스킵이다.

REQ-AIF-010: 자식 프로세스와의 계약면은 종료코드뿐이다. 이 모듈은 자식
stdout을 파싱하지 않는다(정적 grep 검증, AC-AIF-001).

REQ-AIF-060: 자식 스폰 직전 (market, trade_date) 인-플라이트 락을 획득한다 —
XAUTOCLAIM 자기재소유로 인한 중복 자식 스폰을 구조적으로 차단한다.

동기 `redis` 클라이언트를 사용하되 모든 명령을 `asyncio.to_thread()`로 감싸
상주 부모의 이벤트 루프를 막지 않는다.
"""

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from datetime import date
from typing import Any

from redis.exceptions import ResponseError

from analyzer.common.logging import get_logger
from analyzer.common.trace import new_trace_id
from analyzer.data.config import get_db_config
from analyzer.data.repository import build_engine
from analyzer.inference.config import InferenceConfig, get_inference_config
from analyzer.inference.lock import acquire_inflight_lock, release_inflight_lock
from analyzer.inference.redis_client import build_redis_client
from analyzer.inference.spawn import spawn_inference_child
from analyzer.inference.trade_date import resolve_trade_date

logger = get_logger(__name__)

DAILY_COMPLETE_STREAM = "stream:daily:complete"
CONSUMER_GROUP = "analyzer"
CONSUMER_NAME = "analyzer-daily-complete"
DLQ_STREAM = f"stream:dlq:{DAILY_COMPLETE_STREAM}"

MAX_DELIVERY_COUNT = 3
"""REQ-AIF-020: 재전달 횟수가 이 값을 초과한 메시지는 DLQ로 이관한다(무한
재시도 방지)."""

_READ_BATCH_SIZE = 10
_CLAIM_BATCH_SIZE = 100
_DEFAULT_BLOCK_MILLISECONDS = 5_000
_DEFAULT_ERROR_BACKOFF_SECONDS = 5.0

SpawnChild = Callable[..., Coroutine[Any, Any, int]]
TradeDateResolver = Callable[[str], "date | None"]


def _build_default_trade_date_resolver() -> TradeDateResolver:
    """DB 엔진을 최초 이벤트 수신 시점에 지연 생성하는 기본 해석기.

    엔진 생성을 프로세스 기동 시점으로 앞당기지 않는 이유는 `run()`의
    fail-fast 대상(REQ-ATG-002)이 학습 자동화 설정이지 DB 접속이 아니기
    때문이다 — DB 접속 실패는 컨슈머 루프의 재시도 경로로 흡수된다.
    """
    cached: dict[str, Any] = {}

    def _resolve(market: str) -> date | None:
        if "engine" not in cached:
            cached["engine"] = build_engine(get_db_config())
        return resolve_trade_date(cached["engine"], market)

    return _resolve


class StreamConsumer:
    """`stream:daily:complete`를 구독해 시장별 추론 자식 프로세스를 스폰한다.

    @MX:ANCHOR: [AUTO] 추론 서빙 경로의 유일한 이벤트 진입점.
    @MX:REASON: fan_in >= 3 (api/main.py 기동 배선, 자식 스폰, DLQ 이관 경로).
    """

    def __init__(
        self,
        *,
        config: InferenceConfig | None = None,
        client: Any | None = None,
        spawn_child: SpawnChild | None = None,
        resolve_trade_date: TradeDateResolver | None = None,
        block_milliseconds: int = _DEFAULT_BLOCK_MILLISECONDS,
        error_backoff_seconds: float = _DEFAULT_ERROR_BACKOFF_SECONDS,
    ) -> None:
        self._config = config
        self._client = client
        self._spawn_child: SpawnChild = spawn_child or spawn_inference_child
        self._resolve_trade_date: TradeDateResolver = (
            resolve_trade_date or _build_default_trade_date_resolver()
        )
        self.block_milliseconds = block_milliseconds
        self.error_backoff_seconds = error_backoff_seconds
        self._group_ready = False

    async def start(self) -> None:
        """취소될 때까지 폴링을 반복한다.

        일시적 오류(Redis 두절, DB 조회 실패 등)는 백오프 후 재시도한다 —
        상주 부모의 다른 배선(FastAPI, APScheduler)을 죽이지 않는다.
        """
        logger.info(
            "stream consumer starting stream=%s group=%s consumer=%s",
            DAILY_COMPLETE_STREAM,
            CONSUMER_GROUP,
            CONSUMER_NAME,
        )
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                logger.info("stream consumer cancelled")
                raise
            except Exception:
                logger.error("stream consumer poll failed", exc_info=True)
                await asyncio.sleep(self.error_backoff_seconds)

    async def poll_once(self) -> int:
        """재소유 대상 → 신규 메시지 순으로 1회 처리하고 스폰 건수를 반환한다."""
        client = self._get_client()
        await asyncio.to_thread(self._ensure_group, client)

        handled = 0
        for message_id, fields in await asyncio.to_thread(self._claim_stale_messages, client):
            handled += await self._handle_message(client, message_id, fields)
        for message_id, fields in await asyncio.to_thread(self._read_new_messages, client):
            handled += await self._handle_message(client, message_id, fields)
        return handled

    def _get_config(self) -> InferenceConfig:
        if self._config is None:
            self._config = get_inference_config()
        return self._config

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = build_redis_client(self._get_config())
        return self._client

    def _ensure_group(self, client: Any) -> None:
        """컨슈머 그룹을 스트림 말미(`$`)에서 생성한다. 이미 있으면 무시한다.

        `$`는 그룹 생성 시점 이후의 메시지부터 받겠다는 뜻이며, 재기동 시에는
        `BUSYGROUP`으로 생성이 무시되므로 기존 그룹의 마지막 미처리 지점부터
        재개된다 — 재기동이 오프셋을 되감지 않는다.
        """
        if self._group_ready:
            return
        try:
            client.xgroup_create(DAILY_COMPLETE_STREAM, CONSUMER_GROUP, id="$", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._group_ready = True

    def _claim_stale_messages(self, client: Any) -> list[tuple[str, dict[str, str]]]:
        """idle 임계를 초과한 미확인 메시지를 재소유하고, 재전달 한도를 넘긴
        메시지는 DLQ로 보낸다.

        재전달 횟수는 XAUTOCLAIM **이전에** XPENDING으로 읽는다 — XAUTOCLAIM
        자체가 재전달 횟수를 1 증가시키므로 이후에 읽으면 한도 판정이 한 칸씩
        밀린다.
        """
        idle_milliseconds = self._get_config().stream_claim_idle_seconds * 1000

        over_limit = {
            str(entry["message_id"])
            for entry in client.xpending_range(
                DAILY_COMPLETE_STREAM,
                CONSUMER_GROUP,
                min="-",
                max="+",
                count=_CLAIM_BATCH_SIZE,
                idle=idle_milliseconds,
            )
            if int(entry["times_delivered"]) > MAX_DELIVERY_COUNT
        }

        result = client.xautoclaim(
            DAILY_COMPLETE_STREAM,
            CONSUMER_GROUP,
            CONSUMER_NAME,
            min_idle_time=idle_milliseconds,
            start_id="0-0",
            count=_CLAIM_BATCH_SIZE,
        )
        # XAUTOCLAIM 응답은 Redis 버전에 따라 [cursor, messages] 또는
        # [cursor, messages, deleted]로 길이가 다르다 — 두 번째 요소만 쓴다.
        messages = result[1] if len(result) >= 2 else []

        claimed: list[tuple[str, dict[str, str]]] = []
        for message_id, fields in messages:
            if message_id in over_limit:
                self._route_to_dlq(client, message_id, fields)
            else:
                claimed.append((message_id, fields))
        return claimed

    def _read_new_messages(self, client: Any) -> list[tuple[str, dict[str, str]]]:
        response = client.xreadgroup(
            CONSUMER_GROUP,
            CONSUMER_NAME,
            {DAILY_COMPLETE_STREAM: ">"},
            count=_READ_BATCH_SIZE,
            block=self.block_milliseconds,
        )
        if not response:
            return []

        messages: list[tuple[str, dict[str, str]]] = []
        for _stream_name, entries in response:
            messages.extend(entries)
        return messages

    def _route_to_dlq(self, client: Any, message_id: str, fields: dict[str, str]) -> None:
        payload = dict(fields)
        payload["original_id"] = message_id
        payload["original_stream"] = DAILY_COMPLETE_STREAM
        payload["reason"] = "max_delivery_exceeded"

        client.xadd(DLQ_STREAM, payload)
        client.xack(DAILY_COMPLETE_STREAM, CONSUMER_GROUP, message_id)
        logger.warning(
            "daily-complete 이벤트를 DLQ로 이관했다 id=%s dlq=%s market=%s",
            message_id,
            DLQ_STREAM,
            fields.get("market"),
        )

    async def _handle_message(self, client: Any, message_id: str, fields: dict[str, str]) -> int:
        """이벤트 1건을 처리한다. 자식을 스폰했으면 1, 아니면 0을 반환한다."""
        market = fields.get("market")
        if not market:
            logger.warning("daily-complete 이벤트에 market 필드가 없다 id=%s", message_id)
            await self._ack(client, message_id)
            return 0

        trace_id = new_trace_id()
        trade_date = await asyncio.to_thread(self._resolve_trade_date, market)
        if trade_date is None:
            logger.warning(
                "추론 대상 거래일을 산출할 수 없다 market=%s id=%s trace_id=%s",
                market,
                message_id,
                trace_id,
            )
            await self._ack(client, message_id)
            return 0

        token = uuid.uuid4().hex
        ttl_seconds = self._get_config().stream_claim_idle_seconds
        acquired = await asyncio.to_thread(
            acquire_inflight_lock,
            client,
            market=market,
            trade_date=trade_date,
            token=token,
            ttl_seconds=ttl_seconds,
        )
        if not acquired:
            # 확인응답하지 않는다 — 진행 중인 소유자가 완료 후 XACK한다.
            logger.info(
                "동일 (market, trade_date)의 자식이 이미 진행 중이라 스폰을 건너뛴다 "
                "market=%s trade_date=%s id=%s",
                market,
                trade_date,
                message_id,
            )
            return 0

        try:
            exit_code = await self._spawn_child(market, trace_id=trace_id)
        finally:
            await asyncio.to_thread(
                release_inflight_lock,
                client,
                market=market,
                trade_date=trade_date,
                token=token,
            )

        logger.info(
            "inference child finished market=%s trade_date=%s trace_id=%s exit_code=%d",
            market,
            trade_date,
            trace_id,
            exit_code,
        )
        await self._ack(client, message_id)
        return 1

    async def _ack(self, client: Any, message_id: str) -> None:
        await asyncio.to_thread(client.xack, DAILY_COMPLETE_STREAM, CONSUMER_GROUP, message_id)

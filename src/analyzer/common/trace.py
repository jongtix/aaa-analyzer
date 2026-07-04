"""Trace ID 발급/전파 유틸리티.

REQ-ANALYZER-FOUNDATION-014/015: `common/`은 Trace ID 발급/전파 유틸리티를 제공해야
하며, 트레이싱 컨텍스트 내에서 발생한 로그 레코드는 활성 Trace ID 필드를 포함해야
한다.

`contextvars`를 사용해 동일 컨텍스트에서 생성된 asyncio 태스크(부모 FastAPI 요청
처리, 자식 CLI 호출) 전반에 Trace ID가 올바르게 전파되도록 한다.
"""

import contextvars
import uuid

# @MX:ANCHOR: [AUTO] Trace ID 컨텍스트 변수 — 부모(요청 단위)와 자식(호출 단위)
# Trace ID 전파의 공용 진입점.
# @MX:REASON: fan_in >= 3 예상(api 미들웨어, 로깅 포매터, inference CLI).
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "analyzer_trace_id", default=None
)


def new_trace_id() -> str:
    """새 Trace ID를 발급하고 현재 컨텍스트의 활성 Trace ID로 설정한 뒤 반환한다."""
    trace_id = uuid.uuid4().hex
    _trace_id_var.set(trace_id)
    return trace_id


def get_trace_id() -> str | None:
    """현재 컨텍스트의 활성 Trace ID를 반환한다. 설정되지 않았으면 None을 반환한다."""
    return _trace_id_var.get()


def set_trace_id(trace_id: str) -> contextvars.Token[str | None]:
    """활성 Trace ID를 명시적으로 설정하고, 나중에 복원할 수 있는 reset 토큰을 반환한다."""
    return _trace_id_var.set(trace_id)


def reset_trace_id(token: contextvars.Token[str | None]) -> None:
    """Trace ID 컨텍스트를 해당 set 호출 이전 상태로 복원한다."""
    _trace_id_var.reset(token)

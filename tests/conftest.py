"""aaa-analyzer 단위 테스트용 공유 pytest fixture."""

import pytest

from analyzer.common import trace


@pytest.fixture(autouse=True)
def _reset_trace_context():
    """테스트 실행 순서와 무관하게 각 테스트가 활성 Trace ID 없이 시작하도록 보장한다.

    `analyzer.common.trace`는 모듈 레벨 `contextvars.ContextVar`를 사용해 단일
    실행 컨텍스트 내에서 Trace ID가 올바르게 전파되도록 한다. 이 fixture가 없으면
    한 테스트에서 발급된 Trace ID가 다음 테스트로 새어 들어갈 수 있다.
    """
    token = trace._trace_id_var.set(None)
    try:
        yield
    finally:
        trace._trace_id_var.reset(token)

"""analyzer.common.trace 및 analyzer.common.logging에 대한 명세 테스트.

REQ-ANALYZER-FOUNDATION-012/013/014/015 (구조화 JSON 로깅, KST 타임스탬프,
Trace ID 발급/전파 유틸리티, 활성 Trace ID의 로그 레코드 자동 주입).
"""

import json
import logging

from analyzer.common import trace
from analyzer.common.logging import JsonFormatter, get_logger


class TestTraceIdIssueAndPropagate:
    def test_new_trace_id_returns_nonempty_string(self):
        trace_id = trace.new_trace_id()

        assert isinstance(trace_id, str)
        assert trace_id != ""

    def test_new_trace_id_sets_it_as_current(self):
        trace_id = trace.new_trace_id()

        assert trace.get_trace_id() == trace_id

    def test_new_trace_id_generates_unique_values(self):
        first = trace.new_trace_id()
        second = trace.new_trace_id()

        assert first != second

    def test_get_trace_id_returns_none_outside_context(self):
        token = trace.set_trace_id("explicit-trace-id")
        trace.reset_trace_id(token)

        assert trace.get_trace_id() is None

    def test_set_trace_id_propagates_explicit_value(self):
        token = trace.set_trace_id("known-trace-id")
        try:
            assert trace.get_trace_id() == "known-trace-id"
        finally:
            trace.reset_trace_id(token)


class TestJsonFormatter:
    def _make_record(self, message: str = "hello") -> logging.LogRecord:
        return logging.LogRecord(
            name="analyzer.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=None,
            exc_info=None,
        )

    def test_format_produces_valid_json(self):
        formatter = JsonFormatter()
        record = self._make_record()

        output = formatter.format(record)

        parsed = json.loads(output)
        assert parsed["message"] == "hello"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "analyzer.test"

    def test_format_uses_kst_timestamp_offset(self):
        formatter = JsonFormatter()
        record = self._make_record()

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["ts"].endswith("+09:00")

    def test_format_includes_active_trace_id(self):
        formatter = JsonFormatter()
        token = trace.set_trace_id("trace-abc-123")
        try:
            record = self._make_record()
            output = formatter.format(record)
        finally:
            trace.reset_trace_id(token)

        parsed = json.loads(output)
        assert parsed["trace_id"] == "trace-abc-123"

    def test_format_without_active_trace_id_has_no_real_trace(self):
        formatter = JsonFormatter()
        record = self._make_record()

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed.get("trace_id") is None


class TestGetLogger:
    def test_get_logger_returns_logger_with_json_formatter(self):
        logger = get_logger("analyzer.test.module")

        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0].formatter, JsonFormatter)

    def test_get_logger_is_idempotent_for_same_name(self):
        first = get_logger("analyzer.test.idempotent")
        second = get_logger("analyzer.test.idempotent")

        assert first is second
        assert len(first.handlers) == 1

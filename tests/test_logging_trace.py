"""analyzer.common.trace 및 analyzer.common.logging에 대한 명세 테스트.

REQ-ANALYZER-FOUNDATION-012/013/014/015 (구조화 JSON 로깅, KST 타임스탬프,
Trace ID 발급/전파 유틸리티, 활성 Trace ID의 로그 레코드 자동 주입).
REQ-LOGS-001~006 (SPEC-OBSV-LOGS-002): stdout에 더해 회전 파일 sink에도
동일한 JSON 형태로 영속 기록하고, 초기화/기록 실패 시 fail-open으로
stdout 방출을 유지한다.
"""

import io
import json
import logging
import uuid
from logging.handlers import RotatingFileHandler

import pytest

from analyzer.common import logging as logging_module
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
        assert len(first.handlers) == len(second.handlers)

    def test_get_logger_keeps_stdout_handler(self):
        """REQ-LOGS-003: 파일 sink 추가와 무관하게 stdout 방출은 유지된다."""
        logger = get_logger("analyzer.test.stdout_kept")

        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        ]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].stream is __import__("sys").stdout


class TestFileSink:
    """REQ-LOGS-001/002/004/005: 파일 sink 부착, 동일 JSON 형태 보존, 경로/회전값."""

    @pytest.fixture(autouse=True)
    def _isolate_logger_cache(self):
        """이름 기반 로거 캐시가 테스트 간 오염되지 않도록 매 테스트 초기화한다."""
        logging_module._configured_loggers.clear()
        yield
        logging_module._configured_loggers.clear()

    def _unique_name(self) -> str:
        return f"analyzer.test.filesink.{uuid.uuid4().hex}"

    def test_file_handler_attached_and_writes_same_json_shape(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_PATH", str(tmp_path))
        logger = get_logger(self._unique_name())

        file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1

        logger.info("hello file sink")
        for handler in logger.handlers:
            handler.flush()

        log_file = tmp_path / "aaa-analyzer.log"
        assert log_file.exists()

        lines = [line for line in log_file.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["message"] == "hello file sink"
        assert parsed["level"] == "INFO"
        assert set(parsed.keys()) == {"ts", "level", "logger", "message", "trace_id"}

    def test_file_handler_reuses_same_formatter_instance_as_stdout(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_PATH", str(tmp_path))
        logger = get_logger(self._unique_name())

        formatters = {handler.formatter for handler in logger.handlers}
        assert len(formatters) == 1

    def test_log_path_env_var_overrides_default_directory(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom-log-dir"
        custom_dir.mkdir()
        monkeypatch.setenv("LOG_PATH", str(custom_dir))

        logger = get_logger(self._unique_name())
        logger.info("override check")
        for handler in logger.handlers:
            handler.flush()

        assert (custom_dir / "aaa-analyzer.log").exists()

    def test_rotation_constants_are_fixed_and_not_env_overridable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_PATH", str(tmp_path))
        monkeypatch.setenv("LOG_MAX_BYTES", "1")
        monkeypatch.setenv("LOG_BACKUP_COUNT", "1")

        logger = get_logger(self._unique_name())
        file_handler = next(h for h in logger.handlers if isinstance(h, RotatingFileHandler))

        assert file_handler.maxBytes == 10 * 1024 * 1024
        assert file_handler.backupCount == 5


class TestFileSinkFailOpen:
    """REQ-LOGS-006: 초기화 실패/기록 실패 시에도 analyzer는 중단되지 않고 stdout은 유지된다."""

    @pytest.fixture(autouse=True)
    def _isolate_logger_cache(self):
        logging_module._configured_loggers.clear()
        yield
        logging_module._configured_loggers.clear()

    def _unique_name(self) -> str:
        return f"analyzer.test.failopen.{uuid.uuid4().hex}"

    def test_init_time_failure_falls_back_to_stdout_only(self, tmp_path, monkeypatch):
        """로그 디렉토리가 존재하지 않으면(마운트 누락 등) 파일 핸들러 없이 계속 동작한다."""
        missing_dir = tmp_path / "not-mounted" / "nested"
        monkeypatch.setenv("LOG_PATH", str(missing_dir))

        logger = get_logger(self._unique_name())

        file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert file_handlers == []
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        ]
        assert len(stream_handlers) == 1

        logger.info("should not raise")

    def test_write_time_failure_does_not_propagate(self, tmp_path, monkeypatch):
        """표준 RotatingFileHandler는 emit() 중 쓰기 실패를 handleError()로 흡수한다(pinning).

        BaseRotatingHandler.emit()은 rollover+write를 try/except Exception으로 감싸고,
        Handler.handleError()는 예외를 재전파하지 않는다(CPython 표준 동작). 이 테스트는
        해당 표준 라이브러리 동작을 고정(pin)하는 목적이며 별도 방어 코드가 필요하지 않다.
        """
        monkeypatch.setenv("LOG_PATH", str(tmp_path))
        logger = get_logger(self._unique_name())

        file_handler = next(h for h in logger.handlers if isinstance(h, RotatingFileHandler))

        class _BrokenStream(io.TextIOBase):
            def write(self, *_args, **_kwargs):
                raise OSError("disk full")

            def flush(self):
                raise OSError("disk full")

        file_handler.stream = _BrokenStream()  # type: ignore[assignment]

        # Act & Assert — 예외가 전파되지 않아야 fail-open이 성립한다.
        logger.info("this should not raise even though the file stream is broken")

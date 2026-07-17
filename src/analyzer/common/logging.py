"""KST 타임스탬프와 Trace ID 자동 주입을 지원하는 구조화 JSON 로깅.

REQ-ANALYZER-FOUNDATION-012/013/015: 로그는 구조화 JSON이어야 하고, 타임스탬프는
KST 타임존을 사용해야 하며, 트레이싱 컨텍스트 내에서 발생한 로그 레코드는
활성 Trace ID 필드를 포함해야 한다.

REQ-LOGS-001~006 (SPEC-OBSV-LOGS-002): stdout에 더해 회전 파일 sink에도 동일한
JSON 형태로 영속 기록한다. 파일 sink 초기화(부착 시점) 또는 기록(런타임)이
실패해도 analyzer는 중단되지 않고 stdout 방출을 계속한다(fail-open).
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

from analyzer.common.trace import get_trace_id

_KST = ZoneInfo("Asia/Seoul")

_LOG_FILENAME = "aaa-analyzer.log"
_DEFAULT_LOG_PATH = "/var/log/aaa-analyzer"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

_configured_loggers: dict[str, logging.Logger] = {}


class JsonFormatter(logging.Formatter):
    """로그 레코드를 KST 타임스탬프와 Trace ID를 포함한 한 줄 JSON으로 포맷한다."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=_KST).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace_id(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _create_file_handler(formatter: logging.Formatter) -> logging.Handler | None:
    """회전 파일 핸들러를 생성한다. 초기화에 실패하면 None을 반환한다(fail-open).

    REQ-LOGS-004/005: 경로는 `LOG_PATH` 환경변수(기본값 `/var/log/aaa-analyzer`)에
    `aaa-analyzer.log`를 결합해 구성하고, maxBytes/backupCount는 코드 상수로 고정한다.
    REQ-LOGS-006: 로그 디렉토리 미마운트 등으로 핸들러 생성 자체가 실패하면(OSError)
    파일 핸들러 없이 stdout만으로 계속 동작할 수 있도록 예외를 흡수한다.
    """
    log_path = os.environ.get("LOG_PATH", _DEFAULT_LOG_PATH)
    log_file = os.path.join(log_path, _LOG_FILENAME)
    try:
        handler = RotatingFileHandler(log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT)
    except OSError:
        return None

    handler.setFormatter(formatter)
    return handler


def get_logger(name: str) -> logging.Logger:
    """JSON 포매터가 구성된 로거를 반환한다. name당 한 번만 생성한다.

    REQ-LOGS-001/002/003: stdout 방출을 유지하면서 동일 JsonFormatter 인스턴스를
    공유하는 회전 파일 sink를 추가로 부착한다.
    """
    if name in _configured_loggers:
        return _configured_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = JsonFormatter()

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = _create_file_handler(formatter)
    if file_handler is not None:
        logger.addHandler(file_handler)

    _configured_loggers[name] = logger
    return logger

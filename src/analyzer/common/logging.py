"""KST 타임스탬프와 Trace ID 자동 주입을 지원하는 구조화 JSON 로깅.

REQ-ANALYZER-FOUNDATION-012/013/015: 로그는 구조화 JSON이어야 하고, 타임스탬프는
KST 타임존을 사용해야 하며, 트레이싱 컨텍스트 내에서 발생한 로그 레코드는
활성 Trace ID 필드를 포함해야 한다.
"""

import json
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from analyzer.common.trace import get_trace_id

_KST = ZoneInfo("Asia/Seoul")

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


def get_logger(name: str) -> logging.Logger:
    """JSON 포매터가 구성된 로거를 반환한다. name당 한 번만 생성한다."""
    if name in _configured_loggers:
        return _configured_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    _configured_loggers[name] = logger
    return logger

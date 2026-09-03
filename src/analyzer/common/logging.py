"""KST 타임스탬프와 Trace ID 자동 주입을 지원하는 구조화 JSON 로깅.

REQ-ANALYZER-FOUNDATION-012/013/015: 로그는 구조화 JSON이어야 하고, 타임스탬프는
KST 타임존을 사용해야 하며, 트레이싱 컨텍스트 내에서 발생한 로그 레코드는
활성 Trace ID 필드를 포함해야 한다.

REQ-LOGS-001~006 (SPEC-OBSV-LOGS-002): stdout에 더해 회전 파일 sink에도 동일한
JSON 형태로 영속 기록한다. 파일 sink 초기화(부착 시점) 또는 기록(런타임)이
실패해도 analyzer는 중단되지 않고 stdout 방출을 계속한다(fail-open).

collector/notifier(logback ECS 스키마, `@timestamp`/`log.level`/`service.name`)와
달리 이 파일 sink는 analyzer native JSON 형태(`ts`/`level`/`logger`/`message`/
`trace_id`)를 그대로 디스크에 보존한다 — Java(logback)와 Python(표준 `logging`)은
서로 다른 언어/서비스라 스키마를 강제로 맞추지 않는다(SPEC-OBSV-LOGS-002 §3
Exclusion #1, REQ-LOGS-002). VictoriaLogs 조회 시의 필드 정합은 vector의
in-flight transform(DP-2)이 담당하며 디스크 파일은 건드리지 않는다.

회전도 압축이 없다 — logback `SizeAndTimeBasedRollingPolicy`는 회전 시 gzip
압축(`*.log.gz`)을 무상 제공하지만, Python 표준 `RotatingFileHandler`는 압축을
지원하지 않아 회전본이 `aaa-analyzer.log.1`처럼 비압축 상태로 남는다. 호스트
logrotate 등으로 압축을 별도 구현하는 방안은 read-only 컨테이너 내 cron 부재 +
신규 운영 부담 문제로 기각했다(DP-1 B안 기각 사유). 대신 analyzer는 스캐폴딩·
저볼륨 단계임을 고려해 상한 자체를 낮게 잡았다 — maxBytes 10 MiB × backupCount 5
= 약 60 MiB(collector/notifier의 파일당 1GB·총 50GB cap 대비 훨씬 보수적).
압축이 필요해지면(로그량 증가 시) 별도 SPEC/ADR 후보다.

총량 상한(total-size-cap) 패리티 판단 (SPEC-OBSV-LOGS-003 REQ-007/008, DP-2)
---------------------------------------------------------------------------
ADR-011이 "Phase 2에서 별도 ADR로 결정"하겠다고 남겼던 판단을 여기에 기록한다.
결론은 **신규 코드 메커니즘을 도입하지 않는 문서화 등가성**이며 근거는 셋이다.

1. **결정론적 총량 상한**: `RotatingFileHandler(maxBytes=10 MiB,
   backupCount=5)`는 기록량과 무관하게 활성 파일 1개 + 회전본 최대 5개(총 6개)
   만 유지하므로, 총 디스크 사용량은 항상 약 60 MiB 이하로 유계다. Python
   stdlib에는 logback `total-size-cap` 상당의 API가 없지만, 개수 상한이 이미
   볼륨 무관 산술 상한을 제공하므로 별도 총량 관리 코드가 불필요하다.

2. **"동등한 의도" 주장의 범위 한정**: 이 비교는 **디스크 사용량이 무한정
   증가하지 않는다**는 의도 하나에 대해서만 성립하며, logback의 사고 조사용
   보존 기간과의 동등성은 **함의하지 않는다**. collector/notifier는
   `max-history: 30`일(연령 상한, 1차 보존 메커니즘)과 `total-size-cap: 50GB`
   (드물게 발동하는 백스톱)의 이중 안전망을 갖추며 그 목적은 30일 조사 창을
   확보하는 것이다. `RotatingFileHandler`에는 대응하는 연령 기반 보존
   메커니즘이 전혀 없고, ~60 MiB 상한은 logback 50GB 대비 약 850배 작다 —
   analyzer 전체를 logback 이중 정책의 백스톱 절반과만 비교해 "더 강하다"고
   말하는 것은 진정한 패리티 비교가 아니다.

3. **예상 보존 기간(정성적 추정, 정밀 산정 아님)**: 라이브 NAS 실측
   (2026-09-03)에서 활성 `aaa-analyzer.log`는 81KB로 단일 회전 상한(10 MiB)의
   약 0.8%였다 — 현재 저볼륨 스캐폴딩 단계의 로깅 속도로는 회전이 자주
   발생하지 않아 상한 도달까지 상당한 여유가 있는 것으로 관측된다. 다만 이는
   파일이 누적된 기간(마지막 재시작/배포 시점)을 알 수 없는 단일 시점
   스냅샷이므로, 정확한 보존 기간(며칠/몇 주/몇 개월)은 배포·재시작 이력에
   따라 달라지며 이 관측만으로 정밀 산정되지 않았다.

**재검토 조건**: (a) analyzer 로그 볼륨이 크게 증가해 회전 주기가 분 단위로
짧아지는 등 60 MiB 상한이 관측 가능한 히스토리를 과도하게 짧게 만드는 경우,
(b) 위 보존 기간 추정치가 단일 시점 스냅샷 기반이라는 한계가 실제 조사에서
문제가 되는 경우 — 커스텀 총량 관리 또는 `backupCount` 상향을 별도 SPEC으로
재검토한다. 그때 신규 코드 메커니즘을 도입하더라도 REQ-LOGS-006이 확립한
fail-open 원칙을 그대로 따른다(그 메커니즘의 실패가 stdout/파일 로깅을
중단시켜서는 안 된다).
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
        # SPEC-ANALYZER-TRAIN-OBSV-001 REQ-ATO-002/018/019/021: 호출부가
        # `extra={"stage_marker": True}`로 명시한 레코드만 이 필드를 payload에
        # 포함한다 — 원격 채널 드레인 릴레이가 이 필드로 저볼륨 단계 전이
        # 로그만 걸러 릴레이한다(D-NEW-1). 지정하지 않은 기존 호출은 필드
        # 자체가 없어 하위 호환된다(기존 exact-key-set 테스트 불변).
        stage_marker = getattr(record, "stage_marker", None)
        if stage_marker is not None:
            payload["stage_marker"] = stage_marker
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

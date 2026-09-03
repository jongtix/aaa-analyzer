"""트레이너 로그(`trainer_{run_id}.log`) 개수 기반 보존 정책.

SPEC-OBSV-LOGS-003 REQ-001~006. 원격 SSH 학습 디스패치는 stdout/stderr를
`tee`로 `trainer_{run_id}.log`에 기록하는데(SPEC-ANALYZER-TRAIN-AUTOMATION-001),
실행마다 run_id로 명명된 **새 파일**이 생기고 셸 `tee`에는 회전이 없어 어떤
정리 로직도 없이 무한 증가한다(라이브 NAS 실측 2026-09-03: 파일 4개, 38MB×2).
`training/persistence.py`의 "활성 개수 기준 정리" 패턴을 로그 파일에 미러링해
최신 N개만 남긴다.

**glob 대상 경로(REQ-001, spec.md §4 DP-1 v0.2.0 정정)**: sweep은
`common/logging.py`가 이미 쓰는 `LOG_PATH` 환경변수(기본값
`/var/log/aaa-analyzer`) — analyzer 컨테이너 **자신의 로컬** 로그 디렉토리 —
를 대상으로 한다. `TRAIN_AUTOMATION_TRAINER_LOG_BASE_DIR`는 사용하지 않는다:
그 값은 맥북상 SMB 마운트 지점 하위의 macOS 절대경로 문자열이며 원격 SSH 명령
문자열 조립에만 쓰인다(analyzer 프로세스 자신은 로컬 경로로 읽지 않는다).
컨테이너 안에서 그 값을 그대로 `Path().glob()`에 넘기면 존재하지 않는 경로를
대상으로 하여 예외 없이 빈 이터레이터를 반환하고 영구히 조용히 무동작한다.
원격에서 tee로 기록된 트레이너 로그와 컨테이너 로컬 로그는 물리적으로 같은
NAS 디스크 경로를 가리키므로, 로컬 `LOG_PATH` glob으로 동일 파일군에 도달한다.

**fail-open(REQ-005)**: sweep 전체가 예외를 삼킨다 — 보존 정리 실패가 학습
디스패치를 실패시키거나 프로세스를 중단시켜서는 안 된다.
"""

import os
from pathlib import Path

from analyzer.common.logging import get_logger

_logger = get_logger(__name__)

_DEFAULT_LOG_PATH = "/var/log/aaa-analyzer"
_TRAINER_LOG_GLOB = "trainer_*.log"
_RETENTION_COUNT_ENV = "TRAIN_AUTOMATION_TRAINER_LOG_RETENTION_COUNT"

DEFAULT_TRAINER_LOG_RETENTION_COUNT = 10


def _resolve_retention_count() -> int:
    raw = os.environ.get(_RETENTION_COUNT_ENV)
    if raw is None:
        return DEFAULT_TRAINER_LOG_RETENTION_COUNT
    try:
        count = int(raw)
    except ValueError:
        _logger.warning(
            "trainer log retention count env is not an integer, falling back to default value=%r",
            raw,
        )
        return DEFAULT_TRAINER_LOG_RETENTION_COUNT
    if count < 0:
        _logger.warning(
            "trainer log retention count env is negative, falling back to default value=%r", raw
        )
        return DEFAULT_TRAINER_LOG_RETENTION_COUNT
    return count


def sweep_trainer_logs(*, current_run_id: str) -> None:
    """로컬 로그 디렉토리의 `trainer_*.log`를 최신 N개만 남기고 삭제한다.

    `current_run_id`는 이 sweep을 유발한 디스패치가 방금 기록한 파일의 run_id다
    — 그 파일은 mtime 순위와 무관하게 항상 삭제 대상에서 제외되고 보존 정원도
    잠식하지 않는다(REQ-004). 추론하지 않고 항상 호출자가 명시 전달한다.

    보존 개수는 `TRAIN_AUTOMATION_TRAINER_LOG_RETENTION_COUNT` 환경변수로
    오버라이드하며 기본값은 10이다(REQ-003, spec.md §4 DP-1).
    """
    try:
        log_dir = Path(os.environ.get("LOG_PATH", _DEFAULT_LOG_PATH))
        current_log_name = f"trainer_{current_run_id}.log"
        candidates = [
            path for path in log_dir.glob(_TRAINER_LOG_GLOB) if path.name != current_log_name
        ]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)

        for stale in candidates[_resolve_retention_count() :]:
            # 한 파일의 삭제 실패(권한 오류 등)가 나머지 초과분 정리를 영구히
            # 막으면 본 SPEC이 없애려는 무한 증가가 그대로 되살아나므로,
            # 파일 단위로도 예외를 흡수하고 다음 후보로 진행한다.
            try:
                stale.unlink()
            except OSError:
                _logger.error("trainer log retention delete failed path=%s", stale, exc_info=True)
                continue
            _logger.info("trainer log retention removed path=%s", stale)
    except Exception:
        _logger.error("trainer log retention sweep failed", exc_info=True)

"""NAS 측 게이트 어댑터 — `promotion_gate_fn` 어댑터 팩토리 (SPEC-ANALYZER-TRAIN-GATE-001
M4, REQ-ATG-007/008/012).

게이트 본체(챌린저 학습 + 홀드아웃 평가 + 챔피언 재채점 + 매니페스트 갱신)는
맥 원격에서 실행되어야 한다(REQ-ATG-007) — NAS 컨테이너는 SSH로 맥 측 게이트
CLI(`training/gate.py`)를 실행하고 stdout verdict JSON을 역직렬화해
`runner.py`의 기존 `promotion_gate_fn` 훅 계약을 만족하는 클로저를 반환하는
어댑터 팩토리만 담당한다.

E-1(REQ-ATG-012): 게이트 실행 실패 5종(SSH 연결 실패/CLI 비정상 종료코드/
타임아웃/stdout JSON 파싱 실패/verdict 역직렬화 실패) 중 어느 하나가
발생하면 `metrics.record_failure(stage="promotion_gate")`를 **직접 호출**한
뒤 예외를 재발생시킨다 — `handle_training_run_failure()`(failure.py)는
경유하지 않는다(`FailureStage` Literal이 "promotion_gate"를 포함하지 않으며,
failure.py는 PRESERVE 무수정 유지).
"""

import json
import os
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

from analyzer.common.logging import get_logger
from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.promotion_gate import PromotionVerdict
from analyzer.orchestration.ssh_dispatch import SshConnection, connect_with_retry
from analyzer.training.gate import deserialize_verdicts

logger = get_logger(__name__)

GATE_TIMEOUT_ENV_VAR = "TRAIN_AUTOMATION_GATE_TIMEOUT_SECONDS"
GATE_TIMEOUT_SECONDS_DEFAULT: float = 2 * 60 * 60
"""REQ-ATG-008: 게이트 원격 실행 상한 시간 — 기존 주간 4h/월간 36h와 별도의
이름 있는 설정 상수(초기값 2시간, REVISABLE — M7 라이브 검증에서 실측 후
재조정, §B 리스크 5). 환경변수 오버라이드를 지원한다."""

DATA_AS_OF_OFFSET_DAYS: int = 1
"""REQ-ATG-005: data_as_of 산출 오프셋(발화일 전일=1) — 이름 있는 상수(하드코딩 금지)."""


def resolve_gate_timeout_seconds() -> float:
    """REQ-ATG-008: 환경변수 오버라이드를 지원하는 게이트 타임아웃 상수 조회."""
    return float(os.environ.get(GATE_TIMEOUT_ENV_VAR, str(GATE_TIMEOUT_SECONDS_DEFAULT)))


def compute_data_as_of(fire_time: date) -> date:
    """REQ-ATG-005: 발화일 전일(달력일 -1) 고정 — 거래 캘린더 조회를 수행하지
    않는다(사용자 확정)."""
    return fire_time - timedelta(days=DATA_AS_OF_OFFSET_DAYS)


class GatePromotionFailure(RuntimeError):
    """REQ-ATG-012(E-1): 게이트 실행 실패 5종 공통 예외 — 항상
    `metrics.record_failure(stage="promotion_gate")` 직접 호출 이후 발생한다."""


def build_gate_remote_command(
    *,
    active_models_root: Path,
    cache_dir: Path,
    data_as_of: date,
    feature_code_version: str,
    calendar_code: str,
    merged_to_active: bool,
    mount_script_path: Path,
    db_tunnel_host: str,
    db_tunnel_key_path: Path,
    python_executable_path: Path,
    mysql_database: str,
    mysql_trainer_password: str,
    db_tunnel_username: str = "db_tunnel",
    db_tunnel_port: int = 22,
    db_tunnel_local_port: int = 3306,
    db_tunnel_remote_port: int = 3306,
) -> str:
    """REQ-ATG-008: 게이트 CLI(`python -m analyzer.training.gate`) 원격 호출
    명령을 조립한다 — `ssh_dispatch.build_remote_dispatch_command()`와 동일한
    DB 터널 수립 + 멱등 SMB 마운트 확인 + 터널 해제(trap) 패턴을 게이트
    CLI 인자에 맞춰 재구성한다(신규 파일이므로 ssh_dispatch.py 무수정 유지)."""
    quoted_db_tunnel_key_path = shlex.quote(str(db_tunnel_key_path))
    quoted_db_tunnel_username = shlex.quote(db_tunnel_username)
    quoted_db_tunnel_host = shlex.quote(db_tunnel_host)
    quoted_calendar_code = shlex.quote(calendar_code)
    quoted_cache_dir = shlex.quote(str(cache_dir))
    quoted_active_models_root = shlex.quote(str(active_models_root))
    quoted_data_as_of = shlex.quote(data_as_of.isoformat())
    quoted_feature_code_version = shlex.quote(feature_code_version)
    quoted_mount_script_path = shlex.quote(str(mount_script_path))
    quoted_python_executable_path = shlex.quote(str(python_executable_path))
    quoted_mysql_database = shlex.quote(mysql_database)
    quoted_mysql_trainer_password = shlex.quote(mysql_trainer_password)
    merged_flag = "--merged-to-active" if merged_to_active else "--no-merged-to-active"

    tunnel_command = (
        f"ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes "
        f"-i {quoted_db_tunnel_key_path} "
        f"-p {db_tunnel_port} "
        f"-L {db_tunnel_local_port}:127.0.0.1:{db_tunnel_remote_port} "
        f"{quoted_db_tunnel_username}@{quoted_db_tunnel_host}"
    )
    mount_command = quoted_mount_script_path
    gate_command = (
        f"MYSQL_HOST=127.0.0.1 MYSQL_PORT={db_tunnel_local_port} "
        f"MYSQL_DATABASE={quoted_mysql_database} "
        f"MYSQL_TRAINER_PASSWORD={quoted_mysql_trainer_password} "
        f"{quoted_python_executable_path} -m analyzer.training.gate "
        f"--models-root {quoted_active_models_root} "
        f"--cache-dir {quoted_cache_dir} "
        f"--data-as-of {quoted_data_as_of} "
        f"--feature-code-version {quoted_feature_code_version} "
        f"--calendar-code {quoted_calendar_code} "
        f"{merged_flag}"
    )

    tunnel_pattern = f"{db_tunnel_local_port}:127.0.0.1:{db_tunnel_remote_port}"
    return (
        f"set -o pipefail; "
        f"{tunnel_command}; "
        f"TUNNEL_PID=$(pgrep -f '{tunnel_pattern}'); "
        f"trap 'kill $TUNNEL_PID 2>/dev/null' EXIT; "
        f"{mount_command} && {gate_command}; "
        f"exit $?"
    )


def _extract_last_valid_json_line(lines: Sequence[str]) -> str | None:
    """§B 리스크 2: stdout/stderr 혼입 가능성에 대한 관용 파싱 — 수집된
    라인 중 유효한 JSON 문서로 파싱되는 **마지막** 라인을 verdict 페이로드로
    채택한다(게이트 CLI의 진행 로그가 동일 콜백으로 섞여 들어와도 견고)."""
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            continue
        return stripped
    return None


def build_gate_promotion_fn(
    *,
    config: AutomationConfig,
    metrics: TrainingMetrics,
    data_as_of: date,
    connection_factory: Callable[[], SshConnection],
    timeout_seconds: float | None = None,
    ssh_max_retries: int = 6,
    ssh_retry_interval_seconds: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Callable[[bool], Mapping[tuple[str, int, str], PromotionVerdict]]:
    """REQ-ATG-007: `promotion_gate_fn` 어댑터 팩토리 — `runner.py`의 기존
    훅 계약(`Callable[[bool], Mapping[tuple[str,int,str], PromotionVerdict]]`)을
    그대로 만족하는 클로저를 반환한다.

    §B 리스크 4: `promotion_gate_fn(merged_to_active)` 시그니처에는 연결이
    없으므로, 어댑터는 `manual_run`과 동일한 `connection_factory` 클로저로
    **자체 SSH 연결**을 새로 연다(runner 보유 연결 재사용 불가).
    """
    effective_timeout = (
        timeout_seconds if timeout_seconds is not None else resolve_gate_timeout_seconds()
    )

    def _promotion_gate_fn(
        merged_to_active: bool,
    ) -> Mapping[tuple[str, int, str], PromotionVerdict]:
        connection = connection_factory()
        connected = connect_with_retry(
            connection,
            max_retries=ssh_max_retries,
            interval_seconds=ssh_retry_interval_seconds,
            sleep_fn=sleep_fn,
        )
        if not connected:
            metrics.record_failure(stage="promotion_gate")
            raise GatePromotionFailure(f"게이트 SSH 연결 {ssh_max_retries}회 재시도 실패")

        try:
            command = build_gate_remote_command(
                active_models_root=config.active_models_root,
                cache_dir=config.cache_dir,
                data_as_of=data_as_of,
                feature_code_version=config.feature_code_version,
                calendar_code=config.calendar_code,
                merged_to_active=merged_to_active,
                mount_script_path=config.mount_script_path,
                db_tunnel_host=config.db_tunnel_host,
                db_tunnel_key_path=config.db_tunnel_private_key_path,
                python_executable_path=config.python_executable_path,
                mysql_database=config.mysql_database,
                mysql_trainer_password=config.mysql_trainer_password,
                db_tunnel_username=config.db_tunnel_username,
                db_tunnel_port=config.db_tunnel_port,
                db_tunnel_local_port=config.db_tunnel_local_port,
                db_tunnel_remote_port=config.db_tunnel_remote_port,
            )
            captured_lines: list[str] = []
            result = connection.exec_command(
                command,
                timeout_seconds=effective_timeout,
                on_output_line=captured_lines.append,
            )

            if result.timed_out:
                metrics.record_failure(stage="promotion_gate")
                raise GatePromotionFailure(f"게이트 원격 실행 {effective_timeout}초 타임아웃 초과")

            if result.exit_code != 0:
                metrics.record_failure(stage="promotion_gate")
                raise GatePromotionFailure(f"게이트 CLI 비정상 종료코드 {result.exit_code}")

            raw_json = _extract_last_valid_json_line(captured_lines)
            if raw_json is None:
                logger.error("gate stdout JSON parse failed raw_output=%s", captured_lines)
                metrics.record_failure(stage="promotion_gate")
                raise GatePromotionFailure("게이트 stdout JSON 파싱 실패")

            try:
                return deserialize_verdicts(raw_json)
            except (KeyError, ValueError, TypeError) as exc:
                logger.error("gate verdict deserialize failed error=%s raw=%s", exc, raw_json)
                metrics.record_failure(stage="promotion_gate")
                raise GatePromotionFailure(f"verdict 역직렬화 실패: {exc}") from exc
        finally:
            connection.close()

    return _promotion_gate_fn

"""월간 원격 캠페인 후처리 훅 — SSH 디스패치 → 종료코드 해석 → 성공/실패 경로 분기.

SPEC-ANALYZER-TRAIN-TUNING-001 §2.6, REQ-ATT-013/014/015/017.

**REQ-ATT-013(정책 반전)**: 월간 콜백은 `runner.execute_scheduled_training_run()`을
호출하지 않는다(shall not) — 대신 `ssh_dispatch.build_remote_campaign_dispatch_command()`
(REQ-ATT-006)로 원격 캠페인 CLI를 직접 SSH 디스패치하고, `promote_staging_to_active()`나
`promotion_gate_fn` 승격 꼬리(runner.py:221-270)를 경유하지 않는다(shall) — 원격
캠페인 프로세스(`analyzer.training.campaign`)가 종료 전에 이미
`activate_market_horizon_combo()`를 통해 자신의 모델 활성화를 완료했기 때문이다.

**REQ-ATT-014(실패 경로)**: 원격 캠페인 프로세스가 0이 아닌 종료코드로 끝나거나,
타임아웃되거나, SSH 연결이 실패하면 `metrics.record_failure(stage="monthly_tuning")`를
**직접 호출**한다(`metrics.py`의 `stage: str` 시그니처가 신규 값을 무수정 수용) —
`orchestration/failure.py`의 `handle_training_run_failure()`는 경유하지 않는다(shall not,
그 모듈의 `FailureStage` Literal이 "monthly_tuning"을 포함하지 않으며 이 SPEC의
PRESERVE 대상이다). 실패는 항상 `MonthlyCampaignRunError`로 재발생(재-raise)되며
예외를 삼켜 성공으로 위장하지 않는다.

**REQ-ATT-015(성공 경로)**: 종료코드 0이면 GATE-001 REQ-ATG-006의 전-조합 센티널
표기(`market="all"`/`horizon=0`/`algorithm="all"`)로 `metrics.record_success()`를
1회 호출한다 — 조합별 세분화(REQ-ATT-022)는 이 SPEC의 범위 밖이다.

**REQ-ATT-017(보존 정책)**: 성공 이후 `combos`(기본값 `campaign.POINT_COMBOS`,
8개 전체 — REQ-ATT-023: 스킵리스트/필터 인자를 이 SPEC의 신규 디스패치 코드가
도입하지 않는다) 각각에 `persistence.apply_retention_for_combos()`(M2)를 적용한다.
"""

import time
from collections.abc import Callable, Sequence
from datetime import date

from analyzer.common.logging import get_logger
from analyzer.common.trace import reset_trace_id, set_trace_id
from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.log_retention import sweep_trainer_logs
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.ssh_dispatch import (
    SshConnection,
    build_remote_campaign_dispatch_command,
    connect_with_retry,
)
from analyzer.orchestration.wol import WolSender, send_with_retry
from analyzer.training.campaign import POINT_COMBOS
from analyzer.training.persistence import apply_retention_for_combos

_logger = get_logger(__name__)

MONTHLY_OPTUNA_TRIALS: int = 100
"""REQ-ATT-009: 월간 원격 디스패치가 `build_remote_campaign_dispatch_command()`에
명시적으로 전달하는 trial 수(ATE REQ-ATE-029의 "프로덕션 월간 튜닝 약 100 trial"
기준값)."""

_FAILURE_STAGE: str = "monthly_tuning"
"""REQ-ATT-014: `metrics.record_failure(stage=...)`에 전달하는 신규 값 —
`orchestration/failure.py`의 `FailureStage` Literal에는 포함되지 않는다(PRESERVE)."""


class MonthlyCampaignRunError(RuntimeError):
    """REQ-ATT-014: 월간 원격 캠페인 실행 실패.

    `metrics.record_failure(stage="monthly_tuning")` 기록 이후 항상 발생한다 —
    예외를 삼켜 실행을 성공으로 위장하지 않는다(shall not).
    """


def _fail(metrics: TrainingMetrics, run_id: str, message: str) -> None:
    """REQ-ATT-014: 실패 로그 기록 + `metrics.record_failure()` 직접 호출 +
    `MonthlyCampaignRunError` 재발생 — `handle_training_run_failure()`를
    경유하지 않는 대체 실패 경로."""
    _logger.error(
        "monthly campaign run failure run_id=%s message=%s",
        run_id,
        message,
        extra={"stage_marker": True},
    )
    metrics.record_failure(stage=_FAILURE_STAGE)
    raise MonthlyCampaignRunError(message)


def execute_monthly_campaign_run(
    *,
    run_id: str,
    data_as_of: date,
    config: AutomationConfig,
    wol_sender: WolSender,
    connection_factory: Callable[[], SshConnection],
    metrics: TrainingMetrics,
    combos: Sequence[tuple[str, int, str]] = POINT_COMBOS,
    n_trials: int = MONTHLY_OPTUNA_TRIALS,
    wol_wait_seconds: float = 30.0,
    wol_max_retries: int = 3,
    ssh_max_retries: int = 6,
    ssh_retry_interval_seconds: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.time,
) -> None:
    """월간 원격 캠페인 CLI를 SSH로 직접 디스패치하고 종료코드로 성공/실패를
    판정한다(REQ-ATT-013/014/015).

    `combos`는 보존 정책(REQ-ATT-017)을 적용할 (시장, horizon, algorithm) 조합
    목록이다 — 기본값 `campaign.POINT_COMBOS`(8개 전체)를 그대로 순회하며,
    이 함수 자신은 어떤 조합도 사전에 배제하지 않는다(REQ-ATT-022/023).
    """
    trace_id_token = set_trace_id(run_id)
    try:
        wol_result = send_with_retry(
            wol_sender, config.target_mac_address, max_retries=wol_max_retries
        )
        if not wol_result.success:
            _fail(metrics, run_id, wol_result.error or "WoL 매직패킷 송신 실패")

        sleep_fn(wol_wait_seconds)

        connection = connection_factory()
        connected = connect_with_retry(
            connection,
            max_retries=ssh_max_retries,
            interval_seconds=ssh_retry_interval_seconds,
            sleep_fn=sleep_fn,
        )
        if not connected:
            _fail(metrics, run_id, f"SSH 연결 {ssh_max_retries}회 재시도 실패")

        try:
            command = build_remote_campaign_dispatch_command(
                active_models_root=config.active_models_root,
                calendar_code=config.calendar_code,
                cache_dir=config.cache_dir,
                data_as_of=data_as_of,
                feature_code_version=config.feature_code_version,
                optuna_storage_dir=config.monthly_optuna_storage_dir,
                summary_report_path=config.monthly_summary_report_path,
                n_trials=n_trials,
                db_tunnel_host=config.db_tunnel_host,
                db_tunnel_key_path=config.db_tunnel_private_key_path,
                db_tunnel_username=config.db_tunnel_username,
                db_tunnel_port=config.db_tunnel_port,
                db_tunnel_local_port=config.db_tunnel_local_port,
                db_tunnel_remote_port=config.db_tunnel_remote_port,
                mount_script_path=config.mount_script_path,
                python_executable_path=config.python_executable_path,
                mysql_database=config.mysql_database,
                mysql_trainer_password=config.mysql_trainer_password,
                trainer_log_base_dir=config.trainer_log_base_dir,
                run_id=run_id,
            )

            result = connection.exec_command(
                command, timeout_seconds=config.monthly_timeout_seconds
            )
            _logger.info(
                "monthly campaign remote exit code received exit_code=%s timed_out=%s run_id=%s",
                result.exit_code,
                result.timed_out,
                run_id,
                extra={"stage_marker": True},
            )

            if result.timed_out:
                _fail(metrics, run_id, f"{config.monthly_timeout_seconds}초 타임아웃 초과")

            if result.exit_code != 0:
                _fail(metrics, run_id, f"월간 캠페인 종료코드 {result.exit_code}")

            # REQ-ATT-015: 전-조합 센티널 표기 — run-level 성공 기록.
            metrics.record_success(market="all", horizon=0, algorithm="all", timestamp=time_fn())

            # REQ-ATT-017: 보존 정책은 소환만 한다(PRESERVE) — 무결성 검증 실패
            # 시 apply_retention_for_combos()가 전파하는 ValueError는 그대로
            # 다시 전파된다(성공 기록은 이미 마쳤으므로 record_failure를
            # 별도로 호출하지 않는다 — 캠페인 자체는 성공했다).
            # review finding W1: 이 호출은 이전까지 무가드 상태였다 — 예외가
            # APScheduler 내부 로거로만 흘러가 run_id 상관관계가 끊기는 것을
            # 막기 위해 구조화 로그(run_id 포함) 기록 후 재발생시킨다.
            try:
                apply_retention_for_combos(config.active_models_root, combos)
            except Exception:
                _logger.error(
                    "monthly campaign retention step failed after successful run run_id=%s",
                    run_id,
                    exc_info=True,
                    extra={"stage_marker": True},
                )
                raise
        finally:
            # SPEC-OBSV-LOGS-003 REQ-002/006: 주간 학습 경로(runner.py)와
            # 동일한 단일 sweep 함수를 디스패치 완료 직후 호출한다.
            sweep_trainer_logs(current_run_id=run_id)
            connection.close()
    finally:
        reset_trace_id(trace_id_token)

"""학습 실행 오케스트레이션 — WoL+SSH+터널+타임아웃+실패처리+계측을 하나의 흐름으로 조립.

SPEC-ANALYZER-TRAIN-AUTOMATION-001 §2 전체, AC-ATA-001~006/012.

이 모듈은 REQ-ATA-060(통합 실패 처리)과 plan.md §B.5(D6, 사전 차단 모델 보존)의
호출 순서를 강제하는 최상위 오케스트레이터다: `promote_staging_to_active()`는
SSH 종료코드 0 확인 후에만 호출되며, 그 외 모든 경로(WoL 실패/SSH 연결 실패/
학습 스크립트 실패/타임아웃)는 `handle_training_run_failure()`로만 라우팅되어
활성 경로를 절대 건드리지 않는다(REQ-ATA-062) — 이 순서 강제 자체가 절대 보장의
실행 메커니즘이다.

@MX:ANCHOR: [AUTO] execute_scheduled_training_run — 주간/월간 cron 잡(scheduler.py)이
호출하는 유일한 진입점.
@MX:REASON: fan_in >= 3 예상 — 주간 재학습 cron, 월간 튜닝 cron 잡 등록 시
동일 함수(다른 run_kind)로 배선되며, 완료 감지·실패 처리·모델 보존 불변식이
모두 이 함수의 호출 순서에 의존한다(REQ-ATA-062).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal

from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.failure import TrainingRunFailure, handle_training_run_failure
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.ssh_dispatch import (
    SshConnection,
    build_remote_dispatch_command,
    connect_with_retry,
    promote_staging_to_active,
)
from analyzer.orchestration.wol import WolSender, send_with_retry

RunKind = Literal["weekly", "monthly"]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """AC-ATA-001~006/012이 검증하는 학습 실행 오케스트레이션의 최종 결과."""

    success: bool
    failure: TrainingRunFailure | None = None
    promoted: bool = False


def execute_scheduled_training_run(
    *,
    run_kind: RunKind,
    run_id: str,
    market: str,
    horizon: int,
    algorithm: str,
    data_as_of: date,
    config: AutomationConfig,
    wol_sender: WolSender,
    connection_factory: Callable[[], SshConnection],
    metrics: TrainingMetrics,
    wol_wait_seconds: float = 30.0,
    wol_max_retries: int = 3,
    ssh_max_retries: int = 6,
    ssh_retry_interval_seconds: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    time_fn: Callable[[], float] = time.time,
) -> RunOutcome:
    """REQ-ATA-010~062: WoL → 30초 대기(REQ-ATA-020) → SSH 연결(재시도 포함) →
    원격 디스패치(터널 수립 내장) → 완료 감지(종료코드) → 프로모션/실패처리까지
    전체 흐름을 실행한다(AC-ATA-001).

    실패 경로(WoL/SSH/학습스크립트/타임아웃)는 모두 동일한
    `handle_training_run_failure()`로 라우팅된다(REQ-ATA-060) — 실패 유형별로
    별도 분기를 추가하지 않는다.
    """
    wol_result = send_with_retry(wol_sender, config.target_mac_address, max_retries=wol_max_retries)
    if not wol_result.success:
        failure = TrainingRunFailure(
            stage="wol", message=wol_result.error or "WoL 매직패킷 송신 실패", run_id=run_id
        )
        handle_training_run_failure(failure, metrics)
        return RunOutcome(success=False, failure=failure)

    sleep_fn(wol_wait_seconds)  # REQ-ATA-020

    connection = connection_factory()
    connected = connect_with_retry(
        connection,
        max_retries=ssh_max_retries,
        interval_seconds=ssh_retry_interval_seconds,
        sleep_fn=sleep_fn,
    )
    if not connected:
        failure = TrainingRunFailure(
            stage="ssh", message=f"SSH 연결 {ssh_max_retries}회 재시도 실패", run_id=run_id
        )
        handle_training_run_failure(failure, metrics)
        return RunOutcome(success=False, failure=failure)

    timeout_seconds = (
        config.weekly_timeout_seconds if run_kind == "weekly" else config.monthly_timeout_seconds
    )
    staging_path = config.staging_models_root / run_id
    command = build_remote_dispatch_command(
        staging_models_root=staging_path,
        calendar_code=config.calendar_code,
        cache_dir=config.cache_dir,
        data_as_of=data_as_of,
        feature_code_version=config.feature_code_version,
        db_tunnel_host=config.db_tunnel_host,
        db_tunnel_key_path=config.db_tunnel_private_key_path,
        db_tunnel_username=config.db_tunnel_username,
        db_tunnel_port=config.db_tunnel_port,
        db_tunnel_local_port=config.db_tunnel_local_port,
        db_tunnel_remote_port=config.db_tunnel_remote_port,
        mount_script_path=config.mount_script_path,
        python_executable_path=config.python_executable_path,
    )

    try:
        result = connection.exec_command(command, timeout_seconds=timeout_seconds)

        if result.timed_out:
            failure = TrainingRunFailure(
                stage="timeout", message=f"{timeout_seconds}초 타임아웃 초과", run_id=run_id
            )
            handle_training_run_failure(failure, metrics)
            return RunOutcome(success=False, failure=failure)

        if result.exit_code != 0:
            failure = TrainingRunFailure(
                stage="training",
                message=f"학습 스크립트 종료코드 {result.exit_code}",
                run_id=run_id,
            )
            handle_training_run_failure(failure, metrics)
            return RunOutcome(success=False, failure=failure)

        # plan.md §B.5(D6): 이 지점에 도달했다는 것 자체가 SSH 종료코드 0을
        # 의미한다 — 프로모션은 오직 이 성공 경로에서만 호출된다.
        promoted = promote_staging_to_active(connection, staging_path, config.active_models_root)
        metrics.record_success(
            market=market, horizon=horizon, algorithm=algorithm, timestamp=time_fn()
        )
        return RunOutcome(success=True, promoted=promoted)
    finally:
        # REQ-ATA-032/AC-ATA-006: 성공·실패·타임아웃 무관 SSH 세션을 정리한다.
        # db_tunnel 자체의 해제는 원격 셸 스크립트의 trap이 담당한다
        # (ssh_dispatch.build_remote_dispatch_command).
        connection.close()

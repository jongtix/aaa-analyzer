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

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

from analyzer.common.logging import get_logger
from analyzer.common.trace import reset_trace_id, set_trace_id
from analyzer.orchestration.config import AutomationConfig
from analyzer.orchestration.failure import TrainingRunFailure, handle_training_run_failure
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.promotion_gate import PromotionVerdict
from analyzer.orchestration.ssh_dispatch import (
    SshConnection,
    build_remote_dispatch_command,
    connect_with_retry,
    promote_staging_to_active,
)
from analyzer.orchestration.wol import WolSender, send_with_retry

RunKind = Literal["weekly", "monthly"]

_logger = get_logger(__name__)


def _make_stage_marker_relay(logger: logging.Logger) -> Callable[[str], None]:
    """SPEC-ANALYZER-TRAIN-OBSV-001 REQ-ATO-002(D-NEW-1): 채널 드레인 루프가
    소비한 원격 stdout/stderr 라인 중, JSON 로그 레코드의 `stage_marker: true`
    필드로 식별되는 저볼륨 부분집합만 NAS 측 analyzer 자신의 기존 구조화
    로거로 릴레이한다. 파싱 실패 라인이나 `stage_marker`가 없거나 `false`인
    라인(상세 로그)은 릴레이하지 않는다(REQ-ATO-007 — 트레이너 파일이
    원문 전량을 별도로 보존하므로 이중 적재하지 않는다)."""

    def _relay(line: str) -> None:
        try:
            record = json.loads(line)
        except json.JSONDecodeError, TypeError:
            return
        if not isinstance(record, dict) or record.get("stage_marker") is not True:
            return
        logger.info("remote stage marker: %s", record.get("message", line))

    return _relay


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
    promotion_gate_fn: Callable[[bool], Mapping[tuple[str, int, str], PromotionVerdict]]
    | None = None,
) -> RunOutcome:
    """REQ-ATA-010~062: WoL → 30초 대기(REQ-ATA-020) → SSH 연결(재시도 포함) →
    원격 디스패치(터널 수립 내장) → 완료 감지(종료코드) → 프로모션/실패처리까지
    전체 흐름을 실행한다(AC-ATA-001).

    실패 경로(WoL/SSH/학습스크립트/타임아웃)는 모두 동일한
    `handle_training_run_failure()`로 라우팅된다(REQ-ATA-060) — 실패 유형별로
    별도 분기를 추가하지 않는다.

    REQ-ATO-012/013/014(AC-ATO-008): NAS 측 오케스트레이터가 남기는 릴레이·
    단계 전이 로그의 trace_id 필드가 run_id와 일치하도록, 이 함수 시작 시점에
    `set_trace_id(run_id)`로 활성 Trace ID를 설정하고 함수 종료(성공/실패/
    타임아웃 무관) 시 반환된 토큰으로 복원한다 — 동시 실행 중인 무관한
    컨텍스트로 값이 새어나가지 않게 한다.

    REQ-ATE-055(M6): 종료코드 0(SSH 성공)만으로 자동 승격을 기록하지 않는다
    — `promotion_gate_fn`이 주어지면(1차 배포 이후), `promote_staging_to_active()`
    성공 여부(`promoted`)를 인자로 호출해 조합별 `PromotionVerdict` 매핑을
    받고, REQ-ATE-064/065에 따라 그 매핑에 있는 각 (시장,horizon,algorithm)
    조합마다 개별적으로 `record_success(outcome=...)`를 호출한다("success"=
    승격, "held-back"=보류). `promotion_gate_fn`이 `None`이면(1차 배포 이전,
    §B 리스크 6 — 활성 챔피언이 아직 없어 챌린저 개념이 성립하지 않는 상태)
    기존처럼 이 함수에 전달된 단일 (market,horizon,algorithm)에 대해서만
    `record_success(outcome="success")`를 호출한다(하위 호환).
    """
    trace_id_token = set_trace_id(run_id)
    try:
        wol_result = send_with_retry(
            wol_sender, config.target_mac_address, max_retries=wol_max_retries
        )
        # REQ-ATO-021: WoL 송신 결과 단계 전이 로그.
        _logger.info(
            "wol send result success=%s run_id=%s",
            wol_result.success,
            run_id,
            extra={"stage_marker": True},
        )
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
            config.weekly_timeout_seconds
            if run_kind == "weekly"
            else config.monthly_timeout_seconds
        )
        # REQ-ATO-021: 디스패치 시작(run_id + 타임아웃값) 단계 전이 로그.
        _logger.info(
            "dispatch start run_id=%s timeout_seconds=%s",
            run_id,
            timeout_seconds,
            extra={"stage_marker": True},
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
            mysql_database=config.mysql_database,
            mysql_trainer_password=config.mysql_trainer_password,
            trainer_log_base_dir=config.trainer_log_base_dir,
            run_id=run_id,
        )

        try:
            result = connection.exec_command(
                command,
                timeout_seconds=timeout_seconds,
                on_output_line=_make_stage_marker_relay(_logger),
            )
            # REQ-ATO-021: 원격 종료코드 수신 단계 전이 로그.
            _logger.info(
                "remote exit code received exit_code=%s timed_out=%s run_id=%s",
                result.exit_code,
                result.timed_out,
                run_id,
                extra={"stage_marker": True},
            )

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
            promoted = promote_staging_to_active(
                connection, staging_path, config.active_models_root
            )
            # REQ-ATO-021: 프로모션 결과 단계 전이 로그.
            _logger.info(
                "promotion result promoted=%s run_id=%s",
                promoted,
                run_id,
                extra={"stage_marker": True},
            )

            if promotion_gate_fn is None:
                # §B 리스크 6: 활성 챔피언이 아직 없는 1차 배포 이전 상태 —
                # 상시 게이트(§2.10) 경로를 호출하지 않는다.
                metrics.record_success(
                    market=market, horizon=horizon, algorithm=algorithm, timestamp=time_fn()
                )
                return RunOutcome(success=True, promoted=promoted)

            # REQ-ATE-055/064/065: 조합별 승격/보류 판정 → 조합마다 개별
            # record_success(outcome=...) 호출. record_success 자체(카운터
            # 증가 + Rank IC 게이지, REQ-ATE-060/066)가 기존 알림 채널
            # (Prometheus → vmalert → 텔레그램, REQ-ATA-060/061)의 트리거
            # 시그널이다 — 별도 알림 함수를 새로 호출하지 않는다.
            verdicts = promotion_gate_fn(promoted)
            for (v_market, v_horizon, v_algorithm), verdict in verdicts.items():
                outcome = "success" if verdict.promoted else "held-back"
                metrics.record_success(
                    market=v_market,
                    horizon=v_horizon,
                    algorithm=v_algorithm,
                    timestamp=time_fn(),
                    outcome=outcome,
                )
                metrics.record_rank_ic(
                    market=v_market,
                    horizon=v_horizon,
                    algorithm=v_algorithm,
                    rank_ic=verdict.challenger_rank_ic,
                )
                _logger.info(
                    "promotion gate outcome market=%s horizon=%s algorithm=%s outcome=%s run_id=%s",
                    v_market,
                    v_horizon,
                    v_algorithm,
                    outcome,
                    run_id,
                    extra={"stage_marker": True},
                )
            return RunOutcome(success=True, promoted=promoted)

        finally:
            # REQ-ATA-032/AC-ATA-006: 성공·실패·타임아웃 무관 SSH 세션을 정리한다.
            # db_tunnel 자체의 해제는 원격 셸 스크립트의 trap이 담당한다
            # (ssh_dispatch.build_remote_dispatch_command).
            connection.close()
    finally:
        reset_trace_id(trace_id_token)

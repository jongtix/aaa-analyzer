"""상주 부모 프로세스를 위한 asyncio 진입점.

FastAPI 앱을 스트림 컨슈머·스케줄러와 연결하고, uvicorn으로 앱을 서빙한다.

SPEC-ANALYZER-TRAIN-GATE-001 M5(REQ-ATG-001/002/003/004/005/006): 기동 시
주간 재학습 cron 잡 하나만 등록하고 스케줄러를 기동한다 — 월간/일일 잡은
등록하지 않는다(`register_default_jobs()`는 호출하지 않는다). 필수
환경변수 누락 시 `get_automation_config()`의 `MissingConfigError`가 잡히지
않고 전파되어 기동이 실패해야 한다(G-1 fail-fast).
"""

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import uvicorn

from analyzer.api.app import create_app
from analyzer.common.logging import get_logger
from analyzer.common.trace import new_trace_id
from analyzer.orchestration.config import AutomationConfig, get_automation_config
from analyzer.orchestration.consumer import StreamConsumer
from analyzer.orchestration.gate_adapter import build_gate_promotion_fn, compute_data_as_of
from analyzer.orchestration.manual_run import run_training
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.scheduler import (
    WEEKLY_FULL_RETRAIN_JOB_ID,
    SchedulerRegistry,
    weekly_full_retrain_trigger,
)
from analyzer.orchestration.ssh_dispatch import ParamikoSshConnection

logger = get_logger(__name__)

_KST = ZoneInfo("Asia/Seoul")


def wire_weekly_retrain_job(
    scheduler: SchedulerRegistry,
    *,
    config: AutomationConfig,
    metrics: TrainingMetrics,
) -> None:
    """REQ-ATG-001/003/005/006: 주간 재학습 cron 잡 하나만 등록하고 스케줄러를
    기동한다 — `register_default_jobs()`(월간/일일 포함)는 호출하지 않는다
    (shall not, no-op cron 잡음 방지).
    """

    def _weekly_job() -> None:
        run_id = new_trace_id()
        fire_date = datetime.now(_KST).date()
        # REQ-ATG-005: data_as_of는 발화 시점에 한 곳에서 1회 계산되어
        # run_training()과 게이트 CLI 양쪽에 동일하게 주입된다.
        data_as_of = compute_data_as_of(fire_date)

        def connection_factory() -> ParamikoSshConnection:
            return ParamikoSshConnection(
                host=config.ssh_host,
                port=config.ssh_port,
                username=config.ssh_username,
                private_key_path=config.ssh_private_key_path,
                known_hosts_path=config.known_hosts_path,
            )

        promotion_gate_fn = build_gate_promotion_fn(
            config=config,
            metrics=metrics,
            data_as_of=data_as_of,
            connection_factory=connection_factory,
        )

        # REQ-ATG-006: 전-조합 센티널 market="all"/horizon=0/algorithm="all" —
        # 원격 학습 CLI는 조합 인자를 받지 않아 1회 호출이 전 조합을 학습하고,
        # 조합별 성공/보류 귀속은 promotion_gate_fn이 반환하는 verdict별로
        # runner.py가 수행한다(단일 실조합 값 사용 금지).
        try:
            outcome = run_training(
                run_kind="weekly",
                run_id=run_id,
                market="all",
                horizon=0,
                algorithm="all",
                data_as_of=data_as_of,
                config=config,
                metrics=metrics,
                promotion_gate_fn=promotion_gate_fn,
                # REQ-ATG-011: 활성 챔피언 사이드카(M1 리더 재사용)에서 조합별
                # 동결 하이퍼파라미터를 읽어 주간 원격 학습에 주입한다 — 게이트
                # 챌린저와 동일 파라미터로 학습하게 하는 유일한 프로덕션 배선
                # 지점(AC-ATG-011).
                params_from_active_meta=config.active_models_root,
            )
        except Exception:
            # Critical-2 수정: 게이트 실패(GatePromotionFailure)는 runner.py의
            # promotion_gate_fn 호출부에 try/except가 없어 이 지점까지 그대로
            # 전파된다 — 여기서 구조화 로거(run_id 포함)로 기록한 뒤 재발생해야
            # APScheduler 내부 로거로만 흘러가 trace_id 상관관계가 끊기는 것을
            # 막는다. metrics.record_failure()는 다시 호출하지 않는다(gate_adapter가
            # 이미 기록, 이중 기록 방지).
            logger.error("weekly training run raised run_id=%s", run_id, exc_info=True)
            raise
        if not outcome.success:
            # §B 리스크 3: 콜백 레벨에서는 metrics.record_failure()를 다시
            # 호출하지 않는다 — runner 내부 실패 4종은 handle_training_run_failure()가,
            # 게이트 실패(E-1)는 gate_adapter가 이미 기록했다(이중 기록 방지).
            logger.error("weekly training run failed run_id=%s failure=%s", run_id, outcome.failure)
            raise RuntimeError(f"weekly training run failed: run_id={run_id}")

    scheduler.register_cron_job(
        WEEKLY_FULL_RETRAIN_JOB_ID, weekly_full_retrain_trigger(), _weekly_job
    )
    scheduler.start()


async def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """상주 부모 프로세스를 시작한다: FastAPI 앱 + 주간 재학습 cron 잡 배선.

    REQ-ATG-002(G-1): `get_automation_config()` 호출은 이 함수 진입 직후
    수행되며, `MissingConfigError`를 잡지 않는다 — 예외가 전파되어 컨테이너
    기동 자체가 실패해야 한다(CD의 `docker compose up -d --wait` + 자동
    롤백이 배포 시점에 설정 결함을 표면화한다).
    """
    consumer = StreamConsumer()
    config = get_automation_config()
    # REQ-ATG-004: TrainingMetrics는 프로세스당 정확히 1회 생성되어 콜백
    # 클로저에 주입된다 — 콜백 내부에서 발화 시마다 재생성하지 않는다.
    metrics = TrainingMetrics()
    scheduler = SchedulerRegistry()

    wire_weekly_retrain_job(scheduler, config=config, metrics=metrics)

    await consumer.start()
    logger.info(
        "orchestration wired (jobs=%d)",
        len(scheduler.registered_jobs()),
    )

    app = create_app()
    uvicorn_config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(uvicorn_config)
    try:
        await server.serve()
    finally:
        # REQ-ATG-001: 프로세스 종료 경로에 shutdown() 훅을 추가한다 —
        # 기존에는 종료 지점이 없었다.
        scheduler.shutdown()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

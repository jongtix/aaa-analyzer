"""수동 학습 실행 진입점 — `python -m analyzer.orchestration.manual_run`.

SPEC-ANALYZER-TRAIN-AUTOMATION-001: cron 자동 등록(`scheduler.py`
`register_default_jobs()`)을 하기 전, 프로덕션과 동일한 경로(WoL → SSH 디스패치
→ 마운트 확인 → DB 터널 → 학습 CLI → 프로모션)로 몇 회차 수동 실행해 결과를
확인하기 위한 진입점이다(2026-08-13 Stage 1/2 실측 검증 후 도입).

`run_training()`이 배선의 핵심이다 — cron 등록 시점에 `register_default_jobs()`에
넘길 콜백들도 인자만 다르게 채워서 이 함수를 그대로 재사용해야 한다(로직을
다시 쓰지 않는다).
"""

import argparse
import sys
from datetime import date

from analyzer.common.logging import get_logger
from analyzer.common.trace import new_trace_id
from analyzer.orchestration.config import AutomationConfig, get_automation_config
from analyzer.orchestration.metrics import TrainingMetrics
from analyzer.orchestration.runner import RunKind, RunOutcome, execute_scheduled_training_run
from analyzer.orchestration.ssh_dispatch import ParamikoSshConnection
from analyzer.orchestration.wol import UdpBroadcastWolSender

logger = get_logger(__name__)


def run_training(
    *,
    run_kind: RunKind,
    run_id: str,
    market: str,
    horizon: int,
    algorithm: str,
    data_as_of: date,
    config: AutomationConfig,
    metrics: TrainingMetrics,
) -> RunOutcome:
    """설정 로딩 이후의 배선(WoL sender·SSH 연결 팩토리)을 담당한다.

    CLI(`main()`)와 향후 cron 콜백(`register_default_jobs()`에 넘길 함수들)이
    공유하는 지점이다 — market/horizon/algorithm/data_as_of를 누가(사람 vs
    스케줄러) 결정하는지만 다를 뿐, 이 함수 자체는 동일하게 재사용된다.

    `metrics`는 호출자가 프로세스 생애주기당 한 번만 구성해 주입해야 한다 —
    `TrainingMetrics()`는 기본 전역 Prometheus 레지스트리에 메트릭을 등록하므로,
    이 함수 안에서 매 호출마다 새로 만들면 같은 프로세스에서 두 번째 호출부터
    "Duplicated timeseries" 오류로 죽는다(cron이 같은 프로세스에서 반복
    실행되므로 재사용 필수).
    """
    wol_sender = UdpBroadcastWolSender()

    def connection_factory() -> ParamikoSshConnection:
        return ParamikoSshConnection(
            host=config.ssh_host,
            port=config.ssh_port,
            username=config.ssh_username,
            private_key_path=config.ssh_private_key_path,
            known_hosts_path=config.known_hosts_path,
        )

    return execute_scheduled_training_run(
        run_kind=run_kind,
        run_id=run_id,
        market=market,
        horizon=horizon,
        algorithm=algorithm,
        data_as_of=data_as_of,
        config=config,
        wol_sender=wol_sender,
        connection_factory=connection_factory,
        metrics=metrics,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyzer.orchestration.manual_run",
        description=(
            "cron 자동 등록 전 수동 학습 실행 — 프로덕션과 동일한 디스패치 경로를 사용한다."
        ),
    )
    parser.add_argument("--run-kind", choices=["weekly", "monthly"], required=True)
    parser.add_argument("--market", required=True, help="예: domestic, overseas")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument(
        "--data-as-of",
        type=date.fromisoformat,
        required=True,
        help="YYYY-MM-DD (KST 기준)",
    )
    parser.add_argument("--run-id", default=None, help="미지정 시 자동 생성")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or new_trace_id()
    config = get_automation_config()

    logger.info(
        "manual training run start run_id=%s run_kind=%s market=%s horizon=%s "
        "algorithm=%s data_as_of=%s",
        run_id,
        args.run_kind,
        args.market,
        args.horizon,
        args.algorithm,
        args.data_as_of,
    )

    outcome = run_training(
        run_kind=args.run_kind,
        run_id=run_id,
        market=args.market,
        horizon=args.horizon,
        algorithm=args.algorithm,
        data_as_of=args.data_as_of,
        config=config,
        metrics=TrainingMetrics(),
    )

    if outcome.success:
        logger.info("manual training run success run_id=%s promoted=%s", run_id, outcome.promoted)
        return 0

    failure = outcome.failure
    logger.error(
        "manual training run failed run_id=%s stage=%s message=%s",
        run_id,
        failure.stage if failure else "unknown",
        failure.message if failure else "",
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

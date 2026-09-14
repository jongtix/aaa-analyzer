"""자식 프로세스 스폰 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-010).

상주 부모는 (market) 단위로 완결형 자식 CLI를 `asyncio.create_subprocess_exec`
으로 스폰하고, **오직 종료코드만을** 계약면으로 관찰한다. 자식 stdout은 부모의
구조화 로거에 한 줄씩 그대로 릴레이될 뿐 어떤 구조화 데이터로도 파싱되지
않는다(REQ-AIF-010 shall not) — 이 모듈에 stdout 파싱 코드가 존재하지 않는
것은 정적 grep으로 검증된다(AC-AIF-001).
"""

import asyncio
import os
import sys

from prometheus_client import multiprocess

from analyzer.common.logging import get_logger
from analyzer.inference.outcome import EXIT_ALL_SKIPPED, EXIT_PARTIAL_FAILURE, EXIT_SUCCESS

__all__ = [
    "EXIT_ALL_SKIPPED",
    "EXIT_PARTIAL_FAILURE",
    "EXIT_SUCCESS",
    "default_child_argv",
    "spawn_inference_child",
]

logger = get_logger(__name__)


def default_child_argv(market: str, *, trace_id: str | None = None) -> list[str]:
    """`python -m analyzer.inference --market <market>` 인자 벡터를 만든다.

    실행 파일은 부모 자신의 인터프리터(`sys.executable`)를 사용한다 — 컨테이너
    안에서 PATH 해석에 의존하지 않기 위함이다(원격 학습 CLI가
    `python_executable_path`를 명시적으로 요구하는 것과 동일한 취지).

    `trace_id`가 주어지면(SPEC-ANALYZER-PIPELINE-001 REQ-APL-111) `--trace-id`
    인자로 자식에 전파한다 — 부모의 이벤트-수신 시점 trace_id가
    `stream:signal:{market}` 발행까지 상관관계를 유지하게 한다. 부재 시(수동
    CLI 실행 등) 이 인자를 생략해 자식이 자체 생성 폴백을 쓰도록 한다(하위호환).
    """
    argv = [sys.executable, "-m", "analyzer.inference", "--market", market]
    if trace_id is not None:
        argv.extend(["--trace-id", trace_id])
    return argv


async def spawn_inference_child(
    market: str,
    *,
    trace_id: str,
    argv: list[str] | None = None,
) -> int:
    """자식 CLI를 스폰해 완료를 기다리고 종료코드를 반환한다.

    stderr는 stdout으로 합류시켜 단일 스트림으로 릴레이한다 — 자식의 구조화
    JSON 로그와 파이썬 트레이스백이 같은 순서로 부모 로그에 남는다.
    """
    command = argv if argv is not None else default_child_argv(market, trace_id=trace_id)

    logger.info("inference child spawning market=%s trace_id=%s", market, trace_id)
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    if process.stdout is not None:
        async for line in process.stdout:
            # REQ-AIF-010: 로그 상관관계 릴레이 전용 — 이 문자열을 구조화
            # 데이터로 해석하지 않는다.
            logger.info(
                "inference child stdout market=%s trace_id=%s line=%s",
                market,
                trace_id,
                line.decode("utf-8", errors="replace").rstrip(),
            )

    exit_code = await process.wait()

    # REQ-APL-133: 자식이 남긴 pid별 Prometheus 멀티프로세스 덤프 파일을
    # 정리한다. `mark_process_dead()`는 대상 디렉토리에 파일이 없으면
    # 조용히 무동작하지만, `PROMETHEUS_MULTIPROC_DIR` 환경변수 자체가
    # 미설정(둘 다 None)이면 `os.path.join(None, ...)`으로 `TypeError`를
    # 던진다(실측 확인 — SPEC 전제와 달리 라이브러리가 경로 자체의 부재까지
    # 무동작으로 흡수하지는 않는다) — 그래서 이 가드는 REQ-APL-133(shall
    # not)을 충족하기 위해 환경변수가 실제로 설정된 경우에만 호출한다.
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR") or os.environ.get("prometheus_multiproc_dir"):
        multiprocess.mark_process_dead(process.pid)

    return exit_code

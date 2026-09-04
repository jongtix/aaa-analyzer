"""시장 추론을 위한 완결형 자식 프로세스 CLI 진입점.

SPEC-ANALYZER-INFER-001 M1(REQ-AIF-010): `python -m analyzer.inference
--market <market>`은 해당 시장의 추론을 완결하고 종료코드 계약(0=성공,
1=전조합 스킵, 2=부분실패)으로 결과를 부모에게 알린다. 결과를 stdout으로
반환하지 않는다 — stdout은 구조화 로그 전용이다.

M1 시점에는 추론 파이프라인 본체(매니페스트 해석 M2, 분위수 서빙 M3, 등급
경계 M4, 피처 조립+INSERT M5, 밴드 스윕 M6)가 아직 배선되지 않았으므로
`run_market_inference()`는 "처리한 조합 없음"을 반환하고 프로세스는 종료코드
1(전조합 스킵)로 끝난다. FOUNDATION-001 시절의 무조건 exit 0은 아무것도 하지
않은 실행을 성공으로 보고하는 것이어서 이 SPEC이 대체한다.
"""

import argparse
import sys

from analyzer.common.logging import get_logger
from analyzer.common.trace import new_trace_id
from analyzer.inference.outcome import InferenceOutcome, resolve_exit_code

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """자식 CLI 인자를 파싱한다. `--market`은 필수다."""
    parser = argparse.ArgumentParser(
        prog="analyzer.inference",
        description="시장 단위 완결형 추론 CLI(종료코드 0=성공/1=전조합스킵/2=부분실패).",
    )
    parser.add_argument(
        "--market",
        required=True,
        help="대상 시장 식별자(domestic, overseas)",
    )
    return parser.parse_args(argv)


# @MX:TODO: [AUTO] 추론 파이프라인 본체(M2~M6) 배선 지점 — 현재는 처리 조합 0건을 반환한다.
def run_market_inference(market: str) -> InferenceOutcome:
    """`market`의 전 (horizon) 조합에 대해 추론을 수행하고 집계를 반환한다.

    M2~M6이 매니페스트 해석 → 예측 → INSERT → 밴드 스윕 → 발행을 이 함수
    안에 채운다. M1은 종료코드 계약면만 확정하므로 처리 조합 0건을 반환한다.
    """
    logger.info(
        "inference pipeline not wired yet market=%s (SPEC-ANALYZER-INFER-001 M2~M6)",
        market,
    )
    return InferenceOutcome(processed=0, skipped_combinations=0, partial_failures=0)


def main(argv: list[str] | None = None) -> int:
    """추론을 실행하고 종료코드 계약(0/1/2)에 따른 코드를 반환한다."""
    args = parse_args(argv)
    trace_id = new_trace_id()

    outcome = run_market_inference(args.market)
    exit_code = resolve_exit_code(outcome)

    logger.info(
        "inference finished market=%s trace_id=%s processed=%d skipped=%d "
        "partial_failures=%d exit_code=%d",
        args.market,
        trace_id,
        outcome.processed,
        outcome.skipped_combinations,
        outcome.partial_failures,
        exit_code,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

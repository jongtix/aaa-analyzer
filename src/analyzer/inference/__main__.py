"""시장 추론을 위한 완결형 자식 프로세스 CLI 진입점.

SPEC-ANALYZER-INFER-001 M1(REQ-AIF-010): `python -m analyzer.inference
--market <market>`은 해당 시장의 추론을 완결하고 종료코드 계약(0=성공,
1=전조합 스킵, 2=부분실패)으로 결과를 부모에게 알린다. 결과를 stdout으로
반환하지 않는다 — stdout은 구조화 로그 전용이다.

SPEC-ANALYZER-PIPELINE-001 REQ-APL-100: `run_market_inference()`는 이제
`pipeline.run_market_inference()`를 호출하도록 교체됐다. `trade_date`는
`trade_date.resolve_trade_date(engine, market)`로 이 함수 진입 전에
산출한다(design.md §1, REQ-AIF-021 계승) — `None`이면(추론 대상 없음)
파이프라인을 호출하지 않고 즉시 전조합스킵으로 종료한다.
"""

import argparse
import sys

from analyzer.common.logging import get_logger
from analyzer.common.trace import new_trace_id
from analyzer.data.config import get_db_config
from analyzer.data.repository import build_engine
from analyzer.inference.config import get_inference_config
from analyzer.inference.outcome import InferenceOutcome, resolve_exit_code
from analyzer.inference.pipeline import run_market_inference
from analyzer.inference.trade_date import resolve_trade_date

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """자식 CLI 인자를 파싱한다. `--market`은 필수, `--trace-id`는 선택이다."""
    parser = argparse.ArgumentParser(
        prog="analyzer.inference",
        description="시장 단위 완결형 추론 CLI(종료코드 0=성공/1=전조합스킵/2=부분실패).",
    )
    parser.add_argument(
        "--market",
        required=True,
        help="대상 시장 식별자(domestic, overseas)",
    )
    parser.add_argument(
        "--trace-id",
        dest="trace_id",
        default=None,
        help="부모가 이벤트 수신 시점에 생성한 trace_id(REQ-APL-111). 부재 시 자체 생성한다.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """추론을 실행하고 종료코드 계약(0/1/2)에 따른 코드를 반환한다."""
    args = parse_args(argv)
    # design.md §4: 빈 문자열도 자체 생성 폴백과 동일하게 취급한다(falsy 값 통일).
    trace_id = args.trace_id or new_trace_id()

    engine = build_engine(get_db_config())
    try:
        trade_date = resolve_trade_date(engine, args.market)
    finally:
        engine.dispose()

    if trade_date is None:
        logger.info(
            "추론 대상 거래일을 산출할 수 없다 market=%s trace_id=%s",
            args.market,
            trace_id,
        )
        outcome = InferenceOutcome(processed=0, skipped_combinations=0, partial_failures=0)
    else:
        inference_config = get_inference_config()
        outcome = run_market_inference(
            args.market,
            trace_id=trace_id,
            models_root=inference_config.container_models_root,
            trade_date=trade_date,
        )

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

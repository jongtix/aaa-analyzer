"""시장 추론을 위한 완결형 자식 프로세스 CLI 진입점.

REQ-ANALYZER-FOUNDATION-011: `python -m analyzer.inference --market <market>`은
인자를 파싱하고 predict/모델 로드 로직 없이 exit code 0으로 종료한다.
실제 추론(모델 로드, 피처 계산, DB/스트림 I/O)은 본 SPEC 범위 밖이며 INFER-001에서
구현된다.
"""

import argparse
import sys

from analyzer.common.logging import get_logger
from analyzer.common.trace import new_trace_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """자식 CLI 인자를 파싱한다. `--market`은 필수다."""
    parser = argparse.ArgumentParser(
        prog="analyzer.inference",
        description="시장 추론용 자식 CLI 골격(predict 로직: INFER-001).",
    )
    parser.add_argument(
        "--market",
        required=True,
        help="대상 시장 식별자(예: domestic, overseas)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """인자를 파싱하고 한 줄 로그를 남긴 뒤 exit 0. predict 로직 없음(골격만)."""
    args = parse_args(argv)
    new_trace_id()

    logger = get_logger(__name__)
    logger.info("inference invoked market=%s (skeleton — no predict logic yet)", args.market)

    return 0


if __name__ == "__main__":
    sys.exit(main())

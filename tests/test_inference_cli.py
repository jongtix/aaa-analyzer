"""완결형 자식 프로세스 CLI 진입점에 대한 명세 테스트.

REQ-ANALYZER-FOUNDATION-011: `python -m analyzer.inference --market <market>`은
인자를 파싱하고 predict/모델 로드 로직 없이 exit code 0으로 종료한다(실제 추론은
INFER-001 범위). `--market` 누락 시 0이 아닌 코드로 종료해야 한다.
"""

import subprocess
import sys

from analyzer.inference.__main__ import main, parse_args


class TestParseArgs:
    def test_parses_required_market_argument(self):
        args = parse_args(["--market", "domestic"])

        assert args.market == "domestic"


class TestMainFunction:
    def test_main_returns_zero_for_valid_market(self):
        exit_code = main(["--market", "domestic"])

        assert exit_code == 0


class TestSubprocessInvocation:
    def test_module_invocation_exits_zero_with_market(self):
        result = subprocess.run(
            [sys.executable, "-m", "analyzer.inference", "--market", "domestic"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0

    def test_module_invocation_exits_nonzero_without_market(self):
        result = subprocess.run(
            [sys.executable, "-m", "analyzer.inference"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0

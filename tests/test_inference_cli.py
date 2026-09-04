"""완결형 자식 프로세스 CLI 진입점에 대한 명세 테스트.

SPEC-ANALYZER-INFER-001 M1(REQ-AIF-010): `python -m analyzer.inference
--market <market>`은 종료코드 계약(0=성공, 1=전조합 스킵, 2=부분실패)을
따른다. M1 시점에는 추론 파이프라인(M2~M6)이 아직 배선되지 않았으므로
기본 실행은 "처리한 조합 없음" = 종료코드 1이다 — FOUNDATION-001 시절의
무조건 exit 0(성공 위장)을 이 SPEC이 대체한다.
"""

import subprocess
import sys

import pytest

from analyzer.inference.__main__ import main, parse_args, run_market_inference
from analyzer.inference.outcome import InferenceOutcome, resolve_exit_code
from analyzer.inference.spawn import EXIT_ALL_SKIPPED, EXIT_PARTIAL_FAILURE, EXIT_SUCCESS


class TestParseArgs:
    def test_parses_required_market_argument(self):
        args = parse_args(["--market", "domestic"])

        assert args.market == "domestic"


class TestResolveExitCode:
    def test_processed_combinations_without_failures_is_success(self):
        outcome = InferenceOutcome(processed=3, skipped_combinations=0, partial_failures=0)

        assert resolve_exit_code(outcome) == EXIT_SUCCESS

    def test_all_combinations_skipped_is_exit_one(self):
        outcome = InferenceOutcome(processed=0, skipped_combinations=4, partial_failures=0)

        assert resolve_exit_code(outcome) == EXIT_ALL_SKIPPED

    def test_nothing_to_do_is_also_exit_one(self):
        outcome = InferenceOutcome(processed=0, skipped_combinations=0, partial_failures=0)

        assert resolve_exit_code(outcome) == EXIT_ALL_SKIPPED

    def test_partial_failure_takes_precedence_over_success(self):
        outcome = InferenceOutcome(processed=9, skipped_combinations=1, partial_failures=1)

        assert resolve_exit_code(outcome) == EXIT_PARTIAL_FAILURE


class TestMainFunction:
    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            (InferenceOutcome(processed=2, skipped_combinations=0, partial_failures=0), 0),
            (InferenceOutcome(processed=0, skipped_combinations=4, partial_failures=0), 1),
            (InferenceOutcome(processed=2, skipped_combinations=0, partial_failures=1), 2),
        ],
    )
    def test_exit_code_follows_the_outcome(
        self, monkeypatch: pytest.MonkeyPatch, outcome: InferenceOutcome, expected: int
    ):
        import analyzer.inference.__main__ as cli

        monkeypatch.setattr(cli, "run_market_inference", lambda market: outcome)

        assert main(["--market", "domestic"]) == expected

    def test_pipeline_is_not_wired_yet_so_nothing_is_processed(self):
        """M1 시점 파이프라인 자리 표시자 — M2~M6이 실제 조합 처리를 채운다."""
        outcome = run_market_inference("domestic")

        assert outcome.processed == 0


class TestSubprocessInvocation:
    def test_module_invocation_reports_all_skipped_while_unwired(self):
        result = subprocess.run(
            [sys.executable, "-m", "analyzer.inference", "--market", "domestic"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == EXIT_ALL_SKIPPED

    def test_module_invocation_exits_nonzero_without_market(self):
        result = subprocess.run(
            [sys.executable, "-m", "analyzer.inference"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0

"""완결형 자식 프로세스 CLI 진입점에 대한 명세 테스트.

SPEC-ANALYZER-INFER-001 M1(REQ-AIF-010): `python -m analyzer.inference
--market <market>`은 종료코드 계약(0=성공, 1=전조합 스킵, 2=부분실패)을
따른다.

SPEC-ANALYZER-PIPELINE-001 REQ-APL-100: `run_market_inference()`가
`pipeline.run_market_inference()`를 호출하도록 교체됐다 — 거래일 산출
(`trade_date.resolve_trade_date()`)이 `None`이면(추론 대상 없음) 파이프라인을
호출하지 않고 즉시 전조합스킵으로 종료한다.
"""

import subprocess
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from analyzer.inference.__main__ import main, parse_args
from analyzer.inference.outcome import InferenceOutcome, resolve_exit_code
from analyzer.inference.spawn import EXIT_ALL_SKIPPED, EXIT_PARTIAL_FAILURE, EXIT_SUCCESS


class TestParseArgs:
    def test_parses_required_market_argument(self):
        args = parse_args(["--market", "domestic"])

        assert args.market == "domestic"

    def test_trace_id_is_optional_and_defaults_to_none(self):
        """SPEC-ANALYZER-PIPELINE-001 REQ-APL-111: 하위호환 — 수동 CLI 실행은
        --trace-id 없이도 동작한다."""
        args = parse_args(["--market", "domestic"])

        assert args.trace_id is None

    def test_trace_id_is_parsed_when_provided(self):
        args = parse_args(["--market", "domestic", "--trace-id", "abc123"])

        assert args.trace_id == "abc123"


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


def _patch_main_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trade_date: date | None,
    outcome: InferenceOutcome,
) -> MagicMock:
    import analyzer.inference.__main__ as cli

    fake_engine = MagicMock(name="engine")
    monkeypatch.setattr(cli, "build_engine", lambda *_a, **_k: fake_engine)
    monkeypatch.setattr(cli, "get_db_config", lambda: object())
    monkeypatch.setattr(cli, "resolve_trade_date", lambda engine, market: trade_date)
    monkeypatch.setattr(
        cli, "get_inference_config", lambda: MagicMock(container_models_root=Path("/mnt/models"))
    )
    run_market_inference_spy = MagicMock(return_value=outcome)
    monkeypatch.setattr(cli, "run_market_inference", run_market_inference_spy)
    return run_market_inference_spy


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
        _patch_main_dependencies(monkeypatch, trade_date=date(2026, 9, 10), outcome=outcome)

        assert main(["--market", "domestic"]) == expected

    def test_calls_pipeline_run_market_inference_with_resolved_trade_date_and_models_root(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """AC-APL-100: `run_market_inference()`가 더 이상 스캐폴딩 로그를
        남기지 않고 `pipeline.run_market_inference()`를 호출한다."""
        outcome = InferenceOutcome(processed=1, skipped_combinations=0, partial_failures=0)
        spy = _patch_main_dependencies(monkeypatch, trade_date=date(2026, 9, 10), outcome=outcome)

        main(["--market", "domestic", "--trace-id", "abc123"])

        spy.assert_called_once_with(
            "domestic",
            trace_id="abc123",
            models_root=Path("/mnt/models"),
            trade_date=date(2026, 9, 10),
        )

    def test_generates_trace_id_when_not_provided(self, monkeypatch: pytest.MonkeyPatch):
        outcome = InferenceOutcome(processed=1, skipped_combinations=0, partial_failures=0)
        spy = _patch_main_dependencies(monkeypatch, trade_date=date(2026, 9, 10), outcome=outcome)

        main(["--market", "domestic"])

        _args, kwargs = spy.call_args
        assert isinstance(kwargs["trace_id"], str) and kwargs["trace_id"]

    def test_no_trade_date_skips_pipeline_call_and_returns_all_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """design.md §1: `trade_date`가 `None`이면 파이프라인을 호출하지
        않고 처리 대상 없음으로 즉시 종료한다."""
        spy = _patch_main_dependencies(
            monkeypatch,
            trade_date=None,
            outcome=InferenceOutcome(processed=0, skipped_combinations=0, partial_failures=0),
        )

        exit_code = main(["--market", "domestic"])

        spy.assert_not_called()
        assert exit_code == EXIT_ALL_SKIPPED


class TestSubprocessInvocation:
    def test_module_invocation_exits_nonzero_without_market(self):
        result = subprocess.run(
            [sys.executable, "-m", "analyzer.inference"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0

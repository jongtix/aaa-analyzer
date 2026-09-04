"""부모↔자식 IPC 계약 명세 테스트 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-010).

부모는 **오직 자식의 종료코드만을** 계약면으로 관찰한다(0=성공, 1=전조합
스킵, 2=부분실패). 자식 stdout은 로그 상관관계 목적으로만 부모의 구조화
로거에 릴레이되며 어떤 구조화 데이터로도 파싱되지 않는다.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from analyzer.inference.spawn import (
    EXIT_ALL_SKIPPED,
    EXIT_PARTIAL_FAILURE,
    EXIT_SUCCESS,
    default_child_argv,
    spawn_inference_child,
)


def _exiting_argv(code: int, *, stdout_line: str = "child-log") -> list[str]:
    return [
        sys.executable,
        "-c",
        f"import sys; print({stdout_line!r}); sys.exit({code})",
    ]


class TestDefaultChildArgv:
    def test_invokes_the_inference_module_with_market_flag(self):
        argv = default_child_argv("domestic")

        assert argv == [sys.executable, "-m", "analyzer.inference", "--market", "domestic"]


class TestExitCodeContract:
    def test_exit_code_constants_match_the_spec(self):
        assert (EXIT_SUCCESS, EXIT_ALL_SKIPPED, EXIT_PARTIAL_FAILURE) == (0, 1, 2)

    @pytest.mark.parametrize("code", [EXIT_SUCCESS, EXIT_ALL_SKIPPED, EXIT_PARTIAL_FAILURE])
    def test_parent_observes_the_child_exit_code(self, code: int):
        """AC-AIF-001: 정상/전조합스킵/부분실패 3개 시나리오가 각각 0/1/2로
        부모에게 전달된다."""
        observed = asyncio.run(
            spawn_inference_child("domestic", trace_id="trace-1", argv=_exiting_argv(code))
        )

        assert observed == code


class TestStdoutRelay:
    def test_child_stdout_is_relayed_to_the_structured_logger(
        self, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level("INFO"):
            asyncio.run(
                spawn_inference_child(
                    "domestic",
                    trace_id="trace-2",
                    argv=_exiting_argv(0, stdout_line="hello-from-child"),
                )
            )

        assert any("hello-from-child" in record.message for record in caplog.records)

    def test_json_stdout_is_relayed_verbatim_and_never_parsed(
        self, caplog: pytest.LogCaptureFixture
    ):
        """REQ-AIF-010(shall not): 자식이 JSON처럼 보이는 줄을 출력해도 부모는
        그것을 구조화 데이터로 해석하지 않고 문자열 그대로 릴레이한다."""
        with caplog.at_level("INFO"):
            exit_code = asyncio.run(
                spawn_inference_child(
                    "domestic",
                    trace_id="trace-3",
                    argv=_exiting_argv(EXIT_PARTIAL_FAILURE, stdout_line='{"exit": 0}'),
                )
            )

        assert exit_code == EXIT_PARTIAL_FAILURE
        assert any('{"exit": 0}' in record.message for record in caplog.records)


class TestStaticIpcBoundary:
    """AC-AIF-001 정적 grep 검증 — 부모 측 코드에 stdout 파싱/비동기 드라이버가
    존재하지 않아야 한다."""

    _PARENT_SOURCES = (
        Path("src/analyzer/inference/spawn.py"),
        Path("src/analyzer/orchestration/consumer.py"),
    )

    def test_parent_never_parses_child_stdout(self):
        for path in self._PARENT_SOURCES:
            source = path.read_text(encoding="utf-8")

            assert "json.loads" not in source, path
            assert "json.load(" not in source, path

    def test_inference_package_does_not_use_an_async_mysql_driver(self):
        for path in sorted(Path("src/analyzer/inference").glob("*.py")):
            assert "asyncmy" not in path.read_text(encoding="utf-8"), path

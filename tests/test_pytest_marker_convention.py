"""pytest 마커 컨벤션에 대한 스모크 테스트 (REQ-ANALYZER-FOUNDATION-028/029).

`integration`이 등록된 마커임을 확인하고, `-m integration`이 현재 0개 테스트를
정상적으로 선택함(아직 통합 테스트가 없음)을 확인하며, `--strict-markers`가
미등록 마커를 거부함을 확인한다.

exit code에 대한 참고: pytest 자체 관례(`pytest.ExitCode.NO_TESTS_COLLECTED`)에
따르면 마커 표현식이 모든 테스트를 제외시킬 때 exit code는 0이 아니라 5를
반환한다. 이는 문서화된, 에러가 아닌 동작(크래시나 설정 오류와는 구분됨)이므로
여기서는 "0개 선택, 필터가 의도대로 동작함"의 성공 케이스로 취급한다.
"""

import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NO_TESTS_COLLECTED = 5

# non-meta 테스트 파일로만 범위를 한정한다. 이 파일 자체는 서브프로세스 호출의
# collection 대상에 절대 포함되면 안 된다: 포함될 경우 생성된 pytest 프로세스가
# 바로 이 테스트를 재선택(및 재생성)하게 되어, 리소스가 고갈될 때까지 재귀가
# 계속된다.
_NON_META_TEST_FILES = [
    "tests/test_logging_trace.py",
    "tests/test_api.py",
    "tests/test_inference_cli.py",
]


class TestIntegrationMarkerFilter:
    def test_dash_m_integration_selects_zero_tests_successfully(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "integration",
                "--no-cov",
                "-q",
                *_NON_META_TEST_FILES,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_REPO_ROOT,
        )

        assert result.returncode == _NO_TESTS_COLLECTED
        assert "deselected" in result.stdout

    def test_dash_m_not_integration_still_runs_unit_tests(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "not integration",
                "--no-cov",
                "-q",
                *_NON_META_TEST_FILES,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=_REPO_ROOT,
        )

        assert result.returncode == 0
        assert "passed" in result.stdout


class TestStrictMarkersRejectsUnregistered:
    def test_unregistered_marker_usage_fails_collection(self):
        # NOTE: 의도적으로 스크래치 파일을 레포 트리 *내부*에 작성한다
        # (OS temp 디렉터리나 pytest 자체의 tmp_path 하위가 아님). pytest는
        # 대상 파일에서 rootdir까지 위로 올라가며 conftest.py 계보를 해석한다;
        # 대상이 레포 밖 먼 곳(예: macOS의 /var/folders/... 하위)에 있으면 그
        # 탐색이 무관하고 잠재적으로 거대한 디렉터리 트리 전체를 훑게 되며,
        # 이미 실행 중인 외부 pytest 프로세스 *내부에서* 호출될 때는 매우
        # 오래 걸릴 수 있다. 스크래치 파일을 레포 내부에 두면 그 탐색이
        # 사소하게 짧아진다.
        scratch_dir = _REPO_ROOT / ".scratch_marker_test"
        scratch_dir.mkdir(exist_ok=True)
        try:
            bogus_test = scratch_dir / "test_bogus_marker.py"
            bogus_test.write_text(
                "import pytest\n\n"
                "@pytest.mark.totally_unregistered_marker\n"
                "def test_noop():\n"
                "    assert True\n"
            )

            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(bogus_test), "--no-cov", "-q"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=_REPO_ROOT,
            )
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

        assert result.returncode != 0
        assert (
            "not found in `markers`" in result.stdout or "not found in `markers`" in result.stderr
        )

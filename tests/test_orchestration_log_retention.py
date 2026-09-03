"""트레이너 로그 보존 sweep 단위 테스트 (SPEC-OBSV-LOGS-003 M1).

AC-001(N개 초과 삭제) / AC-002(현재 run_id 제외) / AC-004(fail-open) /
AC-005(비-트레이너 파일 제외) / AC-006(기본값 10) + Edge Cases(N 이하 no-op,
빈/부재 디렉토리 no-op)를 검증한다.

sweep 대상 경로는 analyzer 컨테이너 **로컬** 로그 디렉토리(`LOG_PATH`,
기본값 `/var/log/aaa-analyzer`)다 — `TRAIN_AUTOMATION_TRAINER_LOG_BASE_DIR`는
맥북 SMB 마운트 경로 문자열이라 sweep 입력이 아니다(spec.md §4 DP-1 v0.2.0).
"""

import logging
import os
from pathlib import Path

import pytest

from analyzer.orchestration.log_retention import (
    DEFAULT_TRAINER_LOG_RETENTION_COUNT,
    sweep_trainer_logs,
)


def _write_trainer_log(directory: Path, run_id: str, mtime: float) -> Path:
    path = directory / f"trainer_{run_id}.log"
    path.write_text("{}\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOG_PATH", str(tmp_path))
    monkeypatch.delenv("TRAIN_AUTOMATION_TRAINER_LOG_RETENTION_COUNT", raising=False)
    return tmp_path


class TestRetentionCount:
    def test_기본_보존_개수는_10이다(self) -> None:
        assert DEFAULT_TRAINER_LOG_RETENTION_COUNT == 10

    def test_초과분은_mtime_오래된_순으로_삭제된다(self, log_dir: Path) -> None:
        # Arrange: N+3개를 mtime 오름차순으로 생성(run-00이 가장 오래됨)
        keep = DEFAULT_TRAINER_LOG_RETENTION_COUNT
        paths = [
            _write_trainer_log(log_dir, f"run-{i:02d}", 1_700_000_000 + i) for i in range(keep + 3)
        ]

        # Act
        sweep_trainer_logs(current_run_id="unrelated")

        # Assert
        survivors = sorted(p.name for p in log_dir.glob("trainer_*.log"))
        assert survivors == sorted(p.name for p in paths[3:])
        assert len(survivors) == keep

    def test_환경변수로_보존_개수를_오버라이드한다(
        self, log_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRAIN_AUTOMATION_TRAINER_LOG_RETENTION_COUNT", "2")
        for i in range(5):
            _write_trainer_log(log_dir, f"run-{i}", 1_700_000_000 + i)

        sweep_trainer_logs(current_run_id="unrelated")

        assert sorted(p.name for p in log_dir.glob("trainer_*.log")) == [
            "trainer_run-3.log",
            "trainer_run-4.log",
        ]

    def test_보존_개수_이하이면_아무것도_삭제하지_않는다(self, log_dir: Path) -> None:
        for i in range(DEFAULT_TRAINER_LOG_RETENTION_COUNT):
            _write_trainer_log(log_dir, f"run-{i:02d}", 1_700_000_000 + i)

        sweep_trainer_logs(current_run_id="unrelated")

        assert len(list(log_dir.glob("trainer_*.log"))) == DEFAULT_TRAINER_LOG_RETENTION_COUNT


class TestCurrentRunExclusion:
    def test_현재_run_id_파일은_가장_오래되어도_보존된다(self, log_dir: Path) -> None:
        # Arrange: 현재 run의 파일이 가장 오래된 mtime을 갖게 한다
        current = _write_trainer_log(log_dir, "current", 1_700_000_000)
        for i in range(DEFAULT_TRAINER_LOG_RETENTION_COUNT + 3):
            _write_trainer_log(log_dir, f"run-{i:02d}", 1_700_000_100 + i)

        # Act
        sweep_trainer_logs(current_run_id="current")

        # Assert
        assert current.exists()

    def test_현재_run_id_는_보존_정원을_잠식하지_않는다(self, log_dir: Path) -> None:
        current = _write_trainer_log(log_dir, "current", 1_700_000_000)
        keep = DEFAULT_TRAINER_LOG_RETENTION_COUNT
        others = [
            _write_trainer_log(log_dir, f"run-{i:02d}", 1_700_000_100 + i) for i in range(keep)
        ]

        sweep_trainer_logs(current_run_id="current")

        assert current.exists()
        assert all(p.exists() for p in others)


class TestNonTrainerFiles:
    def test_비_트레이너_파일은_건드리지_않는다(self, log_dir: Path) -> None:
        # Arrange: 라이브 NAS 실측상 같은 디렉토리에 공존하는 파일들
        untouched = [log_dir / "aaa-analyzer.log", log_dir / "aaa-analyzer.log.1"]
        for path in untouched:
            path.write_text("{}\n", encoding="utf-8")
            os.utime(path, (1_600_000_000, 1_600_000_000))
        for i in range(DEFAULT_TRAINER_LOG_RETENTION_COUNT + 3):
            _write_trainer_log(log_dir, f"run-{i:02d}", 1_700_000_000 + i)

        # Act
        sweep_trainer_logs(current_run_id="unrelated")

        # Assert
        assert all(path.exists() for path in untouched)


class TestFailOpen:
    def test_삭제_실패는_전파되지_않고_로그로만_남는다(
        self,
        log_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        for i in range(DEFAULT_TRAINER_LOG_RETENTION_COUNT + 3):
            _write_trainer_log(log_dir, f"run-{i:02d}", 1_700_000_000 + i)

        def _raise(self: Path, *args: object, **kwargs: object) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "unlink", _raise)

        # Act & Assert: 예외가 호출자로 전파되지 않는다
        with caplog.at_level(logging.ERROR, logger="analyzer.orchestration.log_retention"):
            sweep_trainer_logs(current_run_id="unrelated")

        assert any(record.levelno >= logging.ERROR for record in caplog.records)

    def test_존재하지_않는_디렉토리는_오류가_아니다(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_PATH", str(tmp_path / "missing"))

        sweep_trainer_logs(current_run_id="unrelated")

    def test_빈_디렉토리는_오류가_아니다(self, log_dir: Path) -> None:
        sweep_trainer_logs(current_run_id="unrelated")

    def test_잘못된_보존_개수_환경변수는_기본값으로_대체된다(
        self, log_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRAIN_AUTOMATION_TRAINER_LOG_RETENTION_COUNT", "not-a-number")
        for i in range(DEFAULT_TRAINER_LOG_RETENTION_COUNT + 2):
            _write_trainer_log(log_dir, f"run-{i:02d}", 1_700_000_000 + i)

        sweep_trainer_logs(current_run_id="unrelated")

        assert len(list(log_dir.glob("trainer_*.log"))) == DEFAULT_TRAINER_LOG_RETENTION_COUNT


class TestEnvExampleDocumented:
    def test_env_example에_보존_개수_항목이_있다(self) -> None:
        env_example = Path(__file__).resolve().parents[1] / ".env.example"
        assert "TRAIN_AUTOMATION_TRAINER_LOG_RETENTION_COUNT" in env_example.read_text(
            encoding="utf-8"
        )

"""src/analyzer/training/db.py trainer 계정 전용 엔진 빌더 테스트 (SPEC-ANALYZER-TRAIN-001 M1).

REQ-AT-010(trainer 계정 분리 상수)/REQ-AT-011(DbConfig 패턴 재사용,
MYSQL_TRAINER_PASSWORD 소비)을 검증한다. 실 DB 접속 없이 `create_engine`을
모킹한 경량 단위 테스트다 — REQ-AT-012(쓰기 권한 부재) 실증은 AC-AT-001의
통합 테스트(`@pytest.mark.integration`) 소관이며 이 파일의 범위 밖이다.
"""

from unittest.mock import MagicMock, patch

import pytest

from analyzer.data.config import DbConfig, MissingConfigError
from analyzer.training.db import build_trainer_engine, get_trainer_db_config


class TestGetTrainerDbConfig:
    """REQ-AT-011: MYSQL_TRAINER_PASSWORD를 소비하는 trainer 전용 DbConfig 로딩."""

    def test_reads_trainer_password_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MYSQL_HOST", "aaa-mysql")
        monkeypatch.setenv("MYSQL_PORT", "3306")
        monkeypatch.setenv("MYSQL_DATABASE", "aaa")
        monkeypatch.setenv("MYSQL_TRAINER_PASSWORD", "trainer-secret")

        config = get_trainer_db_config()

        assert config == DbConfig(
            host="aaa-mysql", port=3306, database="aaa", password="trainer-secret"
        )

    def test_raises_when_trainer_password_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MYSQL_HOST", "aaa-mysql")
        monkeypatch.setenv("MYSQL_PORT", "3306")
        monkeypatch.setenv("MYSQL_DATABASE", "aaa")
        monkeypatch.delenv("MYSQL_TRAINER_PASSWORD", raising=False)

        with pytest.raises(MissingConfigError, match="MYSQL_TRAINER_PASSWORD"):
            get_trainer_db_config()

    def test_analyzer_password_alone_is_not_sufficient(self, monkeypatch: pytest.MonkeyPatch):
        """REQ-AT-010: trainer 계정은 analyzer 계정과 별도 시크릿을 요구한다."""
        monkeypatch.setenv("MYSQL_HOST", "aaa-mysql")
        monkeypatch.setenv("MYSQL_PORT", "3306")
        monkeypatch.setenv("MYSQL_DATABASE", "aaa")
        monkeypatch.setenv("MYSQL_ANALYZER_PASSWORD", "analyzer-secret")
        monkeypatch.delenv("MYSQL_TRAINER_PASSWORD", raising=False)

        with pytest.raises(MissingConfigError):
            get_trainer_db_config()


class TestBuildTrainerEngine:
    """REQ-AT-010/011/012: trainer 계정 전용 SQLAlchemy 엔진 구성."""

    def test_builds_engine_using_trainer_account(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MYSQL_HOST", "aaa-mysql")
        monkeypatch.setenv("MYSQL_PORT", "3306")
        monkeypatch.setenv("MYSQL_DATABASE", "aaa")
        monkeypatch.setenv("MYSQL_TRAINER_PASSWORD", "trainer-secret")

        with patch("analyzer.training.db.create_engine") as mock_create_engine:
            mock_create_engine.return_value = MagicMock()

            build_trainer_engine()

        assert mock_create_engine.call_count == 1
        (url,), kwargs = mock_create_engine.call_args
        assert url == "mysql+pymysql://trainer:trainer-secret@aaa-mysql:3306/aaa"
        assert kwargs.get("pool_pre_ping") is True

    def test_takes_no_arguments(self):
        """AC-AT-001 worked example: `build_trainer_engine()`은 인자 없이 호출된다."""
        import inspect

        sig = inspect.signature(build_trainer_engine)
        assert len(sig.parameters) == 0

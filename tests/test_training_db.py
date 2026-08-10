"""src/analyzer/training/db.py trainer 계정 전용 엔진 빌더 테스트 (SPEC-ANALYZER-TRAIN-001 M1).

REQ-AT-010(trainer 계정 분리 상수)/REQ-AT-011(DbConfig 패턴 재사용,
MYSQL_TRAINER_PASSWORD 소비)을 검증한다. 실 DB 접속 없이 `create_engine`을
모킹한 경량 단위 테스트다 — REQ-AT-012(쓰기 권한 부재) 실증은
`TestTrainerWriteDeniedIntegration`(`@pytest.mark.integration`, 이 파일 하단)
소관이다.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

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


def _trainer_db_config_available() -> bool:
    """MYSQL_*/MYSQL_TRAINER_PASSWORD 환경변수가 이미 프로세스 환경에 로드되어
    있는지 확인한다(.env 파일은 읽지 않음) — `test_data_adjustment_e2e.py`의
    `_db_config_available()`과 동일한 패턴을 trainer 계정에 적용."""
    try:
        get_trainer_db_config()
    except MissingConfigError:
        return False
    return True


@pytest.mark.integration
@pytest.mark.skipif(
    not _trainer_db_config_available(),
    reason="MYSQL_TRAINER_PASSWORD 미설정 — 실 DB 접속 통합 테스트는 로컬/수동 실행 전용",
)
class TestTrainerWriteDeniedIntegration:
    """AC-AT-001 worked example 실증: trainer 계정으로 SELECT는 성공, INSERT는
    MySQL 권한 오류(1142 command denied류)로 실패해야 한다(REQ-AT-012)."""

    def test_select_1_succeeds(self):
        engine = build_trainer_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_insert_into_daily_ohlcv_denied(self):
        engine = build_trainer_engine()
        with engine.connect() as conn, pytest.raises(DBAPIError, match=r"1142"):
            conn.execute(
                text(
                    "INSERT INTO daily_ohlcv "
                    "(stock_code, trade_date, open_price, high_price, low_price, "
                    "close_price, volume) "
                    "VALUES (:stock_code, :trade_date, :open_price, :high_price, "
                    ":low_price, :close_price, :volume)"
                ),
                {
                    "stock_code": "__AC_AT_001_DUMMY__",
                    "trade_date": "1900-01-01",
                    "open_price": 0,
                    "high_price": 0,
                    "low_price": 0,
                    "close_price": 0,
                    "volume": 0,
                },
            )

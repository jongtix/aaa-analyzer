"""src/analyzer/data/config.py config 로딩 패턴 테스트 (SPEC-ANALYZER-DATA-001 M1, spec.md §4.8).

`os.environ` 직접 읽기 관례를 유지한다(pydantic-settings 미도입). DB 접속 정보와
REQ-AD-032 결제주기(settlement) config를 검증한다. `.env`/`.env.*` 파일 자체는
읽지 않는다 — 테스트는 monkeypatch로 환경변수를 주입한다.
"""

import pytest

from analyzer.data.config import (
    DbConfig,
    MissingConfigError,
    get_db_config,
    get_settlement_days,
)


class TestGetDbConfig:
    def test_reads_all_fields_from_environ(self, monkeypatch):
        monkeypatch.setenv("MYSQL_HOST", "aaa-mysql")
        monkeypatch.setenv("MYSQL_PORT", "3306")
        monkeypatch.setenv("MYSQL_DATABASE", "aaa")
        monkeypatch.setenv("MYSQL_ANALYZER_PASSWORD", "secret-value")

        config = get_db_config()

        assert config == DbConfig(
            host="aaa-mysql", port=3306, database="aaa", password="secret-value"
        )

    def test_missing_required_var_raises(self, monkeypatch):
        monkeypatch.delenv("MYSQL_HOST", raising=False)
        monkeypatch.setenv("MYSQL_PORT", "3306")
        monkeypatch.setenv("MYSQL_DATABASE", "aaa")
        monkeypatch.setenv("MYSQL_ANALYZER_PASSWORD", "secret-value")

        with pytest.raises(MissingConfigError, match="MYSQL_HOST"):
            get_db_config()

    def test_error_message_never_includes_password_value(self, monkeypatch):
        monkeypatch.setenv("MYSQL_HOST", "aaa-mysql")
        monkeypatch.setenv("MYSQL_PORT", "3306")
        monkeypatch.setenv("MYSQL_DATABASE", "aaa")
        monkeypatch.delenv("MYSQL_ANALYZER_PASSWORD", raising=False)

        with pytest.raises(MissingConfigError) as exc_info:
            get_db_config()

        assert "secret" not in str(exc_info.value).lower()


class TestGetSettlementDays:
    """REQ-AD-032: T+N 결제주기는 하드코딩이 아니라 config로 구현해야 한다."""

    def test_krx_default_is_t_plus_2(self, monkeypatch):
        monkeypatch.delenv("DIVIDEND_SETTLEMENT_DAYS_KRX", raising=False)

        assert get_settlement_days("KRX") == 2

    def test_us_default_is_t_plus_1(self, monkeypatch):
        monkeypatch.delenv("DIVIDEND_SETTLEMENT_DAYS_US", raising=False)

        assert get_settlement_days("US") == 1

    def test_krx_override_via_env_var(self, monkeypatch):
        """AC-AD-010: config 값(N) 변경 시 파생 ex_date 출력이 달라져야 한다."""
        monkeypatch.setenv("DIVIDEND_SETTLEMENT_DAYS_KRX", "1")

        assert get_settlement_days("KRX") == 1

    def test_unknown_market_raises(self, monkeypatch):
        with pytest.raises(MissingConfigError, match="UNKNOWN"):
            get_settlement_days("UNKNOWN")

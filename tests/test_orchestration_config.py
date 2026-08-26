"""자동화 설정 로딩 테스트 (SPEC-ANALYZER-TRAIN-AUTOMATION-001, plan.md §B.4).

`data/config.py`의 `DbConfig`/`get_db_config()` 패턴(REQ-AT-011 선례)을 그대로
계승한다 — 이미 프로세스 환경에 반영된 환경변수만 읽고 `.env`/`.env.*` 파일 자체는
열지 않는다.
"""

from pathlib import Path

import pytest

from analyzer.data.config import MissingConfigError
from analyzer.orchestration.config import AutomationConfig, get_automation_config

_REQUIRED_ENV = {
    "TRAIN_AUTOMATION_TARGET_MAC": "AA:BB:CC:DD:EE:FF",
    "TRAIN_AUTOMATION_SSH_HOST": "macbook.local",
    "TRAIN_AUTOMATION_SSH_USERNAME": "trainer-dispatch",
    "TRAIN_AUTOMATION_SSH_KEY_PATH": "/run/secrets/dispatch_key",
    "TRAIN_AUTOMATION_KNOWN_HOSTS_PATH": "/run/secrets/known_hosts",
    "TRAIN_AUTOMATION_STAGING_MODELS_ROOT": "/staging/models",
    "TRAIN_AUTOMATION_ACTIVE_MODELS_ROOT": "/models",
    "TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT": "/mnt/container-models",
    "TRAIN_AUTOMATION_DB_TUNNEL_HOST": "nas-host",
    "TRAIN_AUTOMATION_DB_TUNNEL_KEY_PATH": "/run/secrets/db_tunnel_key",
    "TRAIN_AUTOMATION_CACHE_DIR": "/cache",
    "TRAIN_AUTOMATION_MOUNT_SCRIPT_PATH": "/Users/mac/aaa/scripts/mount-nas-hdd1.sh",
    "TRAIN_AUTOMATION_PYTHON_PATH": "/Users/mac/aaa/aaa-analyzer/.venv/bin/python",
    "TRAIN_AUTOMATION_TRAINER_LOG_BASE_DIR": "/Users/mac/aaa/mnt/HDD_1/Development/aaa/logs",
    "MYSQL_DATABASE": "aaa",
    "MYSQL_TRAINER_PASSWORD": "trainer-secret",
}


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


class TestGetAutomationConfig:
    def test_reads_all_required_fields(self, monkeypatch: pytest.MonkeyPatch):
        _set_required_env(monkeypatch)

        config = get_automation_config()

        assert config == AutomationConfig(
            target_mac_address="AA:BB:CC:DD:EE:FF",
            ssh_host="macbook.local",
            ssh_port=22,
            ssh_username="trainer-dispatch",
            ssh_private_key_path=Path("/run/secrets/dispatch_key"),
            known_hosts_path=Path("/run/secrets/known_hosts"),
            db_tunnel_host="nas-host",
            db_tunnel_port=22,
            db_tunnel_username="db_tunnel",
            db_tunnel_private_key_path=Path("/run/secrets/db_tunnel_key"),
            db_tunnel_local_port=3306,
            db_tunnel_remote_port=3306,
            weekly_timeout_seconds=4 * 60 * 60,
            monthly_timeout_seconds=36 * 60 * 60,
            staleness_threshold_days=28,
            staging_models_root=Path("/staging/models"),
            active_models_root=Path("/models"),
            container_models_root=Path("/mnt/container-models"),
            cache_dir=Path("/cache"),
            calendar_code="KRX",
            feature_code_version="v1",
            mount_script_path=Path("/Users/mac/aaa/scripts/mount-nas-hdd1.sh"),
            python_executable_path=Path("/Users/mac/aaa/aaa-analyzer/.venv/bin/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/Users/mac/aaa/mnt/HDD_1/Development/aaa/logs"),
        )

    def test_raises_when_required_env_var_missing(self, monkeypatch: pytest.MonkeyPatch):
        _set_required_env(monkeypatch)
        monkeypatch.delenv("TRAIN_AUTOMATION_TARGET_MAC", raising=False)

        with pytest.raises(MissingConfigError, match="TRAIN_AUTOMATION_TARGET_MAC"):
            get_automation_config()

    def test_raises_listing_all_missing_vars(self, monkeypatch: pytest.MonkeyPatch):
        for key in _REQUIRED_ENV:
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(MissingConfigError) as exc_info:
            get_automation_config()
        for key in _REQUIRED_ENV:
            assert key in str(exc_info.value)

    def test_optional_ssh_port_override(self, monkeypatch: pytest.MonkeyPatch):
        _set_required_env(monkeypatch)
        monkeypatch.setenv("TRAIN_AUTOMATION_SSH_PORT", "2222")

        config = get_automation_config()

        assert config.ssh_port == 2222

    def test_optional_db_tunnel_port_override(self, monkeypatch: pytest.MonkeyPatch):
        """Stage 2 실측 검증(2026-08-13)에서 발견 — 나스 sshd가 비표준 포트를
        쓸 때 터널 SSH 접속 포트를 오버라이드할 수 있어야 한다."""
        _set_required_env(monkeypatch)
        monkeypatch.setenv("TRAIN_AUTOMATION_DB_TUNNEL_SSH_PORT", "55522")

        config = get_automation_config()

        assert config.db_tunnel_port == 55522

    def test_default_db_tunnel_port_is_twenty_two(self, monkeypatch: pytest.MonkeyPatch):
        _set_required_env(monkeypatch)

        config = get_automation_config()

        assert config.db_tunnel_port == 22

    def test_optional_timeout_overrides(self, monkeypatch: pytest.MonkeyPatch):
        _set_required_env(monkeypatch)
        monkeypatch.setenv("TRAIN_AUTOMATION_WEEKLY_TIMEOUT_SECONDS", "1800")
        monkeypatch.setenv("TRAIN_AUTOMATION_MONTHLY_TIMEOUT_SECONDS", "3600")
        monkeypatch.setenv("TRAIN_AUTOMATION_STALENESS_THRESHOLD_DAYS", "14")

        config = get_automation_config()

        assert config.weekly_timeout_seconds == 1800
        assert config.monthly_timeout_seconds == 3600
        assert config.staleness_threshold_days == 14

    def test_default_weekly_timeout_is_four_hours(self, monkeypatch: pytest.MonkeyPatch):
        """REQ-ATA-040: 초기값은 주간 4시간이며, 하드코딩이 아닌 외부화된 설정으로
        제공되어야 한다(오버라이드 미설정 시 기본값)."""
        _set_required_env(monkeypatch)

        config = get_automation_config()

        assert config.weekly_timeout_seconds == 4 * 60 * 60

    def test_default_monthly_timeout_is_thirty_six_hours(self, monkeypatch: pytest.MonkeyPatch):
        """REQ-ATA-040: 초기값은 월간 36시간."""
        _set_required_env(monkeypatch)

        config = get_automation_config()

        assert config.monthly_timeout_seconds == 36 * 60 * 60

    def test_default_staleness_threshold_is_four_weeks(self, monkeypatch: pytest.MonkeyPatch):
        """REQ-ATA-072: 마지막 성공 재학습 후 4주(28일) 초과 시 정체."""
        _set_required_env(monkeypatch)

        config = get_automation_config()

        assert config.staleness_threshold_days == 28

    def test_optional_calendar_code_and_feature_version_override(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _set_required_env(monkeypatch)
        monkeypatch.setenv("TRAIN_AUTOMATION_CALENDAR_CODE", "NASDAQ")
        monkeypatch.setenv("TRAIN_AUTOMATION_FEATURE_CODE_VERSION", "v2")

        config = get_automation_config()

        assert config.calendar_code == "NASDAQ"
        assert config.feature_code_version == "v2"

    def test_mount_script_path_reads_configured_value(self, monkeypatch: pytest.MonkeyPatch):
        """SPEC-ANALYZER-TRAIN-AUTOMATION-001: ssh_key_path/known_hosts_path와
        동일 계열(계정별 절대경로)의 필수 항목이므로 기본값 없이 설정값을 그대로
        읽어야 한다."""
        _set_required_env(monkeypatch)
        monkeypatch.setenv(
            "TRAIN_AUTOMATION_MOUNT_SCRIPT_PATH",
            "/Users/other/Development/aaa/scripts/mount-nas-hdd1.sh",
        )

        config = get_automation_config()

        assert config.mount_script_path == Path(
            "/Users/other/Development/aaa/scripts/mount-nas-hdd1.sh"
        )

    def test_python_executable_path_reads_configured_value(self, monkeypatch: pytest.MonkeyPatch):
        """SPEC-ANALYZER-TRAIN-AUTOMATION-001: mount_script_path와 동일 계열
        (계정별 절대경로)의 필수 항목이므로 기본값 없이 설정값을 그대로
        읽어야 한다(수동 실행 실측, 2026-08-13 — 원격 비대화형 셸 PATH에
        `python`이 없어 종료코드 127 발생 후 도입)."""
        _set_required_env(monkeypatch)
        monkeypatch.setenv(
            "TRAIN_AUTOMATION_PYTHON_PATH",
            "/Users/other/Development/aaa/aaa-analyzer/.venv/bin/python",
        )

        config = get_automation_config()

        assert config.python_executable_path == Path(
            "/Users/other/Development/aaa/aaa-analyzer/.venv/bin/python"
        )

    def test_raises_when_container_models_root_missing(self, monkeypatch: pytest.MonkeyPatch):
        """AC-ATD-003 (SPEC-ANALYZER-TRAIN-STALENESS-001): 나머지 15종이 모두
        설정된 상태에서 신규 필수 변수 TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT만
        누락되면 MissingConfigError 메시지에 그 변수명이 포함되어야 한다 —
        기존 일괄 검증 경로(get_automation_config())에 자동으로 편입된다."""
        _set_required_env(monkeypatch)
        monkeypatch.delenv("TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT", raising=False)

        with pytest.raises(MissingConfigError, match="TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT"):
            get_automation_config()

    def test_container_models_root_reads_configured_value(self, monkeypatch: pytest.MonkeyPatch):
        """AC-ATD-003: 컨테이너 내부 마운트 경로 전용 신규 변수는 기존
        TRAIN_AUTOMATION_ACTIVE_MODELS_ROOT(맥북 SMB 경로)와 별개로, 설정된
        값을 그대로 AutomationConfig.container_models_root에 채운다."""
        _set_required_env(monkeypatch)
        monkeypatch.setenv("TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT", "/mnt/other-container-models")

        config = get_automation_config()

        assert config.container_models_root == Path("/mnt/other-container-models")

    def test_mysql_database_and_trainer_password_reuse_container_env(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """수동 실행 실측(2026-08-13) — 원격 학습 CLI(training/db.py
        get_trainer_db_config())가 요구하는 MYSQL_DATABASE/
        MYSQL_TRAINER_PASSWORD가 원격 비대화형 셸에 전혀 전달되지 않아
        MissingConfigError로 실패했다. 신규 시크릿이 아니라 컨테이너
        자신의 동일 이름 env var를 그대로 재사용한다."""
        _set_required_env(monkeypatch)
        monkeypatch.setenv("MYSQL_DATABASE", "aaa")
        monkeypatch.setenv("MYSQL_TRAINER_PASSWORD", "reused-secret")

        config = get_automation_config()

        assert config.mysql_database == "aaa"
        assert config.mysql_trainer_password == "reused-secret"

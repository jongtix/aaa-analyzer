"""추론 서버 설정 로딩 명세 테스트 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-020).

`InferenceConfig`는 Redis 접속 정보와 XAUTOCLAIM idle 임계를 담는다. 접속
정보의 환경변수명은 `.env.common`이 이미 제공하는 collector와 동일한 4개
(`REDIS_HOST`/`REDIS_PORT`/`REDIS_APPUSER_USERNAME`/`REDIS_APPUSER_PASSWORD`)를
그대로 재사용한다 — analyzer 전용 신규 시크릿을 만들지 않는다.
"""

from pathlib import Path

import pytest

from analyzer.data.config import MissingConfigError
from analyzer.inference.config import (
    DEFAULT_STREAM_CLAIM_IDLE_SECONDS,
    MINIMUM_STREAM_CLAIM_IDLE_SECONDS,
    InferenceConfig,
    get_inference_config,
)

_REQUIRED_ENV = {
    "REDIS_HOST": "redis",
    "REDIS_PORT": "6379",
    "REDIS_APPUSER_USERNAME": "appuser",
    "REDIS_APPUSER_PASSWORD": "redis-secret",
    "TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT": "/mnt/models",
}


def _set_required(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("INFERENCE_STREAM_CLAIM_IDLE_SECONDS", raising=False)


class TestGetInferenceConfig:
    def test_reads_redis_connection_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)

        config = get_inference_config()

        assert config == InferenceConfig(
            redis_host="redis",
            redis_port=6379,
            redis_username="appuser",
            redis_password="redis-secret",
            stream_claim_idle_seconds=DEFAULT_STREAM_CLAIM_IDLE_SECONDS,
            container_models_root=Path("/mnt/models"),
        )

    def test_default_idle_threshold_is_thirty_minutes(self):
        """REQ-AIF-020: 확정값 1800초(30분) — M7 실측(2026-09-12) 근거로
        착수 시점 잠정값(600초)에서 상향."""
        assert DEFAULT_STREAM_CLAIM_IDLE_SECONDS == 1800

    def test_idle_threshold_can_be_overridden(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("INFERENCE_STREAM_CLAIM_IDLE_SECONDS", "900")

        assert get_inference_config().stream_claim_idle_seconds == 900

    def test_short_idle_threshold_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        """REQ-AIF-020(shall not): 실제 추론 사이클보다 짧은 값(예: 30초)은
        자기유발 중복 자식 스폰을 유발하므로 환경변수 오버라이드로도 허용하지
        않는다."""
        _set_required(monkeypatch)
        monkeypatch.setenv("INFERENCE_STREAM_CLAIM_IDLE_SECONDS", "30")

        with pytest.raises(ValueError) as excinfo:
            get_inference_config()

        assert str(MINIMUM_STREAM_CLAIM_IDLE_SECONDS) in str(excinfo.value)

    def test_missing_variables_are_reported_together(self, monkeypatch: pytest.MonkeyPatch):
        for name in _REQUIRED_ENV:
            monkeypatch.delenv(name, raising=False)

        with pytest.raises(MissingConfigError) as excinfo:
            get_inference_config()

        for name in _REQUIRED_ENV:
            assert name in str(excinfo.value)

    def test_secret_value_is_not_exposed_in_error_message(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        monkeypatch.delenv("REDIS_HOST", raising=False)

        with pytest.raises(MissingConfigError) as excinfo:
            get_inference_config()

        assert "redis-secret" not in str(excinfo.value)


class TestContainerModelsRoot:
    """SPEC-ANALYZER-PIPELINE-001 AC-APL-120/REQ-APL-120: `orchestration/
    config.py`의 `AutomationConfig.container_models_root`와 동일한 환경변수명을
    재사용하되, SSH/WoL 등 추론과 무관한 `AutomationConfig` 필드를 요구하지
    않는다."""

    def test_reads_container_models_root_without_automation_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _set_required(monkeypatch)
        # AutomationConfig가 요구하는 SSH/WoL 관련 환경변수는 의도적으로 미설정.
        for name in (
            "TRAIN_AUTOMATION_TARGET_MAC",
            "TRAIN_AUTOMATION_SSH_HOST",
            "TRAIN_AUTOMATION_SSH_USERNAME",
        ):
            monkeypatch.delenv(name, raising=False)

        config = get_inference_config()

        assert config.container_models_root == Path("/mnt/models")

    def test_missing_container_models_root_is_reported(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        monkeypatch.delenv("TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT", raising=False)

        with pytest.raises(MissingConfigError) as excinfo:
            get_inference_config()

        assert "TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT" in str(excinfo.value)

    def test_does_not_import_automation_config_module(self):
        """`inference/config.py`가 `orchestration.config`(AutomationConfig)를
        임포트하지 않는다는 정적 가드(Lazy Load 원칙 위반 방지) — 설명용
        docstring 언급은 허용하고 실제 `import` 구문만 검사한다."""
        source = Path("src/analyzer/inference/config.py").read_text(encoding="utf-8")
        assert "from analyzer.orchestration.config import" not in source
        assert "import analyzer.orchestration.config" not in source


class TestBuildRedisClient:
    def test_client_is_configured_from_config_without_connecting(self):
        """`redis.Redis` 생성자는 소켓을 열지 않으므로, 접속 파라미터가 config
        그대로 전달됐는지 연결 없이 검증할 수 있다."""
        from analyzer.inference.redis_client import build_redis_client

        client = build_redis_client(
            InferenceConfig(
                redis_host="redis-host",
                redis_port=6380,
                redis_username="appuser",
                redis_password="redis-secret",
                stream_claim_idle_seconds=600,
                container_models_root=Path("/mnt/models"),
            )
        )

        kwargs = client.connection_pool.connection_kwargs
        assert kwargs["host"] == "redis-host"
        assert kwargs["port"] == 6380
        assert kwargs["username"] == "appuser"
        assert kwargs["decode_responses"] is True

"""SSH 연결 재시도 + MySQL 터널 + 학습 원격 디스패치 + 타임아웃 강제 테스트.

REQ-ATA-020/021/022/030/031/032/040/041/050/051, AC-ATA-001/003/006/009/011.

`SshConnection`은 `typing.Protocol`로 정의되어 실 네트워크 없이 페이크 구현으로
단위 테스트 가능하다(`WolSender`와 동일한 설계 원칙, plan.md §B.1). 실제 네트워크를
사용하는 통합 테스트만 `@pytest.mark.integration`으로 표시한다(acceptance.md §C).
"""

import shlex
from collections.abc import Callable
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import paramiko
import pytest

from analyzer.orchestration.ssh_dispatch import (
    CommandResult,
    ParamikoSshConnection,
    SshKeyPermissionError,
    build_remote_dispatch_command,
    connect_with_retry,
    promote_staging_to_active,
    validate_private_key_permissions,
)


def _no_sleep(_seconds: float) -> None:
    """시간이 흐르는 것처럼 보이게만 하고 실제로 대기하지 않는 테스트용 sleep_fn."""


def _make_connection_with_mocked_client(
    tmp_path: Path,
    *,
    time_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] = _no_sleep,
) -> ParamikoSshConnection:
    """`self._client`를 `MagicMock()`으로 교체해 실 네트워크 없이 I/O 메서드를
    검증하기 위한 헬퍼(초기화 자체는 실 `paramiko.SSHClient()`로 수행).

    `time_fn`/`sleep_fn`은 REQ-ATO-009 자체 데드라인 추적 폴링 루프를 실제
    대기 없이 결정론적으로 테스트하기 위한 주입 지점이다(기본값은 실제
    `time.monotonic`을 그대로 쓰되 `sleep_fn`만 무동작으로 대체한다)."""
    key_path = tmp_path / "id_ed25519"
    key_path.write_text("fake-key-material")
    key_path.chmod(0o600)
    known_hosts_path = tmp_path / "known_hosts"
    known_hosts_path.write_text("")

    kwargs: dict = {}
    if time_fn is not None:
        kwargs["time_fn"] = time_fn

    connection = ParamikoSshConnection(
        host="macbook.local",
        port=22,
        username="dispatch",
        private_key_path=key_path,
        known_hosts_path=known_hosts_path,
        sleep_fn=sleep_fn,
        **kwargs,
    )
    connection._client = MagicMock()  # pyright: ignore[reportAttributeAccessIssue]
    return connection


class _FakeSshConnection:
    """단위 테스트 전용 페이크 — 연결 시도/명령 실행 호출을 기록한다."""

    def __init__(
        self,
        *,
        connect_failures: int = 0,
        command_results: list[CommandResult] | None = None,
    ) -> None:
        self._connect_failures_remaining = connect_failures
        self._command_results = list(command_results or [])
        self.connect_attempts = 0
        self.executed_commands: list[tuple[str, float]] = []
        self.closed = False

    def connect(self) -> None:
        self.connect_attempts += 1
        if self._connect_failures_remaining > 0:
            self._connect_failures_remaining -= 1
            raise ConnectionError("SSH 연결 실패(페이크)")

    def exec_command(
        self,
        command: str,
        timeout_seconds: float,
        *,
        on_output_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        self.executed_commands.append((command, timeout_seconds))
        if self._command_results:
            return self._command_results.pop(0)
        return CommandResult(exit_code=0)

    def close(self) -> None:
        self.closed = True


class TestValidatePrivateKeyPermissions:
    """REQ-ATA-022: 프라이빗 키 파일 모드는 600이어야 한다."""

    def test_passes_when_mode_is_600(self, tmp_path: Path):
        key_path = tmp_path / "id_ed25519"
        key_path.write_text("fake-key-material")
        key_path.chmod(0o600)

        validate_private_key_permissions(key_path)  # 예외 없이 통과해야 한다

    def test_raises_when_mode_is_644(self, tmp_path: Path):
        key_path = tmp_path / "id_ed25519"
        key_path.write_text("fake-key-material")
        key_path.chmod(0o644)

        with pytest.raises(SshKeyPermissionError, match="600"):
            validate_private_key_permissions(key_path)

    def test_raises_when_mode_is_777(self, tmp_path: Path):
        key_path = tmp_path / "id_ed25519"
        key_path.write_text("fake-key-material")
        key_path.chmod(0o777)

        with pytest.raises(SshKeyPermissionError):
            validate_private_key_permissions(key_path)


class TestParamikoSshConnectionSecurityPolicy:
    """AC-ATA-009: known_hosts 핀 고정 사용, StrictHostKeyChecking=no(AutoAddPolicy) 미사용."""

    def test_uses_reject_policy_not_auto_add(self, tmp_path: Path):
        key_path = tmp_path / "id_ed25519"
        key_path.write_text("fake-key-material")
        key_path.chmod(0o600)
        known_hosts_path = tmp_path / "known_hosts"
        known_hosts_path.write_text("")

        connection = ParamikoSshConnection(
            host="macbook.local",
            port=22,
            username="dispatch",
            private_key_path=key_path,
            known_hosts_path=known_hosts_path,
        )

        policy = connection._client._policy  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(policy, paramiko.RejectPolicy)

    def test_raises_before_connecting_when_key_permissions_invalid(self, tmp_path: Path):
        key_path = tmp_path / "id_ed25519"
        key_path.write_text("fake-key-material")
        key_path.chmod(0o644)
        known_hosts_path = tmp_path / "known_hosts"
        known_hosts_path.write_text("")

        with pytest.raises(SshKeyPermissionError):
            ParamikoSshConnection(
                host="macbook.local",
                port=22,
                username="dispatch",
                private_key_path=key_path,
                known_hosts_path=known_hosts_path,
            )

    def test_loads_known_hosts_file(self, tmp_path: Path):
        """known_hosts 핀 고정 — 파일 내용이 client의 호스트 키 저장소에 로드된다."""
        key_path = tmp_path / "id_ed25519"
        key_path.write_text("fake-key-material")
        key_path.chmod(0o600)
        known_hosts_path = tmp_path / "known_hosts"
        # ssh-keygen -A 스타일 테스트용 더미 known_hosts 라인
        known_hosts_path.write_text(
            "macbook.local ssh-ed25519 "
            "AAAAC3NzaC1lZDI1NTE5AAAAIBhH1v3z0z9j3v3z0z9j3v3z0z9j3v3z0z9j3v3z\n"
        )

        connection = ParamikoSshConnection(
            host="macbook.local",
            port=22,
            username="dispatch",
            private_key_path=key_path,
            known_hosts_path=known_hosts_path,
        )

        loaded_hosts = connection._client.get_host_keys()  # noqa: SLF001
        assert "macbook.local" in loaded_hosts


def _stub_result(exit_code: int = 0, timed_out: bool = False) -> CommandResult:
    return CommandResult(exit_code=exit_code, timed_out=timed_out)


class TestConnectWithRetry:
    """REQ-ATA-021: SSH 연결 실패 시 10초 간격으로 최대 6회까지 재시도."""

    def test_succeeds_on_first_attempt(self):
        connection = _FakeSshConnection(connect_failures=0)
        sleeps: list[float] = []

        result = connect_with_retry(
            connection, max_retries=6, interval_seconds=10, sleep_fn=sleeps.append
        )

        assert result is True
        assert connection.connect_attempts == 1
        assert sleeps == []

    def test_succeeds_after_retries_with_correct_sleep_interval(self):
        connection = _FakeSshConnection(connect_failures=2)
        sleeps: list[float] = []

        result = connect_with_retry(
            connection, max_retries=6, interval_seconds=10, sleep_fn=sleeps.append
        )

        assert result is True
        assert connection.connect_attempts == 3
        assert sleeps == [10, 10]

    def test_fails_after_max_retries_exhausted(self):
        connection = _FakeSshConnection(connect_failures=10)
        sleeps: list[float] = []

        result = connect_with_retry(
            connection, max_retries=6, interval_seconds=10, sleep_fn=sleeps.append
        )

        assert result is False
        assert connection.connect_attempts == 6
        assert sleeps == [10] * 5  # 마지막 실패 시도 뒤에는 대기하지 않는다


class TestBuildRemoteDispatchCommand:
    """REQ-ATA-030/031: MySQL 터널 수립 → TRAIN-001 CLI 원격 호출(스테이징 경로) →
    터널 해제(trap)까지 단일 원격 셸 스크립트로 구성한다."""

    def test_includes_db_tunnel_ssh_command(self):
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-1",
            db_tunnel_username="db_tunnel",
            db_tunnel_local_port=3306,
            db_tunnel_remote_port=3306,
        )

        assert "db_tunnel@nas-host" in command
        assert "-L 3306:127.0.0.1:3306" in command
        assert "/run/secrets/db_tunnel_key" in command
        assert "-p 22" in command  # 미지정 시 기본값 22

    def test_db_tunnel_port_uses_configured_value(self):
        """Stage 2 실측 검증(2026-08-13)에서 발견 — db_tunnel_port는 -L 포워딩
        포트(db_tunnel_local_port/remote_port)와 별개로, 터널 SSH 접속 자체의
        포트여야 한다(나스가 비표준 포트를 쓰는 경우 필수)."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-2",
            db_tunnel_port=55522,
            db_tunnel_local_port=3306,
            db_tunnel_remote_port=3306,
        )

        assert "-p 55522" in command
        assert "-L 3306:127.0.0.1:3306" in command  # 포워딩 포트는 그대로 별개 유지

    def test_does_not_use_strict_host_key_checking_no_for_dispatch(self):
        """REQ-ATA-022 준수 — db_tunnel 터널 SSH 명령도 StrictHostKeyChecking=no를 쓰지 않는다."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-3",
        )

        assert "StrictHostKeyChecking=no" not in command

    def test_includes_training_cli_with_staging_models_root(self):
        """plan.md §B.5(D6): --models-root는 활성 경로가 아니라 스테이징 경로여야 한다."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-4",
        )

        assert "python -m analyzer.training.train" in command
        assert "--models-root /staging/run-1" in command
        assert "--calendar-code KRX" in command
        assert "--cache-dir /cache" in command
        assert "--data-as-of 2026-08-11" in command
        assert "--feature-code-version v1" in command

    def test_uses_configured_python_executable_path_not_bare_python(self):
        """수동 실행 실측(2026-08-13) — 원격 비대화형 셸 PATH엔 `python`이 없고
        (`python3`만 존재), `python3`도 analyzer 패키지가 없는 시스템 파이썬을
        가리켜 종료코드 127로 실패한다. venv python 절대경로를 그대로 써야 한다."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/Users/mac/aaa/aaa-analyzer/.venv/bin/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-5",
        )

        assert "/Users/mac/aaa/aaa-analyzer/.venv/bin/python -m analyzer.training.train" in command

    def test_injects_mysql_env_vars_required_by_train_db_config(self):
        """수동 실행 실측(2026-08-13) — training/db.py get_trainer_db_config()가
        요구하는 MYSQL_HOST/PORT/DATABASE/TRAINER_PASSWORD가 원격 비대화형
        셸에 없어 MissingConfigError로 즉시 실패했다. MYSQL_HOST/PORT는
        컨테이너 자신의 값이 아니라 db_tunnel 로컬 포워딩으로 고정한다."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-6",
            db_tunnel_local_port=13306,
        )

        assert "MYSQL_HOST=127.0.0.1 MYSQL_PORT=13306" in command
        assert "MYSQL_DATABASE=aaa" in command
        assert "MYSQL_TRAINER_PASSWORD=trainer-secret" in command
        assert (
            "MYSQL_TRAINER_PASSWORD=trainer-secret TRAIN_RUN_ID=run-6 "
            "/python -m analyzer.training.train"
        ) in command

    def test_mysql_trainer_password_is_shlex_quoted(self):
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="pass with spaces",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-7",
        )

        assert "MYSQL_TRAINER_PASSWORD='pass with spaces'" in command

    def test_python_executable_path_is_shlex_quoted(self):
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/custom/venv bin/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-8",
        )

        assert "'/custom/venv bin/python' -m analyzer.training.train" in command

    def test_tears_down_tunnel_on_exit_regardless_of_training_outcome(self):
        """REQ-ATA-032: 학습 실행 종료 시(성공·실패 무관) 터널을 해제한다 —
        `trap ... EXIT`로 원격 스크립트 자체 종료 경로에서 항상 정리되도록 구성한다."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-9",
        )

        assert "trap" in command
        assert "EXIT" in command

    def test_propagates_training_script_exit_code(self):
        """학습 스크립트의 실제 종료코드가 최종 exec_command 종료코드로 전달되어야
        한다(REQ-ATA-050 — 종료코드가 1차 완료 감지 근거)."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-10",
        )

        assert command.rstrip().endswith("exit $?") or "exit $TRAIN_EXIT_CODE" in command

    def test_shell_metacharacters_in_interpolated_values_are_safely_quoted(self):
        """F1: 원격 셸 명령 조립 시 각 값은 shlex.quote로 이스케이프되어야 한다 —
        그렇지 않으면 값에 셸 메타문자가 섞여도 별도 명령으로 실행될 수 있다."""
        malicious_calendar_code = "KRX; touch /tmp/pwned"

        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code=malicious_calendar_code,
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-11",
        )

        # 이스케이프되지 않은 형태(따옴표 없이 노출)로는 존재하지 않아야 한다 —
        # shlex.quote로 감싸져 셸이 하나의 인자로만 해석해야 한다.
        assert f"--calendar-code {malicious_calendar_code} " not in command
        assert shlex.quote(malicious_calendar_code) in command

    def test_all_interpolated_values_pass_through_shlex_quote(self):
        """공백을 포함한 모든 보간 값이 안전하게 인용되어 하나의 토큰으로
        유지되어야 한다 — cache_dir, staging_models_root 등."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run 1"),
            calendar_code="KRX",
            cache_dir=Path("/cache dir"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db tunnel key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-12",
        )

        assert shlex.quote("/staging/run 1") in command
        assert shlex.quote("/cache dir") in command
        assert shlex.quote("/run/secrets/db tunnel key") in command

    def test_mount_script_gates_training_cli_via_and(self):
        """마운트 실패 시 학습 CLI가 실행되지 않아야 한다 — 스테이징 경로가
        SMB 마운트포인트 하위이므로 마운트 미완료 상태에서 쓰기가 진행되면
        안 된다."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/custom/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-13",
        )

        assert "/custom/mount.sh && MYSQL_HOST=127.0.0.1" in command
        assert "/python -m analyzer.training.train" in command

    def test_mount_script_path_is_shlex_quoted(self):
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/custom/mount script.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-14",
        )

        assert shlex.quote("/custom/mount script.sh") in command

    def test_includes_trainer_file_redirection_with_run_id_filename(self):
        """AC-ATO-005/REQ-ATO-004/005: 트레이너 파일은 `trainer_<run_id>.log`
        이름으로 `trainer_log_base_dir` 아래 원문 stdout/stderr 전체를 기록한다."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-abc123",
        )

        expected_path = shlex.quote("/logs/analyzer/trainer_run-abc123.log")
        assert f"2>&1 | tee {expected_path}" in command

    def test_trainer_file_redirection_starts_after_mount_gate(self):
        """AC-ATO-018(REQ-ATO-008): 트레이너 파일 리다이렉션(tee)은 마운트
        확인 명령의 `&&` 이후에 위치해야 한다 — 마운트 실패 시 리다이렉션이
        시도되지 않아야 한다."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/custom/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-15",
        )

        mount_gate_index = command.index("/custom/mount.sh &&")
        tee_index = command.index("| tee ")
        assert mount_gate_index < tee_index

    def test_trainer_log_base_dir_is_shlex_quoted(self):
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs dir/analyzer"),
            run_id="run-16",
        )

        assert shlex.quote("/logs dir/analyzer/trainer_run-16.log") in command

    def test_includes_train_run_id_env_var(self):
        """AC-ATO-008(REQ-ATO-012/013): NAS에서 발급된 run_id가 원격 CLI로
        TRAIN_RUN_ID env var로 전달되어야 한다."""
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id="run-xyz789",
        )

        assert "TRAIN_RUN_ID=run-xyz789" in command

    def test_train_run_id_is_shlex_quoted(self):
        """REQ-ATO-014: run_id 전파는 기존 안전한 셸 조립 관례(shlex.quote)를
        그대로 따라야 한다 — 값에 셸 메타문자가 포함돼도 안전하게 전달된다."""
        malicious_run_id = "run; touch /tmp/pwned"
        command = build_remote_dispatch_command(
            staging_models_root=Path("/staging/run-1"),
            calendar_code="KRX",
            cache_dir=Path("/cache"),
            data_as_of=date(2026, 8, 11),
            feature_code_version="v1",
            db_tunnel_host="nas-host",
            db_tunnel_key_path=Path("/run/secrets/db_tunnel_key"),
            mount_script_path=Path("/mount.sh"),
            python_executable_path=Path("/python"),
            mysql_database="aaa",
            mysql_trainer_password="trainer-secret",
            trainer_log_base_dir=Path("/logs/analyzer"),
            run_id=malicious_run_id,
        )

        assert shlex.quote(malicious_run_id) in command
        assert f"TRAIN_RUN_ID={malicious_run_id} " not in command


class TestPromoteStagingToActive:
    """plan.md §B.5(D6) 사전 차단 설계 — SSH 종료코드 0 확인 후에만 호출되어야 한다."""

    def test_issues_promotion_command_and_returns_true_on_success(self):
        connection = _FakeSshConnection(command_results=[_stub_result(exit_code=0)])

        promoted = promote_staging_to_active(connection, Path("/staging/run-1"), Path("/models"))

        assert promoted is True
        assert len(connection.executed_commands) == 1
        command, _timeout = connection.executed_commands[0]
        assert "/staging/run-1" in command
        assert "/models" in command

    def test_returns_false_when_remote_command_fails(self):
        connection = _FakeSshConnection(command_results=[_stub_result(exit_code=1)])

        promoted = promote_staging_to_active(connection, Path("/staging/run-1"), Path("/models"))

        assert promoted is False

    def test_shell_metacharacters_in_paths_are_safely_quoted(self):
        """F1과 동일 계열: staging_path/active_path도 shlex.quote로 이스케이프되어야
        한다 — 그렇지 않으면 공백/셸 메타문자가 섞인 경로가 `rm -rf` 등에서
        의도치 않은 명령 분리·주입으로 이어질 수 있다."""
        connection = _FakeSshConnection(command_results=[_stub_result(exit_code=0)])
        malicious_staging_path = Path("/staging/run 1; rm -rf /")
        malicious_active_path = Path("/models active")

        promote_staging_to_active(connection, malicious_staging_path, malicious_active_path)

        command, _timeout = connection.executed_commands[0]
        assert shlex.quote(str(malicious_staging_path)) in command
        assert shlex.quote(str(malicious_active_path)) in command
        assert f"rm -rf {malicious_staging_path}" not in command


class TestParamikoSshConnectionMockedIO:
    """`self._client`(`paramiko.SSHClient`)를 `MagicMock()`으로 교체해 실 네트워크
    없이 `connect`/`exec_command`/`close`의 위임 로직을 검증한다."""

    def test_connect_delegates_with_expected_arguments(self, tmp_path: Path):
        connection = _make_connection_with_mocked_client(tmp_path)

        connection.connect()

        connection._client.connect.assert_called_once_with(  # pyright: ignore[reportAttributeAccessIssue]
            hostname="macbook.local",
            port=22,
            username="dispatch",
            key_filename=str(tmp_path / "id_ed25519"),
            look_for_keys=False,
            allow_agent=False,
            timeout=connection._connect_timeout_seconds,  # pyright: ignore[reportAttributeAccessIssue]
        )

    def test_connect_passes_explicit_timeout_to_avoid_unbounded_block(self, tmp_path: Path):
        """F2: connect()에 timeout이 없으면 paramiko가 무기한 블로킹할 수 있어
        REQ-ATA-021의 10초×6회 재시도 설계가 무력화된다."""
        connection = _make_connection_with_mocked_client(tmp_path)

        connection.connect()

        _args, kwargs = connection._client.connect.call_args  # pyright: ignore[reportAttributeAccessIssue]
        assert "timeout" in kwargs
        assert isinstance(kwargs["timeout"], (int, float))
        assert kwargs["timeout"] > 0

    def test_connect_timeout_is_configurable_via_constructor(self, tmp_path: Path):
        key_path = tmp_path / "id_ed25519"
        key_path.write_text("fake-key-material")
        key_path.chmod(0o600)
        known_hosts_path = tmp_path / "known_hosts"
        known_hosts_path.write_text("")

        connection = ParamikoSshConnection(
            host="macbook.local",
            port=22,
            username="dispatch",
            private_key_path=key_path,
            known_hosts_path=known_hosts_path,
            connect_timeout_seconds=5.0,
        )
        connection._client = MagicMock()  # pyright: ignore[reportAttributeAccessIssue]

        connection.connect()

        _args, kwargs = connection._client.connect.call_args  # pyright: ignore[reportAttributeAccessIssue]
        assert kwargs["timeout"] == 5.0

    def test_exec_command_returns_exit_code_from_channel(self, tmp_path: Path):
        """D1(계층 B, 의도적 갱신) — 신규 폴링 드레인 루프에 맞춰 mock 설정을
        갱신한다: recv_ready/recv_stderr_ready는 출력 없음(False), exit_status_ready는
        즉시 True로 종료코드 획득 경로만으로 완료를 판정하게 한다."""
        connection = _make_connection_with_mocked_client(tmp_path)
        mock_transport = MagicMock()
        mock_channel = MagicMock()
        mock_channel.recv_ready.return_value = False
        mock_channel.recv_stderr_ready.return_value = False
        mock_channel.exit_status_ready.return_value = True
        mock_channel.recv_exit_status.return_value = 0
        mock_transport.open_session.return_value = mock_channel
        connection._client.get_transport.return_value = mock_transport  # pyright: ignore[reportAttributeAccessIssue]

        result = connection.exec_command("echo hi", timeout_seconds=5.0)

        assert result == CommandResult(exit_code=0)
        # REQ-ATO-010: settimeout()은 더 이상 timeout_seconds를 그대로 넘기지
        # 않는다 — recv_exit_status()가 settimeout 값을 준수하지 않으므로,
        # 타임아웃 강제는 이 메서드 자체의 데드라인 추적이 전담한다(0.0은
        # recv()/recv_stderr() 논블로킹 폴링 전용 설정).
        mock_channel.settimeout.assert_called_once_with(0.0)
        mock_channel.exec_command.assert_called_once_with("echo hi")

    def test_exec_command_returns_timed_out_result_on_timeout(self, tmp_path: Path):
        """D1(계층 B, 의도적 갱신) — 원격 프로세스 종료코드 대기 API(exit_status_ready)가
        영구히 False를 반환하도록(REQ-ATA-041 결함 실측 재현) 시뮬레이션해도,
        읽기 루프 자체의 자체 데드라인 추적이 타임아웃을 강제해야 한다(REQ-ATO-009)."""
        call_count = 0

        def _incrementing_clock() -> float:
            nonlocal call_count
            call_count += 1
            # 첫 호출(데드라인 계산)은 0, 이후 호출은 데드라인을 항상 초과하도록
            # 큰 폭으로 증가시켜 실제 대기 없이 결정론적으로 타임아웃을 유발한다.
            return 0.0 if call_count == 1 else 1_000_000.0

        connection = _make_connection_with_mocked_client(tmp_path, time_fn=_incrementing_clock)
        mock_transport = MagicMock()
        mock_channel = MagicMock()
        mock_channel.recv_ready.return_value = False
        mock_channel.recv_stderr_ready.return_value = False
        mock_channel.exit_status_ready.return_value = False
        mock_transport.open_session.return_value = mock_channel
        connection._client.get_transport.return_value = mock_transport  # pyright: ignore[reportAttributeAccessIssue]

        result = connection.exec_command("sleep 999999", timeout_seconds=1.0)

        assert result == CommandResult(exit_code=-1, timed_out=True)
        mock_channel.close.assert_called_once()

    def test_exec_command_raises_when_not_connected(self, tmp_path: Path):
        connection = _make_connection_with_mocked_client(tmp_path)
        connection._client.get_transport.return_value = None  # pyright: ignore[reportAttributeAccessIssue]

        with pytest.raises(ConnectionError):
            connection.exec_command("echo hi", timeout_seconds=5.0)

    def test_close_delegates_to_client(self, tmp_path: Path):
        connection = _make_connection_with_mocked_client(tmp_path)

        connection.close()

        connection._client.close.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]


class BufferOverflowError(RuntimeError):
    """테스트 전용 — 채널 버퍼 포화를 결정론적으로 시뮬레이션한다(AC-ATO-001/002).

    실제 OS 레벨 write() 블로킹을 재현하지 않고도(테스트가 무한정 멈추는 것을
    피하면서) "소비되지 않았다면 블로킹했을 것"이라는 조건을 결정론적으로
    검증하기 위한 acceptance.md AC-ATO-001 명시 설계다.
    """


class _FakeBufferedChannelConnection:
    """채널 버퍼를 유한 용량 큐로 시뮬레이션하는 `SshConnection` 페이크.

    `exec_command()`가 `on_output_line` 콜백을 받으면 라인마다 즉시 "소비"한
    것으로 간주해 점유량을 즉시 되돌린다 — 신규 읽기 루프가 라인 단위로
    지속 드레인하는 것과 동등한 소비 모델이다(AC-ATO-001). 콜백이 `None`이면
    소비가 전혀 일어나지 않아, 읽기 루프가 없던 구현 이전 경로와 동등하게
    버퍼가 무한정 쌓여 용량 초과 시 `BufferOverflowError`를 던진다(AC-ATO-002).
    """

    def __init__(self, *, buffer_capacity: int, remote_lines: list[str]) -> None:
        self._buffer_capacity = buffer_capacity
        self._remote_lines = list(remote_lines)
        self.max_concurrent_unconsumed = 0

    def connect(self) -> None:
        pass

    def exec_command(
        self,
        command: str,
        timeout_seconds: float,
        *,
        on_output_line=None,
    ) -> CommandResult:
        occupancy = 0
        for line in self._remote_lines:
            occupancy += 1
            self.max_concurrent_unconsumed = max(self.max_concurrent_unconsumed, occupancy)
            if occupancy > self._buffer_capacity:
                raise BufferOverflowError(
                    f"채널 버퍼 용량({self._buffer_capacity}) 초과 — 소비되지 않은 라인이 누적됐다"
                )
            if on_output_line is not None:
                on_output_line(line)
                occupancy -= 1
        return CommandResult(exit_code=0)

    def close(self) -> None:
        pass


class TestChannelBufferSaturationPrevention:
    """AC-ATO-001/002(REQ-ATO-001): 읽기 루프가 라인마다 즉시 소비하면 채널
    버퍼가 사전 정의된 용량을 초과하지 않고, 소비가 없으면(구현 이전과 동등한
    경로) 결함이 실제로 재현된다."""

    def test_draining_callback_prevents_buffer_overflow(self):
        """AC-ATO-001: 콜백(드레인)이 있으면 10,000라인 전체가 예외 없이
        처리되고, 최대 동시 미소비 라인 수가 버퍼 용량을 초과하지 않는다."""
        connection = _FakeBufferedChannelConnection(
            buffer_capacity=10, remote_lines=[f"line-{i}" for i in range(10_000)]
        )
        consumed: list[str] = []

        result = connection.exec_command("cmd", timeout_seconds=5.0, on_output_line=consumed.append)

        assert result == CommandResult(exit_code=0)
        assert len(consumed) == 10_000
        assert connection.max_concurrent_unconsumed <= 10

    def test_characterization_regression_without_drain_reproduces_defect(self):
        """AC-ATO-002(D3): 읽기 루프가 없는(구현 이전과 동등한) 경로 — 콜백을
        전달하지 않으면 채널 버퍼 용량 초과로 `BufferOverflowError`가 발생해야
        한다. 이 특성화 회귀 테스트는 이번 결함이 실제로 재현 가능했음을
        구조적으로 증명하며, 향후 읽기 루프가 실수로 제거되면 즉시 실패해야
        한다(테스트 스위트에 영구 보존)."""
        connection = _FakeBufferedChannelConnection(
            buffer_capacity=10, remote_lines=[f"line-{i}" for i in range(10_000)]
        )

        with pytest.raises(BufferOverflowError):
            connection.exec_command("cmd", timeout_seconds=5.0)


class TestReadLoopCompletionRequiresExitCodePath:
    """AC-ATO-017 Worked example B(REQ-ATO-026, D7): 원격 스트림이 프로세스
    종료 전에 먼저 닫혀도(EOF), 읽기 루프는 종료코드 대기 API가 실제로 값을
    반환할 때까지 계속 대기한 뒤에만 완료를 판정해야 한다."""

    def test_does_not_complete_on_stream_eof_before_exit_status_ready(self, tmp_path: Path):
        connection = _make_connection_with_mocked_client(tmp_path)
        mock_transport = MagicMock()
        mock_channel = MagicMock()
        # stdout/stderr는 즉시 닫힌 것처럼(EOF, recv_ready/recv_stderr_ready
        # 계속 False) 시뮬레이션하고, exit_status_ready는 세 번째 폴링에서만
        # True가 되도록 해 "스트림이 프로세스보다 먼저 닫히는" 실제 가능
        # 시나리오(자식으로 fd 리다이렉트)를 모사한다.
        mock_channel.recv_ready.return_value = False
        mock_channel.recv_stderr_ready.return_value = False
        ready_sequence = iter([False, False, True])
        mock_channel.exit_status_ready.side_effect = lambda: next(ready_sequence)
        mock_channel.recv_exit_status.return_value = 0
        mock_transport.open_session.return_value = mock_channel
        connection._client.get_transport.return_value = mock_transport  # pyright: ignore[reportAttributeAccessIssue]

        result = connection.exec_command("cmd", timeout_seconds=5.0)

        assert result == CommandResult(exit_code=0)
        assert mock_channel.exit_status_ready.call_count == 3


class TestChannelDrainPartialLineBuffering:
    """acceptance.md §B 경계 사례: 라인 경계 없이 부분 라인만 수신한 경우,
    버퍼링 후 다음 read에서 결합해 완전한 라인 단위로만 콜백을 호출해야 한다."""

    def test_partial_line_is_buffered_until_newline_arrives(self, tmp_path: Path):
        connection = _make_connection_with_mocked_client(tmp_path)
        mock_transport = MagicMock()
        mock_channel = MagicMock()
        mock_channel.recv_stderr_ready.return_value = False
        recv_ready_sequence = iter([True, True, False])
        mock_channel.recv_ready.side_effect = lambda: next(recv_ready_sequence, False)
        recv_chunks = iter([b"partial-", b"line\n"])
        mock_channel.recv.side_effect = lambda _n: next(recv_chunks)
        exit_ready_sequence = iter([False, False, True])
        mock_channel.exit_status_ready.side_effect = lambda: next(exit_ready_sequence, True)
        mock_channel.recv_exit_status.return_value = 0

        mock_transport.open_session.return_value = mock_channel
        connection._client.get_transport.return_value = mock_transport  # pyright: ignore[reportAttributeAccessIssue]
        received_lines: list[str] = []

        connection.exec_command("cmd", timeout_seconds=5.0, on_output_line=received_lines.append)

        assert received_lines == ["partial-line"]


class TestNoAlternativeCompletionSignalingMechanism:
    """AC-ATA-011(REQ-ATA-051): 종료코드 판정 외 콜백 엔드포인트/폴링 완료 파일
    메커니즘을 도입하지 않았음을 구조적으로 검증한다(코드 리뷰/grep 방식)."""

    _FORBIDDEN_PATTERNS = (
        "BaseHTTPRequestHandler",
        "http.server",
        "Flask(",
        "@app.route",
        "callback_url",
        "poll_completion_file",
    )

    def test_orchestration_package_has_no_forbidden_signaling_patterns(self):
        orchestration_dir = (
            Path(__file__).resolve().parents[1] / "src" / "analyzer" / "orchestration"
        )

        for py_file in sorted(orchestration_dir.glob("*.py")):
            content = py_file.read_text(encoding="utf-8")
            for pattern in self._FORBIDDEN_PATTERNS:
                assert pattern not in content, (
                    f"{py_file.name}에서 금지된 완료 시그널링 패턴 발견: {pattern} "
                    "(REQ-ATA-051 — 종료코드 판정 외 별도 메커니즘 미도입)"
                )

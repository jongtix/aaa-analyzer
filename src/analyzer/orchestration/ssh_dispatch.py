"""SSH 연결 재시도 + MySQL 터널 관리 + 학습 원격 디스패치 + 타임아웃 강제.

SPEC-ANALYZER-TRAIN-AUTOMATION-001 §2.2/§2.3/§2.4/§2.5,
REQ-ATA-020/021/022/030/031/032/040/041/050/051.

`WolSender`(wol.py)와 동일한 설계 원칙 — 실 네트워크 클라이언트(`paramiko`)를
`SshConnection` `typing.Protocol` 뒤로 추상화해, 이 모듈이 담당하는 재시도/
타임아웃/디스패치 로직을 실 SSH 연결 없이 페이크 구현으로 단위 테스트할 수 있게
한다.

**MySQL 터널 설계 결정(REQ-ATA-031)**: NAS→MacBook SSH 디스패치 세션과 MacBook→
NAS MySQL 접근(`db_tunnel` 계정) 세션은 서로 다른 방향의 별도 SSH 연결이다.
analyzer(NAS)는 MacBook에 직접 로컬 포트포워딩을 열 수 없으므로, `db_tunnel`
터널 수립을 원격 디스패치 명령 문자열 자체에 내장한다 — MacBook에서 백그라운드
`ssh -f -N -L`로 터널을 열고, TRAIN-001 CLI를 실행하고, `trap ... EXIT`로 종료
사유와 무관하게 터널을 정리한 뒤, 학습 스크립트의 실제 종료코드를 그대로
전달한다(REQ-ATA-032, REQ-ATA-050). 이 설계는 paramiko 레벨의 포트포워딩
구현(로컬 리스닝 소켓 + 바이트 펌핑 스레드)을 회피해 순수 문자열 조립 +
종료코드 캡처만으로 검증 가능하게 만든다.

**모델 보존 설계(plan.md §B.5, D6)**: `--models-root`는 항상 스테이징 경로를
가리키며, `promote_staging_to_active()`는 SSH 종료코드 0(성공) 확인 후에만
호출되어야 한다 — 실패/타임아웃 경로는 이 함수를 호출하지 않으므로 활성 경로는
원천적으로 교체되지 않는다(REQ-ATA-062, 상위 오케스트레이터인 `runner.py`가
이 호출 순서를 강제한다).

REQ-ATA-051: 이 모듈은 SSH 종료코드 판정 외의 별도 완료 시그널링 메커니즘
(콜백 엔드포인트, 폴링 완료 파일 등)을 도입하지 않는다.
"""

import shlex
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import paramiko

_DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
"""paramiko 기본값(None=무기한 블로킹)은 REQ-ATA-021의 10초×6회 재시도 설계를
무력화한다 — 첫 시도가 멈추면 재시도 루프에 도달하지 못한다."""


class SshKeyPermissionError(RuntimeError):
    """SSH 프라이빗 키 파일의 mode가 600이 아니다(REQ-ATA-022)."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """원격 명령 실행 결과 — 종료코드가 완료 판정의 1차 근거다(REQ-ATA-050)."""

    exit_code: int
    timed_out: bool = False


class SshConnection(Protocol):
    """SSH 연결 추상화 — 재시도/디스패치/타임아웃 로직이 소비하는 최소 인터페이스."""

    def connect(self) -> None:
        """연결을 수립한다. 실패 시 예외를 던진다."""
        ...

    def exec_command(self, command: str, timeout_seconds: float) -> CommandResult:
        """원격 명령을 실행하고 종료코드를 반환한다.

        `timeout_seconds` 초과 시 SSH 세션을 강제 종료하고
        `CommandResult(exit_code=-1, timed_out=True)`를 반환해야 한다(REQ-ATA-041 —
        "SSH 세션을 강제 종료해야 하며"는 세션 종료를 요구할 뿐, 원격 프로세스
        자체의 종료까지 보장할 필요는 없다).
        """
        ...

    def close(self) -> None:
        """SSH 연결 자체를 종료한다."""
        ...


def validate_private_key_permissions(key_path: Path) -> None:
    """REQ-ATA-022: 프라이빗 키 파일 모드가 600이 아니면 예외를 던진다."""
    mode = stat.S_IMODE(key_path.stat().st_mode)
    if mode != 0o600:
        raise SshKeyPermissionError(
            f"{key_path}의 파일 모드가 {oct(mode)}입니다 — 600이어야 합니다(REQ-ATA-022)"
        )


class ParamikoSshConnection:
    """`paramiko` 기반 실 SSH 연결 구현.

    known_hosts 핀 고정을 사용하며(REQ-ATA-022), `StrictHostKeyChecking=no`에
    대응하는 `paramiko.AutoAddPolicy`는 사용하지 않는다 — known_hosts에 없는
    호스트는 `paramiko.RejectPolicy`로 명시적으로 거부한다.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        private_key_path: Path,
        known_hosts_path: Path,
        connect_timeout_seconds: float = _DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        validate_private_key_permissions(private_key_path)
        self._host = host
        self._port = port
        self._username = username
        self._private_key_path = private_key_path
        self._connect_timeout_seconds = connect_timeout_seconds
        self._client = paramiko.SSHClient()
        self._client.load_host_keys(str(known_hosts_path))
        self._client.set_missing_host_key_policy(paramiko.RejectPolicy())

    def connect(self) -> None:
        self._client.connect(
            hostname=self._host,
            port=self._port,
            username=self._username,
            key_filename=str(self._private_key_path),
            look_for_keys=False,
            allow_agent=False,
            timeout=self._connect_timeout_seconds,
        )

    def exec_command(self, command: str, timeout_seconds: float) -> CommandResult:
        transport = self._client.get_transport()
        if transport is None:
            raise ConnectionError("SSH transport가 없습니다 — connect()를 먼저 호출하세요")
        channel = transport.open_session()
        channel.settimeout(timeout_seconds)
        channel.exec_command(command)
        try:
            exit_code = channel.recv_exit_status()
        except TimeoutError:
            channel.close()
            return CommandResult(exit_code=-1, timed_out=True)
        return CommandResult(exit_code=exit_code)

    def close(self) -> None:
        self._client.close()


def connect_with_retry(
    connection: SshConnection,
    max_retries: int = 6,
    interval_seconds: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """REQ-ATA-021: SSH 연결 실패 시 `interval_seconds` 간격으로 최대
    `max_retries`회까지 재시도한다. 마지막 실패 시도 뒤에는 대기하지 않는다."""
    for attempt in range(1, max_retries + 1):
        try:
            connection.connect()
            return True
        except Exception:  # noqa: BLE001 — 재시도 경계에서 포착
            if attempt < max_retries:
                sleep_fn(interval_seconds)
    return False


def build_remote_dispatch_command(
    *,
    staging_models_root: Path,
    calendar_code: str,
    cache_dir: Path,
    data_as_of: date,
    feature_code_version: str,
    db_tunnel_host: str,
    db_tunnel_key_path: Path,
    mount_script_path: Path,
    python_executable_path: Path,
    mysql_database: str,
    mysql_trainer_password: str,
    db_tunnel_username: str = "db_tunnel",
    db_tunnel_port: int = 22,
    db_tunnel_local_port: int = 3306,
    db_tunnel_remote_port: int = 3306,
) -> str:
    """REQ-ATA-030/031/032: 멱등 SMB 마운트 확인 → MySQL 터널 수립 →
    TRAIN-001 CLI 원격 호출(스테이징 경로) → 터널 해제(trap)까지 단일 원격
    셸 스크립트로 구성한다.

    마운트 확인은 학습 CLI 실행의 선행조건이다(`&&`로 연결) — 스테이징
    경로가 SMB 마운트포인트 하위이므로, 마운트 실패 상태에서 학습 CLI가
    실행되면 로컬 디스크에 조용히 쓰거나 경로 없음으로 실패할 수 있다.
    마운트 실패 시 학습 CLI를 건너뛰고 마운트 스크립트의 종료코드를 그대로
    전달한다.

    `db_tunnel_port`는 터널 SSH 접속 자체의 포트다(나스 sshd 포트) —
    `db_tunnel_local_port`/`db_tunnel_remote_port`(-L 포워딩의 MySQL 포트,
    3306)와는 별개이며, 이 둘을 혼동하면 나스가 비표준 SSH 포트를 쓸 때
    (aaa-infra/docs/TECHSPEC.md §6.2) 터널 접속이 항상 거부된다(Stage 2
    실측 검증 중 발견, 2026-08-13).

    `mount_script_path`는 필수 인자다 — 기본값(맥북 계정별 절대경로)의 유일한
    출처는 `config.py`의 `AutomationConfig`/`get_automation_config()`이며, 이
    함수는 그 값을 그대로 소비할 뿐 자체 기본값을 갖지 않는다(경로 중복 방지).

    TRAIN-001의 확정된 CLI 계약(`training/train.py` `main()`)을 그대로 소비한다
    — 이 SPEC은 그 계약을 재정의하지 않는다(REQ-ATA-030).

    `python_executable_path`는 맥북상 학습 venv의 python 절대경로다 — 원격
    SSH 실행은 비대화형 셸이라 PATH에 pyenv/venv가 안 잡힌다. `python`
    하드코딩은 맥에 시스템 `python`이 없어(`python3`만 존재) 종료코드 127로,
    `python3`로 바꿔도 `analyzer` 패키지가 없는 시스템 파이썬이라 동일하게
    실패한다(수동 실행 실측, 2026-08-13).

    `mysql_database`/`mysql_trainer_password`는 `training/db.py`
    `get_trainer_db_config()`가 요구하는 필수 환경변수다 — analyzer 컨테이너
    자신의 동일 이름 env var를 재사용할 뿐 신규 시크릿이 아니다.
    `MYSQL_HOST`/`MYSQL_PORT`는 컨테이너 자신의 값(도커 네트워크 호스트명)이
    아니라 db_tunnel이 연 로컬 포워딩(`127.0.0.1:{db_tunnel_local_port}`)으로
    고정한다 — 이 값들이 없으면 원격 학습 CLI가 MissingConfigError로 즉시
    실패한다(수동 실행 실측, 2026-08-13).
    """
    quoted_db_tunnel_key_path = shlex.quote(str(db_tunnel_key_path))
    quoted_db_tunnel_username = shlex.quote(db_tunnel_username)
    quoted_db_tunnel_host = shlex.quote(db_tunnel_host)
    quoted_calendar_code = shlex.quote(calendar_code)
    quoted_cache_dir = shlex.quote(str(cache_dir))
    quoted_staging_models_root = shlex.quote(str(staging_models_root))
    quoted_data_as_of = shlex.quote(data_as_of.isoformat())
    quoted_feature_code_version = shlex.quote(feature_code_version)
    quoted_mount_script_path = shlex.quote(str(mount_script_path))
    quoted_python_executable_path = shlex.quote(str(python_executable_path))
    quoted_mysql_database = shlex.quote(mysql_database)
    quoted_mysql_trainer_password = shlex.quote(mysql_trainer_password)

    tunnel_command = (
        f"ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes "
        f"-i {quoted_db_tunnel_key_path} "
        f"-p {db_tunnel_port} "
        f"-L {db_tunnel_local_port}:127.0.0.1:{db_tunnel_remote_port} "
        f"{quoted_db_tunnel_username}@{quoted_db_tunnel_host}"
    )
    mount_command = quoted_mount_script_path
    train_command = (
        f"MYSQL_HOST=127.0.0.1 MYSQL_PORT={db_tunnel_local_port} "
        f"MYSQL_DATABASE={quoted_mysql_database} "
        f"MYSQL_TRAINER_PASSWORD={quoted_mysql_trainer_password} "
        f"{quoted_python_executable_path} -m analyzer.training.train "
        f"--calendar-code {quoted_calendar_code} "
        f"--cache-dir {quoted_cache_dir} "
        f"--models-root {quoted_staging_models_root} "
        f"--data-as-of {quoted_data_as_of} "
        f"--feature-code-version {quoted_feature_code_version}"
    )
    tunnel_pattern = f"{db_tunnel_local_port}:127.0.0.1:{db_tunnel_remote_port}"
    return (
        f"set -o pipefail; "
        f"{tunnel_command}; "
        f"TUNNEL_PID=$(pgrep -f '{tunnel_pattern}'); "
        f"trap 'kill $TUNNEL_PID 2>/dev/null' EXIT; "
        f"{mount_command} && {train_command}; "
        f"exit $?"
    )


def promote_staging_to_active(
    connection: SshConnection,
    staging_path: Path,
    active_path: Path,
    *,
    timeout_seconds: float = 120.0,
) -> bool:
    """plan.md §B.5(D6): SSH 종료코드 0(성공) 확인 후에만 호출되어야 한다.

    스테이징 산출물을 활성 경로로 병합 이동한다 — 활성 경로에 이미 존재하는
    다른 버전 파일은 건드리지 않고(`cp -a`), 성공적으로 복사된 뒤에만 스테이징
    원본을 정리한다. 원격 명령 자체가 실패하면(예: 경로 없음) `False`를 반환하고
    활성 경로는 변경되지 않는다. 이 함수는 `runner.py`의 오케스트레이션 순서
    (실패/타임아웃 시 호출되지 않음)를 통해서만 REQ-ATA-062의 절대 보장을
    충족한다 — 이 함수 자체는 "호출되지 않으면 활성 경로는 불변"이라는 전제
    위에서만 안전하다.
    """
    quoted_active_path = shlex.quote(str(active_path))
    quoted_staging_path = shlex.quote(str(staging_path))
    command = (
        f"mkdir -p {quoted_active_path} && "
        f"cp -a {quoted_staging_path}/. {quoted_active_path}/ && "
        f"rm -rf {quoted_staging_path}"
    )
    result = connection.exec_command(command, timeout_seconds=timeout_seconds)
    return result.exit_code == 0

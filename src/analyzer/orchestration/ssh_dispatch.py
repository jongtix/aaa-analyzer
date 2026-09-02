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

SPEC-ANALYZER-TRAIN-OBSV-001 §2.1/§2.3, REQ-ATO-001/003/009/010/011: `exec_command()`
내부를 **모든** 호출에 보편 적용되는 폴링 드레인 루프로 재구현한다 — SSH 채널
버퍼를 지속 소비해 원격 프로세스의 `write()` 블로킹(15시간 데드락 실측,
2026-08-13~14)을 구조적으로 방지하고, 자체 `time.monotonic()` 데드라인 추적으로
타임아웃을 강제한다.

**REQ-ATO-010 실측 근거(근본 원인)**: `paramiko.Channel.recv_exit_status()`는
`timeout` 인자를 받지 않는 시그니처이며(`channel.settimeout()`이 설정하는
값은 `recv()`/`send()`류의 블로킹 I/O 타임아웃에만 적용된다), 내부적으로
`self.status_event.wait()`를 **인자 없이** 호출한다 — 즉 `recv_exit_status()`
자체는 `settimeout()`으로 설정한 타임아웃을 준수하지 않고 종료 이벤트가 발생할
때까지 무기한 대기한다. 원격 프로세스가 `write()`에서 영구 블록하면 종료
이벤트 자체가 결코 발생하지 않으므로, 기존 `try: channel.recv_exit_status()
except TimeoutError` 경로는 이 경우 결코 도달하지 않는다 — 이것이 REQ-ATA-041
타임아웃이 실측에서 발화하지 않은 근본 원인이다. 이 SPEC은 `recv_exit_status()`
단독 호출 대신 `channel.exit_status_ready()`를 논블로킹 폴링하고 자체
데드라인을 별도로 추적하는 방식으로 재설계해 이 결함을 우회한다.
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

from analyzer.common.logging import get_logger

_logger = get_logger(__name__)


_DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
"""paramiko 기본값(None=무기한 블로킹)은 REQ-ATA-021의 10초×6회 재시도 설계를
무력화한다 — 첫 시도가 멈추면 재시도 루프에 도달하지 못한다."""

_DEFAULT_POLL_INTERVAL_SECONDS = 0.05
"""REQ-ATO-001/009: 채널 드레인 + 자체 데드라인 추적 폴링 루프의 유휴 대기 간격."""

_CHANNEL_READ_CHUNK_BYTES = 4096


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

    def exec_command(
        self,
        command: str,
        timeout_seconds: float,
        *,
        on_output_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        """원격 명령을 실행하고 종료코드를 반환한다.

        `timeout_seconds` 초과 시 SSH 세션을 강제 종료하고
        `CommandResult(exit_code=-1, timed_out=True)`를 반환해야 한다(REQ-ATA-041 —
        "SSH 세션을 강제 종료해야 하며"는 세션 종료를 요구할 뿐, 원격 프로세스
        자체의 종료까지 보장할 필요는 없다).

        SPEC-ANALYZER-TRAIN-OBSV-001 REQ-ATO-001/003/009: 구현은 이 호출 동안
        원격 stdout/stderr를 라인 단위로 지속 소비해야 한다(콜백 인자 전달
        여부와 무관하게 항상 드레인됨) — SSH 채널 버퍼 포화로 인한 원격
        프로세스의 `write()` 블로킹을 방지하기 위함이다. `on_output_line`은
        옵셔널 키워드 인자(기본값 `None`)로, 지정되면 소비한 각 라인을
        전달받는다(값이 `None`이면 드레인은 계속하되 콜백 호출만 생략한다).
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
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        validate_private_key_permissions(private_key_path)
        self._host = host
        self._port = port
        self._username = username
        self._private_key_path = private_key_path
        self._connect_timeout_seconds = connect_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
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

    def exec_command(
        self,
        command: str,
        timeout_seconds: float,
        *,
        on_output_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        transport = self._client.get_transport()
        if transport is None:
            raise ConnectionError("SSH transport가 없습니다 — connect()를 먼저 호출하세요")
        channel = transport.open_session()
        # REQ-ATO-010: recv_exit_status()는 settimeout() 값을 준수하지 않는다
        # (모듈 docstring 참조) — 여기서는 recv()/recv_stderr()류의 개별 논블로킹
        # 폴링만을 위해 0으로 설정하고, 타임아웃 자체는 이 메서드가 직접
        # `time_fn`/데드라인으로 강제한다.
        channel.settimeout(0.0)
        channel.exec_command(command)

        deadline = self._time_fn() + timeout_seconds
        stdout_buffer = b""
        stderr_buffer = b""
        while True:
            made_progress = False

            if channel.recv_ready():
                chunk = channel.recv(_CHANNEL_READ_CHUNK_BYTES)
                if chunk:
                    made_progress = True
                    stdout_buffer = self._drain_lines(stdout_buffer + chunk, on_output_line)

            if channel.recv_stderr_ready():
                chunk = channel.recv_stderr(_CHANNEL_READ_CHUNK_BYTES)
                if chunk:
                    made_progress = True
                    stderr_buffer = self._drain_lines(stderr_buffer + chunk, on_output_line)

            # REQ-ATO-026/D7: 완료 판정의 유일한 근거는 종료코드 획득 경로다 —
            # 스트림 EOF(recv_ready()/recv_stderr_ready()가 False)만으로는
            # 완료를 추론하지 않는다.
            if channel.exit_status_ready():
                exit_code = channel.recv_exit_status()
                return CommandResult(exit_code=exit_code)

            if self._time_fn() >= deadline:
                channel.close()
                return CommandResult(exit_code=-1, timed_out=True)

            if not made_progress:
                self._sleep_fn(self._poll_interval_seconds)

    @staticmethod
    def _drain_lines(buffer: bytes, on_output_line: Callable[[str], None] | None) -> bytes:
        """REQ-ATO-001: 개행 경계로 완성된 라인만 콜백에 전달하고 미완성 잔여
        바이트(부분 라인)는 다음 read에서 결합하도록 그대로 반환한다."""
        if b"\n" not in buffer:
            return buffer
        *complete_lines, remainder = buffer.split(b"\n")
        if on_output_line is not None:
            for raw_line in complete_lines:
                on_output_line(raw_line.decode("utf-8", errors="replace"))
        return remainder

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
            # REQ-ATO-021: SSH 연결 성공/시도 횟수 단계 전이 로그.
            _logger.info(
                "ssh connect succeeded attempt=%d/%d",
                attempt,
                max_retries,
                extra={"stage_marker": True},
            )
            return True
        except Exception:  # noqa: BLE001 — 재시도 경계에서 포착
            if attempt < max_retries:
                sleep_fn(interval_seconds)
    _logger.error(
        "ssh connect failed after max retries attempts=%d",
        max_retries,
        extra={"stage_marker": True},
    )
    return False


def _build_db_tunnel_command(
    *,
    db_tunnel_host: str,
    db_tunnel_key_path: Path,
    db_tunnel_username: str,
    db_tunnel_port: int,
    db_tunnel_local_port: int,
    db_tunnel_remote_port: int,
) -> str:
    """REQ-ATT-005: MySQL 터널 수립 셸 명령 조립 — 공유 골격의 1단계.

    `build_remote_dispatch_command()`(주간 학습)와
    `build_remote_campaign_dispatch_command()`(월간 캠페인) 양쪽이 동일하게
    소비한다. 원본 `build_remote_dispatch_command()`의 `tunnel_command` 지역
    변수 조립 로직을 그대로 옮긴 것으로, 출력 바이트는 변하지 않는다.
    """
    quoted_db_tunnel_key_path = shlex.quote(str(db_tunnel_key_path))
    quoted_db_tunnel_username = shlex.quote(db_tunnel_username)
    quoted_db_tunnel_host = shlex.quote(db_tunnel_host)
    return (
        f"ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes "
        f"-i {quoted_db_tunnel_key_path} "
        f"-p {db_tunnel_port} "
        f"-L {db_tunnel_local_port}:127.0.0.1:{db_tunnel_remote_port} "
        f"{quoted_db_tunnel_username}@{quoted_db_tunnel_host}"
    )


def _wrap_with_tunnel_mount_trap(
    *,
    tunnel_command: str,
    mount_script_path: Path,
    inner_command: str,
    db_tunnel_local_port: int,
    db_tunnel_remote_port: int,
) -> str:
    """REQ-ATT-005: 터널 수립 + 멱등 마운트 확인 + trap 기반 터널 해제 골격.

    `build_remote_dispatch_command()`/`build_remote_campaign_dispatch_command()`
    양쪽이 공유하는 원격 셸 스크립트 뼈대다. `inner_command`는 마운트 확인
    통과 후 실행될 학습/캠페인 CLI 호출(및 그 tee 로깅)을 완성된 문자열로
    그대로 받는다 — 이 함수는 그 내용을 알지 못한다(REQ-ATT-005/006 분리
    원칙 — 골격 추출과 캠페인 CLI 호출 조립은 별개 책임이다).

    원본 `build_remote_dispatch_command()`의 최종 반환문 조립 로직을 그대로
    옮긴 것으로, 출력 바이트는 변하지 않는다(AC-ATT-006 회귀 가드).
    """
    quoted_mount_script_path = shlex.quote(str(mount_script_path))
    tunnel_pattern = f"{db_tunnel_local_port}:127.0.0.1:{db_tunnel_remote_port}"
    return (
        f"set -o pipefail; "
        f"{tunnel_command}; "
        f"TUNNEL_PID=$(pgrep -f '{tunnel_pattern}'); "
        f"trap 'kill $TUNNEL_PID 2>/dev/null' EXIT; "
        f"{quoted_mount_script_path} && {inner_command}; "
        f"exit $?"
    )


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
    trainer_log_base_dir: Path,
    run_id: str,
    db_tunnel_username: str = "db_tunnel",
    db_tunnel_port: int = 22,
    db_tunnel_local_port: int = 3306,
    db_tunnel_remote_port: int = 3306,
    params_from_active_meta: Path | None = None,
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

    `params_from_active_meta`(REQ-ATG-011, additive, §D M3 허용 예외): 지정되면
    `train.py`의 `--params-from-active-meta <경로>` 플래그를 원격 학습 CLI
    호출에 추가한다 — 주간 원격 학습이 챔피언 동결 하이퍼파라미터로 학습하게
    된다. 미지정(기본값 `None`) 시 기존 명령 문자열과 byte-identical하다
    (하위 호환).
    """
    quoted_calendar_code = shlex.quote(calendar_code)
    quoted_cache_dir = shlex.quote(str(cache_dir))
    quoted_staging_models_root = shlex.quote(str(staging_models_root))
    quoted_data_as_of = shlex.quote(data_as_of.isoformat())
    quoted_feature_code_version = shlex.quote(feature_code_version)
    quoted_python_executable_path = shlex.quote(str(python_executable_path))
    quoted_mysql_database = shlex.quote(mysql_database)
    quoted_mysql_trainer_password = shlex.quote(mysql_trainer_password)
    quoted_run_id = shlex.quote(run_id)
    trainer_log_path = trainer_log_base_dir / f"trainer_{run_id}.log"
    quoted_trainer_log_path = shlex.quote(str(trainer_log_path))
    quoted_trainer_log_base_dir = shlex.quote(str(trainer_log_base_dir))
    params_from_active_meta_arg = (
        f" --params-from-active-meta {shlex.quote(str(params_from_active_meta))}"
        if params_from_active_meta is not None
        else ""
    )

    tunnel_command = _build_db_tunnel_command(
        db_tunnel_host=db_tunnel_host,
        db_tunnel_key_path=db_tunnel_key_path,
        db_tunnel_username=db_tunnel_username,
        db_tunnel_port=db_tunnel_port,
        db_tunnel_local_port=db_tunnel_local_port,
        db_tunnel_remote_port=db_tunnel_remote_port,
    )
    train_command = (
        # F3: mkdir -p로 트레이너 로그 디렉터리를 tee보다 먼저 생성한다 —
        # promote_staging_to_active()의 mkdir -p 관례와 동일하게, 디렉터리가
        # 없으면 원격 tee 자체가 실패해 트레이너 파일이 기록되지 않는다.
        f"mkdir -p {quoted_trainer_log_base_dir} && "
        f"MYSQL_HOST=127.0.0.1 MYSQL_PORT={db_tunnel_local_port} "
        f"MYSQL_DATABASE={quoted_mysql_database} "
        f"MYSQL_TRAINER_PASSWORD={quoted_mysql_trainer_password} "
        # REQ-ATO-012/013: NAS에서 발급된 run_id를 원격 CLI로 전달한다 —
        # train.py main()이 이 값을 기존 trace_id 발급/전파 유틸리티(set_trace_id())로
        # 즉시 설정해 릴레이 로그·트레이너 파일 로그 양쪽의 trace_id 필드에 반영한다.
        f"TRAIN_RUN_ID={quoted_run_id} "
        f"{quoted_python_executable_path} -m analyzer.training.train "
        f"--calendar-code {quoted_calendar_code} "
        f"--cache-dir {quoted_cache_dir} "
        f"--models-root {quoted_staging_models_root} "
        f"--data-as-of {quoted_data_as_of} "
        f"--feature-code-version {quoted_feature_code_version}"
        f"{params_from_active_meta_arg} "
        # REQ-ATO-004/007: stdout/stderr 전체를 트레이너 파일에 원문 그대로
        # 영속 기록하면서(tee), 동일 바이트가 여전히 SSH 채널(stdout)로도
        # 흘러가게 유지한다 — 채널 드레인 루프(REQ-ATO-001)가 그 스트림을
        # 계속 소비하고, stage_marker 필드 기반 저볼륨 릴레이(REQ-ATO-002)만
        # 걸러 릴레이한다. `set -o pipefail`(아래)이 tee를 거친 뒤에도
        # train_command 자신의 종료코드가 최종 결과로 전달되게 한다.
        f"2>&1 | tee {quoted_trainer_log_path}"
    )

    return _wrap_with_tunnel_mount_trap(
        tunnel_command=tunnel_command,
        mount_script_path=mount_script_path,
        inner_command=train_command,
        db_tunnel_local_port=db_tunnel_local_port,
        db_tunnel_remote_port=db_tunnel_remote_port,
    )


def build_remote_campaign_dispatch_command(
    *,
    active_models_root: Path,
    calendar_code: str,
    cache_dir: Path,
    data_as_of: date,
    feature_code_version: str,
    optuna_storage_dir: Path,
    summary_report_path: Path,
    n_trials: int,
    db_tunnel_host: str,
    db_tunnel_key_path: Path,
    mount_script_path: Path,
    python_executable_path: Path,
    mysql_database: str,
    mysql_trainer_password: str,
    trainer_log_base_dir: Path,
    run_id: str,
    db_tunnel_username: str = "db_tunnel",
    db_tunnel_port: int = 22,
    db_tunnel_local_port: int = 3306,
    db_tunnel_remote_port: int = 3306,
) -> str:
    """REQ-ATT-006/007/009: 월간 원격 캠페인 CLI(`python -m
    analyzer.training.campaign`) 디스패치 명령을, `build_remote_dispatch_command()`
    (주간 학습)와 동일한 REQ-ATT-005 공유 골격(터널 수립 + 멱등 마운트 확인 +
    trap 기반 터널 해제) 위에 조립한다.

    `active_models_root`는 REQ-ATT-007에 따라 반드시 `AutomationConfig.
    active_models_root`여야 한다(호출자 책임 — 이 함수는 전달값을 그대로
    `--models-root`에 사용할 뿐 스테이징/활성 여부를 스스로 판정하지 않는다).
    캠페인 자신의 `run_walk_forward_campaign_and_activate()`가
    `activate_market_horizon_combo()`를 통해 이 경로에 직접 활성화를 기록하므로,
    주간 학습의 `promote_staging_to_active()` 승격 단계는 이 경로에 존재하지
    않는다(REQ-ATT-013).

    `n_trials`는 호출자가 명시적으로 전달해야 한다(REQ-ATT-009 — 월간
    원격 디스패치는 100을 전달한다) — 이 함수 자체는 기본값을 갖지 않는다.

    캠페인 CLI의 확정된 인자 관례(campaign.py `main()`) —
    `--calendar-code`/`--cache-dir`/`--models-root`/`--data-as-of`/
    `--feature-code-version`/`--optuna-storage-dir`/`--summary-report-path` —
    를 그대로 전달하며 재정의하지 않는다(REQ-ATT-006). 캠페인도
    `build_trainer_engine()`(`training/db.py`)을 통해 동일한 `trainer` 계정
    MySQL 접속을 요구하므로, 주간 학습 경로와 동일한 db_tunnel MySQL 환경변수
    주입 + `mkdir -p` + tee 트레이너 로그 관례를 재사용한다.
    """
    quoted_calendar_code = shlex.quote(calendar_code)
    quoted_cache_dir = shlex.quote(str(cache_dir))
    quoted_active_models_root = shlex.quote(str(active_models_root))
    quoted_data_as_of = shlex.quote(data_as_of.isoformat())
    quoted_feature_code_version = shlex.quote(feature_code_version)
    quoted_optuna_storage_dir = shlex.quote(str(optuna_storage_dir))
    quoted_summary_report_path = shlex.quote(str(summary_report_path))
    quoted_python_executable_path = shlex.quote(str(python_executable_path))
    quoted_mysql_database = shlex.quote(mysql_database)
    quoted_mysql_trainer_password = shlex.quote(mysql_trainer_password)
    quoted_run_id = shlex.quote(run_id)
    trainer_log_path = trainer_log_base_dir / f"trainer_{run_id}.log"
    quoted_trainer_log_path = shlex.quote(str(trainer_log_path))
    quoted_trainer_log_base_dir = shlex.quote(str(trainer_log_base_dir))

    tunnel_command = _build_db_tunnel_command(
        db_tunnel_host=db_tunnel_host,
        db_tunnel_key_path=db_tunnel_key_path,
        db_tunnel_username=db_tunnel_username,
        db_tunnel_port=db_tunnel_port,
        db_tunnel_local_port=db_tunnel_local_port,
        db_tunnel_remote_port=db_tunnel_remote_port,
    )
    campaign_command = (
        f"mkdir -p {quoted_trainer_log_base_dir} && "
        f"MYSQL_HOST=127.0.0.1 MYSQL_PORT={db_tunnel_local_port} "
        f"MYSQL_DATABASE={quoted_mysql_database} "
        f"MYSQL_TRAINER_PASSWORD={quoted_mysql_trainer_password} "
        f"TRAIN_RUN_ID={quoted_run_id} "
        f"{quoted_python_executable_path} -m analyzer.training.campaign "
        f"--calendar-code {quoted_calendar_code} "
        f"--cache-dir {quoted_cache_dir} "
        f"--models-root {quoted_active_models_root} "
        f"--data-as-of {quoted_data_as_of} "
        f"--feature-code-version {quoted_feature_code_version} "
        f"--optuna-storage-dir {quoted_optuna_storage_dir} "
        f"--summary-report-path {quoted_summary_report_path} "
        f"--n-trials {n_trials} "
        f"2>&1 | tee {quoted_trainer_log_path}"
    )

    return _wrap_with_tunnel_mount_trap(
        tunnel_command=tunnel_command,
        mount_script_path=mount_script_path,
        inner_command=campaign_command,
        db_tunnel_local_port=db_tunnel_local_port,
        db_tunnel_remote_port=db_tunnel_remote_port,
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

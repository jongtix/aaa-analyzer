"""자동화 설정 로딩 — 타임아웃/스케줄/MAC/SSH·터널 접속 정보 (plan.md §B.4).

`data/config.py`의 `DbConfig`/`get_db_config()` 패턴(REQ-AT-011 선례)을 그대로
계승한다 — 항목 수가 적어 `pydantic-settings` 도입 비용이 이익을 상회하므로
`os.environ`을 직접 읽는다. `.env`/`.env.*` 파일 자체는 절대 읽지 않는다(이미
프로세스 환경에 반영된 값만 사용).

타임아웃 초기값(REQ-ATA-040)은 주간 4시간/월간 14시간(REQ-ATT-012, 2026-09-01
36시간→14시간 명시적 수정)이며, 운영 중 튜닝될 것으로 예상되므로 하드코딩하지
않고 환경변수 오버라이드를 지원한다. 모델 정체 임계값(REQ-ATA-072)의 초기값은
4주(28일)다.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from analyzer.data.config import MissingConfigError

_DEFAULT_SSH_PORT = 22
_DEFAULT_DB_TUNNEL_USERNAME = "db_tunnel"
_DEFAULT_DB_TUNNEL_SSH_PORT = 22
_DEFAULT_DB_TUNNEL_PORT = 3306
_DEFAULT_WEEKLY_TIMEOUT_SECONDS = 4 * 60 * 60
_DEFAULT_MONTHLY_TIMEOUT_SECONDS = 14 * 60 * 60
"""REQ-ATT-012 (2026-09-01, 36h→14h 명시적 수정): 캠페인 자신의 내부
graceful-stop 상수 CAMPAIGN_TIMEOUT_HOURS=12.0(campaign.py:85) + SSH 세션
여유 2시간. 환경변수명/dataclass 필드명은 무수정 — `register_default_jobs()`가
프로덕션에서 호출된 적이 없어(SPEC-ANALYZER-TRAIN-TUNING-001 §1.1) 이 필드의
기존 프로덕션 소비자가 0건이므로 하위호환 이슈가 없다."""
_DEFAULT_STALENESS_THRESHOLD_DAYS = 28
_DEFAULT_CALENDAR_CODE = "KRX"
_DEFAULT_FEATURE_CODE_VERSION = "v1"

_REQUIRED_ENV_VARS = (
    "TRAIN_AUTOMATION_TARGET_MAC",
    "TRAIN_AUTOMATION_SSH_HOST",
    "TRAIN_AUTOMATION_SSH_USERNAME",
    "TRAIN_AUTOMATION_SSH_KEY_PATH",
    "TRAIN_AUTOMATION_KNOWN_HOSTS_PATH",
    "TRAIN_AUTOMATION_STAGING_MODELS_ROOT",
    "TRAIN_AUTOMATION_ACTIVE_MODELS_ROOT",
    "TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT",
    "TRAIN_AUTOMATION_DB_TUNNEL_HOST",
    "TRAIN_AUTOMATION_DB_TUNNEL_KEY_PATH",
    "TRAIN_AUTOMATION_CACHE_DIR",
    "TRAIN_AUTOMATION_MOUNT_SCRIPT_PATH",
    "TRAIN_AUTOMATION_PYTHON_PATH",
    "TRAIN_AUTOMATION_TRAINER_LOG_BASE_DIR",
    "MYSQL_DATABASE",
    "MYSQL_TRAINER_PASSWORD",
    "TRAIN_AUTOMATION_MONTHLY_OPTUNA_STORAGE_DIR",
    "TRAIN_AUTOMATION_MONTHLY_SUMMARY_REPORT_PATH",
)


@dataclass(frozen=True, slots=True)
class AutomationConfig:
    """WoL+SSH 원격 학습 자동화에 필요한 접속/타임아웃/스케줄 설정(REQ-ATA-040/081)."""

    target_mac_address: str
    """WoL 대상 MAC. 반드시 하드웨어(번인) MAC이어야 한다 — `ifconfig`류가
    보고하는 활성 인터페이스 MAC(macOS Private Wi-Fi Address로 순환 가능)을
    그대로 사용해서는 안 된다(plan.md §D, 2026-08-11 NAS 실측 근거)."""

    ssh_host: str
    ssh_port: int
    ssh_username: str
    ssh_private_key_path: Path
    """REQ-ATA-022: 읽기 전용 컨테이너 마운트 볼륨, 파일 모드 600."""
    known_hosts_path: Path
    """REQ-ATA-022: known_hosts 핀 고정 — `StrictHostKeyChecking=no` 금지."""

    db_tunnel_host: str
    """REQ-ATA-031: 기존 `db_tunnel` 계정이 위치한 NAS 호스트."""
    db_tunnel_port: int
    """터널 SSH 접속 자체의 포트(나스 sshd 포트, aaa-infra/docs/TECHSPEC.md §6.2
    비표준 55522 사용 참고) — db_tunnel_local_port/remote_port(-L 포워딩의 MySQL
    포트, 3306)와는 별개다. 혼동해서 하나로 합치지 말 것(Stage 2 실측 검증 중
    발견된 결함, 2026-08-13)."""
    db_tunnel_username: str
    db_tunnel_private_key_path: Path
    db_tunnel_local_port: int
    db_tunnel_remote_port: int

    weekly_timeout_seconds: int
    monthly_timeout_seconds: int
    staleness_threshold_days: int

    staging_models_root: Path
    """plan.md §B.5(D6): 사전 차단 설계의 임시 스테이징 경로."""
    active_models_root: Path
    """TRAIN-001의 활성 모델 경로 관례를 그대로 소비 — 이 SPEC이 재정의하지 않는다."""
    container_models_root: Path
    """SPEC-ANALYZER-TRAIN-STALENESS-001 REQ-ATD-003: `detect_stale_models()`가
    스캔하는 컨테이너 내부 경로 — NAS analyzer 컨테이너에 read-only bind
    mount된 활성 모델 디렉토리를 가리킨다. `active_models_root`(맥북 SMB 경로
    전용, `promote_staging_to_active()`가 소비)와 별개 값이며 서로 대체할 수
    없다 — 재사용 시 원격 프로모션 경로가 깨진다."""
    cache_dir: Path
    calendar_code: str
    feature_code_version: str

    mount_script_path: Path
    """SPEC-ANALYZER-TRAIN-AUTOMATION-001: 맥북상 SMB 무인 마운트 스크립트
    (`scripts/mount-nas-hdd1.sh`) 경로 — 원격 학습 CLI 실행 전 선행조건으로
    호출된다. ssh_key_path/known_hosts_path와 동일 계열(계정별 절대경로)이므로
    필수 항목이다 — 기본값을 두면 계정 불일치가 원격 SSH 실행 시점에야 불명확한
    셸 에러로 드러난다."""

    python_executable_path: Path
    """맥북상 TRAIN-001 학습 venv의 python 절대경로(예:
    `.venv/bin/python`) — 원격 SSH 실행은 비대화형 셸이라 `.zshrc` 등을
    읽지 않으므로 PATH에 pyenv/uv venv가 잡히지 않는다. `python` 하드코딩은
    이 맥에 시스템 `python`이 없어(`python3`만 존재) 종료코드 127로 실패하고,
    `python3`로 바꿔도 `analyzer` 패키지가 없는 시스템 파이썬을 가리켜
    동일하게 실패한다(Stage 실전 수동 실행 중 발견, 2026-08-13). mount_script_path와
    동일 계열(계정별 절대경로)이므로 기본값을 두지 않는다."""

    mysql_database: str
    """맥 원격 학습 CLI(`training/db.py` `get_trainer_db_config()`)에 전달할
    DB 스키마명 — analyzer 컨테이너 자신의 `MYSQL_DATABASE`와 동일 값을
    재사용한다(신규 시크릿 아님). 원격 SSH 실행은 비대화형 셸이라 이 값이
    전달되지 않으면 MissingConfigError로 학습이 즉시 실패한다(수동 실행
    실측, 2026-08-13)."""
    mysql_trainer_password: str
    """맥 원격 학습 CLI에 전달할 trainer 계정 비밀번호 — analyzer 컨테이너
    자신의 `MYSQL_TRAINER_PASSWORD`와 동일 값을 재사용한다(신규 시크릿
    아님, TRAIN-001 REQ-AT-010/011 계약)."""

    trainer_log_base_dir: Path
    """SPEC-ANALYZER-TRAIN-OBSV-001 REQ-ATO-005: 트레이너 파일(`trainer_<run_id>.log`)
    목적지 디렉토리 — 맥 로컬 SMB 마운트 지점의 절대경로다. `mount_script_path`/
    `python_executable_path`와 동일 계열(계정별 절대경로)이므로 기본값을 두지
    않는다. analyzer 프로세스는 `aaa-infra` 전용 셸 변수(`AAA_HDD_BASE`)를
    직접 읽지 않는다 — 이 필드가 이미 프로세스 환경에 반영된 리터럴 절대경로
    값만 소비한다(spec.md §4.5)."""

    monthly_optuna_storage_dir: Path
    """SPEC-ANALYZER-TRAIN-TUNING-001 REQ-ATT-010: 월간 cron 재실행 전용
    Optuna storage 디렉터리 — 수동/애드혹 캠페인 재실행이 사용하는 storage
    디렉터리와 분리된다(SQLite 파일 락 경합 회피). `tuning.create_or_resume_study()`의
    `load_if_exists=True`(REQ-AT-100)는 무수정이므로 월간 재실행 간 trial이
    이 디렉터리 안에서 누적된다."""
    monthly_summary_report_path: Path
    """SPEC-ANALYZER-TRAIN-TUNING-001 REQ-ATT-011: 월간 캠페인 요약 리포트
    출력 경로."""


def get_automation_config() -> AutomationConfig:
    """`TRAIN_AUTOMATION_*` 환경변수를 읽어 `AutomationConfig`를 구성한다.

    필수 환경변수가 누락되면 `MissingConfigError`를 발생시키며, 누락된 모든
    변수명을 한 번에 나열한다(`data/config.py` `get_db_config()`와 동일한
    일괄 검증 패턴).
    """
    missing = [name for name in _REQUIRED_ENV_VARS if name not in os.environ]
    if missing:
        raise MissingConfigError(f"필수 환경변수 누락: {', '.join(missing)}")

    return AutomationConfig(
        target_mac_address=os.environ["TRAIN_AUTOMATION_TARGET_MAC"],
        ssh_host=os.environ["TRAIN_AUTOMATION_SSH_HOST"],
        ssh_port=int(os.environ.get("TRAIN_AUTOMATION_SSH_PORT", str(_DEFAULT_SSH_PORT))),
        ssh_username=os.environ["TRAIN_AUTOMATION_SSH_USERNAME"],
        ssh_private_key_path=Path(os.environ["TRAIN_AUTOMATION_SSH_KEY_PATH"]),
        known_hosts_path=Path(os.environ["TRAIN_AUTOMATION_KNOWN_HOSTS_PATH"]),
        db_tunnel_host=os.environ["TRAIN_AUTOMATION_DB_TUNNEL_HOST"],
        db_tunnel_port=int(
            os.environ.get("TRAIN_AUTOMATION_DB_TUNNEL_SSH_PORT", str(_DEFAULT_DB_TUNNEL_SSH_PORT))
        ),
        db_tunnel_username=os.environ.get(
            "TRAIN_AUTOMATION_DB_TUNNEL_USERNAME", _DEFAULT_DB_TUNNEL_USERNAME
        ),
        db_tunnel_private_key_path=Path(os.environ["TRAIN_AUTOMATION_DB_TUNNEL_KEY_PATH"]),
        db_tunnel_local_port=int(
            os.environ.get("TRAIN_AUTOMATION_DB_TUNNEL_LOCAL_PORT", str(_DEFAULT_DB_TUNNEL_PORT))
        ),
        db_tunnel_remote_port=int(
            os.environ.get("TRAIN_AUTOMATION_DB_TUNNEL_REMOTE_PORT", str(_DEFAULT_DB_TUNNEL_PORT))
        ),
        weekly_timeout_seconds=int(
            os.environ.get(
                "TRAIN_AUTOMATION_WEEKLY_TIMEOUT_SECONDS", str(_DEFAULT_WEEKLY_TIMEOUT_SECONDS)
            )
        ),
        monthly_timeout_seconds=int(
            os.environ.get(
                "TRAIN_AUTOMATION_MONTHLY_TIMEOUT_SECONDS", str(_DEFAULT_MONTHLY_TIMEOUT_SECONDS)
            )
        ),
        staleness_threshold_days=int(
            os.environ.get(
                "TRAIN_AUTOMATION_STALENESS_THRESHOLD_DAYS",
                str(_DEFAULT_STALENESS_THRESHOLD_DAYS),
            )
        ),
        staging_models_root=Path(os.environ["TRAIN_AUTOMATION_STAGING_MODELS_ROOT"]),
        active_models_root=Path(os.environ["TRAIN_AUTOMATION_ACTIVE_MODELS_ROOT"]),
        container_models_root=Path(os.environ["TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT"]),
        cache_dir=Path(os.environ["TRAIN_AUTOMATION_CACHE_DIR"]),
        calendar_code=os.environ.get("TRAIN_AUTOMATION_CALENDAR_CODE", _DEFAULT_CALENDAR_CODE),
        feature_code_version=os.environ.get(
            "TRAIN_AUTOMATION_FEATURE_CODE_VERSION", _DEFAULT_FEATURE_CODE_VERSION
        ),
        mount_script_path=Path(os.environ["TRAIN_AUTOMATION_MOUNT_SCRIPT_PATH"]),
        python_executable_path=Path(os.environ["TRAIN_AUTOMATION_PYTHON_PATH"]),
        mysql_database=os.environ["MYSQL_DATABASE"],
        mysql_trainer_password=os.environ["MYSQL_TRAINER_PASSWORD"],
        trainer_log_base_dir=Path(os.environ["TRAIN_AUTOMATION_TRAINER_LOG_BASE_DIR"]),
        monthly_optuna_storage_dir=Path(os.environ["TRAIN_AUTOMATION_MONTHLY_OPTUNA_STORAGE_DIR"]),
        monthly_summary_report_path=Path(
            os.environ["TRAIN_AUTOMATION_MONTHLY_SUMMARY_REPORT_PATH"]
        ),
    )

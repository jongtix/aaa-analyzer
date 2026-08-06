"""DB 접속 정보 및 결제주기(settlement) config 로딩.

SPEC-ANALYZER-DATA-001 §4.8: config 항목 수가 적어(analyzer 전체 선례는 `LOG_PATH`
1건뿐) `pydantic-settings` 도입 비용이 이익을 상회한다 — 기존 관례대로
`os.environ`을 직접 읽는다. `.env`/`.env.*` 파일 자체는 절대 읽지 않는다(런타임에
이미 로드된 환경변수만 사용).
"""

import os
from dataclasses import dataclass

# REQ-AD-032: 시장별 국내 결제주기(T+N)는 하드코딩 상수가 아니라 config여야 한다.
# 시장별 기본값 — KRX 기본 T+2, 미국 기본 T+1(§4.2).
_DEFAULT_SETTLEMENT_DAYS: dict[str, int] = {
    "KRX": 2,
    "US": 1,
}


class MissingConfigError(RuntimeError):
    """필수 환경변수가 누락되었거나 지원하지 않는 시장 코드가 주어졌을 때 발생한다."""


@dataclass(frozen=True, slots=True)
class DbConfig:
    """REQ-AD-010: 커넥션 풀링을 지원하는 DB 엔진 구성에 필요한 접속 정보."""

    host: str
    port: int
    database: str
    password: str


def get_db_config() -> DbConfig:
    """`MYSQL_HOST`/`MYSQL_PORT`/`MYSQL_DATABASE`/`MYSQL_ANALYZER_PASSWORD`를 읽는다.

    앞의 셋은 `.env.common`, 마지막은 `.env.analyzer`에서 로드되나(REQ-AD-010),
    이 함수는 이미 프로세스 환경에 반영된 값만 `os.environ`으로 읽을 뿐 파일
    자체는 열지 않는다. 값 자체는 로그/예외 메시지에 노출하지 않는다.
    """
    missing = [
        name
        for name in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_ANALYZER_PASSWORD")
        if name not in os.environ
    ]
    if missing:
        raise MissingConfigError(f"필수 환경변수 누락: {', '.join(missing)}")

    return DbConfig(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ["MYSQL_PORT"]),
        database=os.environ["MYSQL_DATABASE"],
        password=os.environ["MYSQL_ANALYZER_PASSWORD"],
    )


def get_settlement_days(market: str) -> int:
    """시장별 결제주기 N(영업일)을 반환한다(REQ-AD-032).

    `DIVIDEND_SETTLEMENT_DAYS_<MARKET>` 환경변수가 설정되어 있으면 그 값을
    우선 사용하고, 없으면 `_DEFAULT_SETTLEMENT_DAYS`의 시장별 기본값을 사용한다.
    지원하지 않는 시장 코드는 `MissingConfigError`를 발생시킨다.
    """
    if market not in _DEFAULT_SETTLEMENT_DAYS:
        raise MissingConfigError(f"지원하지 않는 시장 코드: {market}")

    env_var = f"DIVIDEND_SETTLEMENT_DAYS_{market}"
    override = os.environ.get(env_var)
    if override is not None:
        return int(override)

    return _DEFAULT_SETTLEMENT_DAYS[market]

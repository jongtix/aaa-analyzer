"""`trainer` 계정 전용 SQLAlchemy 엔진 빌더 (SELECT 전용, 쓰기 금지).

SPEC-ANALYZER-TRAIN-001 M1(REQ-AT-010/011/012): 학습 파이프라인은 `analyzer`
계정(`data/repository.py`, SELECT+INSERT+UPDATE 일부)이 아니라 `trainer`
계정(SELECT 전용, SCHEMA-001에서 이미 프로비저닝 완료)으로 DB에 연결한다.
`_ANALYZER_DB_USER`(`data/repository.py`)와 명확히 구분되는 상수
(`_TRAINER_DB_USER`)를 이 신규 모듈에 정의한다(plan.md §B.1/§D).

`get_trainer_db_config()`는 `data/config.py`의 `get_db_config()`/`DbConfig`
패턴을 그대로 재사용하되(REQ-AT-011), `MYSQL_ANALYZER_PASSWORD` 대신 확정된
신규 환경변수 `MYSQL_TRAINER_PASSWORD`를 소비한다 — `.env`/`.env.*` 파일
자체는 읽지 않고 이미 프로세스에 반영된 환경변수만 사용한다.

`trainer` 계정 자체가 MySQL GRANT 레벨에서 SELECT 전용이므로(REQ-AT-012),
이 모듈이 발행하는 쿼리를 코드 레벨에서 추가로 제한하지 않는다 — 권한
강제는 DB 서버가 담당하고, 이 모듈은 연결 경로만 제공한다.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from analyzer.data.config import DbConfig, MissingConfigError

_TRAINER_DB_USER = "trainer"


def get_trainer_db_config() -> DbConfig:
    """`MYSQL_HOST`/`MYSQL_PORT`/`MYSQL_DATABASE`/`MYSQL_TRAINER_PASSWORD`를 읽는다.

    앞의 셋은 analyzer 계정과 동일한 DB 서버 접속 정보를 공유하며, 마지막
    시크릿만 `MYSQL_TRAINER_PASSWORD`로 별도 관리한다(REQ-AT-010/011).
    """
    missing = [
        name
        for name in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_TRAINER_PASSWORD")
        if name not in os.environ
    ]
    if missing:
        raise MissingConfigError(f"필수 환경변수 누락: {', '.join(missing)}")

    return DbConfig(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ["MYSQL_PORT"]),
        database=os.environ["MYSQL_DATABASE"],
        password=os.environ["MYSQL_TRAINER_PASSWORD"],
    )


def build_trainer_engine() -> Engine:
    """환경변수로부터 `trainer` 계정 전용 SQLAlchemy 엔진을 구성한다(REQ-AT-010/011/012)."""
    config = get_trainer_db_config()
    url = (
        f"mysql+pymysql://{_TRAINER_DB_USER}:{config.password}"
        f"@{config.host}:{config.port}/{config.database}"
    )
    return create_engine(url, pool_pre_ping=True)

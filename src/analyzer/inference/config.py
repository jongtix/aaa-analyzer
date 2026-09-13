"""추론 서버 설정 로딩 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-020).

`data/config.py`/`orchestration/config.py`의 선례를 그대로 계승한다 — 항목
수가 적어 `pydantic-settings` 도입 비용이 이익을 상회하므로 `os.environ`을
직접 읽는다. `.env`/`.env.*` 파일 자체는 절대 읽지 않는다(이미 프로세스
환경에 반영된 값만 사용).

Redis 접속 정보의 환경변수명은 collector가 이미 소비 중인 `.env.common`의
4개(`REDIS_HOST`/`REDIS_PORT`/`REDIS_APPUSER_USERNAME`/
`REDIS_APPUSER_PASSWORD`)를 그대로 재사용한다 — analyzer 전용 신규 시크릿을
만들지 않는다. 실제로 이 계정이 스트림 명령(XREADGROUP/XGROUP/XACK/
XAUTOCLAIM/XADD) ACL 권한을 보유하는지에 대한 NAS 실측은 REQ-AIF-141(M9)
소관이며, 실측 결과에 따라 전용 계정으로 교체될 수 있다.
"""

import os
from dataclasses import dataclass

from analyzer.data.config import MissingConfigError

DEFAULT_STREAM_CLAIM_IDLE_SECONDS = 1800
"""REQ-AIF-020 확정값(30분, M7 실측 2026-09-12) — 착수 시점 잠정값(600초/
10분)에서 상향. NAS 프로덕션 DB·실제 챔피언 모델로 (feature 조립 + 포인트
모델 배치 predict + 밴드 스윕) 전체 사이클을 시장 전체 A·B등급 유니버스에
대해 1회 측정한 결과, domestic(단일 활성 horizon D60, 62종목) 전체
사이클이 745.88초(≈12.4분)로 종전 600초 기본값을 **이미 초과**했다 —
600초는 자기유발 중복 스폰(XAUTOCLAIM 자가-재소유) 위험이 실측으로
확인된 안전하지 않은 값이었다. overseas(D20/D60, 53종목)는 306.86초로
더 짧다(D60은 별도 발견된 프로덕션 결함으로 predict 자체가 실패해 사실상
D20만 반영됨 — progress.md M7 참조). domestic 측정치(worst case) 대비
약 2.4배 마진을 두고 REQ-AIF-142의 vmalert 데드맨 N값(30분)과 동일하게
맞췄다 — 두 값 모두 "사이클이 정상보다 오래 걸린다"는 동일 실패 부류를
서로 다른 계층(내부 자가-재수령 가드 vs 외부 알람)에서 방어하므로 값을
맞추면 운용자가 기억·튜닝하기 쉽다. 도메스틱에 향후 horizon D20이 추가
활성화되면(현재는 챔피언 부재로 비활성) 사이클이 대략 2배로 늘어날 수
있어(약 1465초 추정) 재실측이 필요하다 — 이 값은 실측 시점의 활성 조합
구성에 종속적이다."""

MINIMUM_STREAM_CLAIM_IDLE_SECONDS = 60
"""REQ-AIF-020(shall not): 실제 추론 사이클 소요 시간에 근접하거나 못 미치는
짧은 값(예: 30초)은 아직 처리 중인 자기 자신의 메시지를 XAUTOCLAIM으로
재소유해 동일 (market, trade_date)에 대한 중복 자식 프로세스를 스폰시킨다 —
환경변수 오버라이드로도 이 하한 미만을 허용하지 않는다.

이 하한은 **환경변수 파싱 경로에만** 적용된다. `InferenceConfig`를 직접
생성하는 테스트(AC-AIF-002가 요구하는 짧은 idle 임계 통합 테스트 등)는
이 검증을 거치지 않는다."""

_REQUIRED_ENV_VARS = (
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_APPUSER_USERNAME",
    "REDIS_APPUSER_PASSWORD",
)

_IDLE_SECONDS_ENV_VAR = "INFERENCE_STREAM_CLAIM_IDLE_SECONDS"


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """추론 상주 부모가 필요로 하는 Redis 접속 정보와 스트림 소비 파라미터."""

    redis_host: str
    redis_port: int
    redis_username: str
    redis_password: str

    stream_claim_idle_seconds: int
    """XAUTOCLAIM 재소유 임계(초). 인-플라이트 락 TTL로도 동일 값을 사용한다
    (REQ-AIF-060 — 자식이 비정상 종료했을 때 락이 TTL로 자연 해제되도록)."""


def get_inference_config() -> InferenceConfig:
    """`REDIS_*` 환경변수를 읽어 `InferenceConfig`를 구성한다.

    필수 환경변수가 누락되면 누락된 모든 변수명을 한 번에 나열한
    `MissingConfigError`를 발생시킨다(`get_db_config()`/
    `get_automation_config()`와 동일한 일괄 검증 패턴). 값 자체는 예외
    메시지에 노출하지 않는다.
    """
    missing = [name for name in _REQUIRED_ENV_VARS if name not in os.environ]
    if missing:
        raise MissingConfigError(f"필수 환경변수 누락: {', '.join(missing)}")

    idle_seconds = int(
        os.environ.get(_IDLE_SECONDS_ENV_VAR, str(DEFAULT_STREAM_CLAIM_IDLE_SECONDS))
    )
    if idle_seconds < MINIMUM_STREAM_CLAIM_IDLE_SECONDS:
        raise ValueError(
            f"{_IDLE_SECONDS_ENV_VAR}는 {MINIMUM_STREAM_CLAIM_IDLE_SECONDS}초 이상이어야 한다"
            f" (입력값: {idle_seconds})"
        )

    return InferenceConfig(
        redis_host=os.environ["REDIS_HOST"],
        redis_port=int(os.environ["REDIS_PORT"]),
        redis_username=os.environ["REDIS_APPUSER_USERNAME"],
        redis_password=os.environ["REDIS_APPUSER_PASSWORD"],
        stream_claim_idle_seconds=idle_seconds,
    )

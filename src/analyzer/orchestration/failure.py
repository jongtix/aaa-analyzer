"""통합 실패 처리 경로 (SPEC-ANALYZER-TRAIN-AUTOMATION-001 §2.6, REQ-ATA-060/061/062).

WoL 실패(§2.1)·SSH 연결 실패(§2.2)·학습 스크립트 실패(§2.5)·타임아웃(§2.4) 중
어느 하나가 발생해도 동일한 통합 실패 처리 경로(`handle_training_run_failure`)를
실행한다(REQ-ATA-060) — 실패 유형별로 별도 처리 로직이나 별도 알림 타이밍을
분기하지 않는다(shall not). 이 원칙을 코드 구조로 강제하기 위해 4가지 실패 유형을
단일 `TrainingRunFailure` 데이터클래스(`stage: Literal[...]`)로 표현한다
(plan.md §B.2 결정 2).

08:00 KST 알림(REQ-ATA-061)은 신규 알림 채널을 도입하지 않고 기존 텔레그램/
vmalert 경로를 재사용한다(REQ-ATA-012) — 이 경로는 §2.7이 발행하는 Prometheus
알람 트리거 시그널을 vmalert가 소비해 트리거하는 구조이므로, 이 함수의 책임은
로그 기록과 Prometheus 알람 트리거 시그널 발행까지다. 실제 알림 발송 타이밍
로직(vmalert 규칙 YAML)은 REQ-ATA-094에 따라 `aaa-infra` 레포 소관이며 이 SPEC이
작성하지 않는다.

활성 모델 보존(REQ-ATA-062)은 이 함수가 직접 강제하지 않는다 — `ssh_dispatch`
모듈의 사전 차단(스테이징 디렉토리 + 성공 시에만 원자적 rename) 설계(plan.md
§B.5, D6)가 실패 시 애초에 활성 경로에 아무것도 쓰지 않도록 보장하므로, 이
경로는 로그+계측 책임만 진다.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from analyzer.orchestration.metrics import TrainingMetrics

logger = logging.getLogger(__name__)

FailureStage = Literal["wol", "ssh", "training", "timeout"]
"""REQ-ATA-060이 열거하는 4가지 실패 유형 — 이 집합 밖의 값은 타입 체커가 거부한다."""


@dataclass(frozen=True, slots=True)
class TrainingRunFailure:
    """4가지 실패 유형을 단일 타입으로 표현하는 통합 실패 이벤트(plan.md §B.2 결정 2)."""

    stage: FailureStage
    message: str
    run_id: str


def handle_training_run_failure(failure: TrainingRunFailure, metrics: TrainingMetrics) -> None:
    """REQ-ATA-060/061: 실패 유형과 무관하게 동일한 통합 경로를 실행한다.

    1. 구조화된 로그 기록(`stage`/`run_id`/`message`).
    2. Prometheus 알람 트리거 시그널 발행(`metrics.record_failure`) — 기존
       텔레그램/vmalert 경로가 이 시그널을 소비해 08:00 KST 알림을 트리거한다
       (알림 발송 자체는 이 SPEC의 스코프 밖, REQ-ATA-094).
    """
    logger.error(
        "training run failure stage=%s run_id=%s message=%s",
        failure.stage,
        failure.run_id,
        failure.message,
    )
    metrics.record_failure(stage=failure.stage)

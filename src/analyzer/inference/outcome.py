"""자식 프로세스 종료코드 계약 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-010).

부모는 **오직 자식의 종료코드만을** IPC 계약면으로 관찰한다. 자식 stdout은
로그 상관관계 목적으로만 릴레이되며 어떤 구조화 데이터로도 파싱되지 않는다
— stdout 로그 핸들러와 "결과를 stdout으로 반환"하는 패턴이 충돌하는 것을
원천 차단하기 위한 설계 결정이다.
"""

from dataclasses import dataclass

EXIT_SUCCESS = 0
"""정상 — 최소 1개 조합을 처리했고 부분 실패가 없다."""

EXIT_ALL_SKIPPED = 1
"""스킵 — 처리한 조합이 하나도 없다(해당 시장 전체 조합에 매니페스트 없음
등 정상 사유). 실패가 아니라 정상 종료의 한 갈래다."""

EXIT_PARTIAL_FAILURE = 2
"""부분실패 — 일부 종목/조합 스킵이 있었으나 프로세스 자체는 완주했다."""


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    """자식 프로세스 1회 실행의 집계 결과 — 종료코드 산출의 유일한 입력."""

    processed: int
    """신호를 실제로 산출한 (시장,horizon) 조합 수."""

    skipped_combinations: int
    """매니페스트 부재/SHA 불일치 등 정상 사유로 건너뛴 조합 수."""

    partial_failures: int
    """조합은 완주했으나 일부 종목이 실패해 스킵된 건수."""


def resolve_exit_code(outcome: InferenceOutcome) -> int:
    """집계 결과를 종료코드 0/1/2로 환원한다.

    부분실패(2)가 성공(0)보다 우선한다 — 일부 종목이 누락된 배치를 무조건
    성공으로 보고하면 부모가 관찰할 수 있는 유일한 신호가 사라진다.
    """
    if outcome.partial_failures > 0:
        return EXIT_PARTIAL_FAILURE
    if outcome.processed == 0:
        return EXIT_ALL_SKIPPED
    return EXIT_SUCCESS

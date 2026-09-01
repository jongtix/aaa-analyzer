"""운영자 호출용 활성화 롤백 CLI (SPEC-ANALYZER-TRAIN-TUNING-001 M1).

REQ-ATT-018: `enumerate_model_versions()`로 active 버전을 열거하고,
`--target-trained-date`와 일치하는 `ModelVersion`을 찾아 그 시점에 이미 계산된
`sha256`을 그대로 `rollback_activation_manifest()`에 전달한다(재해시하지 않음).

REQ-ATT-019: 대상이 열거 결과에 없으면 0이 아닌 종료코드로 종료하고 사용 가능한
`trained_date` 목록을 에러 메시지에 포함한다 — 가장 가까운 날짜로 조용히
대체하지 않는다.

REQ-ATT-020: `rollback_activation_manifest()` 호출 전 명시적 `--confirm`을
요구한다. `--confirm` 없이 호출되면 대기 중인 롤백 내용만 출력하고 매니페스트를
변경하지 않은 채 종료한다.

REQ-ATT-024: `--confirm` 게이트 통과 후 `rollback_activation_manifest()` 호출
직전에 매니페스트를 재조회해, 대상 조회 시점에 관측한 기준선(`trained_date` +
`promoted_at`)과 비교한다 — 불일치하면(월간 자동 프로모션이 조회-실행 사이에
끼어든 경쟁 상황) 롤백을 호출하지 않고 0이 아닌 종료코드로 종료한다(낙관적
동시성 검사). 이 검사 로직은 전부 이 모듈 안에 있으며 `activation.py`의
기존 시그니처는 무수정이다(PRESERVE).

이 SPEC의 완전 자동배포 정책(spec.md §1.2)에서 이 CLI는 부가 기능이 아니라
하중을 짊어지는(load-bearing) 안전망이다.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from analyzer.orchestration.activation import (
    ActivationManifest,
    read_activation_manifest,
    rollback_activation_manifest,
)
from analyzer.training.persistence import enumerate_model_versions


def _manifest_state(manifest: ActivationManifest) -> tuple[str, str]:
    """낙관적 동시성 비교 기준선 — (`trained_date`, `promoted_at`)(REQ-ATT-024)."""
    return manifest.trained_date.isoformat(), manifest.promoted_at


def _describe(manifest: ActivationManifest | None) -> str:
    if manifest is None:
        return "(매니페스트 없음)"
    trained_date, promoted_at = _manifest_state(manifest)
    return f"trained_date={trained_date}, promoted_at={promoted_at}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyzer.training.rollback",
        description="활성화 매니페스트를 지정한 trained_date 버전으로 되돌린다(운영자 수동 호출).",
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--target-trained-date", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="실제 롤백을 수행한다. 미지정 시 대기 중인 롤백 내용만 출력한다(REQ-ATT-020).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점 — 성공 시 `0`, 실패 시 `1`을 반환한다.

    @MX:WARN: [AUTO] 서빙 대상 전환 경로 — `--confirm` 게이트와 낙관적 동시성
    검사(REQ-ATT-024)를 우회하는 지름길을 추가해서는 안 된다.
    @MX:REASON: 이 CLI는 완전 자동배포(월간 무인 재승격)의 유일한 안전망이며,
    검사를 건너뛰면 롤백이 방금 전 자동 프로모션을 무신호로 덮어쓴다.
    """
    args = _build_parser().parse_args(argv)
    combo = f"{args.market}/{args.horizon}/{args.algorithm}"

    # (a) active 버전 열거 + (b) 대상 조회 — 이 시점의 매니페스트가 기준선이다.
    versions = enumerate_model_versions(args.models_root, args.market, args.horizon, args.algorithm)
    target = next((v for v in versions if v.trained_date == args.target_trained_date), None)
    baseline = read_activation_manifest(args.models_root, args.market, args.horizon, args.algorithm)

    if target is None:
        available = ", ".join(v.trained_date.isoformat() for v in versions) or "(없음)"
        print(
            f"롤백 대상 trained_date={args.target_trained_date.isoformat()}이 {combo}의 "
            f"active 버전에 없다. 사용 가능한 trained_date: {available}",
            file=sys.stderr,
        )
        return 1

    if baseline is None:
        print(f"롤백할 활성화 매니페스트가 존재하지 않는다: {combo}", file=sys.stderr)
        return 1

    if not args.confirm:
        print(
            f"[대기 중인 롤백] {combo}: "
            f"현재 trained_date={baseline.trained_date.isoformat()} -> "
            f"대상 trained_date={target.trained_date.isoformat()}\n"
            "--confirm이 지정되지 않아 매니페스트를 변경하지 않는다."
        )
        return 0

    # (c) 낙관적 동시성 검사 — 호출 직전 재조회 후 기준선과 대조(REQ-ATT-024).
    current = read_activation_manifest(args.models_root, args.market, args.horizon, args.algorithm)
    if current is None or _manifest_state(current) != _manifest_state(baseline):
        print(
            f"동시쓰기 경쟁 감지 — {combo}의 활성화 매니페스트가 대상 조회 이후 변경되었다. "
            f"롤백을 중단한다(자동 프로모션을 덮어쓰지 않음). "
            f"조회 당시: {_describe(baseline)} / 재조회 시점: {_describe(current)}",
            file=sys.stderr,
        )
        return 1

    # (d) 열거 시점에 이미 계산된 사이드카 해시를 그대로 전달한다(재해시 없음).
    rolled_back = rollback_activation_manifest(
        args.models_root,
        market=args.market,
        horizon=args.horizon,
        algorithm=args.algorithm,
        target_trained_date=target.trained_date,
        target_sidecar_sha256=target.sha256,
    )
    print(
        f"[롤백 완료] {combo}: trained_date={rolled_back.trained_date.isoformat()} "
        f"(이전 {baseline.trained_date.isoformat()})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""활성화 매니페스트 + 스코어링 전략 매니페스트 원자적 읽기/쓰기 + 롤백
(SPEC-ANALYZER-TRAIN-EVAL-001 M6, design.md §6).

REQ-ATE-048/049: (시장,horizon,algorithm) 조합별로 현재 서빙 대상 모델
아티팩트를 가리키는 활성화 매니페스트를 유지한다 — `trained_date`, 사이드카
SHA-256, 프로모션 시각, 프로모션 근거 지표 요약을 포함하며, 갱신은 임시
파일 작성 후 `os.replace()`(원자적 치환)로 수행한다.

REQ-ATE-050: (시장,horizon)별 스코어링 전략(lgbm 단독/xgb 단독/앙상블)
매니페스트를 활성화 매니페스트와 독립적으로 갱신 가능한 별도 파일로 유지한다.

REQ-ATE-051: `promote_staging_to_active()`(`ssh_dispatch.py`, cp -a)의 병합
메커니즘 자체는 이 모듈이 건드리지 않는다 — 이 모듈은 "저장 이후 선택"
레이어만 담당한다(design.md §6.2 책임 분리).

REQ-ATE-052(F3): 매니페스트 갱신은 (a) 아티팩트가 활성 저장소에 이미
병합되었고 AND (b) §2.8(1차 배포 경로) 또는 §2.10(상시 게이트 경로) 둘 중
하나의 판정을 통과한 경우에만 트리거된다 — `promote_activation_manifest()`가
이 두 조건을 단일 게이트로 강제한다.

REQ-ATE-053: 롤백은 매니페스트의 `trained_date`를 이전 값으로 재기록하는
것만으로 완료된다 — 어떤 모델 파일도 이동/복사/삭제하지 않는다(2단계
보존 정책이 이미 최근 12개 버전을 active 경로에 보존하므로).

REQ-ATE-054: 매니페스트가 가리키는 `trained_date`가 2단계 보존 정책에 의해
아카이브로 이동된 경우, 그 댕글링 상태를 감지한다.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

_ACTIVATION_MANIFEST_FILENAME = "activation_manifest.json"
_STRATEGY_MANIFEST_FILENAME = "strategy_manifest.json"


def activation_manifest_path(models_root: Path, market: str, horizon: int, algorithm: str) -> Path:
    """(시장,horizon,algorithm) 조합별 활성화 매니페스트 경로(REQ-ATE-048).

    `persistence.model_dir()`와 동일한 관례(`models/{market}/{horizon}/
    {algorithm}/`)의 인접 파일로 둔다 — `persistence.py` 자체는 무수정.
    """
    return models_root / market / str(horizon) / algorithm / _ACTIVATION_MANIFEST_FILENAME


def strategy_manifest_path(models_root: Path, market: str, horizon: int) -> Path:
    """(시장,horizon)별 스코어링 전략 매니페스트 경로(REQ-ATE-050)."""
    return models_root / market / str(horizon) / _STRATEGY_MANIFEST_FILENAME


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """임시 파일 작성 후 `os.replace()`로 원자적 치환한다(REQ-ATE-049).

    쓰기 도중 예외가 발생해도(임시 파일 작성 단계에서 실패) 대상 경로의
    기존 내용은 손상되지 않는다 — `os.replace()` 호출 자체는 원자적이며,
    이 함수는 그 이전에 완전히 직렬화된 임시 파일만 치환 대상으로 삼는다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


@dataclass(frozen=True, slots=True)
class ActivationManifest:
    """(시장,horizon,algorithm) 조합 1개의 활성화 매니페스트(REQ-ATE-048)."""

    market: str
    horizon: int
    algorithm: str
    trained_date: date
    sidecar_sha256: str
    promoted_at: str
    promotion_basis: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "horizon": self.horizon,
            "algorithm": self.algorithm,
            "trained_date": self.trained_date.isoformat(),
            "sidecar_sha256": self.sidecar_sha256,
            "promoted_at": self.promoted_at,
            "promotion_basis": dict(self.promotion_basis),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ActivationManifest:
        return cls(
            market=payload["market"],
            horizon=payload["horizon"],
            algorithm=payload["algorithm"],
            trained_date=date.fromisoformat(payload["trained_date"]),
            sidecar_sha256=payload["sidecar_sha256"],
            promoted_at=payload["promoted_at"],
            promotion_basis=payload.get("promotion_basis", {}),
        )


def write_activation_manifest(models_root: Path, manifest: ActivationManifest) -> Path:
    """활성화 매니페스트를 원자적으로 기록한다(REQ-ATE-049)."""
    path = activation_manifest_path(
        models_root, manifest.market, manifest.horizon, manifest.algorithm
    )
    _atomic_write_json(path, manifest.to_payload())
    return path


def read_activation_manifest(
    models_root: Path, market: str, horizon: int, algorithm: str
) -> ActivationManifest | None:
    """활성화 매니페스트를 읽는다 — 아직 존재하지 않으면(1차 배포 이전) None을
    반환한다(§B 리스크 6, "매니페스트 없음 = 아직 배포 안 됨")."""
    path = activation_manifest_path(models_root, market, horizon, algorithm)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ActivationManifest.from_payload(payload)


def promote_activation_manifest(
    models_root: Path,
    *,
    market: str,
    horizon: int,
    algorithm: str,
    merged_to_active: bool,
    gate_passed: bool,
    trained_date: date,
    sidecar_sha256: str,
    promotion_basis: Mapping[str, Any],
) -> ActivationManifest | None:
    """REQ-ATE-052(F3): 매니페스트 갱신을 (병합 성공 AND 게이트 통과) 단일
    조건으로 게이팅한다 — §2.8(1차 배포, `gate_passed`=안정화+챔피언 선정
    통과)과 §2.10(상시 게이트, `gate_passed`=승격 판정 통과) 양쪽 경로가
    이 함수 하나를 공유한다. 조건을 만족하지 않으면 쓰기 없이 `None`을
    반환한다(병합 성공만으로 매니페스트가 갱신되지 않음, AC-ATE-038 4번째
    시나리오).

    @MX:ANCHOR: [AUTO] 활성화 매니페스트의 유일한 쓰기 경로 — 1차 배포
    (campaign.py `activate_market_horizon_combo()`)와 상시 게이트
    (promotion_gate.py `evaluate_and_promote()`) 양쪽이 이 함수를 공유한다.
    @MX:REASON: fan_in >= 2, 서빙 대상 모델을 가리키는 유일한 진실원(SSOT)을
    갱신하는 지점이므로 우회 경로가 생기면 활성화 상태 불일치가 발생한다.
    """
    if not (merged_to_active and gate_passed):
        return None
    manifest = ActivationManifest(
        market=market,
        horizon=horizon,
        algorithm=algorithm,
        trained_date=trained_date,
        sidecar_sha256=sidecar_sha256,
        promoted_at=datetime.now(UTC).isoformat(),
        promotion_basis=dict(promotion_basis),
    )
    write_activation_manifest(models_root, manifest)
    return manifest


def rollback_activation_manifest(
    models_root: Path,
    *,
    market: str,
    horizon: int,
    algorithm: str,
    target_trained_date: date,
    target_sidecar_sha256: str,
) -> ActivationManifest:
    """REQ-ATE-053: 매니페스트의 `trained_date`를 이전 값으로 재기록한다 —
    어떤 파일 시스템 조작(이동/복사/삭제)도 수행하지 않는다(2단계 보존
    정책이 이미 최근 12개 버전을 active 경로에 보존하므로, 되돌릴 대상
    파일이 항상 존재한다). `target_sidecar_sha256`은 호출자가 롤백 대상
    버전의 사이드카에서 직접 조회해 전달한다(이 함수는 파일을 읽지 않음).

    @MX:WARN: [AUTO] 배포 롤백 경로 — 활성 서빙 대상을 가리키는 상태를
    되돌리는 유일한 함수. 잘못된 `target_sidecar_sha256`을 전달하면
    실제 아티팩트와 매니페스트가 불일치한 상태로 롤백될 수 있다.
    @MX:REASON: 호출자가 파일 시스템 대신 대상 버전의 사이드카를 직접
    조회해 전달해야 하는 계약(REQ-ATE-053)이며, 이 계약을 어기면
    무결성 검증 없이 서빙 대상이 전환된다.
    """
    existing = read_activation_manifest(models_root, market, horizon, algorithm)
    if existing is None:
        raise ValueError(
            f"롤백할 활성화 매니페스트가 존재하지 않는다: {market}/{horizon}/{algorithm}"
        )
    rolled_back = ActivationManifest(
        market=market,
        horizon=horizon,
        algorithm=algorithm,
        trained_date=target_trained_date,
        sidecar_sha256=target_sidecar_sha256,
        promoted_at=datetime.now(UTC).isoformat(),
        promotion_basis={
            **dict(existing.promotion_basis),
            "rollback_from_trained_date": existing.trained_date.isoformat(),
        },
    )
    write_activation_manifest(models_root, rolled_back)
    return rolled_back


def detect_dangling_manifest(
    manifest: ActivationManifest, active_trained_dates: Sequence[date]
) -> bool:
    """REQ-ATE-054: 매니페스트가 가리키는 `trained_date`가 현재 active 경로에
    보존된 버전 목록(`active_trained_dates`, `persistence.apply_retention_policy()`
    호출 결과의 `RetentionResult.active`에서 유도)에 없으면 댕글링 상태로
    판정한다 — 2단계 보존 정책에 의해 아카이브로 이동되었음을 의미한다.
    """
    return manifest.trained_date not in set(active_trained_dates)


@dataclass(frozen=True, slots=True)
class ScoringStrategyManifest:
    """(시장,horizon) 조합의 활성 스코어링 전략 매니페스트(REQ-ATE-050)."""

    market: str
    horizon: int
    active_strategy: str
    updated_at: str
    basis: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "horizon": self.horizon,
            "active_strategy": self.active_strategy,
            "updated_at": self.updated_at,
            "basis": dict(self.basis),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ScoringStrategyManifest:
        return cls(
            market=payload["market"],
            horizon=payload["horizon"],
            active_strategy=payload["active_strategy"],
            updated_at=payload["updated_at"],
            basis=payload.get("basis", {}),
        )


def write_strategy_manifest(
    models_root: Path,
    *,
    market: str,
    horizon: int,
    active_strategy: str,
    basis: Mapping[str, Any] | None = None,
) -> Path:
    """스코어링 전략 매니페스트를 원자적으로 기록한다(REQ-ATE-050) — 활성화
    매니페스트(조합별)와 독립적으로 갱신 가능한 별도 파일이다."""
    manifest = ScoringStrategyManifest(
        market=market,
        horizon=horizon,
        active_strategy=active_strategy,
        updated_at=datetime.now(UTC).isoformat(),
        basis=dict(basis or {}),
    )
    path = strategy_manifest_path(models_root, market, horizon)
    _atomic_write_json(path, manifest.to_payload())
    return path


def read_strategy_manifest(
    models_root: Path, market: str, horizon: int
) -> ScoringStrategyManifest | None:
    """스코어링 전략 매니페스트를 읽는다 — 존재하지 않으면 None."""
    path = strategy_manifest_path(models_root, market, horizon)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScoringStrategyManifest.from_payload(payload)

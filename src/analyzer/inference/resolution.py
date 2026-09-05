"""매니페스트 기반 모델 해석 — Lazy Load 대상 경량 조회 + SHA-256 검증 +
G2/G3 정책 (SPEC-ANALYZER-INFER-001 M2, design.md §2).

`training/gate.py`의 `resolve_champion_model_paths()`가 갖는 "활성화
매니페스트 부재 조합은 매핑에서 제외" 동작 패턴을 계승하되, optuna/paramiko를
임포트하는 `gate.py`(맥 측 CLI 전용 모듈) 전체를 자식 프로세스 임포트
그래프에 끌고 오지 않기 위해 이 경량 모듈로 로직을 이전한다(REQ-AIF-030,
AC-AIF-004) — `gate.py`는 무수정이며 이 모듈은 그것을 임포트하지 않는다.

이 모듈은 `orchestration/activation.py`·`training/persistence.py`를
무수정 재사용한다(PRESERVE) — 매니페스트/모델 파일에 대해 읽기 전용
소비자로만 동작하며 어떤 쓰기 로직도 갖지 않는다.

범위 경계: 분위수 모델 해석(`resolve_latest_quantile_manifest()`,
REQ-AIF-050)과 confidence 가드(REQ-AIF-051)는 M3 소관이다 — 이 모듈은
포인트 모델(lightgbm/xgboost) 대상 해석과 G3 score 컬럼 정책
(REQ-AIF-040/041)까지만 다룬다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from analyzer.orchestration.activation import (
    ActivationManifest,
    read_activation_manifest,
    read_strategy_manifest,
)
from analyzer.training.ensemble import compute_ensemble_score
from analyzer.training.persistence import model_dir, model_filename, verify_model_integrity

_POINT_ALGORITHMS: tuple[str, ...] = ("lightgbm", "xgboost")


class SkipReason(StrEnum):
    """REQ-AIF-130 스킵 사유 레이블 중 이 모듈이 산출하는 두 가지.

    나머지 레이블(quantile_missing/degenerate_quantile은 M3,
    feature_insufficient는 M5, manifest_race는 M6/M7)은 이 모듈의 범위 밖이다.
    """

    NO_MANIFEST = "no_manifest"
    SHA_MISMATCH = "sha_mismatch"


@dataclass(frozen=True, slots=True)
class ServingPlan:
    """(시장,horizon) 조합의 서빙 대상 — 포인트 모델 알고리즘 집합 + 각
    알고리즘의 활성화 매니페스트 + SHA-256 검증을 통과한 모델 파일 경로
    (design.md §2)."""

    market: str
    horizon: int
    active_strategy: str
    algorithms: tuple[str, ...]
    manifests: Mapping[str, ActivationManifest]
    model_paths: Mapping[str, Path]


def resolve_serving_targets(
    models_root: Path, market: str, horizon: int
) -> ServingPlan | SkipReason:
    """design.md §2 알고리즘을 그대로 구현한다 — 전략 매니페스트로 알고리즘
    집합을 결정(G3)하고, 각 알고리즘의 활성화 매니페스트를 조회(G2)한 뒤,
    로드 전 SHA-256 사이드카 검증을 수행한다(REQ-AIF-032).

    반환값은 `ServingPlan`(전부 성공) 또는 `SkipReason`(스킵 사유) — 폴백
    모델은 어떤 스킵 경로에서도 사용하지 않는다(REQ-AIF-031, G2).
    """
    strategy = read_strategy_manifest(models_root, market, horizon)
    if strategy is None:
        return SkipReason.NO_MANIFEST  # REQ-AIF-031: 조합 자체 미배포

    if strategy.active_strategy == "ensemble":
        algorithms = _POINT_ALGORITHMS
    else:
        algorithms = (strategy.active_strategy,)

    manifests: dict[str, ActivationManifest] = {}
    for algorithm in algorithms:
        manifest = read_activation_manifest(models_root, market, horizon, algorithm)
        if manifest is None:
            # REQ-AIF-031: strategy_manifest.json은 있으나 참조 알고리즘의
            # activation_manifest.json이 없는 불일치 상태 — 조합 전체 스킵.
            return SkipReason.NO_MANIFEST
        manifests[algorithm] = manifest

    model_paths: dict[str, Path] = {}
    for algorithm, manifest in manifests.items():
        path = model_dir(models_root, market, horizon, algorithm) / model_filename(
            market, horizon, algorithm, manifest.trained_date
        )
        sidecar_path = path.with_suffix(path.suffix + ".sha256")
        if not path.is_file() or not sidecar_path.is_file():
            return SkipReason.SHA_MISMATCH  # REQ-AIF-032: 로드 실패로 처리
        if not verify_model_integrity(path, sidecar_path):
            return SkipReason.SHA_MISMATCH
        model_paths[algorithm] = path

    return ServingPlan(
        market=market,
        horizon=horizon,
        active_strategy=strategy.active_strategy,
        algorithms=algorithms,
        manifests=manifests,
        model_paths=model_paths,
    )


@dataclass(frozen=True, slots=True)
class ScoreColumns:
    """G3 score 컬럼 산출 결과 — `trading_signals.lgbm_score`/`xgb_score`/
    `score` 3개 컬럼에 그대로 대응한다(REQ-AIF-040/041)."""

    lgbm_score: float | None
    xgb_score: float | None
    score: float


def compute_score_columns(active_strategy: str, predictions: Mapping[str, float]) -> ScoreColumns:
    """G3 score 컬럼 채움 규칙(REQ-AIF-040/041, AC-AIF-007/008).

    `active_strategy == "ensemble"`이면 `compute_ensemble_score()`(ADR-033
    공식, `training/ensemble.py`, PRESERVE — 시그니처·내부 로직 무수정)로
    앙상블 score를 산출하고 두 알고리즘의 원 예측값을 각 컬럼에 채운다.
    단독 전략(`"lightgbm"`/`"xgboost"`)이면 score는 그 알고리즘의 원
    예측값이며, 예측되지 않은 반대쪽 알고리즘의 컬럼은 NULL(`None`)로
    기록한다(collector Flyway NOT NULL 완화 마이그레이션 선행 필요,
    spec.md §4.1) — 앙상블 공식은 적용하지 않는다.
    """
    if active_strategy == "ensemble":
        lgbm_score = predictions["lightgbm"]
        xgb_score = predictions["xgboost"]
        result = compute_ensemble_score(lgbm_score, xgb_score)
        return ScoreColumns(lgbm_score=lgbm_score, xgb_score=xgb_score, score=result.score_ensemble)

    if active_strategy == "lightgbm":
        score = predictions["lightgbm"]
        return ScoreColumns(lgbm_score=score, xgb_score=None, score=score)

    if active_strategy == "xgboost":
        score = predictions["xgboost"]
        return ScoreColumns(lgbm_score=None, xgb_score=score, score=score)

    raise ValueError(f"알 수 없는 active_strategy: {active_strategy!r}")

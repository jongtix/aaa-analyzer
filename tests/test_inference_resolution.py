"""매니페스트 기반 모델 해석 명세 테스트 (SPEC-ANALYZER-INFER-001 M2,
REQ-AIF-030/031/032/040/041).
"""

import hashlib
from datetime import date
from pathlib import Path

import pytest

from analyzer.inference.resolution import (
    ScoreColumns,
    ServingPlan,
    SkipReason,
    compute_score_columns,
    resolve_serving_targets,
)
from analyzer.orchestration import activation as activation_module
from analyzer.training import gate as gate_module
from analyzer.training import persistence as persistence_module


def _write_strategy(models_root: Path, market: str, horizon: int, active_strategy: str) -> None:
    activation_module.write_strategy_manifest(
        models_root, market=market, horizon=horizon, active_strategy=active_strategy
    )


def _write_activation(
    models_root: Path, market: str, horizon: int, algorithm: str, trained_date: date
) -> None:
    activation_module.write_activation_manifest(
        models_root,
        activation_module.ActivationManifest(
            market=market,
            horizon=horizon,
            algorithm=algorithm,
            trained_date=trained_date,
            sidecar_sha256="deadbeef",
            promoted_at="2026-08-19T00:00:00+00:00",
            promotion_basis={},
        ),
    )


def _write_model_with_sidecar(
    models_root: Path,
    market: str,
    horizon: int,
    algorithm: str,
    trained_date: date,
    *,
    content: bytes = b"dummy-model-bytes",
    corrupt_sidecar: bool = False,
) -> Path:
    target_dir = persistence_module.model_dir(models_root, market, horizon, algorithm)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / persistence_module.model_filename(
        market, horizon, algorithm, trained_date
    )
    model_path.write_bytes(content)
    sha256 = hashlib.sha256(content).hexdigest()
    sidecar_path = model_path.with_suffix(model_path.suffix + ".sha256")
    sidecar_path.write_text("0" * 64 if corrupt_sidecar else sha256, encoding="utf-8")
    return model_path


def _write_full_combo(
    models_root: Path, market: str, horizon: int, algorithm: str, trained_date: date
) -> None:
    _write_activation(models_root, market, horizon, algorithm, trained_date)
    _write_model_with_sidecar(models_root, market, horizon, algorithm, trained_date)


class TestResolveServingTargetsEnsemble:
    """AC-AIF-007 전제 — ensemble 조합은 두 알고리즘 모두 해석돼야 한다."""

    def test_resolves_both_algorithms_when_manifests_and_artifacts_present(self, tmp_path: Path):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_strategy(models_root, "domestic", 20, "ensemble")
        _write_full_combo(models_root, "domestic", 20, "lightgbm", trained_date)
        _write_full_combo(models_root, "domestic", 20, "xgboost", trained_date)

        plan = resolve_serving_targets(models_root, "domestic", 20)

        assert isinstance(plan, ServingPlan)
        assert plan.active_strategy == "ensemble"
        assert plan.algorithms == ("lightgbm", "xgboost")
        assert set(plan.manifests.keys()) == {"lightgbm", "xgboost"}
        assert set(plan.model_paths.keys()) == {"lightgbm", "xgboost"}


class TestResolveServingTargetsSolo:
    """AC-AIF-008 전제 — 단독 전략 조합은 그 알고리즘 하나만 해석돼야 한다."""

    def test_resolves_single_algorithm_for_solo_strategy(self, tmp_path: Path):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_strategy(models_root, "domestic", 60, "xgboost")
        _write_full_combo(models_root, "domestic", 60, "xgboost", trained_date)

        plan = resolve_serving_targets(models_root, "domestic", 60)

        assert isinstance(plan, ServingPlan)
        assert plan.active_strategy == "xgboost"
        assert plan.algorithms == ("xgboost",)
        assert set(plan.manifests.keys()) == {"xgboost"}


class TestResolveServingTargetsG2Skip:
    """AC-AIF-005: 매니페스트 부재 조합은 스킵돼야 한다 — 폴백 모델 금지."""

    def test_no_strategy_manifest_is_skipped(self, tmp_path: Path):
        models_root = tmp_path / "models"
        # domestic/20 조합 자체가 미배포(현재 라이브 상태 재현).

        result = resolve_serving_targets(models_root, "domestic", 20)

        assert result is SkipReason.NO_MANIFEST

    def test_strategy_manifest_without_referenced_activation_manifest_is_skipped(
        self, tmp_path: Path
    ):
        """§B 경계 사례: strategy_manifest는 있으나 참조 activation_manifest가 없음."""
        models_root = tmp_path / "models"
        _write_strategy(models_root, "domestic", 20, "ensemble")
        _write_full_combo(models_root, "domestic", 20, "lightgbm", date(2026, 8, 19))
        # xgboost activation_manifest는 기록하지 않음 — 불일치 상태.

        result = resolve_serving_targets(models_root, "domestic", 20)

        assert result is SkipReason.NO_MANIFEST


class TestResolveServingTargetsShaMismatch:
    """AC-AIF-006: SHA-256 검증 실패는 해당 조합만 스킵돼야 한다."""

    def test_corrupted_sidecar_is_skipped(self, tmp_path: Path):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_strategy(models_root, "domestic", 60, "xgboost")
        _write_activation(models_root, "domestic", 60, "xgboost", trained_date)
        _write_model_with_sidecar(
            models_root, "domestic", 60, "xgboost", trained_date, corrupt_sidecar=True
        )

        result = resolve_serving_targets(models_root, "domestic", 60)

        assert result is SkipReason.SHA_MISMATCH

    def test_missing_model_file_is_skipped(self, tmp_path: Path):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_strategy(models_root, "domestic", 60, "xgboost")
        _write_activation(models_root, "domestic", 60, "xgboost", trained_date)
        # 모델 파일/사이드카 자체를 디스크에 쓰지 않음.

        result = resolve_serving_targets(models_root, "domestic", 60)

        assert result is SkipReason.SHA_MISMATCH

    def test_only_the_failing_combo_is_skipped(self, tmp_path: Path):
        """다른 조합은 정상 처리돼야 한다(REQ-AIF-032, shall not 프로세스 전체 중단)."""
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_strategy(models_root, "domestic", 60, "xgboost")
        _write_activation(models_root, "domestic", 60, "xgboost", trained_date)
        _write_model_with_sidecar(
            models_root, "domestic", 60, "xgboost", trained_date, corrupt_sidecar=True
        )
        _write_strategy(models_root, "overseas", 60, "xgboost")
        _write_full_combo(models_root, "overseas", 60, "xgboost", trained_date)

        broken = resolve_serving_targets(models_root, "domestic", 60)
        healthy = resolve_serving_targets(models_root, "overseas", 60)

        assert broken is SkipReason.SHA_MISMATCH
        assert isinstance(healthy, ServingPlan)


class TestResolveServingTargetsPathParity:
    """AC-AIF-004: 매니페스트 경로 해석 결과는 resolve_champion_model_paths()와
    동일한 조합을 반환해야 한다."""

    def test_model_path_matches_resolve_champion_model_paths(self, tmp_path: Path):
        models_root = tmp_path / "models"
        trained_date = date(2026, 8, 19)
        _write_strategy(models_root, "domestic", 60, "xgboost")
        _write_full_combo(models_root, "domestic", 60, "xgboost", trained_date)

        plan = resolve_serving_targets(models_root, "domestic", 60)
        champion_paths = gate_module.resolve_champion_model_paths(models_root)

        assert isinstance(plan, ServingPlan)
        assert plan.model_paths["xgboost"] == champion_paths[("domestic", 60, "xgboost")]


class TestStaticImportBoundary:
    """AC-AIF-004: 자식 프로세스는 optuna/paramiko를 임포트하는 gate.py 전체를
    임포트해서는 안 된다 — 정적 검사.

    docstring 산문에 "optuna/paramiko"라는 단어가 등장하는 것은 허용된다
    (이 모듈이 회피하는 대상을 설명하는 문서화 목적) — 실제 import 구문
    라인만 검사 대상이다.
    """

    def test_resolution_module_does_not_import_gate_module(self):
        source_lines = (
            Path("src/analyzer/inference/resolution.py").read_text(encoding="utf-8").splitlines()
        )
        import_lines = [
            line.strip() for line in source_lines if line.strip().startswith(("import ", "from "))
        ]

        assert not any(
            "training.gate" in line or "training import gate" in line for line in import_lines
        )
        assert not any("optuna" in line for line in import_lines)
        assert not any("paramiko" in line for line in import_lines)


class TestComputeScoreColumnsEnsemble:
    """AC-AIF-007: ensemble 조합은 compute_ensemble_score()로 score를 산출하고
    lgbm_score/xgb_score를 각 모델 원 예측값으로 채운다."""

    def test_ensemble_score_matches_adr033_formula(self):
        columns = compute_score_columns("ensemble", {"lightgbm": 0.04, "xgboost": 0.03})

        assert columns == ScoreColumns(lgbm_score=0.04, xgb_score=0.03, score=0.03)


class TestComputeScoreColumnsSolo:
    """AC-AIF-008: 단독 전략 조합은 score를 원 예측값으로 채우고 반대쪽
    컬럼은 NULL(None)로 기록한다."""

    def test_xgboost_solo_leaves_lgbm_score_null(self):
        columns = compute_score_columns("xgboost", {"xgboost": 0.021})

        assert columns == ScoreColumns(lgbm_score=None, xgb_score=0.021, score=0.021)

    def test_lightgbm_solo_leaves_xgb_score_null(self):
        columns = compute_score_columns("lightgbm", {"lightgbm": -0.015})

        assert columns == ScoreColumns(lgbm_score=-0.015, xgb_score=None, score=-0.015)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            compute_score_columns("unknown", {})

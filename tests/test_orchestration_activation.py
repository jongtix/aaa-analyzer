"""활성화 매니페스트 + 스코어링 전략 매니페스트 원자적 쓰기/롤백/댕글링 감지
테스트 (SPEC-ANALYZER-TRAIN-EVAL-001 M6, REQ-ATE-048~054).
"""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from analyzer.orchestration.activation import (
    ActivationManifest,
    activation_manifest_path,
    detect_dangling_manifest,
    promote_activation_manifest,
    read_activation_manifest,
    read_strategy_manifest,
    rollback_activation_manifest,
    write_activation_manifest,
    write_strategy_manifest,
)


class TestActivationManifestAtomicWrite:
    """REQ-ATE-049: 원자적 쓰기 — 쓰기 도중 중단 시뮬레이션에서도 이전
    매니페스트 상태를 보존한다."""

    def test_interrupted_write_preserves_prior_manifest(self, tmp_path: Path):
        models_root = tmp_path / "models"
        manifest_v1 = ActivationManifest(
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            trained_date=date(2026, 8, 10),
            sidecar_sha256="v1hash",
            promoted_at="2026-08-10T00:00:00+00:00",
            promotion_basis={"gate1_rolling_mean_rank_ic": 0.03},
        )
        write_activation_manifest(models_root, manifest_v1)

        manifest_v2 = ActivationManifest(
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            trained_date=date(2026, 8, 17),
            sidecar_sha256="v2hash",
            promoted_at="2026-08-17T00:00:00+00:00",
            promotion_basis={},
        )

        original_write_text = Path.write_text

        def _raise_on_tmp(self: Path, *args, **kwargs):
            if self.name.endswith(".tmp"):
                raise OSError("simulated interrupt mid-write")
            return original_write_text(self, *args, **kwargs)

        with patch.object(Path, "write_text", _raise_on_tmp):
            with pytest.raises(OSError, match="simulated interrupt"):
                write_activation_manifest(models_root, manifest_v2)

        # 이전(v1) 매니페스트 내용이 손상 없이 그대로 남아 있어야 한다.
        restored = read_activation_manifest(models_root, "domestic", 20, "lightgbm")
        assert restored is not None
        assert restored.trained_date == date(2026, 8, 10)
        assert restored.sidecar_sha256 == "v1hash"

        # 임시 파일이 남아 있지 않아야 한다는 요구는 없으나, 최종 목적 파일이
        # v2로 치환되지 않았음이 핵심 불변식이다.
        path = activation_manifest_path(models_root, "domestic", 20, "lightgbm")
        assert path.exists()

    def test_write_then_read_roundtrip(self, tmp_path: Path):
        models_root = tmp_path / "models"
        manifest = ActivationManifest(
            market="overseas",
            horizon=60,
            algorithm="xgboost",
            trained_date=date(2026, 8, 17),
            sidecar_sha256="abc123",
            promoted_at="2026-08-17T01:00:00+00:00",
            promotion_basis={"gate2_mean_rank_ic": 0.02},
        )
        write_activation_manifest(models_root, manifest)

        loaded = read_activation_manifest(models_root, "overseas", 60, "xgboost")
        assert loaded == manifest

    def test_read_returns_none_when_manifest_absent(self, tmp_path: Path):
        """§B 리스크 6: 매니페스트 없음 = 아직 배포 안 됨."""
        assert read_activation_manifest(tmp_path / "models", "domestic", 20, "lightgbm") is None


class TestActivationManifestUpdateGate:
    """REQ-ATE-052(F3): 매니페스트 갱신은 (병합 성공 AND 게이트 통과)일 때만
    트리거된다 — §2.8(1차 배포) 또는 §2.10(상시 게이트) 어느 경로든 이
    단일 게이트를 공유한다."""

    def test_merge_alone_does_not_trigger_manifest_write(self, tmp_path: Path):
        result = promote_activation_manifest(
            tmp_path / "models",
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            merged_to_active=True,
            gate_passed=False,
            trained_date=date(2026, 8, 17),
            sidecar_sha256="x",
            promotion_basis={},
        )

        assert result is None
        assert read_activation_manifest(tmp_path / "models", "domestic", 20, "lightgbm") is None

    def test_gate_pass_without_merge_does_not_trigger_manifest_write(self, tmp_path: Path):
        result = promote_activation_manifest(
            tmp_path / "models",
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            merged_to_active=False,
            gate_passed=True,
            trained_date=date(2026, 8, 17),
            sidecar_sha256="x",
            promotion_basis={},
        )

        assert result is None
        assert read_activation_manifest(tmp_path / "models", "domestic", 20, "lightgbm") is None

    def test_both_conditions_true_writes_manifest_initial_deployment_path(self, tmp_path: Path):
        """§2.8 1차 배포 경로 — AC-ATE-034 두 번째 시나리오."""
        result = promote_activation_manifest(
            tmp_path / "models",
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            merged_to_active=True,
            gate_passed=True,
            trained_date=date(2026, 8, 17),
            sidecar_sha256="deadbeef",
            promotion_basis={"gate1_rolling_mean_rank_ic": 0.03},
        )

        assert result is not None
        loaded = read_activation_manifest(tmp_path / "models", "domestic", 20, "lightgbm")
        assert loaded is not None
        assert loaded.trained_date == date(2026, 8, 17)

    def test_both_conditions_true_writes_manifest_standing_gate_path(self, tmp_path: Path):
        """§2.10 상시 게이트 경로 — AC-ATE-034 세 번째 시나리오. 기존 매니페스트가
        이미 있어도(승격 이전 버전) 새 trained_date로 갱신된다."""
        models_root = tmp_path / "models"
        write_activation_manifest(
            models_root,
            ActivationManifest(
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                trained_date=date(2026, 8, 10),
                sidecar_sha256="old",
                promoted_at="2026-08-10T00:00:00+00:00",
                promotion_basis={},
            ),
        )

        promote_activation_manifest(
            models_root,
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            merged_to_active=True,
            gate_passed=True,
            trained_date=date(2026, 8, 17),
            sidecar_sha256="new",
            promotion_basis={"path": "standing_gate"},
        )

        loaded = read_activation_manifest(models_root, "domestic", 20, "lightgbm")
        assert loaded is not None
        assert loaded.trained_date == date(2026, 8, 17)
        assert loaded.sidecar_sha256 == "new"


class TestScoringStrategyManifestIndependence:
    """REQ-ATE-050: 스코어링 전략 매니페스트는 활성화 매니페스트(조합별)와
    독립적으로 갱신 가능한 별도 파일이다."""

    def test_strategy_manifest_is_a_separate_file_from_activation_manifest(self, tmp_path: Path):
        models_root = tmp_path / "models"
        write_activation_manifest(
            models_root,
            ActivationManifest(
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                trained_date=date(2026, 8, 17),
                sidecar_sha256="x",
                promoted_at="2026-08-17T00:00:00+00:00",
                promotion_basis={},
            ),
        )
        write_strategy_manifest(
            models_root, market="domestic", horizon=20, active_strategy="ensemble"
        )

        strategy = read_strategy_manifest(models_root, "domestic", 20)
        assert strategy is not None
        assert strategy.active_strategy == "ensemble"

        # 활성화 매니페스트는 별도 파일에 손상 없이 남아 있어야 한다.
        activation = read_activation_manifest(models_root, "domestic", 20, "lightgbm")
        assert activation is not None
        assert activation.trained_date == date(2026, 8, 17)

    def test_strategy_manifest_update_does_not_require_activation_manifest_update(
        self, tmp_path: Path
    ):
        models_root = tmp_path / "models"
        write_strategy_manifest(
            models_root, market="overseas", horizon=60, active_strategy="lightgbm"
        )
        assert read_activation_manifest(models_root, "overseas", 60, "lightgbm") is None
        assert read_strategy_manifest(models_root, "overseas", 60) is not None


class TestRollbackNoFilesystemMutation:
    """REQ-ATE-053: 롤백은 매니페스트 재기록만으로 완료된다 — 어떤 모델
    파일도 이동/복사/삭제하지 않는다."""

    def test_rollback_rewrites_trained_date_only(self, tmp_path: Path):
        models_root = tmp_path / "models"
        write_activation_manifest(
            models_root,
            ActivationManifest(
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                trained_date=date(2026, 8, 17),
                sidecar_sha256="new",
                promoted_at="2026-08-17T00:00:00+00:00",
                promotion_basis={},
            ),
        )

        rolled_back = rollback_activation_manifest(
            models_root,
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            target_trained_date=date(2026, 8, 10),
            target_sidecar_sha256="old",
        )

        assert rolled_back.trained_date == date(2026, 8, 10)
        loaded = read_activation_manifest(models_root, "domestic", 20, "lightgbm")
        assert loaded is not None
        assert loaded.trained_date == date(2026, 8, 10)

    def test_rollback_invokes_no_filesystem_move_or_copy(self, tmp_path: Path):
        """파일 시스템 조작(이동/복사) 0건을 shutil 모킹으로 검증한다."""
        models_root = tmp_path / "models"
        write_activation_manifest(
            models_root,
            ActivationManifest(
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                trained_date=date(2026, 8, 17),
                sidecar_sha256="new",
                promoted_at="2026-08-17T00:00:00+00:00",
                promotion_basis={},
            ),
        )

        with (
            patch("shutil.move") as move_spy,
            patch("shutil.copy") as copy_spy,
            patch("shutil.copy2") as copy2_spy,
        ):
            rollback_activation_manifest(
                models_root,
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                target_trained_date=date(2026, 8, 10),
                target_sidecar_sha256="old",
            )

        move_spy.assert_not_called()
        copy_spy.assert_not_called()
        copy2_spy.assert_not_called()

    def test_rollback_without_existing_manifest_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="롤백할 활성화 매니페스트가 존재하지 않는다"):
            rollback_activation_manifest(
                tmp_path / "models",
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                target_trained_date=date(2026, 8, 10),
                target_sidecar_sha256="old",
            )


class TestDanglingManifestDetection:
    """REQ-ATE-054: 매니페스트가 가리키는 trained_date가 2단계 보존 정책에
    의해 아카이브로 이동되었으면 댕글링 상태로 감지한다."""

    def test_detects_dangling_when_trained_date_not_in_active_list(self):
        manifest = ActivationManifest(
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            trained_date=date(2025, 1, 1),
            sidecar_sha256="x",
            promoted_at="2025-01-01T00:00:00+00:00",
            promotion_basis={},
        )
        active_trained_dates = [date(2026, 8, 10), date(2026, 8, 3)]

        assert detect_dangling_manifest(manifest, active_trained_dates) is True

    def test_not_dangling_when_trained_date_still_active(self):
        manifest = ActivationManifest(
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            trained_date=date(2026, 8, 10),
            sidecar_sha256="x",
            promoted_at="2026-08-10T00:00:00+00:00",
            promotion_basis={},
        )
        active_trained_dates = [date(2026, 8, 10), date(2026, 8, 3)]

        assert detect_dangling_manifest(manifest, active_trained_dates) is False

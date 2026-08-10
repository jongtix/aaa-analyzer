"""src/analyzer/training/persistence.py 모델 영속화 테스트 (SPEC-ANALYZER-TRAIN-001 M6).

REQ-AT-090/091/092(네이티브 포맷 저장 + 경로 관례 + SHA-256 라운드트립)/
REQ-AT-093/094/095(2단계 보존 + tar 무결성 + 영구 미삭제)를 검증한다.
AC-AT-008/AC-AT-009의 worked example을 그대로 구현한다.
"""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pytest

from analyzer.training.persistence import (
    ModelVersion,
    apply_retention_policy,
    model_dir,
    model_filename,
    save_model_native,
    verify_model_integrity,
)


def _trained_lgbm_model() -> lgb.LGBMRegressor:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    y = x @ np.array([0.02, -0.01, 0.015]) + rng.normal(scale=0.01, size=60)
    model = lgb.LGBMRegressor(n_estimators=10, verbosity=-1)
    model.fit(x, y)
    return model


class TestModelDirAndFilename:
    """REQ-AT-091: 경로 관례 models/{market}/{horizon}/{algorithm}/, 파일명 계약."""

    def test_model_dir_convention(self, tmp_path: Path):
        result = model_dir(tmp_path, "domestic", 20, "lightgbm")

        assert result == tmp_path / "domestic" / "20" / "lightgbm"

    def test_model_filename_convention(self):
        filename = model_filename("domestic", 20, "lightgbm", date(2026, 8, 8))

        assert filename == "domestic_20_lightgbm_2026-08-08.txt"


class TestSaveModelNative:
    """AC-AT-008: 네이티브 포맷 저장 + 저장 직후 SHA-256 라운드트립 검증."""

    def test_ac_at_008_saves_as_txt_not_pickle(self, tmp_path: Path):
        model = _trained_lgbm_model()

        saved = save_model_native(model, tmp_path, "domestic", 20, "lightgbm", date(2026, 8, 8))

        assert saved.model_path.name == "domestic_20_lightgbm_2026-08-08.txt"
        assert saved.model_path.exists()
        assert saved.model_path.suffix == ".txt"

    def test_ac_at_008_sidecar_hash_matches_recomputed_hash(self, tmp_path: Path):
        model = _trained_lgbm_model()

        saved = save_model_native(model, tmp_path, "domestic", 20, "lightgbm", date(2026, 8, 8))

        assert saved.sidecar_path.exists()
        assert verify_model_integrity(saved.model_path, saved.sidecar_path)

    def test_saved_file_content_is_not_pickle(self, tmp_path: Path):
        """REQ-AT-090: pickle 등 언어/버전 종속 직렬화를 사용하지 않는다(shall not)."""
        model = _trained_lgbm_model()

        saved = save_model_native(model, tmp_path, "domestic", 20, "lightgbm", date(2026, 8, 8))

        # LightGBM 네이티브 텍스트 포맷은 사람이 읽을 수 있는 형식으로 시작한다
        # (pickle의 바이너리 매직 바이트와 다름).
        content_head = saved.model_path.read_text(encoding="utf-8")[:20]
        assert "tree" in content_head.lower() or "version" in content_head.lower()

    def test_rejects_unsupported_algorithm(self, tmp_path: Path):
        model = _trained_lgbm_model()

        with pytest.raises(ValueError, match="algorithm"):
            save_model_native(model, tmp_path, "domestic", 20, "unknown_algo", date(2026, 8, 8))


class TestVerifyModelIntegrity:
    def test_returns_false_when_file_corrupted(self, tmp_path: Path):
        model = _trained_lgbm_model()
        saved = save_model_native(model, tmp_path, "domestic", 20, "lightgbm", date(2026, 8, 8))

        saved.model_path.write_bytes(b"corrupted content")

        assert verify_model_integrity(saved.model_path, saved.sidecar_path) is False


class TestApplyRetentionPolicy:
    """AC-AT-009: 최근 12개 active 유지, 이전은 월별 아카이브, 검증 실패 시 원본 보존."""

    def _make_versions(self, tmp_path: Path, count: int) -> list[ModelVersion]:
        versions: list[ModelVersion] = []
        for i in range(count):
            model = _trained_lgbm_model()
            trained_date = date(2026, 1, 1 + i)
            saved = save_model_native(
                model, tmp_path / "models", "domestic", 20, "lightgbm", trained_date
            )
            versions.append(
                ModelVersion(
                    trained_date=trained_date,
                    model_path=saved.model_path,
                    sidecar_path=saved.sidecar_path,
                    sha256=saved.sha256,
                )
            )
        return versions

    def test_ac_at_009_recent_12_stay_active_uncompressed(self, tmp_path: Path):
        versions = self._make_versions(tmp_path, 15)

        result = apply_retention_policy(versions, tmp_path / "archive", active_count=12)

        assert len(result.active) == 12
        for v in result.active:
            assert v.model_path.exists()
        # active는 가장 최근 12개(날짜 내림차순 정렬 시 앞쪽)여야 한다.
        assert result.active == sorted(versions, key=lambda v: v.trained_date)[-12:]

    def test_ac_at_009_oldest_3_move_to_monthly_archive(self, tmp_path: Path):
        versions = self._make_versions(tmp_path, 15)

        result = apply_retention_policy(versions, tmp_path / "archive", active_count=12)

        assert len(result.archived_months) == 1
        archive_path = tmp_path / "archive" / f"{result.archived_months[0]}.tar.zst"
        assert archive_path.exists()

    def test_boundary_exactly_13_versions_yields_12_active_1_archived(self, tmp_path: Path):
        """§B 경계 사례: 정확히 13번째 버전(active 12개 + 아카이브 1개 전환점).

        `<=12`(전부 active, 아카이브 없음) vs `<13`(1개만 아카이브)의 오적용을
        구분하는 명시적 테스트 — 12개는 `<=12` 분기로 잘못 빠지면 전부 active로
        남아 이 테스트가 실패한다.
        """
        versions = self._make_versions(tmp_path, 13)

        result = apply_retention_policy(versions, tmp_path / "archive", active_count=12)

        assert len(result.active) == 12
        assert result.active == sorted(versions, key=lambda v: v.trained_date)[-12:]
        assert len(result.archived_months) == 1
        archived_version = sorted(versions, key=lambda v: v.trained_date)[0]
        assert archived_version not in result.active

    def test_ac_at_009_staged_originals_deleted_after_successful_archive(self, tmp_path: Path):
        versions = self._make_versions(tmp_path, 15)
        to_be_archived = sorted(versions, key=lambda v: v.trained_date)[:3]

        apply_retention_policy(versions, tmp_path / "archive", active_count=12)

        for v in to_be_archived:
            assert not v.model_path.exists()
            assert not v.sidecar_path.exists()

    def test_no_archival_needed_when_at_or_below_active_count(self, tmp_path: Path):
        versions = self._make_versions(tmp_path, 5)

        result = apply_retention_policy(versions, tmp_path / "archive", active_count=12)

        assert len(result.active) == 5
        assert result.archived_months == []

    def test_ac_at_009_verification_failure_preserves_staged_originals(self, tmp_path: Path):
        """REQ-AT-094: 검증 실패를 시뮬레이션하면 원본이 삭제되지 않아야 한다."""
        versions = self._make_versions(tmp_path, 15)
        to_be_archived = sorted(versions, key=lambda v: v.trained_date)[:3]

        with patch("analyzer.training.persistence._verify_archive_integrity", return_value=False):
            with pytest.raises(ValueError, match="무결성"):
                apply_retention_policy(versions, tmp_path / "archive", active_count=12)

        for v in to_be_archived:
            assert v.model_path.exists()
            assert v.sidecar_path.exists()

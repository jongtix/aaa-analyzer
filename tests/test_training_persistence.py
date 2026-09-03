"""src/analyzer/training/persistence.py 모델 영속화 테스트 (SPEC-ANALYZER-TRAIN-001 M6).

REQ-AT-090/091/092(네이티브 포맷 저장 + 경로 관례 + SHA-256 라운드트립)/
REQ-AT-093/094/095(2단계 보존 + tar 무결성 + 영구 미삭제)를 검증한다.
AC-AT-008/AC-AT-009의 worked example을 그대로 구현한다.
"""

import hashlib
import inspect
from datetime import date
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pytest

from analyzer.training.persistence import (
    MONTHLY_ACTIVE_COUNT,
    ModelVersion,
    apply_retention_for_combos,
    apply_retention_policy,
    combo_archive_root,
    enumerate_model_versions,
    model_dir,
    model_filename,
    quantile_model_filename,
    save_model_native,
    save_quantile_model,
    verify_model_integrity,
)


def _write_version(
    models_root: Path,
    market: str,
    horizon: int,
    algorithm: str,
    trained_date: date,
    payload: bytes,
) -> tuple[Path, Path, str]:
    """실제 학습 없이 파일명 관례에 맞는 모델 파일 + 사이드카 쌍을 기록한다."""
    target_dir = model_dir(models_root, market, horizon, algorithm)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_filename(market, horizon, algorithm, trained_date)
    model_path.write_bytes(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    sidecar_path = model_path.with_suffix(model_path.suffix + ".sha256")
    sidecar_path.write_text(sha256, encoding="utf-8")
    return model_path, sidecar_path, sha256


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


class TestEnumerateModelVersions:
    """AC-ATT-017(REQ-ATT-016): `model_dir()` 스캔으로 네이티브 모델 파일 +
    `.sha256` 사이드카 쌍을 `ModelVersion` 시퀀스로 구성한다."""

    _write_version = staticmethod(_write_version)

    def test_ac_att_017_returns_five_versions_sorted_ascending_with_sidecar_hashes(
        self, tmp_path: Path
    ):
        expected: dict[date, str] = {}
        for day in (12, 5, 26, 19, 1):
            trained_date = date(2026, 8, day)
            _, _, sha256 = self._write_version(
                tmp_path, "domestic", 20, "xgboost", trained_date, f"model-{day}".encode()
            )
            expected[trained_date] = sha256

        versions = enumerate_model_versions(tmp_path, "domestic", 20, "xgboost")

        assert len(versions) == 5
        assert [v.trained_date for v in versions] == sorted(expected)
        for version in versions:
            assert version.sha256 == expected[version.trained_date]
            assert version.sha256 == version.sidecar_path.read_text(encoding="utf-8").strip()
            assert version.model_path.exists()

    def test_returns_empty_list_when_directory_absent(self, tmp_path: Path):
        assert enumerate_model_versions(tmp_path, "overseas", 5, "lightgbm") == []

    def test_skips_model_file_without_sidecar(self, tmp_path: Path):
        self._write_version(tmp_path, "domestic", 20, "xgboost", date(2026, 8, 1), b"paired")
        _, orphan_sidecar, _ = self._write_version(
            tmp_path, "domestic", 20, "xgboost", date(2026, 8, 8), b"orphan"
        )
        orphan_sidecar.unlink()

        versions = enumerate_model_versions(tmp_path, "domestic", 20, "xgboost")

        assert [v.trained_date for v in versions] == [date(2026, 8, 1)]

    def test_ignores_other_combinations_in_the_same_root(self, tmp_path: Path):
        self._write_version(tmp_path, "domestic", 20, "xgboost", date(2026, 8, 1), b"a")
        self._write_version(tmp_path, "domestic", 20, "lightgbm", date(2026, 8, 1), b"b")
        self._write_version(tmp_path, "overseas", 20, "xgboost", date(2026, 8, 1), b"c")

        versions = enumerate_model_versions(tmp_path, "domestic", 20, "xgboost")

        assert len(versions) == 1
        assert versions[0].model_path.name == "domestic_20_xgboost_2026-08-01.json"

    def test_rejects_unsupported_algorithm(self, tmp_path: Path):
        with pytest.raises(ValueError, match="지원하지 않는 algorithm"):
            enumerate_model_versions(tmp_path, "domestic", 20, "catboost")


class TestApplyRetentionForCombos:
    """AC-ATT-018(REQ-ATT-017): 월간 성공 후처리가 호출할 보존 정책 배선 지점.

    `enumerate_model_versions()`(M1) 결과를 `apply_retention_policy()`에 넘기는
    조합별 호출 지점으로, `active_count=36`을 명시적으로 전달한다. 함수 자체의
    기본값 12는 무수정(PRESERVE)이며 여기서 재정의하지 않는다.
    """

    @staticmethod
    def _write_n_versions(
        models_root: Path,
        market: str,
        horizon: int,
        algorithm: str,
        count: int,
        start: date = date(2026, 1, 1),
    ) -> list[date]:
        dates = [date.fromordinal(start.toordinal() + i) for i in range(count)]
        for i, trained_date in enumerate(dates):
            _write_version(
                models_root, market, horizon, algorithm, trained_date, f"payload-{i}".encode()
            )
        return dates

    def test_ac_att_018_forty_versions_leave_exactly_36_active_and_archive_four(
        self, tmp_path: Path
    ):
        combo = ("domestic", 20, "xgboost")
        dates = self._write_n_versions(tmp_path, *combo, count=40)

        results = apply_retention_for_combos(tmp_path, [combo])

        result = results[combo]
        assert len(result.active) == 36
        assert [v.trained_date for v in result.active] == dates[4:]
        # 오래된 4개는 아카이브로 이동하고 active 경로에서 사라진다.
        assert enumerate_model_versions(tmp_path, *combo) == result.active
        assert result.archived_months == ["2026-01"]
        assert (combo_archive_root(tmp_path, *combo) / "2026-01.tar.zst").exists()

    def test_passes_active_count_36_explicitly_to_apply_retention_policy(self, tmp_path: Path):
        combo = ("domestic", 20, "xgboost")
        self._write_n_versions(tmp_path, *combo, count=3)

        with patch(
            "analyzer.training.persistence.apply_retention_policy",
            wraps=apply_retention_policy,
        ) as spy:
            apply_retention_for_combos(tmp_path, [combo])

        assert spy.call_count == 1
        assert spy.call_args.kwargs["active_count"] == 36
        assert MONTHLY_ACTIVE_COUNT == 36

    def test_apply_retention_policy_own_default_active_count_stays_12(self):
        """PRESERVE: 호출 지점이 36을 넘기더라도 함수 자체 기본값은 12로 유지된다."""
        assert inspect.signature(apply_retention_policy).parameters["active_count"].default == 12

    def test_archive_root_follows_models_root_archive_combo_convention(self, tmp_path: Path):
        assert combo_archive_root(tmp_path, "domestic", 20, "xgboost") == (
            tmp_path / "archive" / "domestic" / "20" / "xgboost"
        )

    def test_same_month_across_combos_does_not_overwrite_each_other(self, tmp_path: Path):
        first = ("domestic", 20, "xgboost")
        second = ("overseas", 20, "xgboost")
        self._write_n_versions(tmp_path, *first, count=38)
        self._write_n_versions(tmp_path, *second, count=38)

        results = apply_retention_for_combos(tmp_path, [first, second])

        assert results[first].archived_months == ["2026-01"]
        assert results[second].archived_months == ["2026-01"]
        assert (combo_archive_root(tmp_path, *first) / "2026-01.tar.zst").exists()
        assert (combo_archive_root(tmp_path, *second) / "2026-01.tar.zst").exists()
        assert combo_archive_root(tmp_path, *first) != combo_archive_root(tmp_path, *second)

    def test_combo_without_versions_yields_empty_result(self, tmp_path: Path):
        combo = ("overseas", 5, "lightgbm")

        results = apply_retention_for_combos(tmp_path, [combo])

        assert results[combo].active == []
        assert results[combo].archived_months == []

    def test_integrity_failure_propagates_and_preserves_originals(self, tmp_path: Path):
        combo = ("domestic", 20, "xgboost")
        self._write_n_versions(tmp_path, *combo, count=40)
        before = enumerate_model_versions(tmp_path, *combo)

        with patch("analyzer.training.persistence._verify_archive_integrity", return_value=False):
            with pytest.raises(ValueError, match="무결성"):
                apply_retention_for_combos(tmp_path, [combo])

        assert enumerate_model_versions(tmp_path, *combo) == before


class TestQuantileModelFilenameAndSave:
    """`train.py`의 사설(private) 헬퍼에서 이 모듈로 이전된 공유 헬퍼
    (`quantile_model_filename()`/`save_quantile_model()`) 회귀 안전성 —
    이전 전 `train.py` 사설 버전과 파일명/저장 동작이 byte-identical해야
    한다(주간 정기 재학습 경로 동작 불변, `test_training_train.py`
    `TestQuantileModelFilenameCollision`이 이미 검증하는 파일명 관례와
    동일한 기댓값을 여기서도 독립적으로 재확인한다)."""

    def test_filename_matches_point_model_stem_plus_alpha_tag(self):
        trained_date = date(2026, 8, 17)
        assert (
            quantile_model_filename("domestic", 20, 0.10, trained_date)
            == "domestic_20_lightgbm_2026-08-17_q10.txt"
        )
        assert (
            quantile_model_filename("domestic", 20, 0.90, trained_date)
            == "domestic_20_lightgbm_2026-08-17_q90.txt"
        )

    def test_save_quantile_model_round_trips_and_shares_dir_with_point_model(self, tmp_path: Path):
        market, horizon = "overseas", 60
        trained_date = date(2026, 8, 19)
        model = _trained_lgbm_model()

        saved = save_quantile_model(model, tmp_path, market, horizon, 0.10, trained_date)

        assert saved.model_path.exists()
        assert saved.sidecar_path.exists()
        assert verify_model_integrity(saved.model_path, saved.sidecar_path)
        assert saved.model_path.parent == model_dir(tmp_path, market, horizon, "lightgbm")
        assert saved.model_path.name == "overseas_60_lightgbm_2026-08-19_q10.txt"
        # 스테이징 임시 디렉토리가 최종 트리에 잔존하지 않아야 한다.
        assert set((tmp_path / market / str(horizon) / "lightgbm").iterdir()) == {
            saved.model_path,
            saved.sidecar_path,
        }

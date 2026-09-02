"""운영자 호출용 롤백 CLI 테스트 (SPEC-ANALYZER-TRAIN-TUNING-001 M1).

REQ-ATT-018(열거 → 대상 조회 → `rollback_activation_manifest()` 소환)/
REQ-ATT-019(대상 불일치 시 후보 목록 포함 실패)/REQ-ATT-020(`--confirm` 게이트)/
REQ-ATT-024(조회-실행 사이 동시쓰기 경쟁 감지)를 검증한다.
AC-ATT-019(폐쇄 게이트)/020/021/024를 그대로 구현한다.
"""

import hashlib
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from analyzer.orchestration import activation
from analyzer.orchestration.activation import (
    ActivationManifest,
    activation_manifest_path,
    promote_activation_manifest,
    read_activation_manifest,
)
from analyzer.training import rollback
from analyzer.training.persistence import model_dir, model_filename

MARKET = "domestic"
HORIZON = 20
ALGORITHM = "xgboost"
D1 = date(2026, 8, 1)
D2 = date(2026, 8, 8)
D3 = date(2026, 8, 15)


def _write_version(models_root: Path, trained_date: date) -> str:
    """모델 파일 + `.sha256` 사이드카 쌍을 기록하고 사이드카 해시를 돌려준다."""
    target_dir = model_dir(models_root, MARKET, HORIZON, ALGORITHM)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_filename(MARKET, HORIZON, ALGORITHM, trained_date)
    payload = f"model-{trained_date.isoformat()}".encode()
    model_path.write_bytes(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    model_path.with_suffix(model_path.suffix + ".sha256").write_text(sha256, encoding="utf-8")
    return sha256


def _read(models_root: Path) -> ActivationManifest:
    """현재 활성화 매니페스트 — 존재하지 않으면 테스트 실패."""
    manifest = read_activation_manifest(models_root, MARKET, HORIZON, ALGORITHM)
    assert manifest is not None
    return manifest


def _promote(models_root: Path, trained_date: date, sha256: str) -> None:
    """실제 `promote_activation_manifest()`를 호출한다(모킹 없음)."""
    promoted = promote_activation_manifest(
        models_root,
        market=MARKET,
        horizon=HORIZON,
        algorithm=ALGORITHM,
        merged_to_active=True,
        gate_passed=True,
        trained_date=trained_date,
        sidecar_sha256=sha256,
        promotion_basis={"gate1_rolling_mean_rank_ic": 0.03},
    )
    assert promoted is not None


def _argv(models_root: Path, target: date, *, confirm: bool) -> list[str]:
    argv = [
        "--models-root",
        str(models_root),
        "--market",
        MARKET,
        "--horizon",
        str(HORIZON),
        "--algorithm",
        ALGORITHM,
        "--target-trained-date",
        target.isoformat(),
    ]
    if confirm:
        argv.append("--confirm")
    return argv


class TestRollbackClosureGate:
    """AC-ATT-019(REQ-ATT-018, 폐쇄 게이트 — 필수): 2회 승격 후 최초 버전으로
    롤백이 실제로 동작함을 실제 호출 경로로 검증한다(모킹 대체 불가)."""

    def test_ac_att_019_double_promotion_then_rollback_to_first_version(self, tmp_path: Path):
        models_root = tmp_path / "models"
        sha_d1 = _write_version(models_root, D1)
        sha_d2 = _write_version(models_root, D2)

        _promote(models_root, D1, sha_d1)
        _promote(models_root, D2, sha_d2)
        assert _read(models_root).trained_date == D2

        exit_code = rollback.main(_argv(models_root, D1, confirm=True))

        assert exit_code == 0
        manifest = _read(models_root)
        assert manifest.trained_date == D1
        assert manifest.sidecar_sha256 == sha_d1
        assert manifest.promotion_basis["rollback_from_trained_date"] == D2.isoformat()


class TestRollbackTargetNotFound:
    """AC-ATT-020(REQ-ATT-019): 열거된 active 버전 중 어느 것과도 일치하지
    않으면 0이 아닌 종료코드 + 후보 목록, 매니페스트는 불변."""

    def test_ac_att_020_unknown_target_fails_with_candidate_list(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        models_root = tmp_path / "models"
        sha_d1 = _write_version(models_root, D1)
        _write_version(models_root, D2)
        _promote(models_root, D1, sha_d1)
        manifest_path = activation_manifest_path(models_root, MARKET, HORIZON, ALGORITHM)
        mtime_before = manifest_path.stat().st_mtime_ns

        exit_code = rollback.main(_argv(models_root, date(2026, 7, 1), confirm=True))

        assert exit_code != 0
        stderr = capsys.readouterr().err
        assert D1.isoformat() in stderr
        assert D2.isoformat() in stderr
        assert manifest_path.stat().st_mtime_ns == mtime_before
        assert _read(models_root).trained_date == D1

    def test_missing_manifest_fails_without_raising(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        models_root = tmp_path / "models"
        _write_version(models_root, D1)

        exit_code = rollback.main(_argv(models_root, D1, confirm=True))

        assert exit_code != 0
        assert "매니페스트" in capsys.readouterr().err


class TestRollbackAlgorithmChoicesValidation:
    """review finding W3: `--algorithm`에 오타(존재하지 않는 값)를 주면
    argparse 자체가 `choices` 제약으로 즉시 거부해야 한다 —
    `enumerate_model_versions()`가 던지는 raw `ValueError` 트레이스백이
    운영자에게 노출되지 않는다."""

    def test_invalid_algorithm_exits_via_argparse_not_raw_valueerror(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        models_root = tmp_path / "models"
        argv = [
            "--models-root",
            str(models_root),
            "--market",
            MARKET,
            "--horizon",
            str(HORIZON),
            "--algorithm",
            "xgboot",
            "--target-trained-date",
            D1.isoformat(),
            "--confirm",
        ]

        with pytest.raises(SystemExit) as exc_info:
            rollback.main(argv)

        assert exc_info.value.code != 0
        stderr = capsys.readouterr().err
        assert "xgboot" in stderr


class TestRollbackConfirmGate:
    """AC-ATT-021(REQ-ATT-020): `--confirm` 미지정 시 대기 중인 롤백 내용만
    출력하고 매니페스트를 변경하지 않는다."""

    def test_ac_att_021_dry_run_prints_pending_rollback_and_leaves_manifest_untouched(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        models_root = tmp_path / "models"
        sha_d1 = _write_version(models_root, D1)
        sha_d2 = _write_version(models_root, D2)
        _promote(models_root, D1, sha_d1)
        _promote(models_root, D2, sha_d2)
        manifest_path = activation_manifest_path(models_root, MARKET, HORIZON, ALGORITHM)
        mtime_before = manifest_path.stat().st_mtime_ns

        with patch.object(
            rollback, "rollback_activation_manifest", wraps=activation.rollback_activation_manifest
        ) as spy:
            exit_code = rollback.main(_argv(models_root, D1, confirm=False))

        assert exit_code == 0
        assert spy.call_count == 0
        stdout = capsys.readouterr().out
        assert D2.isoformat() in stdout
        assert D1.isoformat() in stdout
        assert manifest_path.stat().st_mtime_ns == mtime_before
        assert _read(models_root).trained_date == D2


class TestRollbackOptimisticConcurrency:
    """AC-ATT-024(REQ-ATT-024): 대상 조회 시점과 `--confirm` 실행 시점 사이에
    자동 프로모션이 끼어들면 롤백을 중단하고 명시적으로 실패한다."""

    def test_ac_att_024_concurrent_promotion_between_lookup_and_confirm_aborts_rollback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        models_root = tmp_path / "models"
        sha_d1 = _write_version(models_root, D1)
        sha_d2 = _write_version(models_root, D2)
        sha_d3 = _write_version(models_root, D3)
        _promote(models_root, D1, sha_d1)
        _promote(models_root, D2, sha_d2)
        baseline = _read(models_root)

        calls: list[int] = []

        def _read_then_race(*args, **kwargs):
            result = activation.read_activation_manifest(*args, **kwargs)
            calls.append(1)
            if len(calls) == 1:
                # 대상 조회(기준선 관측) 직후 월간 자동 프로모션이 끼어든다.
                _promote(models_root, D3, sha_d3)
            return result

        with (
            patch.object(rollback, "read_activation_manifest", side_effect=_read_then_race),
            patch.object(
                rollback,
                "rollback_activation_manifest",
                wraps=activation.rollback_activation_manifest,
            ) as rollback_spy,
        ):
            exit_code = rollback.main(_argv(models_root, D1, confirm=True))

        assert exit_code != 0
        assert rollback_spy.call_count == 0

        stderr = capsys.readouterr().err
        assert baseline.trained_date.isoformat() in stderr
        assert baseline.promoted_at in stderr
        assert D3.isoformat() in stderr

        current = _read(models_root)
        assert current.trained_date == D3
        assert current.sidecar_sha256 == sha_d3
        assert current.promoted_at in stderr
        assert "rollback_from_trained_date" not in current.promotion_basis

    def test_unchanged_manifest_between_lookup_and_confirm_proceeds(self, tmp_path: Path):
        models_root = tmp_path / "models"
        sha_d1 = _write_version(models_root, D1)
        sha_d2 = _write_version(models_root, D2)
        _promote(models_root, D1, sha_d1)
        _promote(models_root, D2, sha_d2)

        real_rollback = Mock(wraps=activation.rollback_activation_manifest)
        with patch.object(rollback, "rollback_activation_manifest", real_rollback):
            exit_code = rollback.main(_argv(models_root, D1, confirm=True))

        assert exit_code == 0
        assert real_rollback.call_count == 1
        assert real_rollback.call_args.kwargs["target_sidecar_sha256"] == sha_d1
        assert _read(models_root).trained_date == D1

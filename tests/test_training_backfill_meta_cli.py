"""운영 백필 CLI 테스트 (SPEC-ANALYZER-TRAIN-META-001 M6, REQ-TM-009).

실 DB/네트워크 없이 로컬 임시 디렉터리의 가짜 모델 파일만으로 dry-run/apply/
idempotency/모델 바이너리 무변경을 검증한다. `resolve_feature_columns()`가
실제로 소비할 수 있는 사이드카 스키마(축소 스키마, M1)를 그대로 검증한다.
"""

import json
from pathlib import Path

import pytest

from analyzer.features.classification import FEATURE_REGISTRY, FeatureClass, classify_feature
from analyzer.training import backfill_meta_cli
from analyzer.training.backfill_meta_cli import (
    ParsedModelFilename,
    apply_backfill,
    main,
    parse_model_filename,
    plan_backfill,
    resolve_backfill_feature_columns,
)
from analyzer.training.campaign_metrics import sidecar_path_for


def _make_model_file(tmp_path: Path, name: str, content: bytes = b"fake-model-bytes") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# resolve_backfill_feature_columns() — 안전성 불변식
# ---------------------------------------------------------------------------


def test_resolve_backfill_feature_columns_overseas_excludes_all_frozen_columns() -> None:
    """overseas feature_columns는 FROZEN(수급) 컬럼을 하나도 포함하지
    않는다 — investor_trend 구조적 결측(spec.md §0)."""
    columns = resolve_backfill_feature_columns("overseas")

    assert columns  # 비어 있지 않음(PRICE_DERIVED 25개는 항상 존재)
    for column in columns:
        assert classify_feature(column) == FeatureClass.PRICE_DERIVED
    assert all(c in FEATURE_REGISTRY for c in columns)


def test_resolve_backfill_feature_columns_overseas_matches_technical_feature_order() -> None:
    """컬럼 순서는 `compute_technical_features()`의 실제 생성 순서(윈도 우선)를
    따라야 한다 — `FEATURE_REGISTRY`의 딕셔너리 삽입 순서(통계량 우선)와
    달라야 정상이다. 순서가 틀리면 예측 시 위치 기반 매핑이 어긋난다."""
    columns = resolve_backfill_feature_columns("overseas")

    # compute_technical_features()가 만드는 실제 순서: KBAR 5개 →
    # 윈도(5,10,20,60)마다 ROC/MA/STD/RANK/CORR.
    expected_prefix = ["KMID", "KLEN", "KUP", "KLOW", "KSFT"]
    assert columns[:5] == expected_prefix
    assert columns[5:10] == ["ROC_5", "MA_5", "STD_5", "RANK_5", "CORR_5"]
    assert columns[10:15] == ["ROC_10", "MA_10", "STD_10", "RANK_10", "CORR_10"]
    assert len(columns) == 25  # 5 KBAR + 4 windows * 5 stats


def test_resolve_backfill_feature_columns_rejects_unsupported_market() -> None:
    """domestic 등 investor_trend 실데이터가 존재할 수 있는 market은
    합성 프레임만으로 안전하게 재현할 수 없으므로 명시적으로 거부한다."""
    with pytest.raises(NotImplementedError, match="overseas"):
        resolve_backfill_feature_columns("domestic")


# ---------------------------------------------------------------------------
# parse_model_filename()
# ---------------------------------------------------------------------------


def test_parse_model_filename_point_model() -> None:
    parsed = parse_model_filename(Path("overseas_60_xgboost_2026-09-05.json"))

    assert parsed == ParsedModelFilename(
        market="overseas",
        horizon=60,
        algorithm="xgboost",
        trained_date="2026-09-05",
        is_quantile=False,
    )


def test_parse_model_filename_quantile_model() -> None:
    parsed = parse_model_filename(Path("overseas_20_lightgbm_2026-09-05_q10.txt"))

    assert parsed == ParsedModelFilename(
        market="overseas",
        horizon=20,
        algorithm="lightgbm",
        trained_date="2026-09-05",
        is_quantile=True,
    )


def test_parse_model_filename_rejects_unknown_pattern() -> None:
    with pytest.raises(ValueError, match="일치하지 않는다"):
        parse_model_filename(Path("not-a-model-file.bin"))


# ---------------------------------------------------------------------------
# plan_backfill() / apply_backfill() — dry-run vs apply vs idempotency
# ---------------------------------------------------------------------------


def test_plan_backfill_missing_model_file(tmp_path: Path) -> None:
    plan = plan_backfill(tmp_path / "overseas_60_xgboost_2026-09-05.json")

    assert plan.action == "skip_model_missing"


def test_plan_backfill_writes_when_sidecar_absent(tmp_path: Path) -> None:
    model_path = _make_model_file(tmp_path, "overseas_60_xgboost_2026-09-05.json")

    plan = plan_backfill(model_path)

    assert plan.action == "write"
    assert plan.market == "overseas"
    assert plan.horizon == 60
    assert plan.algorithm == "xgboost"
    assert len(plan.feature_columns) == 25


def test_plan_backfill_skips_when_sidecar_already_exists(tmp_path: Path) -> None:
    model_path = _make_model_file(tmp_path, "overseas_60_xgboost_2026-09-05.json")
    sidecar_path_for(model_path).write_text("{}", encoding="utf-8")

    plan = plan_backfill(model_path)

    assert plan.action == "skip_sidecar_exists"


def test_apply_backfill_writes_reduced_schema_sidecar(tmp_path: Path) -> None:
    """--apply 경로: `feature_columns`를 포함하고 `frozen_hyperparameters`는
    생략된 축소 스키마(M1)가 기록된다."""
    model_path = _make_model_file(tmp_path, "overseas_60_xgboost_2026-09-05.json")
    plan = plan_backfill(model_path)

    written = apply_backfill(plan)

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["market"] == "overseas"
    assert payload["horizon"] == 60
    assert payload["algorithm"] == "xgboost"
    assert payload["feature_columns"] == list(plan.feature_columns)
    assert "frozen_hyperparameters" not in payload
    assert "aggregate_metrics" not in payload
    assert "final_fold_train_row_count" not in payload


def test_apply_backfill_rejects_non_write_plan(tmp_path: Path) -> None:
    plan = plan_backfill(tmp_path / "missing.json")

    with pytest.raises(ValueError, match="action='write'"):
        apply_backfill(plan)


def test_apply_backfill_never_touches_model_binary(tmp_path: Path) -> None:
    """모델 바이너리 파일의 mtime·내용이 사이드카 기록 전후로 완전히
    동일함을 확인한다 — 이 스크립트는 모델 파일을 절대 열거나 수정하지
    않는다."""
    model_path = _make_model_file(tmp_path, "overseas_60_xgboost_2026-09-05.json")
    original_bytes = model_path.read_bytes()
    original_mtime_ns = model_path.stat().st_mtime_ns

    plan = plan_backfill(model_path)
    apply_backfill(plan)

    assert model_path.read_bytes() == original_bytes
    assert model_path.stat().st_mtime_ns == original_mtime_ns


# ---------------------------------------------------------------------------
# main() CLI 진입점 — dry-run vs --apply vs idempotency(2회 실행)
# ---------------------------------------------------------------------------


def test_main_dry_run_by_default_does_not_write_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = _make_model_file(tmp_path, "overseas_60_xgboost_2026-09-05.json")

    exit_code = main([str(model_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[DRY-RUN]" in captured.out
    assert not sidecar_path_for(model_path).exists()


def test_main_apply_writes_sidecar(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_path = _make_model_file(tmp_path, "overseas_60_xgboost_2026-09-05.json")

    exit_code = main([str(model_path), "--apply"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[APPLIED]" in captured.out
    assert sidecar_path_for(model_path).exists()


def test_main_apply_twice_is_idempotent_second_run_skips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--apply로 두 번 실행 — 두 번째 실행은 스킵하며 기존 사이드카를
    덮어쓰지 않는다(내용·mtime 불변)."""
    model_path = _make_model_file(tmp_path, "overseas_60_xgboost_2026-09-05.json")

    first_exit_code = main([str(model_path), "--apply"])
    sidecar_path = sidecar_path_for(model_path)
    first_content = sidecar_path.read_text(encoding="utf-8")
    first_mtime_ns = sidecar_path.stat().st_mtime_ns

    capsys.readouterr()  # 첫 실행 출력 비우기
    second_exit_code = main([str(model_path), "--apply"])
    captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert "[SKIP]" in captured.out
    assert sidecar_path.read_text(encoding="utf-8") == first_content
    assert sidecar_path.stat().st_mtime_ns == first_mtime_ns


def test_main_reports_error_and_nonzero_exit_for_missing_model_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_path = tmp_path / "overseas_60_xgboost_2026-09-05.json"

    exit_code = main([str(missing_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[ERROR]" in captured.err


def test_main_reports_error_and_nonzero_exit_for_unparseable_filename(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_path = _make_model_file(tmp_path, "not-a-model-file.bin")

    exit_code = main([str(bad_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[ERROR]" in captured.err
    assert not sidecar_path_for(bad_path).exists()


def test_main_reports_error_and_nonzero_exit_for_unsupported_market(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """domestic 등 investor_trend 실데이터가 존재할 수 있는 market은
    main()을 통해서도 명시적으로 거부되고, 아무것도 기록하지 않는다."""
    domestic_path = _make_model_file(tmp_path, "domestic_60_xgboost_2026-09-05.json")

    exit_code = main([str(domestic_path), "--apply"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[ERROR]" in captured.err
    assert not sidecar_path_for(domestic_path).exists()


def test_main_handles_both_quantile_files_for_d20(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-TM-009 실제 대상 3개 파일 중 D20 분위수 쌍(q10/q90)을
    한 번의 호출로 처리할 수 있음을 확인한다."""
    q10_path = _make_model_file(tmp_path, "overseas_20_lightgbm_2026-09-05_q10.txt")
    q90_path = _make_model_file(tmp_path, "overseas_20_lightgbm_2026-09-05_q90.txt")

    exit_code = main([str(q10_path), str(q90_path), "--apply"])

    assert exit_code == 0
    assert sidecar_path_for(q10_path).exists()
    assert sidecar_path_for(q90_path).exists()
    q10_payload = json.loads(sidecar_path_for(q10_path).read_text(encoding="utf-8"))
    assert q10_payload["algorithm"] == "lightgbm"
    assert q10_payload["horizon"] == 20


def test_main_module_entrypoint_is_importable() -> None:
    """`python -m analyzer.training.backfill_meta_cli`로 실행 가능하도록
    모듈이 `main`을 노출하는지 확인한다."""
    assert callable(backfill_meta_cli.main)

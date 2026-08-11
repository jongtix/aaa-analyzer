"""모델 정체(staleness) 감지 테스트 (REQ-ATA-072/083, AC-ATA-007).

활성 모델 파일명 관례 `{market}_{horizon}_{algorithm}_{trained_date}`(TRAIN-001이
확립, `training/persistence.py` `model_filename()`)에서 `trained_date`를 파싱해
정체를 감지한다 — `persistence.py`는 건드리지 않고(이 SPEC의 PRESERVE 대상 밖) 이
모듈이 독립적으로 파일명을 파싱한다.
"""

from datetime import date
from pathlib import Path

from analyzer.orchestration.staleness import detect_stale_models


def _touch_model_file(
    models_root: Path,
    market: str,
    horizon: int,
    algorithm: str,
    trained_date: date,
    ext: str = "txt",
) -> Path:
    target_dir = models_root / market / str(horizon) / algorithm
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{market}_{horizon}_{algorithm}_{trained_date.isoformat()}.{ext}"
    path = target_dir / filename
    path.write_text("dummy-model-content")
    return path


class TestDetectStaleModels:
    """REQ-ATA-072: 마지막 성공 재학습 후 4주(기본 28일) 초과 시 정체로 표시한다."""

    def test_marks_stale_when_older_than_threshold(self, tmp_path: Path):
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", date(2026, 1, 1))

        results = detect_stale_models(tmp_path, threshold_days=28, as_of=date(2026, 2, 15))

        assert len(results) == 1
        assert results[0].market == "domestic"
        assert results[0].horizon == 5
        assert results[0].algorithm == "lightgbm"
        assert results[0].most_recent_trained_date == date(2026, 1, 1)
        assert results[0].is_stale is True

    def test_marks_fresh_when_within_threshold(self, tmp_path: Path):
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", date(2026, 2, 1))

        results = detect_stale_models(tmp_path, threshold_days=28, as_of=date(2026, 2, 15))

        assert len(results) == 1
        assert results[0].is_stale is False

    def test_boundary_exactly_at_threshold_is_not_stale(self, tmp_path: Path):
        """경계값 — 정확히 threshold_days 경과는 초과(exceed)가 아니므로 정체가 아니다."""
        trained = date(2026, 1, 1)
        as_of = date(2026, 1, 29)  # trained + 28일
        assert (as_of - trained).days == 28
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", trained)

        results = detect_stale_models(tmp_path, threshold_days=28, as_of=as_of)

        assert results[0].is_stale is False

    def test_boundary_one_day_past_threshold_is_stale(self, tmp_path: Path):
        trained = date(2026, 1, 1)
        as_of = date(2026, 1, 30)  # trained + 29일
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", trained)

        results = detect_stale_models(tmp_path, threshold_days=28, as_of=as_of)

        assert results[0].is_stale is True

    def test_uses_most_recent_trained_date_among_multiple_versions(self, tmp_path: Path):
        """동일 (market, horizon, algorithm) 조합에 여러 버전이 있으면 가장 최근
        trained_date를 기준으로 정체를 판정한다(active 최대 12개 보존 정책과 무관)."""
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", date(2026, 1, 1))
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", date(2026, 2, 1))

        results = detect_stale_models(tmp_path, threshold_days=28, as_of=date(2026, 2, 15))

        assert len(results) == 1
        assert results[0].most_recent_trained_date == date(2026, 2, 1)
        assert results[0].is_stale is False

    def test_evaluates_each_market_horizon_algorithm_combo_independently(self, tmp_path: Path):
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", date(2026, 1, 1))
        _touch_model_file(tmp_path, "overseas", 20, "xgboost", date(2026, 2, 10), ext="json")

        results = detect_stale_models(tmp_path, threshold_days=28, as_of=date(2026, 2, 15))

        by_combo = {(r.market, r.horizon, r.algorithm): r for r in results}
        assert by_combo[("domestic", 5, "lightgbm")].is_stale is True
        assert by_combo[("overseas", 20, "xgboost")].is_stale is False

    def test_ignores_non_model_files(self, tmp_path: Path):
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", date(2026, 1, 1))
        (tmp_path / "README.md").write_text("not a model")
        sidecar_dir = tmp_path / "domestic" / "5" / "lightgbm"
        (sidecar_dir / "domestic_5_lightgbm_2026-01-01.txt.sha256").write_text("deadbeef")

        results = detect_stale_models(tmp_path, threshold_days=28, as_of=date(2026, 2, 15))

        assert len(results) == 1

    def test_returns_empty_list_for_empty_models_root(self, tmp_path: Path):
        results = detect_stale_models(tmp_path, threshold_days=28, as_of=date(2026, 2, 15))

        assert results == []

    def test_defaults_to_today_when_as_of_not_provided(self, tmp_path: Path):
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", date.today())

        results = detect_stale_models(tmp_path, threshold_days=28)

        assert len(results) == 1
        assert results[0].is_stale is False

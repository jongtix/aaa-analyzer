"""모델 정체(staleness) 감지 테스트 (REQ-ATA-072/083, AC-ATA-007).

활성 모델 파일명 관례 `{market}_{horizon}_{algorithm}_{trained_date}`(TRAIN-001이
확립, `training/persistence.py` `model_filename()`)에서 `trained_date`를 파싱해
정체를 감지한다 — `persistence.py`는 건드리지 않고(이 SPEC의 PRESERVE 대상 밖) 이
모듈이 독립적으로 파일명을 파싱한다.
"""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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

    def test_returns_empty_list_when_models_root_does_not_exist(self, tmp_path: Path):
        """models_root 부재 시나리오(§B 리스크 3) — Python 3.14 pathlib.Path.rglob()는
        존재하지 않는 디렉터리에서 예외를 던지지 않고 빈 이터레이터를 반환한다(실측
        확인됨). detect_stale_models()는 이 동작을 그대로 전파하며 별도 방어 코드를
        추가하지 않는다 — 도메인 함수는 예외를 삼키지 않는 순수 함수로 유지한다(plan.md
        §B 리스크 3, §D 제약)."""
        missing_root = tmp_path / "does-not-exist"
        assert not missing_root.exists()

        results = detect_stale_models(missing_root, threshold_days=28, as_of=date(2026, 2, 15))

        assert results == []

    def test_propagates_permission_error_from_rglob(self, tmp_path: Path, monkeypatch):
        """권한 거부 시나리오(§B 리스크 3) — 샌드박스 환경에서는 `chmod 000`이 실제
        권한 거부를 재현하지 못해(실측: root/샌드박스 우회 확인) `Path.rglob()`을
        직접 모킹해 PermissionError를 재현한다. `detect_stale_models()`는 예외를
        삼키지 않고 그대로 전파하는 순수 함수로 유지한다(plan.md §B 리스크 3 —
        예외 처리·기록 책임은 콜백(api/main.py)에 둔다, 이 마일스톤 범위 밖)."""

        def _raise_permission_error(self, pattern):  # noqa: ARG001 - Path.rglob 시그니처 모방
            raise PermissionError("permission denied")

        monkeypatch.setattr(Path, "rglob", _raise_permission_error)

        try:
            detect_stale_models(tmp_path, threshold_days=28, as_of=date(2026, 2, 15))
        except PermissionError:
            pass
        else:
            raise AssertionError("detect_stale_models()가 PermissionError를 삼켰다 — 전파돼야 한다")

    def test_excludes_multi_dot_sidecar_meta_json_files(self, tmp_path: Path):
        """REQ-ATD-008: `.meta.json`처럼 확장자 앞에 점이 하나 더 있는 사이드카
        파일은 배제한다 — `\\w+`는 `.`를 포함하지 않으므로 `trained_date` 뒤
        `\\.` 앵커 이후 남은 문자열이 `meta.json`(내부에 `.` 포함)이면 전체 패턴이
        `$`까지 매칭되지 않아 이미 배제된다(실측 확인). allowlist(`txt|json`)
        강화 후에도 동일하게 배제됨을 회귀 고정한다."""
        target_dir = tmp_path / "domestic" / "5" / "lightgbm"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "domestic_5_lightgbm_2026-01-01.meta.json").write_text("{}")

        results = detect_stale_models(tmp_path, threshold_days=28, as_of=date(2026, 2, 15))

        assert results == []

    def test_rejects_unexpected_single_extensions(self, tmp_path: Path):
        """REQ-ATD-008: allowlist 강화 후 `txt`/`json` 외 확장자(예: `.bin`, `.pkl`,
        `.pt`)는 이전에는 permissive `\\w+`가 매칭했으나 강화된 allowlist에서는
        배제돼야 한다."""
        for ext in ("bin", "pkl", "pt"):
            _touch_model_file(tmp_path, "domestic", 5, "lightgbm", date(2026, 1, 1), ext=ext)

        results = detect_stale_models(tmp_path, threshold_days=28, as_of=date(2026, 2, 15))

        assert results == []

    def test_defaults_to_kst_calendar_date_not_system_local(self, tmp_path: Path, monkeypatch):
        """이 프로젝트는 모든 시각을 KST로 해석한다 — 시스템 로컬 타임존이 달라도
        `as_of` 미지정 시 KST 캘린더 날짜를 기준으로 판정해야 한다."""
        import analyzer.orchestration.staleness as staleness_module

        kst_now = datetime(2026, 2, 15, 0, 30, tzinfo=ZoneInfo("Asia/Seoul"))
        # KST 00:30은 UTC 전날 15:30 — 시스템이 UTC라면 today()는 하루 전 날짜를 반환한다.
        utc_now = kst_now.astimezone(ZoneInfo("UTC"))
        assert utc_now.date() != kst_now.date()

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is not None:
                    return kst_now.astimezone(tz)
                return utc_now.replace(tzinfo=None)

        monkeypatch.setattr(staleness_module, "datetime", _FrozenDatetime)
        _touch_model_file(tmp_path, "domestic", 5, "lightgbm", kst_now.date())

        results = detect_stale_models(tmp_path, threshold_days=28)

        assert len(results) == 1
        assert results[0].is_stale is False
        assert results[0].most_recent_trained_date == kst_now.date()

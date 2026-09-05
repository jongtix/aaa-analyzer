"""confidence 산출 가드 명세 테스트 (SPEC-ANALYZER-INFER-001 M3,
REQ-AIF-051, AC-AIF-010).
"""

from datetime import date
from pathlib import Path

from analyzer.inference.resolution import QuantileManifest, SkipReason
from analyzer.inference.scoring import resolve_confidence_for_stock


def _make_manifest(tmp_path: Path) -> QuantileManifest:
    p10_path = tmp_path / "p10.txt"
    p90_path = tmp_path / "p90.txt"
    p10_path.write_bytes(b"dummy")
    p90_path.write_bytes(b"dummy")
    return QuantileManifest(
        market="domestic",
        horizon=20,
        trained_date=date(2026, 8, 25),
        p10_path=p10_path,
        p90_path=p90_path,
    )


class TestResolveConfidenceForStockQuantileMissing:
    """AC-AIF-010: 분위수 모델 자체가 없으면(조합 단위) confidence를
    시도조차 하지 않고 QUANTILE_MISSING으로 스킵돼야 한다."""

    def test_missing_manifest_routes_to_quantile_missing(self):
        result = resolve_confidence_for_stock(None, score=0.04, p10=-0.01, p90=0.01)

        assert result is SkipReason.QUANTILE_MISSING

    def test_missing_manifest_ignores_score_and_bounds(self):
        """매니페스트가 없으면 score/p10/p90 값과 무관하게 항상 스킵된다 —
        confidence 계산 자체를 시도하지 않는다는 것을 증명."""
        result = resolve_confidence_for_stock(None, score=0.0, p10=0.5, p90=0.5)

        assert result is SkipReason.QUANTILE_MISSING


class TestResolveConfidenceForStockDegenerateQuantile:
    """AC-AIF-010: 매니페스트는 있으나 종목의 예측이 축퇴 분포(p10==p90)면
    그 종목만 DEGENERATE_QUANTILE로 스킵돼야 한다(ensemble.py 무수정)."""

    def test_degenerate_distribution_routes_to_degenerate_quantile(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)

        result = resolve_confidence_for_stock(manifest, score=0.04, p10=0.02, p90=0.02)

        assert result is SkipReason.DEGENERATE_QUANTILE

    def test_valid_distribution_returns_confidence_float(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)

        result = resolve_confidence_for_stock(manifest, score=0.04, p10=-0.02, p90=0.06)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_zero_score_returns_neutral_confidence(self, tmp_path: Path):
        """score_ensemble == 0.0이면 ensemble.py의 계약대로 0.5(방향 무정보)를
        반환한다 — p10==p90이어도 이 분기는 ValueError 이전에 처리된다."""
        manifest = _make_manifest(tmp_path)

        result = resolve_confidence_for_stock(manifest, score=0.0, p10=0.02, p90=0.02)

        assert result == 0.5


class TestResolveConfidenceForStockBatchIndependence:
    """AC-AIF-010: 배치 내 한 종목의 스킵(사유 무관)이 같은 배치의 다른
    종목 처리 결과에 영향을 주지 않아야 한다 — 순수 함수 검증."""

    def test_degenerate_stock_does_not_affect_healthy_stock_in_same_batch(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)

        # 종목 A: 축퇴 분포 → 스킵.
        result_a = resolve_confidence_for_stock(manifest, score=0.03, p10=0.01, p90=0.01)
        # 종목 B: 정상 분포 → 같은 매니페스트, 같은 호출 시퀀스에서 정상 산출.
        result_b = resolve_confidence_for_stock(manifest, score=0.03, p10=-0.02, p90=0.05)
        # 종목 C: 다시 정상 — A의 스킵이 이후 호출에 잔류 상태를 남기지 않음을 재확인.
        result_c = resolve_confidence_for_stock(manifest, score=0.02, p10=-0.01, p90=0.04)

        assert result_a is SkipReason.DEGENERATE_QUANTILE
        assert isinstance(result_b, float)
        assert isinstance(result_c, float)

    def test_quantile_missing_combo_does_not_affect_other_combo_with_manifest(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)

        # 조합 A: 분위수 모델 미배포 → 조합 전체 스킵.
        result_a = resolve_confidence_for_stock(None, score=0.03, p10=0.0, p90=0.0)
        # 조합 B: 분위수 모델 배포됨 → 정상 산출.
        result_b = resolve_confidence_for_stock(manifest, score=0.03, p10=-0.02, p90=0.05)

        assert result_a is SkipReason.QUANTILE_MISSING
        assert isinstance(result_b, float)

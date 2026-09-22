"""추론 Prometheus 계측 테스트 (SPEC-ANALYZER-INFER-001 M8, REQ-AIF-130,
AC-AIF-021).

`TrainingMetrics`(`orchestration/metrics.py`)와 동일하게 격리된
`CollectorRegistry`를 주입해 테스트 간 전역 레지스트리 오염과 "Duplicated
timeseries" 재등록 오류를 피한다.
"""

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from analyzer.inference.metrics import (
    INFERENCE_CYCLE_DURATION_NAME,
    INFERENCE_LAST_CYCLE_NAME,
    INFERENCE_SIGNALS_TOTAL_NAME,
    INFERENCE_SKIP_TOTAL_NAME,
    InferenceMetrics,
)
from analyzer.inference.resolution import SkipReason


class TestInferenceMetricsNaming:
    """B-INFER-1: design.md §8은 `analyzer_inference_*`로 표기했으나, 실제
    코드베이스 관례(`orchestration/metrics.py`의 `aaa_analyzer_training_*`,
    REQ-AIF-130이 "패턴 계승"을 명시)를 따라 `aaa_analyzer_` 접두사를
    붙인다. AC-AIF-021이 요구하는 `analyzer_inference_*` 문자열은 부분
    문자열로 그대로 포함되므로 `/metrics` 조회 검증도 충족한다."""

    def test_all_four_metrics_follow_codebase_prefix(self):
        for name in (
            INFERENCE_SKIP_TOTAL_NAME,
            INFERENCE_SIGNALS_TOTAL_NAME,
            INFERENCE_CYCLE_DURATION_NAME,
            INFERENCE_LAST_CYCLE_NAME,
        ):
            assert name.startswith("aaa_analyzer_inference_")

    def test_design_md_names_remain_substrings(self):
        """design.md §8 명명을 부분 문자열로 보존한다(AC-AIF-021 grep 충족)."""
        assert "analyzer_inference_skip_total" in INFERENCE_SKIP_TOTAL_NAME
        assert "analyzer_inference_signals_total" in INFERENCE_SIGNALS_TOTAL_NAME
        assert "analyzer_inference_cycle_duration_seconds" in INFERENCE_CYCLE_DURATION_NAME


class TestInferenceMetricsSkipCounter:
    """REQ-AIF-130 (a): 조합/종목 단위 스킵 카운터(market/horizon/reason)."""

    def test_record_skip_increments_counter_per_reason(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        metrics.record_skip(market="domestic", horizon=20, reason=SkipReason.NO_MANIFEST)
        metrics.record_skip(market="domestic", horizon=20, reason=SkipReason.NO_MANIFEST)
        metrics.record_skip(market="domestic", horizon=20, reason=SkipReason.SHA_MISMATCH)

        no_manifest = registry.get_sample_value(
            INFERENCE_SKIP_TOTAL_NAME,
            {"market": "domestic", "horizon": "20", "reason": "no_manifest"},
        )
        sha_mismatch = registry.get_sample_value(
            INFERENCE_SKIP_TOTAL_NAME,
            {"market": "domestic", "horizon": "20", "reason": "sha_mismatch"},
        )
        assert no_manifest == 2.0
        assert sha_mismatch == 1.0

    def test_all_seven_skip_reasons_are_recordable(self):
        """REQ-AIF-130 (a)가 열거한 6개 사유 + SPEC-ANALYZER-PIPELINE-001
        REQ-APL-103의 `UNEXPECTED_ERROR`(일반 예외 경계) 총 7개 레이블
        전체가 기록 가능해야 한다 — 레이블 어휘는 `resolution.SkipReason`을
        그대로 재사용한다(신규 문자열 도입 금지)."""
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        expected = {
            "no_manifest",
            "sha_mismatch",
            "quantile_missing",
            "feature_insufficient",
            "degenerate_quantile",
            "manifest_race",
            "unexpected_error",
        }
        assert {member.value for member in SkipReason} == expected

        for reason in SkipReason:
            metrics.record_skip(market="overseas", horizon=60, reason=reason)

        for reason in SkipReason:
            value = registry.get_sample_value(
                INFERENCE_SKIP_TOTAL_NAME,
                {"market": "overseas", "horizon": "60", "reason": reason.value},
            )
            assert value == 1.0

    def test_plain_string_reason_is_accepted(self):
        """호출부가 `SkipReason` 대신 문자열을 넘겨도 동일 레이블로 기록된다."""
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        metrics.record_skip(market="domestic", horizon=60, reason="manifest_race")

        value = registry.get_sample_value(
            INFERENCE_SKIP_TOTAL_NAME,
            {"market": "domestic", "horizon": "60", "reason": "manifest_race"},
        )
        assert value == 1.0

    def test_unknown_reason_label_is_rejected(self):
        """REQ-AIF-130이 확정한 6개 어휘 밖 문자열은 거부한다 — 오타 레이블이
        조용히 새 시계열을 만들어 알람 규칙을 무력화하는 것을 막는다."""
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        with pytest.raises(ValueError):
            metrics.record_skip(market="domestic", horizon=20, reason="no_manifests")


class TestInferenceMetricsSignalCounter:
    """REQ-AIF-130 (b): 신호 생성 카운터(market/horizon/signal_class)."""

    def test_record_signal_increments_counter_per_class(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        metrics.record_signal(market="domestic", horizon=20, signal_class="STRONG_BUY")
        metrics.record_signal(market="domestic", horizon=20, signal_class="STRONG_BUY")
        metrics.record_signal(market="domestic", horizon=20, signal_class="HOLD")

        strong_buy = registry.get_sample_value(
            INFERENCE_SIGNALS_TOTAL_NAME,
            {"market": "domestic", "horizon": "20", "signal_class": "STRONG_BUY"},
        )
        hold = registry.get_sample_value(
            INFERENCE_SIGNALS_TOTAL_NAME,
            {"market": "domestic", "horizon": "20", "signal_class": "HOLD"},
        )
        assert strong_buy == 2.0
        assert hold == 1.0

    def test_all_five_grade_classes_are_recordable(self):
        """등급 5클래스(`training/boundaries.GRADE_ORDER`)가 전부 기록 가능하다."""
        from analyzer.training.boundaries import GRADE_ORDER

        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        for signal_class in GRADE_ORDER:
            metrics.record_signal(market="overseas", horizon=60, signal_class=signal_class)

        for signal_class in GRADE_ORDER:
            value = registry.get_sample_value(
                INFERENCE_SIGNALS_TOTAL_NAME,
                {"market": "overseas", "horizon": "60", "signal_class": signal_class},
            )
            assert value == 1.0


class TestInferenceMetricsCycleDuration:
    """REQ-AIF-130 (c): 추론 사이클 소요 시간 히스토그램(market 레이블)."""

    def test_observe_cycle_duration_records_sum_and_count(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        metrics.observe_cycle_duration(market="domestic", seconds=745.88)
        metrics.observe_cycle_duration(market="domestic", seconds=300.0)

        total = registry.get_sample_value(
            f"{INFERENCE_CYCLE_DURATION_NAME}_sum", {"market": "domestic"}
        )
        count = registry.get_sample_value(
            f"{INFERENCE_CYCLE_DURATION_NAME}_count", {"market": "domestic"}
        )
        assert total == pytest.approx(1045.88)
        assert count == 2.0

    def test_markets_are_recorded_independently(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        metrics.observe_cycle_duration(market="domestic", seconds=745.88)
        metrics.observe_cycle_duration(market="overseas", seconds=306.86)

        domestic = registry.get_sample_value(
            f"{INFERENCE_CYCLE_DURATION_NAME}_count", {"market": "domestic"}
        )
        overseas = registry.get_sample_value(
            f"{INFERENCE_CYCLE_DURATION_NAME}_count", {"market": "overseas"}
        )
        assert domestic == 1.0
        assert overseas == 1.0

    def test_buckets_cover_the_m7_measured_cycle_range(self):
        """M7 실측(domestic 745.88초 / overseas 306.86초)과 REQ-AIF-142 알람
        임계(30분)가 기본 버킷(최대 10초)으로는 전부 +Inf에 몰려 관측 불가하다
        — 분 단위 버킷을 명시해야 한다."""
        from analyzer.inference.metrics import CYCLE_DURATION_BUCKETS

        assert max(CYCLE_DURATION_BUCKETS) >= 1800
        assert any(300 <= bucket <= 900 for bucket in CYCLE_DURATION_BUCKETS)

        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        metrics.observe_cycle_duration(market="domestic", seconds=745.88)

        below = registry.get_sample_value(
            f"{INFERENCE_CYCLE_DURATION_NAME}_bucket", {"market": "domestic", "le": "600.0"}
        )
        at_ceiling = registry.get_sample_value(
            f"{INFERENCE_CYCLE_DURATION_NAME}_bucket", {"market": "domestic", "le": "1800.0"}
        )
        assert below == 0.0
        assert at_ceiling == 1.0


class TestInferenceMetricsLastCycleGauge:
    """SPEC-OBSV-ANALYZER-DEADMAN-001 REQ-DMR-001/005: 데드맨 스위치
    재설계가 도입한 재기동-생존 게이지."""

    def test_record_last_cycle_sets_value_for_market(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        metrics.record_last_cycle(market="domestic", epoch_seconds=1_758_000_000.0)

        value = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        assert value == 1_758_000_000.0

    def test_markets_are_recorded_independently(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        metrics.record_last_cycle(market="domestic", epoch_seconds=1_758_000_000.0)
        metrics.record_last_cycle(market="overseas", epoch_seconds=1_758_003_600.0)

        domestic = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        overseas = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "overseas"})
        assert domestic == 1_758_000_000.0
        assert overseas == 1_758_003_600.0

    def test_re_recording_overwrites_the_previous_value(self):
        """단일-프로세스 게이지 자체는 그냥 `.set()`이다 — 재기동-생존은
        `multiprocess_mode="max"`(REQ-DMR-005)가 pid 파일 병합 단계에서
        제공하는 속성이며, 여기서는 단일 프로세스 내 최신값 덮어쓰기만
        확인한다(멀티프로세스 병합 자체는 test_api.py 복원-시뮬레이션이
        검증한다)."""
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        metrics.record_last_cycle(market="domestic", epoch_seconds=1_758_000_000.0)
        metrics.record_last_cycle(market="domestic", epoch_seconds=1_758_003_600.0)

        value = registry.get_sample_value(INFERENCE_LAST_CYCLE_NAME, {"market": "domestic"})
        assert value == 1_758_003_600.0

    def test_gauge_is_registered_with_max_multiprocess_mode(self):
        """REQ-DMR-005의 병합 의미론이 실제로 `multiprocess_mode="max"`로
        선언돼 있는지 직접 확인한다 — `orchestration/metrics.py`의
        `last_success_timestamp`/`model_stale` 게이지와 동일 관례."""
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)

        assert metrics.inference_last_cycle._multiprocess_mode == "max"  # noqa: SLF001


class TestInferenceMetricsRegistryIsolation:
    """AC-AIF-021 스모크: 주입 레지스트리에서 실제 스크레이프 가능한지 확인
    (`TrainingMetrics` 테스트의 동일 패턴 계승)."""

    def test_registry_is_scrapeable_via_generate_latest(self):
        registry = CollectorRegistry()
        metrics = InferenceMetrics(registry=registry)
        metrics.record_skip(market="domestic", horizon=20, reason=SkipReason.NO_MANIFEST)
        metrics.record_signal(market="domestic", horizon=20, signal_class="BUY")
        metrics.observe_cycle_duration(market="domestic", seconds=1.0)
        metrics.record_last_cycle(market="domestic", epoch_seconds=1_758_000_000.0)

        output = generate_latest(registry).decode("utf-8")

        assert INFERENCE_SKIP_TOTAL_NAME in output
        assert INFERENCE_SIGNALS_TOTAL_NAME in output
        assert INFERENCE_CYCLE_DURATION_NAME in output
        assert INFERENCE_LAST_CYCLE_NAME in output

    def test_uses_injected_registry_not_default(self):
        from prometheus_client import REGISTRY

        isolated_registry = CollectorRegistry()
        InferenceMetrics(registry=isolated_registry)

        default_output = generate_latest(REGISTRY).decode("utf-8")
        isolated_output = generate_latest(isolated_registry).decode("utf-8")

        assert INFERENCE_SKIP_TOTAL_NAME not in default_output
        assert INFERENCE_SKIP_TOTAL_NAME in isolated_output

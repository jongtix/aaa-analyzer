"""Prometheus 계측 테스트 (REQ-ATA-070/071, acceptance.md §C.1 매핑 보강).

`TrainingMetrics`는 격리된 `CollectorRegistry`를 주입받아 테스트 간 전역 레지스트리
상태 오염을 피한다. 이 SPEC의 analyzer 측 책임은 올바른 메트릭 발행에 한정되며
(REQ-ATA-071), vmalert 알람 규칙 YAML은 작성하지 않는다 — 아래 스모크 테스트는
`prometheus_client` 레지스트리에서 실제로 스크레이프 가능한지만 검증한다.
"""

from datetime import date

from prometheus_client import CollectorRegistry, generate_latest

from analyzer.orchestration.metrics import (
    LAST_SUCCESS_TIMESTAMP_NAME,
    MODEL_STALE_NAME,
    RANK_IC_NAME,
    TRAINING_RUN_TOTAL_NAME,
    TrainingMetrics,
)
from analyzer.orchestration.staleness import ModelStalenessInfo


class TestTrainingMetricsSuccess:
    """AC-ATA-001: 성공 경로 Prometheus 메트릭 발행(REQ-ATA-070)."""

    def test_record_success_increments_counter_and_sets_gauge(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        metrics.record_success(market="domestic", horizon=5, algorithm="lightgbm", timestamp=1000.0)

        counter_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "success"}
        )
        gauge_value = registry.get_sample_value(
            LAST_SUCCESS_TIMESTAMP_NAME,
            {"market": "domestic", "horizon": "5", "algorithm": "lightgbm"},
        )
        assert counter_value == 1.0
        assert gauge_value == 1000.0


class TestTrainingMetricsFailure:
    """AC-ATA-002~005: 실패 경로 Prometheus 알람 트리거 시그널(REQ-ATA-061 경유)."""

    def test_record_failure_increments_counter_per_stage(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        metrics.record_failure(stage="wol")
        metrics.record_failure(stage="wol")
        metrics.record_failure(stage="ssh")

        wol_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "wol", "outcome": "failure"}
        )
        ssh_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "ssh", "outcome": "failure"}
        )
        assert wol_value == 2.0
        assert ssh_value == 1.0

    def test_all_four_failure_stages_are_recordable(self):
        """REQ-ATA-060: WoL/SSH/학습스크립트/타임아웃 4종 실패가 모두 동일 경로로 기록 가능하다."""
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        for stage in ("wol", "ssh", "training", "timeout"):
            metrics.record_failure(stage=stage)

        for stage in ("wol", "ssh", "training", "timeout"):
            value = registry.get_sample_value(
                TRAINING_RUN_TOTAL_NAME, {"stage": stage, "outcome": "failure"}
            )
            assert value == 1.0


class TestTrainingMetricsStaleness:
    """AC-ATA-007: 모델 정체 감지 알람 메트릭(REQ-ATA-072)."""

    def test_record_staleness_sets_gauge_to_one_when_stale(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        metrics.record_staleness(market="overseas", horizon=20, algorithm="xgboost", is_stale=True)

        value = registry.get_sample_value(
            MODEL_STALE_NAME, {"market": "overseas", "horizon": "20", "algorithm": "xgboost"}
        )
        assert value == 1.0

    def test_record_staleness_sets_gauge_to_zero_when_fresh(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        metrics.record_staleness(market="overseas", horizon=20, algorithm="xgboost", is_stale=False)

        value = registry.get_sample_value(
            MODEL_STALE_NAME, {"market": "overseas", "horizon": "20", "algorithm": "xgboost"}
        )
        assert value == 0.0


class TestTrainingMetricsOutcomeAwareRecordSuccess:
    """M6(REQ-ATE-064/065): `record_success()`가 outcome 레이블(success/
    held-back)로 카운터를 구분 증가시킨다 — held-back은 게이지를 갱신하지
    않는다(기존 챔피언이 계속 서빙 중이므로 "마지막 성공"을 갱신할 근거 없음)."""

    def test_outcome_success_increments_counter_and_updates_gauge(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        metrics.record_success(
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            timestamp=1000.0,
            outcome="success",
        )

        counter_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "success"}
        )
        gauge_value = registry.get_sample_value(
            LAST_SUCCESS_TIMESTAMP_NAME,
            {"market": "domestic", "horizon": "20", "algorithm": "lightgbm"},
        )
        assert counter_value == 1.0
        assert gauge_value == 1000.0

    def test_outcome_held_back_increments_counter_but_not_gauge(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        metrics.record_success(
            market="domestic",
            horizon=20,
            algorithm="xgboost",
            timestamp=2000.0,
            outcome="held-back",
        )

        counter_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "held-back"}
        )
        gauge_value = registry.get_sample_value(
            LAST_SUCCESS_TIMESTAMP_NAME,
            {"market": "domestic", "horizon": "20", "algorithm": "xgboost"},
        )
        assert counter_value == 1.0
        assert gauge_value is None  # 게이지가 아예 갱신되지 않았어야 한다.

    def test_default_outcome_is_success_backward_compatible(self):
        """기존 호출자(`outcome` 미지정)는 outcome="success"로 하위 호환된다."""
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        metrics.record_success(market="domestic", horizon=20, algorithm="lightgbm", timestamp=1.0)

        counter_value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "success", "outcome": "success"}
        )
        assert counter_value == 1.0


class TestTrainingMetricsRankIcGauge:
    """REQ-ATE-066(M6): (시장,horizon,algorithm)별 Rank IC 게이지 — 8개
    조합 각각에 대해 독립적인 레이블 조합으로 관측 가능해야 한다(AC-ATE-051)."""

    def test_record_rank_ic_sets_gauge_per_combo(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        combos = [
            ("domestic", 20, "lightgbm", 0.03),
            ("domestic", 20, "xgboost", 0.025),
            ("domestic", 60, "lightgbm", -0.01),
            ("overseas", 20, "lightgbm", 0.02),
            ("overseas", 20, "xgboost", 0.015),
            ("overseas", 60, "lightgbm", 0.04),
            ("overseas", 60, "xgboost", 0.05),
            ("domestic", 60, "xgboost", 0.06),
        ]
        for market, horizon, algorithm, rank_ic in combos:
            metrics.record_rank_ic(
                market=market, horizon=horizon, algorithm=algorithm, rank_ic=rank_ic
            )

        for market, horizon, algorithm, rank_ic in combos:
            value = registry.get_sample_value(
                RANK_IC_NAME,
                {"market": market, "horizon": str(horizon), "algorithm": algorithm},
            )
            assert value == rank_ic

    def test_rank_ic_gauge_follows_existing_naming_convention(self):
        assert RANK_IC_NAME.startswith("aaa_analyzer_training_")


class TestTrainingMetricsStalenessBatch:
    """SPEC-ANALYZER-TRAIN-STALENESS-001 M3(REQ-ATD-009, M4 선행 구현 —
    일일 콜백 배선(M3)이 이 메서드 없이는 정상 동작할 수 없어 M3 커밋에 앞당겨
    구현한다): `record_staleness_batch()`는 스캔 시작 시 `model_stale` 게이지
    패밀리 전체를 초기화(clear)한 뒤 전달된 결과만 재기록한다 — 삭제된 조합의
    값이 영구 잔존해서는 안 된다. 기존 `record_staleness()`(단건) 시그니처는
    무수정이다."""

    def test_clears_combo_not_present_in_new_batch(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)
        metrics.record_staleness(market="domestic", horizon=20, algorithm="lightgbm", is_stale=True)

        metrics.record_staleness_batch([])

        value = registry.get_sample_value(
            MODEL_STALE_NAME, {"market": "domestic", "horizon": "20", "algorithm": "lightgbm"}
        )
        assert value is None

    def test_records_all_combos_in_batch(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        results = [
            ModelStalenessInfo(
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                most_recent_trained_date=date(2026, 1, 1),
                is_stale=True,
            ),
            ModelStalenessInfo(
                market="overseas",
                horizon=60,
                algorithm="xgboost",
                most_recent_trained_date=date(2026, 8, 1),
                is_stale=False,
            ),
        ]

        metrics.record_staleness_batch(results)

        stale_value = registry.get_sample_value(
            MODEL_STALE_NAME, {"market": "domestic", "horizon": "20", "algorithm": "lightgbm"}
        )
        fresh_value = registry.get_sample_value(
            MODEL_STALE_NAME, {"market": "overseas", "horizon": "60", "algorithm": "xgboost"}
        )
        assert stale_value == 1.0
        assert fresh_value == 0.0

    def test_stale_combo_from_prior_batch_is_cleared_when_absent_from_next(self):
        """이전 스캔에서 정체로 기록된 조합이 모델 파일 삭제로 다음 스캔
        결과에서 사라지면, 게이지 값이 영구 잔존하지 않고 사라져야 한다
        (research.md D-11)."""
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)
        metrics.record_staleness_batch(
            [
                ModelStalenessInfo(
                    market="domestic",
                    horizon=20,
                    algorithm="lightgbm",
                    most_recent_trained_date=date(2026, 1, 1),
                    is_stale=True,
                )
            ]
        )

        metrics.record_staleness_batch(
            [
                ModelStalenessInfo(
                    market="overseas",
                    horizon=60,
                    algorithm="xgboost",
                    most_recent_trained_date=date(2026, 8, 1),
                    is_stale=False,
                )
            ]
        )

        stale_gone = registry.get_sample_value(
            MODEL_STALE_NAME, {"market": "domestic", "horizon": "20", "algorithm": "lightgbm"}
        )
        assert stale_gone is None


class TestPrometheusScrapeSmoke:
    """§E Self-Verification: 로컬 `prometheus_client` 레지스트리 스크레이프 스모크 테스트."""

    def test_registry_is_scrapeable_via_generate_latest(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)
        metrics.record_success(market="domestic", horizon=5, algorithm="lightgbm", timestamp=1000.0)
        metrics.record_failure(stage="ssh")
        metrics.record_staleness(market="domestic", horizon=5, algorithm="lightgbm", is_stale=True)

        output = generate_latest(registry).decode("utf-8")

        assert TRAINING_RUN_TOTAL_NAME in output
        assert LAST_SUCCESS_TIMESTAMP_NAME in output
        assert MODEL_STALE_NAME in output

    def test_uses_injected_registry_not_default(self):
        """레지스트리를 주입하면 `prometheus_client` 전역 기본 레지스트리를
        오염시키지 않는다 — 테스트 격리 확인(전역 레지스트리 스크레이프 결과에
        이 인스턴스의 메트릭명이 나타나지 않아야 한다)."""
        from prometheus_client import REGISTRY

        isolated_registry = CollectorRegistry()
        TrainingMetrics(registry=isolated_registry)

        default_output = generate_latest(REGISTRY).decode("utf-8")
        isolated_output = generate_latest(isolated_registry).decode("utf-8")

        assert TRAINING_RUN_TOTAL_NAME not in default_output
        assert TRAINING_RUN_TOTAL_NAME in isolated_output

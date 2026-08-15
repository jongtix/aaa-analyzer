"""통합 실패 처리 경로 테스트 (REQ-ATA-060/061/062, plan.md §B.2).

WoL/SSH/학습스크립트/타임아웃 4종 실패를 단일 `TrainingRunFailure` 타입과
`handle_training_run_failure()` 단일 경로로 통일해, 유형별 별도 처리 로직
분기(REQ-ATA-060, shall not)를 구조적으로 방지한다.
"""

import logging

import pytest
from prometheus_client import CollectorRegistry

from analyzer.common.logging import JsonFormatter
from analyzer.orchestration import failure as failure_module
from analyzer.orchestration.failure import TrainingRunFailure, handle_training_run_failure
from analyzer.orchestration.metrics import TRAINING_RUN_TOTAL_NAME, TrainingMetrics


class TestTrainingRunFailure:
    def test_holds_stage_message_and_run_id(self):
        failure = TrainingRunFailure(stage="wol", message="3회 재시도 실패", run_id="run-1")

        assert failure.stage == "wol"
        assert failure.message == "3회 재시도 실패"
        assert failure.run_id == "run-1"

    def test_is_frozen(self):
        failure = TrainingRunFailure(stage="wol", message="x", run_id="run-1")
        try:
            failure.stage = "ssh"  # type: ignore[misc]
        except AttributeError:
            pass
        else:
            raise AssertionError("TrainingRunFailure는 불변이어야 한다")


class TestHandleTrainingRunFailure:
    """REQ-ATA-060: 4가지 실패 유형이 모두 동일한 통합 경로를 실행한다."""

    def test_logs_error_with_stage_and_run_id(self, caplog: pytest.LogCaptureFixture):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)
        failure = TrainingRunFailure(stage="ssh", message="6회 재시도 실패", run_id="run-42")

        with caplog.at_level(logging.ERROR, logger="analyzer.orchestration.failure"):
            handle_training_run_failure(failure, metrics)

        assert any(
            "ssh" in record.message and "run-42" in record.message for record in caplog.records
        )

    def test_uses_structured_json_logger_not_raw_stdlib_logger(self):
        """AC-ATO-014(REQ-ATO-022): raw 표준 로거가 아니라 기존 구조화 JSON
        로거(get_logger())를 사용해야 한다 — 평문 stderr 유출을 제거한다."""
        assert isinstance(failure_module.logger.handlers[0].formatter, JsonFormatter)

    def test_records_prometheus_failure_metric(self):
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)
        failure = TrainingRunFailure(stage="timeout", message="4시간 초과", run_id="run-7")

        handle_training_run_failure(failure, metrics)

        value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "timeout", "outcome": "failure"}
        )
        assert value == 1.0

    def test_all_four_failure_stages_route_through_same_function(self):
        """REQ-ATA-060: WoL/SSH/학습스크립트/타임아웃 4종 모두 동일한 함수 1개로 처리된다
        (유형별 별도 처리 로직 분기가 없음을 구조적으로 검증)."""
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)

        for stage in ("wol", "ssh", "training", "timeout"):
            failure = TrainingRunFailure(stage=stage, message=f"{stage} 실패", run_id="run-x")
            handle_training_run_failure(failure, metrics)

        for stage in ("wol", "ssh", "training", "timeout"):
            value = registry.get_sample_value(
                TRAINING_RUN_TOTAL_NAME, {"stage": stage, "outcome": "failure"}
            )
            assert value == 1.0

    def test_does_not_raise_when_called_repeatedly(self):
        """반복 호출(재시도 사이클 여러 번)에도 예외 없이 누적 기록된다."""
        registry = CollectorRegistry()
        metrics = TrainingMetrics(registry=registry)
        failure = TrainingRunFailure(stage="wol", message="x", run_id="run-y")

        handle_training_run_failure(failure, metrics)
        handle_training_run_failure(failure, metrics)

        value = registry.get_sample_value(
            TRAINING_RUN_TOTAL_NAME, {"stage": "wol", "outcome": "failure"}
        )
        assert value == 2.0

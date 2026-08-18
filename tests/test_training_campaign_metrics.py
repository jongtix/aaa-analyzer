"""src/analyzer/training/campaign_metrics.py 테스트 (SPEC-ANALYZER-TRAIN-EVAL-001 M4).

사이드카 명명 충돌 회피(REQ-ATE-032), JSONL 스키마/append-only 계약
(REQ-ATE-037/038), 500개 폴드 시뮬레이션 규모(design.md §7), 신규
MySQL 스키마 미도입(REQ-ATE-035, 정적 grep 가드)을 검증한다.
"""

import json
from pathlib import Path

import pandas as pd

from analyzer.training.backtest import BacktestMetrics
from analyzer.training.campaign_metrics import (
    ComboGateVerdictStub,
    append_fold_metrics,
    compute_aggregate_metrics,
    fold_metrics_jsonl_filename,
    sidecar_path_for,
    write_campaign_summary_report,
    write_sidecar_metadata,
)


def _metrics(rank_ic: float = 0.03) -> BacktestMetrics:
    return BacktestMetrics(
        hit_rate=0.55,
        pearson_ic=0.04,
        rank_ic=rank_ic,
        precision=0.5,
        sharpe_ratio=0.8,
        max_drawdown=-0.1,
        confidence_calibration=0.9,
    )


class TestSidecarFilenameNoCollisionWithXgboostNativeExtension:
    def test_xgboost_json_model_gets_double_suffix(self, tmp_path: Path):
        model_path = tmp_path / "domestic_20_xgboost_2026-08-17.json"
        sidecar = sidecar_path_for(model_path)

        assert sidecar.name == "domestic_20_xgboost_2026-08-17.json.meta.json"
        assert sidecar != model_path

    def test_lightgbm_txt_model_gets_double_suffix(self, tmp_path: Path):
        model_path = tmp_path / "domestic_20_lightgbm_2026-08-17.txt"
        sidecar = sidecar_path_for(model_path)

        assert sidecar.name == "domestic_20_lightgbm_2026-08-17.txt.meta.json"


class TestSidecarMetadataSchema:
    def test_required_fields_present(self, tmp_path: Path):
        model_path = tmp_path / "domestic_20_lightgbm_2026-08-17.txt"
        aggregate = compute_aggregate_metrics([0.02, 0.03, 0.04])

        sidecar_path = write_sidecar_metadata(
            model_path,
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            aggregate_metrics=aggregate,
            final_fold_train_row_count=12345,
            frozen_hyperparameters={"n_estimators": 100, "learning_rate": 0.05},
            feature_columns=["KMID", "ROC_5"],
            fold_metrics_jsonl_relative_path="domestic_20_lightgbm_campaign_folds.jsonl",
        )

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["market"] == "domestic"
        assert payload["horizon"] == 20
        assert payload["algorithm"] == "lightgbm"
        assert payload["aggregate_metrics"]["mean_rank_ic"] == aggregate.mean_rank_ic
        assert payload["aggregate_metrics"]["stddev_rank_ic"] == aggregate.stddev_rank_ic
        assert payload["aggregate_metrics"]["icir"] == aggregate.icir
        assert payload["final_fold_train_row_count"] == 12345
        assert payload["frozen_hyperparameters"] == {"n_estimators": 100, "learning_rate": 0.05}
        assert payload["feature_columns"] == ["KMID", "ROC_5"]
        assert payload["fold_metrics_jsonl"] == "domestic_20_lightgbm_campaign_folds.jsonl"

    def test_no_inline_fold_time_series_in_sidecar(self, tmp_path: Path):
        """REQ-ATE-033: 폴드별 시계열 전체를 사이드카에 인라인 포함하지 않는다."""
        model_path = tmp_path / "domestic_20_lightgbm_2026-08-17.txt"
        aggregate = compute_aggregate_metrics([0.02, 0.03])

        sidecar_path = write_sidecar_metadata(
            model_path,
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            aggregate_metrics=aggregate,
            final_fold_train_row_count=100,
            frozen_hyperparameters={},
            feature_columns=[],
            fold_metrics_jsonl_relative_path="domestic_20_lightgbm_campaign_folds.jsonl",
        )

        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert "fold_metrics" not in payload
        assert "folds" not in payload


class TestComputeAggregateMetrics:
    def test_mean_stddev_icir_computed_correctly(self):
        result = compute_aggregate_metrics([0.02, 0.04])
        assert result.mean_rank_ic == 0.03
        assert result.stddev_rank_ic > 0
        assert result.icir == result.mean_rank_ic / result.stddev_rank_ic

    def test_single_value_stddev_is_zero_no_division_error(self):
        result = compute_aggregate_metrics([0.05])
        assert result.mean_rank_ic == 0.05
        assert result.stddev_rank_ic == 0.0
        assert result.icir == 0.0

    def test_empty_sequence_returns_zeros(self):
        result = compute_aggregate_metrics([])
        assert result.mean_rank_ic == 0.0
        assert result.stddev_rank_ic == 0.0
        assert result.icir == 0.0


class TestAppendFoldMetricsJsonl:
    def test_each_line_is_valid_single_json_object(self, tmp_path: Path):
        train_end = pd.Timestamp("2026-01-01")
        val_start = pd.Timestamp("2026-01-08")
        val_end = pd.Timestamp("2026-01-15")

        for fold_index in range(3):
            append_fold_metrics(
                tmp_path,
                "domestic",
                20,
                "lightgbm",
                fold_index,
                train_end,
                val_start,
                val_end,
                _metrics(rank_ic=0.01 * fold_index),
            )

        jsonl_path = tmp_path / fold_metrics_jsonl_filename("domestic", 20, "lightgbm")
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["fold_index"] == i
            assert record["train_end"] == "2026-01-01"
            assert record["val_start"] == "2026-01-08"
            assert record["val_end"] == "2026-01-15"
            assert set(record.keys()) == {
                "fold_index",
                "train_end",
                "val_start",
                "val_end",
                "hit_rate",
                "pearson_ic",
                "rank_ic",
                "precision",
                "sharpe_ratio",
                "max_drawdown",
                "confidence_calibration",
            }

    def test_none_val_end_serializes_to_null(self, tmp_path: Path):
        append_fold_metrics(
            tmp_path,
            "domestic",
            20,
            "lightgbm",
            0,
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-08"),
            None,
            _metrics(),
        )
        jsonl_path = tmp_path / fold_metrics_jsonl_filename("domestic", 20, "lightgbm")
        record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
        assert record["val_end"] is None


class TestFiveHundredFoldSimulationScale:
    """design.md §7: 약 500개 이상 폴드 규모에서도 사이드카는 조합당 1개만
    생성되고, 개별 폴드 사이드카가 폭발하지 않아야 한다.
    """

    def test_500_folds_produce_single_sidecar_and_full_jsonl(self, tmp_path: Path):
        jsonl_dir = tmp_path / "metrics"
        train_end = pd.Timestamp("2016-01-04")
        val_start = pd.Timestamp("2016-01-11")
        val_end = pd.Timestamp("2016-01-18")

        rank_ics: list[float] = []
        for fold_index in range(500):
            rank_ic = 0.001 * (fold_index % 7)
            rank_ics.append(rank_ic)
            append_fold_metrics(
                jsonl_dir,
                "domestic",
                20,
                "lightgbm",
                fold_index,
                train_end,
                val_start,
                val_end,
                _metrics(rank_ic=rank_ic),
            )

        model_path = tmp_path / "models" / "domestic_20_lightgbm_2026-08-17.txt"
        model_path.parent.mkdir(parents=True)
        write_sidecar_metadata(
            model_path,
            market="domestic",
            horizon=20,
            algorithm="lightgbm",
            aggregate_metrics=compute_aggregate_metrics(rank_ics),
            final_fold_train_row_count=999,
            frozen_hyperparameters={"n_estimators": 50},
            feature_columns=["KMID"],
            fold_metrics_jsonl_relative_path=fold_metrics_jsonl_filename(
                "domestic", 20, "lightgbm"
            ),
        )

        sidecar_files = list(model_path.parent.glob("*.meta.json"))
        assert len(sidecar_files) == 1

        jsonl_path = jsonl_dir / fold_metrics_jsonl_filename("domestic", 20, "lightgbm")
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 500
        for line in lines:
            json.loads(line)  # 각 라인이 유효한 단일 JSON 객체여야 한다.


class TestNoNewMySQLSchemaStaticGuard:
    """REQ-ATE-035/036: 이 모듈은 파일시스템 전용이며 신규 MySQL 스키마를
    도입하지 않는다 — 소스 코드에 DDL/DML 관련 토큰이 없어야 한다.
    """

    def test_source_contains_no_sql_ddl_dml_tokens(self):
        import analyzer.training.campaign_metrics as module

        source_path = Path(module.__file__)
        source = source_path.read_text(encoding="utf-8").upper()

        forbidden_tokens = (
            "CREATE TABLE",
            "ALTER TABLE",
            "INSERT INTO",
            "SQLALCHEMY",
            "PD.READ_SQL",
        )
        for token in forbidden_tokens:
            assert token not in source, f"forbidden SQL/DDL token found: {token}"


class TestCampaignSummaryReport:
    def test_report_includes_adjacent_fold_correlation_caveat(self, tmp_path: Path):
        report_path = tmp_path / "summary.md"
        write_campaign_summary_report(report_path, [])

        content = report_path.read_text(encoding="utf-8")
        assert "인접 폴드는 통계적으로 독립적인 표본이 아니다" in content

    def test_report_includes_gate_stub_per_combo(self, tmp_path: Path):
        report_path = tmp_path / "summary.md"
        verdicts = [
            ComboGateVerdictStub(
                market="domestic",
                horizon=20,
                algorithm="lightgbm",
                gate_verdict="not_yet_evaluated",
                supporting_metrics={"mean_rank_ic": 0.03},
            ),
            ComboGateVerdictStub(market="overseas", horizon=60, algorithm="xgboost"),
        ]
        write_campaign_summary_report(report_path, verdicts)

        content = report_path.read_text(encoding="utf-8")
        assert "domestic / D20 / lightgbm" in content
        assert "overseas / D60 / xgboost" in content
        assert "not_yet_evaluated" in content
        assert "mean_rank_ic: 0.03" in content

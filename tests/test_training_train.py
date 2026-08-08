"""src/analyzer/training/train.py 오케스트레이션 진입점 테스트 (SPEC-ANALYZER-TRAIN-001 M7).

design.md §4 진입점 계약: 각 단계(DB 조회→데이터셋 조립→학습→저장)가
올바른 순서로 호출되는지, 실패 시 비정상 종료코드를 반환하는지를
mock/stub으로 검증한다. 실 DB 접속(trainer 계정, SSH 터널)은 필요하지
않다 — 모든 외부 I/O 경계를 모킹한다.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from analyzer.data.models import TradingCalendar
from analyzer.training import train as train_module
from analyzer.training.models import HORIZONS, MARKETS
from analyzer.training.train import TrainingPipelineResult, main, run_training_pipeline


def _dummy_assembled_dataset() -> pd.DataFrame:
    dates = [date(2026, 1, i + 1) for i in range(30)]
    return pd.DataFrame(
        {
            "stock_code": ["A1"] * 30,
            "trade_date": dates,
            "KMID": [0.01] * 30,
            "label_D20": [0.02] * 30,
            "label_D60": [0.03] * 30,
        }
    )


class TestRunTrainingPipelineOrchestration:
    def test_calls_each_stage_in_order(self, tmp_path: Path):
        call_log: list[str] = []

        def _log(name):
            def _side_effect(*args, **kwargs):
                call_log.append(name)
                if name == "fetch_market_calendar":
                    return TradingCalendar(calendar_code="KRX", trading_days=frozenset())
                if name == "fetch_stock_universe":
                    return pd.DataFrame(
                        {"stock_code": ["A1"], "grade": ["A"], "delisted_at": [None]}
                    )
                if name == "fetch_market_data":
                    return ({}, {}, {})
                if name == "assemble_dataset_cached":
                    return _dummy_assembled_dataset()
                if name == "train_pooled_models":
                    model = MagicMock()
                    return {
                        (m, h, algo): model
                        for m in MARKETS
                        for h in HORIZONS
                        for algo in ("lightgbm", "xgboost")
                    }
                if name == "save_model_native":
                    from analyzer.training.persistence import SavedModel

                    return SavedModel(
                        model_path=tmp_path / "m.txt",
                        sidecar_path=tmp_path / "m.txt.sha256",
                        sha256="deadbeef",
                    )
                return None

            return _side_effect

        with (
            patch.object(
                train_module, "fetch_market_calendar", side_effect=_log("fetch_market_calendar")
            ),
            patch.object(
                train_module, "fetch_stock_universe", side_effect=_log("fetch_stock_universe")
            ),
            patch.object(train_module, "fetch_market_data", side_effect=_log("fetch_market_data")),
            patch.object(
                train_module.cache_module,
                "assemble_dataset_cached",
                side_effect=_log("assemble_dataset_cached"),
            ),
            patch.object(
                train_module, "train_pooled_models", side_effect=_log("train_pooled_models")
            ),
            patch.object(
                train_module.persistence_module,
                "save_model_native",
                side_effect=_log("save_model_native"),
            ),
        ):
            result = run_training_pipeline(
                trainer_engine=MagicMock(),
                calendar_code="KRX",
                cache_dir=tmp_path / "cache",
                models_root=tmp_path / "models",
                data_as_of=date(2026, 8, 8),
                feature_code_version="v1",
            )

        assert result.success is True
        # 캘린더 조회가 가장 먼저, 저장이 가장 나중이어야 한다.
        assert call_log[0] == "fetch_market_calendar"
        assert call_log[-1] == "save_model_native"
        assert call_log.index("train_pooled_models") > call_log.index("assemble_dataset_cached")
        assert call_log.index("save_model_native") > call_log.index("train_pooled_models")
        # 시장(MARKETS) 개수만큼 데이터셋 조립이 호출되어야 한다.
        assert call_log.count("assemble_dataset_cached") == len(MARKETS)

    def test_returns_failure_result_when_a_stage_raises(self, tmp_path: Path):
        with patch.object(
            train_module, "fetch_market_calendar", side_effect=RuntimeError("DB 연결 실패")
        ):
            result = run_training_pipeline(
                trainer_engine=MagicMock(),
                calendar_code="KRX",
                cache_dir=tmp_path / "cache",
                models_root=tmp_path / "models",
                data_as_of=date(2026, 8, 8),
                feature_code_version="v1",
            )

        assert result.success is False
        assert result.error is not None
        assert "DB 연결 실패" in result.error

    def test_success_result_is_a_dataclass_with_saved_paths(self):
        result = TrainingPipelineResult(success=True, saved_model_paths=[Path("/tmp/a.txt")])

        assert result.success is True
        assert result.saved_model_paths == [Path("/tmp/a.txt")]
        assert result.error is None


class TestFetchStockUniverseMarketCodeMapping:
    def test_domestic_maps_to_krx(self):
        engine = MagicMock()
        with patch.object(train_module.pd, "read_sql", return_value=pd.DataFrame()) as mock_read:
            train_module.fetch_stock_universe(engine, "domestic")

        params = mock_read.call_args[1]["params"]
        assert params["market_code"] == "KRX"

    def test_overseas_maps_to_us(self):
        engine = MagicMock()
        with patch.object(train_module.pd, "read_sql", return_value=pd.DataFrame()) as mock_read:
            train_module.fetch_stock_universe(engine, "overseas")

        params = mock_read.call_args[1]["params"]
        assert params["market_code"] == "US"


class TestMainCliExitCode:
    """오케스트레이션 진입점 CLI: 성공 시 0, 실패 시 1 (design.md §4 계약)."""

    def test_main_returns_0_on_success(self, tmp_path: Path):
        with (
            patch.object(train_module, "build_trainer_engine", return_value=MagicMock()),
            patch.object(
                train_module,
                "run_training_pipeline",
                return_value=TrainingPipelineResult(success=True, saved_model_paths=[]),
            ),
        ):
            exit_code = main(
                [
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--models-root",
                    str(tmp_path / "models"),
                    "--data-as-of",
                    "2026-08-08",
                    "--feature-code-version",
                    "v1",
                ]
            )

        assert exit_code == 0

    def test_main_returns_1_on_failure(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        with (
            patch.object(train_module, "build_trainer_engine", return_value=MagicMock()),
            patch.object(
                train_module,
                "run_training_pipeline",
                return_value=TrainingPipelineResult(success=False, error="테스트 실패"),
            ),
        ):
            exit_code = main(
                [
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--models-root",
                    str(tmp_path / "models"),
                    "--data-as-of",
                    "2026-08-08",
                    "--feature-code-version",
                    "v1",
                ]
            )

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "테스트 실패" in captured.err

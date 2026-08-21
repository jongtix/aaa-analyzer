"""src/analyzer/training/train.py 오케스트레이션 진입점 테스트 (SPEC-ANALYZER-TRAIN-001 M7).

design.md §4 진입점 계약: 각 단계(DB 조회→데이터셋 조립→학습→저장)가
올바른 순서로 호출되는지, 실패 시 비정상 종료코드를 반환하는지를
mock/stub으로 검증한다. 실 DB 접속(trainer 계정, SSH 터널)은 필요하지
않다 — 모든 외부 I/O 경계를 모킹한다.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightgbm as lgb
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
        # 캘린더 조회가 (시장별로) 가장 먼저 오고, 저장이 가장 나중이어야 한다.
        assert call_log[0] == "fetch_market_calendar"
        assert call_log[-1] == "save_model_native"
        assert call_log.index("train_pooled_models") > call_log.index("assemble_dataset_cached")
        assert call_log.index("save_model_native") > call_log.index("train_pooled_models")
        # 시장(MARKETS) 개수만큼 데이터셋 조립 + 캘린더 조회가 호출되어야 한다
        # (시장별 캘린더 — domestic=KRX/overseas=NYSE는 서로 다른 캘린더라
        # 공유하면 안 된다, 2026-08-13 발견 회귀 버그).
        assert call_log.count("assemble_dataset_cached") == len(MARKETS)
        assert call_log.count("fetch_market_calendar") == len(MARKETS)

    def test_fetches_krx_calendar_for_domestic_and_nyse_for_overseas(self, tmp_path: Path):
        """domestic/overseas가 동일 캘린더를 공유하면 미국 종목의 레이블이
        KRX 개장일 기준으로 계산되고, KRX 개장+미국 휴장일(추수감사절 등)이
        거래정지로 오판된다(회귀 버그, 2026-08-13 발견). `market_calendar` 실측:
        calendar_code='NYSE'의 최초 거래일(2007-08-20)이
        `DEFAULT_START_DATES["overseas"]`와 정확히 일치 — 해외용으로 이미
        시딩돼 있었으나 코드가 소비하지 않고 있었다."""
        calendar_code_log: list[str] = []

        def _fetch_market_calendar(_engine, calendar_code):
            calendar_code_log.append(calendar_code)
            return TradingCalendar(calendar_code=calendar_code, trading_days=frozenset())

        with (
            patch.object(train_module, "fetch_market_calendar", side_effect=_fetch_market_calendar),
            patch.object(
                train_module,
                "fetch_stock_universe",
                return_value=pd.DataFrame({"stock_code": [], "grade": [], "delisted_at": []}),
            ),
            patch.object(train_module, "fetch_market_data", return_value=({}, {}, {})),
            patch.object(
                train_module.cache_module,
                "assemble_dataset_cached",
                return_value=_dummy_assembled_dataset(),
            ),
            patch.object(train_module, "train_pooled_models", return_value={}),
        ):
            run_training_pipeline(
                trainer_engine=MagicMock(),
                calendar_code="KRX",
                cache_dir=tmp_path / "cache",
                models_root=tmp_path / "models",
                data_as_of=date(2026, 8, 8),
                feature_code_version="v1",
            )

        assert calendar_code_log == ["KRX", "NYSE"]

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

    def test_failure_logs_full_traceback_and_error_field_stays_string(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """AC-ATO-012(REQ-ATO-020): 실패 시 전체 traceback이 로그로 남고,
        TrainingPipelineResult.error는 여전히 문자열 메시지만 담는다(반환 타입 불변)."""
        with (
            caplog.at_level("ERROR", logger="analyzer.training.train"),
            patch.object(
                train_module, "fetch_market_calendar", side_effect=RuntimeError("DB 연결 실패")
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

        assert isinstance(result.error, str)
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(error_records) >= 1
        assert error_records[0].exc_info is not None
        assert "RuntimeError" in caplog.text
        assert "DB 연결 실패" in caplog.text

    def test_success_result_is_a_dataclass_with_saved_paths(self):

        result = TrainingPipelineResult(success=True, saved_model_paths=[Path("/tmp/a.txt")])

        assert result.success is True
        assert result.saved_model_paths == [Path("/tmp/a.txt")]
        assert result.error is None

    def test_saved_combos_field_defaults_empty_and_is_additive(self):
        """REQ-ATE-063(M6): 신규 필드는 additive — 기존 `saved_model_paths`만
        지정한 생성 경로가 그대로 동작하고 `saved_combos`는 빈 리스트 기본값."""
        result = TrainingPipelineResult(success=True, saved_model_paths=[Path("/tmp/a.txt")])

        assert result.saved_combos == []
        # 기존 필드 타입/의미는 이 확장 전후로 불변이어야 한다(REQ-ATE-063).
        assert result.saved_model_paths == [Path("/tmp/a.txt")]

    def test_saved_combos_populated_with_deduped_market_horizon_algorithm_tuples(
        self, tmp_path: Path
    ):
        """REQ-ATE-062/064: `run_training_pipeline()`이 저장한 각 (시장,horizon,
        algorithm) 조합을 구조화된 필드로 반환한다 — 분위수 보조 모델
        (`lightgbm_quantile`)은 포인트 LightGBM과 동일한 algorithm="lightgbm"
        으로 해석되므로 동일 조합이 중복 없이 1회만 나타나야 한다."""
        model = MagicMock()
        model.spec = lgb.LGBMRegressor  # isinstance 체크 통과용

        def _fake_train_pooled_models(*args, **kwargs):
            models: dict[tuple, object] = {}
            for m in MARKETS:
                for h in HORIZONS:
                    models[(m, h, "lightgbm")] = MagicMock(spec=lgb.LGBMRegressor)
                    models[(m, h, "xgboost")] = MagicMock()
                    for alpha in (0.10, 0.90):
                        models[(m, h, "lightgbm_quantile", alpha)] = MagicMock(
                            spec=lgb.LGBMRegressor
                        )
            return models

        def _fake_save_model_native(model, models_root, market, horizon, algorithm, trained_date):
            from analyzer.training.persistence import SavedModel

            path = tmp_path / f"{market}_{horizon}_{algorithm}.bin"
            return SavedModel(model_path=path, sidecar_path=path, sha256="deadbeef")

        def _fake_save_quantile_model(model, models_root, market, horizon, alpha, trained_date):
            from analyzer.training.persistence import SavedModel

            path = tmp_path / f"{market}_{horizon}_lightgbm_q{alpha}.bin"
            return SavedModel(model_path=path, sidecar_path=path, sha256="deadbeef")

        with (
            patch.object(
                train_module,
                "fetch_market_calendar",
                return_value=TradingCalendar(calendar_code="KRX", trading_days=frozenset()),
            ),
            patch.object(
                train_module,
                "fetch_stock_universe",
                return_value=pd.DataFrame({"stock_code": [], "grade": [], "delisted_at": []}),
            ),
            patch.object(train_module, "fetch_market_data", return_value=({}, {}, {})),
            patch.object(
                train_module.cache_module,
                "assemble_dataset_cached",
                return_value=_dummy_assembled_dataset(),
            ),
            patch.object(
                train_module, "train_pooled_models", side_effect=_fake_train_pooled_models
            ),
            patch.object(
                train_module.persistence_module,
                "save_model_native",
                side_effect=_fake_save_model_native,
            ),
            patch.object(
                train_module, "_save_quantile_model", side_effect=_fake_save_quantile_model
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
        expected_combos = {
            (m, h, algo) for m in MARKETS for h in HORIZONS for algo in ("lightgbm", "xgboost")
        }
        assert set(result.saved_combos) == expected_combos
        # 8 포인트 조합(중복 없음) — 분위수 보조 모델이 별도 조합으로
        # 이중 계산되지 않았음을 확인.
        assert len(result.saved_combos) == len(expected_combos)


class TestFrozenParamsByComboInjection:
    """REQ-ATG-011: 챔피언 동결 하이퍼파라미터를 (market,horizon,algorithm)별로
    주간 학습 파이프라인에 주입한다 — 게이트 챌린저(gate.py)와 동일 리더를
    공유해 조합별로 동일 파라미터임을 보장한다(AC-ATG-011)."""

    def test_none_preserves_single_call_backward_compat(self, tmp_path: Path):
        with (
            patch.object(
                train_module,
                "fetch_market_calendar",
                return_value=TradingCalendar(calendar_code="KRX", trading_days=frozenset()),
            ),
            patch.object(
                train_module,
                "fetch_stock_universe",
                return_value=pd.DataFrame({"stock_code": [], "grade": [], "delisted_at": []}),
            ),
            patch.object(train_module, "fetch_market_data", return_value=({}, {}, {})),
            patch.object(
                train_module.cache_module,
                "assemble_dataset_cached",
                return_value=_dummy_assembled_dataset(),
            ),
            patch.object(train_module, "train_pooled_models", return_value={}) as tpm_spy,
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
        tpm_spy.assert_called_once()
        _, kwargs = tpm_spy.call_args
        assert kwargs["lgbm_params"] is None
        assert kwargs["xgb_params"] is None

    def test_frozen_params_grouped_calls_train_pooled_models_per_distinct_signature(
        self, tmp_path: Path
    ):
        trained_date = date(2026, 8, 8)
        frozen_params_by_combo = {
            ("domestic", 60, "xgboost"): {"n_estimators": 38},
            ("overseas", 20, "xgboost"): {"n_estimators": 71},
            ("overseas", 60, "xgboost"): {"n_estimators": 71},  # overseas/20과 동일 시그니처
        }

        call_kwargs_log: list[dict] = []
        saved_combos_log: list[tuple] = []

        def _fake_train_pooled_models(data_by_combo, lgbm_params=None, xgb_params=None, **_):
            call_kwargs_log.append({"lgbm_params": lgbm_params, "xgb_params": xgb_params})
            models: dict[tuple, object] = {}
            for market, horizon in data_by_combo:
                models[(market, horizon, "lightgbm")] = MagicMock(spec=lgb.LGBMRegressor)
                models[(market, horizon, "xgboost")] = MagicMock()
            return models

        def _fake_save_model_native(model, models_root, market, horizon, algorithm, trained_date):
            from analyzer.training.persistence import SavedModel

            saved_combos_log.append((market, horizon, algorithm))
            path = tmp_path / f"{market}_{horizon}_{algorithm}.bin"
            return SavedModel(model_path=path, sidecar_path=path, sha256="deadbeef")

        with (
            patch.object(
                train_module,
                "fetch_market_calendar",
                return_value=TradingCalendar(calendar_code="KRX", trading_days=frozenset()),
            ),
            patch.object(
                train_module,
                "fetch_stock_universe",
                return_value=pd.DataFrame({"stock_code": [], "grade": [], "delisted_at": []}),
            ),
            patch.object(train_module, "fetch_market_data", return_value=({}, {}, {})),
            patch.object(
                train_module.cache_module,
                "assemble_dataset_cached",
                return_value=_dummy_assembled_dataset(),
            ),
            patch.object(
                train_module, "train_pooled_models", side_effect=_fake_train_pooled_models
            ),
            patch.object(
                train_module.persistence_module,
                "save_model_native",
                side_effect=_fake_save_model_native,
            ),
        ):
            result = run_training_pipeline(
                trainer_engine=MagicMock(),
                calendar_code="KRX",
                cache_dir=tmp_path / "cache",
                models_root=tmp_path / "models",
                data_as_of=trained_date,
                feature_code_version="v1",
                frozen_params_by_combo=frozen_params_by_combo,
            )

        assert result.success is True
        # domestic/20(기본값), domestic/60(38), overseas/20+overseas/60(71 공유) — 3개 그룹.
        assert len(call_kwargs_log) == 3
        xgb_param_values = [k["xgb_params"] for k in call_kwargs_log]
        assert {"n_estimators": 38} in xgb_param_values
        assert {"n_estimators": 71} in xgb_param_values
        assert None in xgb_param_values
        # 매 조합이 정확히 1회씩만 저장되어야 한다(그룹 간 중복 저장 없음).
        expected_combos = {
            (m, h, algo) for m in MARKETS for h in HORIZONS for algo in ("lightgbm", "xgboost")
        }
        assert set(saved_combos_log) == expected_combos
        assert len(saved_combos_log) == len(expected_combos)

    def test_combo_without_frozen_entry_falls_back_to_base_params(self, tmp_path: Path):
        from analyzer.training.persistence import SavedModel

        call_kwargs_log: list[dict] = []

        def _fake_train_pooled_models(data_by_combo, lgbm_params=None, xgb_params=None, **_):
            call_kwargs_log.append({"lgbm_params": lgbm_params, "xgb_params": xgb_params})
            return {
                (m, h, algo): MagicMock()
                for m, h in data_by_combo
                for algo in ("lightgbm", "xgboost")
            }

        def _fake_save_model_native(model, models_root, market, horizon, algorithm, trained_date):
            path = tmp_path / f"{market}_{horizon}_{algorithm}.bin"
            return SavedModel(model_path=path, sidecar_path=path, sha256="deadbeef")

        with (
            patch.object(
                train_module,
                "fetch_market_calendar",
                return_value=TradingCalendar(calendar_code="KRX", trading_days=frozenset()),
            ),
            patch.object(
                train_module,
                "fetch_stock_universe",
                return_value=pd.DataFrame({"stock_code": [], "grade": [], "delisted_at": []}),
            ),
            patch.object(train_module, "fetch_market_data", return_value=({}, {}, {})),
            patch.object(
                train_module.cache_module,
                "assemble_dataset_cached",
                return_value=_dummy_assembled_dataset(),
            ),
            patch.object(
                train_module, "train_pooled_models", side_effect=_fake_train_pooled_models
            ),
            patch.object(
                train_module.persistence_module,
                "save_model_native",
                side_effect=_fake_save_model_native,
            ),
        ):
            run_training_pipeline(
                trainer_engine=MagicMock(),
                calendar_code="KRX",
                cache_dir=tmp_path / "cache",
                models_root=tmp_path / "models",
                data_as_of=date(2026, 8, 8),
                feature_code_version="v1",
                frozen_params_by_combo={},  # 빈 매핑 — 전 조합이 기본값으로 폴백
            )

        assert len(call_kwargs_log) == 1
        assert call_kwargs_log[0] == {"lgbm_params": None, "xgb_params": None}


class TestParamsFromActiveMetaFlag:
    """REQ-ATG-011: `--params-from-active-meta` 플래그 — gate.py의 리더를
    재사용해 frozen_params_by_combo를 구성해 run_training_pipeline()에
    전달한다(AC-ATG-011 "게이트 챌린저와 동일 리더 공유")."""

    def test_flag_absent_forwards_none_frozen_params(self, tmp_path: Path):
        captured: dict = {}

        def _fake_run_training_pipeline(**kwargs):
            captured.update(kwargs)
            return TrainingPipelineResult(success=True, saved_model_paths=[])

        with (
            patch.object(train_module, "build_trainer_engine", return_value=MagicMock()),
            patch.object(
                train_module, "run_training_pipeline", side_effect=_fake_run_training_pipeline
            ),
        ):
            main(
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

        assert captured["frozen_params_by_combo"] is None

    def test_flag_present_reads_frozen_params_via_gate_reader(self, tmp_path: Path):
        import json as json_module

        from analyzer.orchestration import activation as activation_module
        from analyzer.training import campaign_metrics as campaign_metrics_module
        from analyzer.training import persistence as persistence_module

        active_models_root = tmp_path / "active_models"
        trained_date = date(2026, 8, 19)
        activation_module.write_activation_manifest(
            active_models_root,
            activation_module.ActivationManifest(
                market="domestic",
                horizon=60,
                algorithm="xgboost",
                trained_date=trained_date,
                sidecar_sha256="x",
                promoted_at="2026-08-19T00:00:00+00:00",
                promotion_basis={},
            ),
        )
        model_dir = persistence_module.model_dir(active_models_root, "domestic", 60, "xgboost")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / persistence_module.model_filename(
            "domestic", 60, "xgboost", trained_date
        )
        model_path.write_text("dummy", encoding="utf-8")
        sidecar_path = campaign_metrics_module.sidecar_path_for(model_path)
        sidecar_path.write_text(
            json_module.dumps({"frozen_hyperparameters": {"n_estimators": 38}}), encoding="utf-8"
        )

        captured: dict = {}

        def _fake_run_training_pipeline(**kwargs):
            captured.update(kwargs)
            return TrainingPipelineResult(success=True, saved_model_paths=[])

        with (
            patch.object(train_module, "build_trainer_engine", return_value=MagicMock()),
            patch.object(
                train_module, "run_training_pipeline", side_effect=_fake_run_training_pipeline
            ),
        ):
            main(
                [
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--models-root",
                    str(tmp_path / "models"),
                    "--data-as-of",
                    "2026-08-08",
                    "--feature-code-version",
                    "v1",
                    "--params-from-active-meta",
                    str(active_models_root),
                ]
            )

        assert captured["frozen_params_by_combo"] == {
            ("domestic", 60, "xgboost"): {"n_estimators": 38}
        }


class TestFetchStockUniverseMarketCodeMapping:
    """`stocks.market`은 거래소 코드(KOSPI/KOSDAQ/NYSE/NASDAQ/AMEX)로 저장된다
    (aaa-collector `Market` enum 실측) — `KRX`/`US`는 지수 종목(`asset_type=INDEX`)
    전용 값이라 개별 종목(`asset_type=STOCK`) 유니버스 조회에는 쓰이지 않는다.
    2026-08-13 NAS DB 실측: market='KRX' AND asset_type='STOCK' → 0행,
    market='US' AND asset_type='STOCK' → 0행(이전 회귀 버그)."""

    def test_domestic_maps_to_kospi_and_kosdaq(self):
        engine = MagicMock()
        with patch.object(train_module.pd, "read_sql", return_value=pd.DataFrame()) as mock_read:
            train_module.fetch_stock_universe(engine, "domestic")

        params = mock_read.call_args[1]["params"]
        assert params["market_codes"] == ("KOSPI", "KOSDAQ")

    def test_overseas_maps_to_nyse_nasdaq_amex(self):
        engine = MagicMock()
        with patch.object(train_module.pd, "read_sql", return_value=pd.DataFrame()) as mock_read:
            train_module.fetch_stock_universe(engine, "overseas")

        params = mock_read.call_args[1]["params"]
        assert params["market_codes"] == ("NYSE", "NASDAQ", "AMEX")


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


class TestMainTraceIdPropagation:
    """AC-ATO-008(REQ-ATO-012/013/014): TRAIN_RUN_ID env var가 있으면
    trace_id로 즉시 설정되어야 한다 — 없으면(fail-open) 그대로 진행된다."""

    def test_sets_trace_id_from_train_run_id_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from analyzer.common.trace import get_trace_id

        monkeypatch.setenv("TRAIN_RUN_ID", "run-xyz789")
        with (
            patch.object(train_module, "build_trainer_engine", return_value=MagicMock()),
            patch.object(
                train_module,
                "run_training_pipeline",
                return_value=TrainingPipelineResult(success=True, saved_model_paths=[]),
            ),
        ):
            main(
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

        assert get_trace_id() == "run-xyz789"

    def test_missing_train_run_id_does_not_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("TRAIN_RUN_ID", raising=False)
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


class TestFetchMarketDataDataAsOfBoundary:
    """AC-ATE-001(REQ-ATE-001/002/003/005/006): fetch_market_data()가
    data_as_of 상한을 fetch_daily_ohlcv(..., end_date=data_as_of)로 강제해야
    한다 — 합성 미래 원주가 행이 조회 결과에서 배제됨을 확인한다."""

    def test_passes_data_as_of_as_end_date_to_fetch_daily_ohlcv(self):
        stocks = pd.DataFrame({"stock_code": ["A1", "B2"]})
        captured_end_dates: list[date | None] = []

        def _fake_fetch_daily_ohlcv(engine, stock_code, start_date=None, end_date=None):
            captured_end_dates.append(end_date)
            return pd.DataFrame({"trade_date": []})

        with (
            patch.object(train_module, "fetch_daily_ohlcv", side_effect=_fake_fetch_daily_ohlcv),
            patch.object(train_module, "fetch_corporate_events", return_value=pd.DataFrame()),
            patch.object(train_module, "fetch_investor_trend", return_value=pd.DataFrame()),
        ):
            train_module.fetch_market_data(MagicMock(), stocks, date(2026, 8, 10))

        assert captured_end_dates == [date(2026, 8, 10), date(2026, 8, 10)]

    def test_synthetic_future_rows_are_excluded_from_fetched_result(self):
        """AC-ATE-001 Given-When-Then을 그대로 구현: 종목 A의 daily_ohlcv에
        2026-08-01~2026-08-20 원주가가 있고 data_as_of=2026-08-10이면, 조립된
        결과의 종목 A 행 중 trade_date > 2026-08-10인 행이 0건이어야 한다."""
        stocks = pd.DataFrame({"stock_code": ["A1"]})
        full_history = pd.DataFrame(
            {
                "trade_date": [date(2026, 8, d) for d in range(1, 21)],
                "close_price": list(range(20)),
            }
        )

        def _fake_fetch_daily_ohlcv(engine, stock_code, start_date=None, end_date=None):
            df = full_history
            if end_date is not None:
                df = df[df["trade_date"] <= end_date]
            return df.reset_index(drop=True)

        with (
            patch.object(train_module, "fetch_daily_ohlcv", side_effect=_fake_fetch_daily_ohlcv),
            patch.object(train_module, "fetch_corporate_events", return_value=pd.DataFrame()),
            patch.object(train_module, "fetch_investor_trend", return_value=pd.DataFrame()),
        ):
            ohlcv_by_stock, _, _ = train_module.fetch_market_data(
                MagicMock(), stocks, date(2026, 8, 10)
            )

        assembled = ohlcv_by_stock["A1"]
        assert (assembled["trade_date"] > date(2026, 8, 10)).sum() == 0
        assert len(assembled) == 10


class TestQuantileModelFilenameCollision:
    """AC-ATE-003(REQ-ATE-007/008/010): 동일 (시장,horizon) 조합의 포인트
    LightGBM 모델 + 분위수 보조 모델(alpha=0.10) + 분위수 보조 모델(alpha=0.90)
    3개가 서로 다른 파일 경로에 저장되고, persistence.py의 실제
    save_model_native()(무수정)를 그대로 재사용해 각 파일이 저장 직후
    SHA-256 라운드트립 검증(REQ-AT-092)을 통과함을 확인한다."""

    @staticmethod
    def _trained_lgbm_model():
        import numpy as np

        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 3))
        y = x @ np.array([0.02, -0.01, 0.015]) + rng.normal(scale=0.01, size=40)

        model = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
        model.fit(x, y)
        return model

    def test_three_lightgbm_family_models_do_not_collide(self, tmp_path: Path):
        from analyzer.training import persistence as persistence_module

        market, horizon = "domestic", 20
        trained_date = date(2026, 8, 17)

        point_model = self._trained_lgbm_model()
        quantile_10 = self._trained_lgbm_model()
        quantile_90 = self._trained_lgbm_model()

        point_saved = persistence_module.save_model_native(
            point_model, tmp_path, market, horizon, "lightgbm", trained_date
        )
        q10_saved = train_module._save_quantile_model(
            quantile_10, tmp_path, market, horizon, 0.10, trained_date
        )
        q90_saved = train_module._save_quantile_model(
            quantile_90, tmp_path, market, horizon, 0.90, trained_date
        )

        paths = {point_saved.model_path, q10_saved.model_path, q90_saved.model_path}
        assert len(paths) == 3
        for saved in (point_saved, q10_saved, q90_saved):
            assert saved.model_path.exists()
            assert persistence_module.verify_model_integrity(saved.model_path, saved.sidecar_path)

        # 포인트 모델의 파일명·경로는 기존 관례(models/{market}/{horizon}/lightgbm/
        # {market}_{horizon}_lightgbm_{trained_date}.txt)와 byte-identical해야 한다.
        assert point_saved.model_path.name == "domestic_20_lightgbm_2026-08-17.txt"
        # 분위수 보조 모델은 alpha 구분자가 파일명 세그먼트에 추가되어야 한다.
        assert q10_saved.model_path.name == "domestic_20_lightgbm_2026-08-17_q10.txt"
        assert q90_saved.model_path.name == "domestic_20_lightgbm_2026-08-17_q90.txt"
        # 셋 다 같은 algorithm 디렉토리(models/domestic/20/lightgbm/) 아래에 있어야 한다.
        assert point_saved.model_path.parent == q10_saved.model_path.parent
        assert point_saved.model_path.parent == q90_saved.model_path.parent

        # 스테이징 임시 디렉토리가 최종 트리에 잔존하지 않아야 한다.
        leftover_dirs = [p for p in tmp_path.iterdir() if p.name.startswith("tmp")]
        assert leftover_dirs == []


class TestSplitFeaturesAndLabelsFeatureAllowlist:
    """AC-ATE-053(REQ-ATE-074): 피처 컬럼 목록은 FEATURE_REGISTRY.keys()와
    assembled.columns의 교집합과 정확히 일치해야 하며, 원시 OHLCV 컬럼은
    배제되어야 한다."""

    def test_raw_ohlcv_columns_excluded_from_feature_columns(self):
        from analyzer.features.classification import FEATURE_REGISTRY

        n = 30
        assembled = pd.DataFrame(
            {
                "stock_code": ["A1"] * n,
                "trade_date": [date(2026, 1, i + 1) for i in range(n)],
                "open_price": [100.0] * n,
                "high_price": [101.0] * n,
                "low_price": [99.0] * n,
                "close_price": [100.5] * n,
                "volume": [1000] * n,
                "KMID": [0.01] * n,
                "ROC_5": [0.02] * n,
                "label_D20": [0.03] * n,
                "label_D20_exclude_reason": [None] * n,
                "label_D60": [0.04] * n,
                "label_D60_exclude_reason": [None] * n,
            }
        )

        feature_columns, x, _y = train_module._split_features_and_labels(assembled, horizon=20)

        assert set(feature_columns) == {"KMID", "ROC_5"}
        assert "open_price" not in feature_columns
        assert "high_price" not in feature_columns
        assert "low_price" not in feature_columns
        assert "close_price" not in feature_columns
        assert "volume" not in feature_columns
        assert "stock_code" not in feature_columns
        assert "trade_date" not in feature_columns
        assert set(feature_columns) == set(assembled.columns) & set(FEATURE_REGISTRY.keys())
        assert set(x.columns) == set(feature_columns)

    def test_partial_coverage_combo_only_includes_present_registry_columns(self):
        """REQ-AT-064: 해외 종목처럼 일부 FEATURE_REGISTRY 키가 assembled.columns에
        존재하지 않는 조합에서도, 교집합 로직이 자연스럽게 존재하는 컬럼만 반환한다."""
        n = 10
        assembled = pd.DataFrame(
            {
                "stock_code": ["OS1"] * n,
                "trade_date": [date(2026, 1, i + 1) for i in range(n)],
                "close_price": [50.0] * n,
                "KMID": [0.01] * n,
                # 수급 피처(foreign_net_ratio 등)는 존재하지 않음 — 해외 결측 시뮬레이션
                "label_D20": [0.02] * n,
            }
        )

        feature_columns, _x, _y = train_module._split_features_and_labels(assembled, horizon=20)

        assert feature_columns == ["KMID"]


class TestTruncateThenRecomputeEquivalence:
    """AC-ATE-054(REQ-ATE-075, design.md §3.4): 전체 패널을 조립한 뒤 T로
    슬라이스한 피처와, 원주가 자체를 T로 먼저 절단한 뒤 재조립한 피처가
    동일해야 한다 — point-in-time 불변식(후행 롤링 윈도우만 사용) 검증.

    합성 데이터는 기업이벤트(SPLIT/DIVIDEND)를 포함하지 않는다 — 이렇게
    하면 adjust_prices()의 as_of_date 의존성(design.md §3.3의 알려진
    한계, 캠페인 실행일 기준 소급 조정)이 개입하지 않고, 순수하게
    compute_technical_features()의 point-in-time 불변식만 격리해 검증할
    수 있다."""

    @staticmethod
    def _empty_events() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "event_type",
                "event_date",
                "stock_rate",
                "cash_amount",
                "event_subtype",
                "ex_dividend_date",
                "currency_code",
            ]
        )

    @staticmethod
    def _weekdays(start: date, end: date) -> list[date]:
        from datetime import timedelta

        days: list[date] = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                days.append(current)
            current += timedelta(days=1)
        return days

    def _ohlcv(self, stock_code: str, dates: list[date], seed: int) -> pd.DataFrame:
        import numpy as np

        rng = np.random.default_rng(seed)
        n = len(dates)
        base = 100.0 + np.cumsum(rng.normal(scale=0.5, size=n))
        return pd.DataFrame(
            {
                "stock_code": [stock_code] * n,
                "trade_date": dates,
                "open_price": base,
                "high_price": base + 1.0,
                "low_price": base - 1.0,
                "close_price": base + 0.2,
                "volume": rng.integers(1000, 5000, size=n),
            }
        )

    def test_technical_features_identical_before_and_after_truncation(self):
        from analyzer.data.models import TradingCalendar
        from analyzer.training.dataset import assemble_dataset

        full_start = date(2016, 1, 4)
        full_end = date(2016, 12, 30)
        truncate_at = date(2016, 6, 15)

        all_dates = self._weekdays(full_start, full_end)
        calendar = TradingCalendar(calendar_code="TEST", trading_days=frozenset(all_dates))

        stocks = pd.DataFrame(
            {
                "stock_code": ["A1", "A2"],
                "grade": ["A", "A"],
                "delisted_at": [None, None],
            }
        )
        full_ohlcv_by_stock = {
            "A1": self._ohlcv("A1", all_dates, seed=1),
            "A2": self._ohlcv("A2", all_dates, seed=2),
        }
        events_by_stock = {"A1": self._empty_events(), "A2": self._empty_events()}
        investor_trend_by_stock: dict[str, pd.DataFrame] = {}

        full_assembled = assemble_dataset(
            stocks=stocks,
            ohlcv_by_stock=full_ohlcv_by_stock,
            events_by_stock=events_by_stock,
            investor_trend_by_stock=investor_trend_by_stock,
            calendar=calendar,
            market="domestic",
        )
        full_sliced = full_assembled.loc[full_assembled["trade_date"] <= truncate_at]

        truncated_ohlcv_by_stock = {
            code: df.loc[df["trade_date"] <= truncate_at].reset_index(drop=True)
            for code, df in full_ohlcv_by_stock.items()
        }
        truncated_assembled = assemble_dataset(
            stocks=stocks,
            ohlcv_by_stock=truncated_ohlcv_by_stock,
            events_by_stock=events_by_stock,
            investor_trend_by_stock=investor_trend_by_stock,
            calendar=calendar,
            market="domestic",
        )

        assert not full_sliced.empty
        assert not truncated_assembled.empty

        feature_columns, _x, _y = train_module._split_features_and_labels(full_sliced, horizon=20)
        assert feature_columns  # sanity: FEATURE_REGISTRY 교집합이 비어있지 않아야 함

        merged = full_sliced.merge(
            truncated_assembled,
            on=["stock_code", "trade_date"],
            suffixes=("_full", "_truncated"),
        )
        assert len(merged) == len(full_sliced)

        import numpy as np

        for column in feature_columns:
            full_values = merged[f"{column}_full"].to_numpy(dtype=float)
            truncated_values = merged[f"{column}_truncated"].to_numpy(dtype=float)
            # 두 계산 모두 초반(윈도 미충족) 구간에서 동일한 위치에 NaN을
            # 산출해야 하므로 NaN을 서로 동일하다고 취급한다(equal_nan=True).
            np.testing.assert_allclose(
                full_values,
                truncated_values,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"point-in-time invariant violated for feature: {column}",
            )

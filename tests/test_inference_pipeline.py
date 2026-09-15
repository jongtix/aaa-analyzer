"""추론 오케스트레이션 계층 명세 테스트 (SPEC-ANALYZER-PIPELINE-001,
REQ-APL-100~107/110/111/121).

`inference/pipeline.py`는 INFER-001이 이미 완성한 9개 순수 모듈을 실제
흐름으로 조립하는 오케스트레이션 계층이다 — 이 테스트는 그 모듈들을 mock
주입 가능한 콜러블로 대체해 조립 로직(조합 단위 스킵, 종목 단위 예외 경계,
INSERT→발행→밴드 스윕 순서 계약)만 검증한다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from analyzer.inference import pipeline as pipeline_module
from analyzer.inference.pipeline import (
    MARKET_CALENDAR_CODE,
    resolve_calendar_code,
    resolve_model_version,
    run_market_inference,
)
from analyzer.inference.resolution import ServingPlan, SkipReason
from analyzer.inference.writer import InsertOutcome


def _manifest(trained_date: date):
    manifest = MagicMock()
    manifest.trained_date = trained_date
    return manifest


class TestCalendarCodeMapping:
    """REQ-APL-121: `training/train.py`의 비공개 심볼을 임포트하지 않고
    `inference/` 패키지 내부에 독립 정의한다."""

    def test_domestic_maps_to_krx(self):
        assert resolve_calendar_code("domestic") == "KRX"

    def test_overseas_maps_to_nyse(self):
        assert resolve_calendar_code("overseas") == "NYSE"

    def test_mapping_constant_does_not_import_private_train_symbol(self):
        from pathlib import Path

        source = Path("src/analyzer/inference/pipeline.py").read_text(encoding="utf-8")
        code_lines = [line for line in source.splitlines() if not line.strip().startswith("#")]
        code = "\n".join(code_lines)
        assert "from analyzer.training.train import" not in code
        assert "import analyzer.training.train" not in code

    def test_market_calendar_code_dict_has_expected_entries(self):
        assert MARKET_CALENDAR_CODE == {"domestic": "KRX", "overseas": "NYSE"}


class TestResolveModelVersion:
    """AC-APL-110: 앙상블 조합은 두 알고리즘 trained_date 중 최신값(max)을
    사용하고, 단독 전략은 그 알고리즘의 trained_date를 그대로 사용한다."""

    def test_ensemble_uses_the_more_recent_trained_date(self):
        serving_plan = ServingPlan(
            market="domestic",
            horizon=60,
            active_strategy="ensemble",
            algorithms=("lightgbm", "xgboost"),
            manifests={
                "lightgbm": _manifest(date(2026, 8, 19)),
                "xgboost": _manifest(date(2026, 8, 29)),
            },
            model_paths={},
        )

        assert resolve_model_version(serving_plan) == "domestic_60_ensemble_2026-08-29"

    def test_ensemble_uses_the_more_recent_trained_date_regardless_of_order(self):
        serving_plan = ServingPlan(
            market="overseas",
            horizon=20,
            active_strategy="ensemble",
            algorithms=("lightgbm", "xgboost"),
            manifests={
                "lightgbm": _manifest(date(2026, 9, 1)),
                "xgboost": _manifest(date(2026, 8, 1)),
            },
            model_paths={},
        )

        assert resolve_model_version(serving_plan) == "overseas_20_ensemble_2026-09-01"

    def test_solo_strategy_uses_its_own_trained_date(self):
        serving_plan = ServingPlan(
            market="domestic",
            horizon=60,
            active_strategy="xgboost",
            algorithms=("xgboost",),
            manifests={"xgboost": _manifest(date(2026, 8, 29))},
            model_paths={},
        )

        assert resolve_model_version(serving_plan) == "domestic_60_xgboost_2026-08-29"


def _solo_serving_plan(market: str, horizon: int, trained_date: date) -> ServingPlan:
    return ServingPlan(
        market=market,
        horizon=horizon,
        active_strategy="xgboost",
        algorithms=("xgboost",),
        manifests={"xgboost": _manifest(trained_date)},
        model_paths={"xgboost": Path(f"/models/{market}_{horizon}_xgboost.txt")},
    )


def _universe_df(rows: list[tuple[int, str, str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["stock_id", "stock_code", "grade", "delisted_at"])


def _constant_batch(df: pd.DataFrame):
    """모든 종목에 동일한 피처 DataFrame을 배정하는
    `assemble_inference_features_batch` 대역 — horizon마다 재조립되지 않고
    시장당 1회만 호출됨을 가정하는 테스트에서 반환값 형태(`dict[str,
    DataFrame]`)만 맞추면 되는 경우에 사용한다."""

    def _fake(engine, calendar, stock_codes, as_of_date):  # noqa: ANN001
        return {stock_code: df for stock_code in stock_codes}

    return _fake


class _FakeMetrics:
    """`InferenceMetrics()`를 대체하는 스파이 — 프로세스당 인스턴스화 횟수와
    각 record_* 호출을 관측 가능하게 만든다(REQ-APL-106/134/150)."""

    instantiation_count = 0

    def __init__(self, registry: object | None = None) -> None:
        type(self).instantiation_count += 1
        self.skip_calls: list[tuple[str, int, SkipReason | str]] = []
        self.signal_calls: list[tuple[str, int, str]] = []
        self.cycle_duration_calls: list[tuple[str, float]] = []

    def record_skip(self, *, market: str, horizon: int, reason: SkipReason | str) -> None:
        self.skip_calls.append((market, horizon, reason))

    def record_signal(self, *, market: str, horizon: int, signal_class: str) -> None:
        self.signal_calls.append((market, horizon, signal_class))

    def observe_cycle_duration(self, *, market: str, seconds: float) -> None:
        self.cycle_duration_calls.append((market, seconds))


class _FakeBoundariesArtifact:
    def classify(self, market: str, horizon: int, scores, boundary_set=None):  # noqa: ANN001
        return ["BUY" for _ in scores]


def _patch_common(monkeypatch: pytest.MonkeyPatch, *, universe_rows: list[tuple]) -> _FakeMetrics:
    """조립 로직 검증에 필요한 공통 의존성을 패치하고, 관측용 metrics 인스턴스를
    반환한다."""
    fake_engine = MagicMock(name="engine")
    fake_redis = MagicMock(name="redis_client")

    monkeypatch.setattr(pipeline_module, "build_engine", lambda *_a, **_k: fake_engine)
    monkeypatch.setattr(pipeline_module, "build_redis_client", lambda *_a, **_k: fake_redis)
    monkeypatch.setattr(pipeline_module, "get_db_config", lambda: object())
    monkeypatch.setattr(pipeline_module, "get_inference_config", lambda: object())
    monkeypatch.setattr(pipeline_module, "load_grade_boundaries", lambda: _FakeBoundariesArtifact())
    monkeypatch.setattr(pipeline_module, "fetch_market_calendar", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(
        pipeline_module, "fetch_stock_universe", lambda *_a, **_k: _universe_df(universe_rows)
    )

    _FakeMetrics.instantiation_count = 0
    fake_metrics_holder: dict[str, _FakeMetrics] = {}

    def _metrics_factory(*_a, **_k) -> _FakeMetrics:
        instance = _FakeMetrics()
        fake_metrics_holder["instance"] = instance
        return instance

    monkeypatch.setattr(pipeline_module, "InferenceMetrics", _metrics_factory)
    monkeypatch.setattr(pipeline_module, "publish_trading_signal", MagicMock())

    class _Holder:
        def __getattr__(self, name: str):
            return getattr(fake_metrics_holder["instance"], name)

    return _Holder()  # type: ignore[return-value]


class TestRunMarketInferenceCombinationSkip:
    """AC-APL-101: 조합 단위 스킵 — NO_MANIFEST 등으로 서빙 대상 해석이
    실패하면 그 조합의 어떤 종목도 처리하지 않는다."""

    def test_no_manifest_skip_records_exactly_once_and_processes_no_stock(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        metrics = _patch_common(monkeypatch, universe_rows=[(1, "005930", "A", None)])
        monkeypatch.setattr(
            pipeline_module, "resolve_serving_targets", lambda *_a, **_k: SkipReason.NO_MANIFEST
        )
        assemble_spy = MagicMock()
        monkeypatch.setattr(pipeline_module, "assemble_inference_features_batch", assemble_spy)

        outcome = run_market_inference(
            "domestic",
            trace_id="trace-1",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        assert outcome.skipped_combinations == 2  # HORIZONS = (20, 60), 둘 다 스킵
        assert outcome.processed == 0
        assert outcome.partial_failures == 0
        assemble_spy.assert_not_called()
        assert len(metrics.skip_calls) == 2
        assert all(reason == SkipReason.NO_MANIFEST for (_m, _h, reason) in metrics.skip_calls)


class TestRunMarketInferenceStockLevelExceptionBoundary:
    """AC-APL-102: 종목 단위 예외 경계 — 한 종목의 실패가 같은 조합의 다른
    종목 처리를 중단시키지 않는다."""

    def test_feature_insufficient_and_unexpected_error_do_not_abort_the_combination(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        universe_rows = [
            (1, "AAA", "A", None),
            (2, "BBB", "A", None),
            (3, "CCC", "A", None),
        ]
        metrics = _patch_common(monkeypatch, universe_rows=universe_rows)

        serving_plan = _solo_serving_plan("domestic", 20, date(2026, 8, 1))

        def _fake_resolve_serving_targets(models_root, market, horizon):  # noqa: ANN001
            if horizon == 20:
                return serving_plan
            return SkipReason.NO_MANIFEST

        monkeypatch.setattr(
            pipeline_module, "resolve_serving_targets", _fake_resolve_serving_targets
        )
        monkeypatch.setattr(
            pipeline_module,
            "resolve_latest_quantile_manifest",
            lambda *_a, **_k: MagicMock(),
        )

        def _fake_predict_point_models(serving_plan, features):  # noqa: ANN001
            if features is not None and "raise" in features.columns:
                raise ValueError("피처 컬럼 누락")
            return {"xgboost": 0.5}

        monkeypatch.setattr(pipeline_module, "predict_point_models", _fake_predict_point_models)

        def _fake_assemble_batch_with_raise(engine, calendar, stock_codes, as_of_date):  # noqa: ANN001
            results: dict[str, object] = {}
            for stock_code in stock_codes:
                if stock_code == "AAA":
                    results[stock_code] = SkipReason.FEATURE_INSUFFICIENT
                elif stock_code == "BBB":
                    results[stock_code] = pd.DataFrame({"raise": [1.0]})
                else:
                    results[stock_code] = pd.DataFrame({"f1": [1.0]})
            return results

        monkeypatch.setattr(
            pipeline_module, "assemble_inference_features_batch", _fake_assemble_batch_with_raise
        )
        monkeypatch.setattr(pipeline_module, "predict_quantile_models", lambda *_a: (0.1, 0.9))
        monkeypatch.setattr(pipeline_module, "resolve_confidence_for_stock", lambda *_a, **_k: 0.8)
        insert_spy = MagicMock(return_value=InsertOutcome.INSERTED)
        monkeypatch.setattr(pipeline_module, "insert_trading_signal", insert_spy)
        monkeypatch.setattr(
            pipeline_module, "sweep_and_write_price_bands", lambda **_k: {"PROMOTE": "inserted"}
        )

        outcome = run_market_inference(
            "domestic",
            trace_id="trace-2",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        # AAA=FEATURE_INSUFFICIENT, BBB=UNEXPECTED_ERROR(ValueError), CCC=성공
        skip_reasons = {reason for (_m, _h, reason) in metrics.skip_calls}
        assert SkipReason.FEATURE_INSUFFICIENT in skip_reasons
        assert SkipReason.UNEXPECTED_ERROR in skip_reasons
        insert_spy.assert_called_once()  # CCC만 INSERT까지 도달
        assert outcome.partial_failures == 1  # (domestic, 20) 조합은 부분실패
        assert outcome.skipped_combinations == 1  # (domestic, 60)는 NO_MANIFEST


class TestRunMarketInferencePublishAlwaysOnNormalReturn:
    """AC-APL-104: insert_trading_signal()이 예외 없이 반환하면(INSERTED/
    SKIPPED_DUPLICATE 무관) 항상 publish_trading_signal()이 호출된다."""

    @pytest.mark.parametrize(
        "insert_outcome", [InsertOutcome.INSERTED, InsertOutcome.SKIPPED_DUPLICATE]
    )
    def test_publish_is_called_regardless_of_insert_outcome(
        self, monkeypatch: pytest.MonkeyPatch, insert_outcome: str
    ):
        universe_rows = [(1, "AAA", "A", None)]
        _patch_common(monkeypatch, universe_rows=universe_rows)

        serving_plan = _solo_serving_plan("domestic", 20, date(2026, 8, 1))
        monkeypatch.setattr(
            pipeline_module,
            "resolve_serving_targets",
            lambda models_root, market, horizon: (
                serving_plan if horizon == 20 else SkipReason.NO_MANIFEST
            ),
        )
        monkeypatch.setattr(
            pipeline_module, "resolve_latest_quantile_manifest", lambda *_a, **_k: MagicMock()
        )
        monkeypatch.setattr(
            pipeline_module,
            "assemble_inference_features_batch",
            _constant_batch(pd.DataFrame({"f1": [1.0]})),
        )
        monkeypatch.setattr(
            pipeline_module, "predict_point_models", lambda *_a, **_k: {"xgboost": 0.5}
        )
        monkeypatch.setattr(pipeline_module, "predict_quantile_models", lambda *_a: (0.1, 0.9))
        monkeypatch.setattr(pipeline_module, "resolve_confidence_for_stock", lambda *_a, **_k: 0.8)
        monkeypatch.setattr(
            pipeline_module, "insert_trading_signal", lambda *_a, **_k: insert_outcome
        )
        publish_spy = MagicMock()
        monkeypatch.setattr(pipeline_module, "publish_trading_signal", publish_spy)
        monkeypatch.setattr(
            pipeline_module, "sweep_and_write_price_bands", lambda **_k: {"PROMOTE": "inserted"}
        )

        run_market_inference(
            "domestic",
            trace_id="trace-3",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        publish_spy.assert_called_once()


class TestRunMarketInferenceBandSweepDoesNotRollback:
    """AC-APL-105: 밴드 스윕 스킵은 이미 완료된 점 신호 INSERT/발행을
    되돌리지 않는다."""

    def test_manifest_race_during_sweep_keeps_point_signal_committed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        universe_rows = [(1, "AAA", "A", None)]
        metrics = _patch_common(monkeypatch, universe_rows=universe_rows)

        serving_plan = _solo_serving_plan("domestic", 20, date(2026, 8, 1))
        monkeypatch.setattr(
            pipeline_module,
            "resolve_serving_targets",
            lambda models_root, market, horizon: (
                serving_plan if horizon == 20 else SkipReason.NO_MANIFEST
            ),
        )
        monkeypatch.setattr(
            pipeline_module, "resolve_latest_quantile_manifest", lambda *_a, **_k: MagicMock()
        )
        monkeypatch.setattr(
            pipeline_module,
            "assemble_inference_features_batch",
            _constant_batch(pd.DataFrame({"f1": [1.0]})),
        )
        monkeypatch.setattr(
            pipeline_module, "predict_point_models", lambda *_a, **_k: {"xgboost": 0.5}
        )
        monkeypatch.setattr(pipeline_module, "predict_quantile_models", lambda *_a: (0.1, 0.9))
        monkeypatch.setattr(pipeline_module, "resolve_confidence_for_stock", lambda *_a, **_k: 0.8)
        insert_spy = MagicMock(return_value=InsertOutcome.INSERTED)
        monkeypatch.setattr(pipeline_module, "insert_trading_signal", insert_spy)
        publish_spy = MagicMock()
        monkeypatch.setattr(pipeline_module, "publish_trading_signal", publish_spy)
        monkeypatch.setattr(
            pipeline_module, "sweep_and_write_price_bands", lambda **_k: SkipReason.MANIFEST_RACE
        )

        outcome = run_market_inference(
            "domestic",
            trace_id="trace-4",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        insert_spy.assert_called_once()
        publish_spy.assert_called_once()
        assert any(reason == SkipReason.MANIFEST_RACE for (_m, _h, reason) in metrics.skip_calls)
        assert outcome.partial_failures == 1


class TestRunMarketInferenceSignalCounterNoDoubleCount:
    """AC-APL-106: 밴드 스윕 성공이 신호 카운터를 중복 증가시키지 않는다."""

    def test_record_signal_called_exactly_once_per_stock(self, monkeypatch: pytest.MonkeyPatch):
        universe_rows = [(1, "AAA", "A", None)]
        metrics = _patch_common(monkeypatch, universe_rows=universe_rows)

        serving_plan = _solo_serving_plan("domestic", 20, date(2026, 8, 1))
        monkeypatch.setattr(
            pipeline_module,
            "resolve_serving_targets",
            lambda models_root, market, horizon: (
                serving_plan if horizon == 20 else SkipReason.NO_MANIFEST
            ),
        )
        monkeypatch.setattr(
            pipeline_module, "resolve_latest_quantile_manifest", lambda *_a, **_k: MagicMock()
        )
        monkeypatch.setattr(
            pipeline_module,
            "assemble_inference_features_batch",
            _constant_batch(pd.DataFrame({"f1": [1.0]})),
        )
        monkeypatch.setattr(
            pipeline_module, "predict_point_models", lambda *_a, **_k: {"xgboost": 0.5}
        )
        monkeypatch.setattr(pipeline_module, "predict_quantile_models", lambda *_a: (0.1, 0.9))
        monkeypatch.setattr(pipeline_module, "resolve_confidence_for_stock", lambda *_a, **_k: 0.8)
        monkeypatch.setattr(
            pipeline_module, "insert_trading_signal", lambda *_a, **_k: InsertOutcome.INSERTED
        )
        monkeypatch.setattr(pipeline_module, "publish_trading_signal", MagicMock())
        monkeypatch.setattr(
            pipeline_module,
            "sweep_and_write_price_bands",
            lambda **_k: {"PROMOTE": "inserted", "DEMOTE": "inserted"},
        )

        run_market_inference(
            "domestic",
            trace_id="trace-5",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        assert len(metrics.signal_calls) == 1


class TestRunMarketInferenceUniverseFiltering:
    """AC-APL-107: grade IN ('A','B') AND delisted_at IS NULL만 처리 대상 —
    시장당 1회만 조회한다."""

    def test_filters_out_low_grade_and_delisted_and_fetches_universe_once(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        universe_rows = [
            (1, "A1", "A", None),
            (2, "A2", "B", None),
            (3, "C1", "C", None),
            (4, "D1", "A", date(2020, 1, 1)),
        ]
        _patch_common(monkeypatch, universe_rows=universe_rows)
        fetch_spy = MagicMock(return_value=_universe_df(universe_rows))
        monkeypatch.setattr(pipeline_module, "fetch_stock_universe", fetch_spy)

        serving_plan = _solo_serving_plan("domestic", 20, date(2026, 8, 1))
        serving_plan_60 = _solo_serving_plan("domestic", 60, date(2026, 8, 1))
        monkeypatch.setattr(
            pipeline_module,
            "resolve_serving_targets",
            lambda models_root, market, horizon: serving_plan if horizon == 20 else serving_plan_60,
        )
        monkeypatch.setattr(
            pipeline_module, "resolve_latest_quantile_manifest", lambda *_a, **_k: MagicMock()
        )
        batch_spy = MagicMock(
            side_effect=lambda engine, calendar, stock_codes, as_of_date: {
                stock_code: pd.DataFrame({"f1": [1.0]}) for stock_code in stock_codes
            }
        )
        monkeypatch.setattr(pipeline_module, "assemble_inference_features_batch", batch_spy)
        monkeypatch.setattr(
            pipeline_module, "predict_point_models", lambda *_a, **_k: {"xgboost": 0.5}
        )
        monkeypatch.setattr(pipeline_module, "predict_quantile_models", lambda *_a: (0.1, 0.9))
        monkeypatch.setattr(pipeline_module, "resolve_confidence_for_stock", lambda *_a, **_k: 0.8)
        monkeypatch.setattr(
            pipeline_module, "insert_trading_signal", lambda *_a, **_k: InsertOutcome.INSERTED
        )
        monkeypatch.setattr(pipeline_module, "publish_trading_signal", MagicMock())
        monkeypatch.setattr(
            pipeline_module, "sweep_and_write_price_bands", lambda **_k: {"PROMOTE": "inserted"}
        )

        run_market_inference(
            "domestic",
            trace_id="trace-6",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        # A1/A2만 처리(grade A/B, 상장유지)
        assert fetch_spy.call_count == 1  # 조합마다 반복되지 않고 시장당 1회
        # W1: 배치 피처 조립도 horizon(20,60)마다가 아니라 시장당 정확히 1회만
        # 호출되고, 그 1회에 필터링된 유니버스(A1/A2) 전체가 전달된다.
        batch_spy.assert_called_once()
        assert set(batch_spy.call_args.args[2]) == {"A1", "A2"}


class TestRunMarketInferenceMetricsLifecycle:
    """AC-APL-134/150: InferenceMetrics는 프로세스당 정확히 1회 생성되고,
    observe_cycle_duration은 정확히 1회만 호출된다."""

    def test_single_metrics_instance_and_single_cycle_observation(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        universe_rows: list[tuple] = []
        metrics = _patch_common(monkeypatch, universe_rows=universe_rows)
        monkeypatch.setattr(
            pipeline_module,
            "resolve_serving_targets",
            lambda *_a, **_k: SkipReason.NO_MANIFEST,
        )

        run_market_inference(
            "domestic",
            trace_id="trace-7",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        assert _FakeMetrics.instantiation_count == 1
        assert len(metrics.cycle_duration_calls) == 1
        assert metrics.cycle_duration_calls[0][0] == "domestic"


class TestRunMarketInferenceResourceCleanup:
    """plan.md §B 리스크 8: engine/redis client를 프로세스 종료 전 명시적으로
    정리한다."""

    def test_engine_dispose_and_redis_close_are_called(self, monkeypatch: pytest.MonkeyPatch):
        fake_engine = MagicMock(name="engine")
        fake_redis = MagicMock(name="redis_client")
        monkeypatch.setattr(pipeline_module, "build_engine", lambda *_a, **_k: fake_engine)
        monkeypatch.setattr(pipeline_module, "build_redis_client", lambda *_a, **_k: fake_redis)
        monkeypatch.setattr(pipeline_module, "get_db_config", lambda: object())
        monkeypatch.setattr(pipeline_module, "get_inference_config", lambda: object())
        monkeypatch.setattr(
            pipeline_module, "load_grade_boundaries", lambda: _FakeBoundariesArtifact()
        )
        monkeypatch.setattr(pipeline_module, "fetch_market_calendar", lambda *_a, **_k: MagicMock())
        monkeypatch.setattr(
            pipeline_module, "fetch_stock_universe", lambda *_a, **_k: _universe_df([])
        )
        monkeypatch.setattr(pipeline_module, "InferenceMetrics", lambda *_a, **_k: _FakeMetrics())
        monkeypatch.setattr(
            pipeline_module, "resolve_serving_targets", lambda *_a, **_k: SkipReason.NO_MANIFEST
        )

        run_market_inference(
            "domestic",
            trace_id="trace-8",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        fake_engine.dispose.assert_called_once()
        fake_redis.close.assert_called_once()


class TestAaaInfraIssue163Containment:
    """AC-APL-140: overseas D60 챔피언에 `.meta.json`이 없어 FEATURE_REGISTRY
    전체로 폴백하고 investor_trend가 빈 결과라 `ValueError`가 발생해도
    프로세스가 크래시하지 않는다 — 그 종목만 UNEXPECTED_ERROR로 스킵되고
    자식은 exit code 0 또는 2로 정상 종료한다(REQ-APL-140/102/103)."""

    def test_predict_valueerror_from_missing_meta_json_is_contained(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        universe_rows = [(1, "OVERSEAS_A", "A", None)]
        metrics = _patch_common(monkeypatch, universe_rows=universe_rows)

        serving_plan = _solo_serving_plan("overseas", 60, date(2026, 8, 1))
        monkeypatch.setattr(
            pipeline_module,
            "resolve_serving_targets",
            lambda models_root, market, horizon: (
                serving_plan if horizon == 60 else SkipReason.NO_MANIFEST
            ),
        )
        monkeypatch.setattr(
            pipeline_module, "resolve_latest_quantile_manifest", lambda *_a, **_k: MagicMock()
        )
        monkeypatch.setattr(
            pipeline_module,
            "assemble_inference_features_batch",
            _constant_batch(pd.DataFrame({"f1": [1.0]})),
        )

        def _raise_missing_feature_columns(*_a, **_k):
            # aaa-infra#163 증상 경로: .meta.json 부재 → FEATURE_REGISTRY 전체
            # 폴백 → predict.py의 컬럼 선택이 investor_trend 부재 컬럼 누락을
            # ValueError로 던짐(research.md §6).
            raise ValueError("예측에 필요한 피처 컬럼이 누락되었다: ['foreign_net_value']")

        monkeypatch.setattr(pipeline_module, "predict_point_models", _raise_missing_feature_columns)

        outcome = run_market_inference(
            "overseas",
            trace_id="trace-163",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        assert SkipReason.UNEXPECTED_ERROR in {reason for (_m, _h, reason) in metrics.skip_calls}
        # 크래시 없이 정상 종료 — exit code 0(다른 조합 성공) 또는 2(부분실패).
        from analyzer.inference.outcome import resolve_exit_code

        assert resolve_exit_code(outcome) in (0, 2)
        assert outcome.partial_failures == 1

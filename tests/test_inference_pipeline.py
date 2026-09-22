"""추론 오케스트레이션 계층 명세 테스트 (SPEC-ANALYZER-PIPELINE-001,
REQ-APL-100~107/110/111/121).

`inference/pipeline.py`는 INFER-001이 이미 완성한 9개 순수 모듈을 실제
흐름으로 조립하는 오케스트레이션 계층이다 — 이 테스트는 그 모듈들을 mock
주입 가능한 콜러블로 대체해 조립 로직(조합 단위 스킵, 종목 단위 예외 경계,
INSERT→발행→밴드 스윕 순서 계약)만 검증한다.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from analyzer.data.models import TradingCalendar
from analyzer.inference import pipeline as pipeline_module
from analyzer.inference.pipeline import (
    MARKET_CALENDAR_CODE,
    resolve_calendar_code,
    resolve_model_version,
    run_market_inference,
)
from analyzer.inference.resolution import ServingPlan, SkipReason
from analyzer.inference.sweep import _union_feature_columns
from analyzer.inference.writer import InsertOutcome


def _weekdays(start: date, end: date) -> list[date]:
    """`tests/test_inference_features.py`의 동명 헬퍼와 동일한 요일 필터 —
    실제 DB 접속 없이 `assemble_inference_features_batch()`의 실제 조립
    경로(모킹하지 않음)를 태우는 M7b 테스트 전용으로 이 파일에 독립
    복제한다(cross-test-file import 결합 회피)."""
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _calendar(start: date, end: date) -> TradingCalendar:
    return TradingCalendar(calendar_code="TEST", trading_days=frozenset(_weekdays(start, end)))


def _ohlcv(stock_code: str, dates: list[date]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "stock_code": [stock_code] * n,
            "trade_date": dates,
            "open_price": [100.0 + i * 0.1 for i in range(n)],
            "high_price": [101.0 + i * 0.1 for i in range(n)],
            "low_price": [99.0 + i * 0.1 for i in range(n)],
            "close_price": [100.5 + i * 0.1 for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        }
    )


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


def _empty_trend() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "stock_code",
            "trade_date",
            "foreign_net_value",
            "institution_net_value",
            "individual_net_value",
            "total_trading_value",
        ]
    )


def _trend(stock_code: str, dates: list[date]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "stock_code": [stock_code] * n,
            "trade_date": dates,
            "foreign_net_value": [1_000_000 + i * 100 for i in range(n)],
            "institution_net_value": [-500_000 + i * 50 for i in range(n)],
            "individual_net_value": [-500_000 - i * 50 for i in range(n)],
            "total_trading_value": [10_000_000 + i * 1000 for i in range(n)],
        }
    )


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

    def _fake(engine, calendar, stock_codes, as_of_date, **_kwargs):  # noqa: ANN001
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
        self.last_cycle_calls: list[tuple[str, float]] = []

    def record_skip(self, *, market: str, horizon: int, reason: SkipReason | str) -> None:
        self.skip_calls.append((market, horizon, reason))

    def record_signal(self, *, market: str, horizon: int, signal_class: str) -> None:
        self.signal_calls.append((market, horizon, signal_class))

    def observe_cycle_duration(self, *, market: str, seconds: float) -> None:
        self.cycle_duration_calls.append((market, seconds))

    def record_last_cycle(self, *, market: str, epoch_seconds: float) -> None:
        self.last_cycle_calls.append((market, epoch_seconds))


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

        def _fake_assemble_batch_with_raise(  # noqa: ANN001
            engine, calendar, stock_codes, as_of_date, **_kwargs
        ):
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
            side_effect=lambda engine, calendar, stock_codes, as_of_date, **_kwargs: {
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


class TestRunMarketInferenceLastCycleWiring:
    """SPEC-OBSV-ANALYZER-DEADMAN-001 REQ-DMR-002: 사이클 완료 시
    `record_cycle_completion()`이 정확히 1회 호출되어 게이지 갱신 + Redis
    영속화가 함께 수행된다 — `observe_cycle_duration()`과 동일한 시점(사이클
    종료 직후, `finally` 이전)에 배선된다."""

    def test_record_last_cycle_called_once_after_cycle_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        metrics = _patch_common(monkeypatch, universe_rows=[])
        monkeypatch.setattr(
            pipeline_module,
            "resolve_serving_targets",
            lambda *_a, **_k: SkipReason.NO_MANIFEST,
        )

        before = time.time()
        run_market_inference(
            "domestic",
            trace_id="trace-lastcycle-1",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )
        after = time.time()

        assert len(metrics.last_cycle_calls) == 1
        market, epoch_seconds = metrics.last_cycle_calls[0]
        assert market == "domestic"
        assert before <= epoch_seconds <= after

    def test_markets_are_recorded_independently(self, monkeypatch: pytest.MonkeyPatch):
        metrics = _patch_common(monkeypatch, universe_rows=[])
        monkeypatch.setattr(
            pipeline_module,
            "resolve_serving_targets",
            lambda *_a, **_k: SkipReason.NO_MANIFEST,
        )

        run_market_inference(
            "overseas",
            trace_id="trace-lastcycle-2",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        assert len(metrics.last_cycle_calls) == 1
        assert metrics.last_cycle_calls[0][0] == "overseas"

    def test_cycle_still_completes_when_redis_save_raises(self, monkeypatch: pytest.MonkeyPatch):
        """REQ-DMR-004 fail-open을 파이프라인 경유로 재확인: `redis_client`가
        `RedisError`를 던져도 사이클 자체는 정상 완주(예외 미전파)한다."""
        import redis.exceptions

        fake_engine = MagicMock(name="engine")

        class _FailingRedis:
            def set(self, *_a, **_k):
                raise redis.exceptions.ConnectionError("연결 불가(시뮬레이션)")

            def get(self, *_a, **_k):
                raise redis.exceptions.ConnectionError("연결 불가(시뮬레이션)")

            def close(self) -> None:
                pass

        fake_redis = _FailingRedis()
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

        outcome = run_market_inference(
            "domestic",
            trace_id="trace-lastcycle-3",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        assert outcome.skipped_combinations == 2


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


class TestRunMarketInferencePassesFeatureColumnsToBatchAssembly:
    """SPEC-ANALYZER-TRAIN-META-001 M7b(REQ-TM-011, AC-TM-010/AC-TM-010b):
    호출부(`run_market_inference()`)가 M7이 `assemble_inference_features_
    batch()`에 심어둔 opt-in `feature_columns` 가드를 실제로 활성화해야
    한다 — M7의 함수 단위 테스트(`tests/test_inference_features.py`)만으로는
    이 호출부 배선 누락을 검출할 수 없다(M7 스스로가 "함수 단위 방어는
    맞았으나 pipeline.py 호출부는 손대지 않았다"는 블로커를 반환한 이유)."""

    def test_ac_tm_010_frozen_gap_stock_is_feature_insufficient_not_unexpected_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Given: investor_trend가 비어 있고, 서빙 모델에 `.meta.json`
        사이드카가 없어 `FEATURE_REGISTRY`(FROZEN 수급 컬럼 포함) 전체로
        폴백하는 종목 1건.
        When: `run_market_inference()`를 실행한다 — `assemble_inference_
        features_batch()`/`predict_point_models()` 둘 다 모킹하지 않고
        실제 조립·예측 경로를 그대로 태운다.
        Then(수정 전=RED): 호출부가 `feature_columns`를 넘기지 않아 M7
        가드가 비활성 상태로 남고, `predict.py::_select_feature_columns()`의
        무방비 `ValueError`까지 도달해 `SkipReason.UNEXPECTED_ERROR`로
        오분류된다.
        Then(수정 후=GREEN): `feature_columns=_union_feature_columns(
        serving_plan)`가 실전달되어 `has_supply_demand_gap()`이 조기
        판별하고, `SkipReason.FEATURE_INSUFFICIENT`로 정확히 분류되며
        `assemble_inference_features()`(따라서 `fetch_daily_ohlcv`)에는
        도달조차 하지 않는다."""
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        trade_date = dates[-1]
        universe_rows = [(1, "GAP1", "A", None)]
        metrics = _patch_common(monkeypatch, universe_rows=universe_rows)
        monkeypatch.setattr(
            pipeline_module,
            "fetch_market_calendar",
            lambda *_a, **_k: _calendar(date(2025, 12, 1), date(2026, 4, 10)),
        )

        # 사이드카 없는 모델 경로 → resolve_feature_columns()가
        # FEATURE_REGISTRY(FROZEN 컬럼 포함) 전체로 폴백한다.
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

        ohlcv_fetch_calls: list[str] = []

        def _tracking_fetch_daily_ohlcv(_engine, stock_code, **_kwargs):  # noqa: ANN001
            ohlcv_fetch_calls.append(stock_code)
            return _ohlcv(stock_code, dates)

        monkeypatch.setattr(
            "analyzer.inference.features.fetch_daily_ohlcv", _tracking_fetch_daily_ohlcv
        )
        monkeypatch.setattr(
            "analyzer.inference.features.fetch_corporate_events",
            lambda *_a, **_k: _empty_events(),
        )
        monkeypatch.setattr(
            "analyzer.inference.features.fetch_investor_trend",
            lambda *_a, **_k: _empty_trend(),
        )
        # predict_point_models/predict_quantile_models은 의도적으로 모킹하지
        # 않는다 — 실제 predict.py의 무방비 ValueError 호출부가 방어되지
        # 않으면 이 테스트가 RED로 실패해야 한다.

        outcome = run_market_inference(
            "domestic",
            trace_id="trace-tm010",
            models_root=Path("/models"),
            trade_date=trade_date,
        )

        skip_reasons = {reason for (_m, _h, reason) in metrics.skip_calls}
        assert SkipReason.FEATURE_INSUFFICIENT in skip_reasons, (
            f"GAP1이 FEATURE_INSUFFICIENT로 분류되지 않았다 — 실제 skip_calls={metrics.skip_calls}"
        )
        assert SkipReason.UNEXPECTED_ERROR not in skip_reasons
        # M7 가드가 조기에 스킵했다면 assemble_inference_features()(따라서
        # fetch_daily_ohlcv)에는 도달조차 하지 않아야 한다.
        assert ohlcv_fetch_calls == []
        assert outcome.skipped_combinations == 1  # (domestic, 60)는 NO_MANIFEST
        assert outcome.partial_failures == 1  # (domestic, 20)는 GAP1 스킵으로 부분실패

    def test_call_site_passes_union_feature_columns_of_the_active_serving_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """호출부가 `_union_feature_columns(serving_plan)`을 그대로
        `feature_columns` 키워드 인자로 전달하는지 직접 검증한다 — 이전
        루프 반복에서 남은 낡은(stale) `serving_plan`이 아니라 현재
        (market, horizon) 조합에서 막 해석된 것이어야 한다."""
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
        batch_spy = MagicMock(return_value={"AAA": pd.DataFrame({"f1": [1.0]})})
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
            trace_id="trace-tm010-wiring",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        batch_spy.assert_called_once()
        expected_columns = _union_feature_columns(serving_plan)
        assert batch_spy.call_args.kwargs.get("feature_columns") == expected_columns
        assert len(expected_columns) > 0  # FEATURE_REGISTRY 폴백이 공집합이 아님을 확인

    def test_ac_tm_010b_unrelated_missing_column_still_surfaces_as_unexpected_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """AC-TM-010b 회귀 가드: `feature_columns` 실배선이 `investor_trend`
        결측과 무관한 진짜 컬럼 누락(REQ-APL-140 일반 예외 경계가 원래
        처리하던 사례)까지 삼켜버리지 않아야 한다(shall not) — 새 조기
        스킵 가드는 `has_supply_demand_gap()`의 FROZEN+trend.empty 조건에만
        정확히 스코프돼 있으므로, investor_trend가 존재하는 종목에서
        발생한 무관한 예외는 여전히 기존 `except Exception:` 일반 경계를
        통해 `SkipReason.UNEXPECTED_ERROR`로 도달해야 한다."""
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
        # investor_trend와 무관한 정상 피처 DataFrame — has_supply_demand_gap()
        # 조기 스킵 가드가 절대 트리거되지 않는(비-FROZEN 결측) 시나리오다.
        monkeypatch.setattr(
            pipeline_module,
            "assemble_inference_features_batch",
            _constant_batch(pd.DataFrame({"f1": [1.0]})),
        )

        def _raise_unrelated_valueerror(*_a, **_k):
            raise ValueError("예측에 필요한 피처 컬럼이 누락되었다: ['some_unrelated_col']")

        monkeypatch.setattr(pipeline_module, "predict_point_models", _raise_unrelated_valueerror)

        outcome = run_market_inference(
            "domestic",
            trace_id="trace-tm010b",
            models_root=Path("/models"),
            trade_date=date(2026, 9, 10),
        )

        skip_reasons = {reason for (_m, _h, reason) in metrics.skip_calls}
        assert SkipReason.UNEXPECTED_ERROR in skip_reasons
        assert SkipReason.FEATURE_INSUFFICIENT not in skip_reasons
        assert outcome.partial_failures == 1

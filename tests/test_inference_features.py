"""61거래일 룩백 피처 조립기 + feature_columns 해석 테스트
(SPEC-ANALYZER-INFER-001 M5, REQ-AIF-070/071, design.md §1).

REQ-AIF-070(AC-AIF-012)/REQ-AIF-071(AC-AIF-013)/REQ-AIF-060 전반부
(AC-AIF-011 종목 단위 스킵 시나리오)를 검증한다. 실 DB 접속 없이
`analyzer.data.repository`의 fetch 함수를 모킹한 단위 테스트다.
"""

import json
import subprocess
import sys
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from analyzer.data.models import TradingCalendar
from analyzer.features.classification import FEATURE_REGISTRY
from analyzer.inference.features import (
    LOOKBACK_TRADING_DAYS,
    assemble_inference_features,
    assemble_inference_features_batch,
    has_supply_demand_gap,
    resolve_feature_columns,
)
from analyzer.inference.resolution import SkipReason


def _weekdays(start: date, end: date) -> list[date]:
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


class TestAssembleInferenceFeaturesSufficientHistory:
    """REQ-AIF-070: 61거래일 이상 이력이 있는 종목은 마지막 1행의 피처가
    조립되고 원본 함수(compute_technical_features)를 재사용한다."""

    def test_returns_single_row_with_technical_features(self):
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        assert len(dates) >= LOOKBACK_TRADING_DAYS
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()

        with (
            patch(
                "analyzer.inference.features.fetch_daily_ohlcv", return_value=_ohlcv("A1", dates)
            ),
            patch(
                "analyzer.inference.features.fetch_corporate_events", return_value=_empty_events()
            ),
            patch("analyzer.inference.features.fetch_investor_trend", return_value=_empty_trend()),
        ):
            result = assemble_inference_features(engine, calendar, "A1", as_of_date)

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["trade_date"] == as_of_date
        assert "ROC_60" in result.columns
        assert pd.notna(result.iloc[0]["ROC_60"])

    def test_merges_supply_demand_features_when_investor_trend_present(self):
        """REQ-AIF-070: 수급 데이터가 존재하면 `compute_supply_demand_features()`
        결과가 기술적 피처와 병합돼야 한다(FEATURE-001 재사용, dataset.py와
        동일한 조건부 병합 규칙)."""
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()

        with (
            patch(
                "analyzer.inference.features.fetch_daily_ohlcv", return_value=_ohlcv("A1", dates)
            ),
            patch(
                "analyzer.inference.features.fetch_corporate_events", return_value=_empty_events()
            ),
            patch(
                "analyzer.inference.features.fetch_investor_trend",
                return_value=_trend("A1", dates),
            ),
        ):
            result = assemble_inference_features(engine, calendar, "A1", as_of_date)

        assert result is not None
        assert len(result) == 1
        assert "foreign_net_ratio" in result.columns
        assert "institution_net_cum_60" in result.columns
        assert pd.notna(result.iloc[0]["foreign_net_ratio"])


class TestAssembleInferenceFeaturesInsufficientHistory:
    """AC-AIF-011 첫 시나리오: 이력이 61거래일 미만인 종목은 None(호출자가
    FEATURE_INSUFFICIENT로 라우팅)을 반환해야 하며(shall), 예외를 던지지
    않아야 한다(shall not)."""

    def test_short_history_returns_none(self):
        dates = _weekdays(date(2026, 3, 1), date(2026, 3, 20))  # < 61 거래일
        as_of_date = dates[-1]
        calendar = _calendar(date(2026, 1, 1), date(2026, 4, 1))
        engine = MagicMock()

        with (
            patch(
                "analyzer.inference.features.fetch_daily_ohlcv", return_value=_ohlcv("NEW1", dates)
            ),
            patch(
                "analyzer.inference.features.fetch_corporate_events", return_value=_empty_events()
            ),
            patch("analyzer.inference.features.fetch_investor_trend", return_value=_empty_trend()),
        ):
            result = assemble_inference_features(engine, calendar, "NEW1", as_of_date)

        assert result is None


class TestAssembleInferenceFeaturesBatchPartialFailure:
    """AC-AIF-011: 종목 10개 중 1개가 피처 결측(이력 <61거래일)이면 그
    종목만 스킵되고 나머지 9개는 정상 처리돼야 한다(shall) — 조합 전체를
    abort시키지 않는다(shall not, REQ-AIF-060 전반부)."""

    def test_one_insufficient_stock_does_not_abort_batch(self):
        long_dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        short_dates = _weekdays(date(2026, 3, 1), date(2026, 3, 20))
        as_of_date = long_dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()

        healthy_codes = [f"H{i}" for i in range(9)]
        stock_codes = [*healthy_codes, "SHORT1"]

        def _fake_fetch_daily_ohlcv(_engine, stock_code, start_date=None, end_date=None):
            if stock_code == "SHORT1":
                return _ohlcv(stock_code, short_dates)
            return _ohlcv(stock_code, long_dates)

        with (
            patch(
                "analyzer.inference.features.fetch_daily_ohlcv",
                side_effect=_fake_fetch_daily_ohlcv,
            ),
            patch(
                "analyzer.inference.features.fetch_corporate_events", return_value=_empty_events()
            ),
            patch("analyzer.inference.features.fetch_investor_trend", return_value=_empty_trend()),
        ):
            results = assemble_inference_features_batch(engine, calendar, stock_codes, as_of_date)

        assert results["SHORT1"] is SkipReason.FEATURE_INSUFFICIENT
        for code in healthy_codes:
            assert isinstance(results[code], pd.DataFrame)
            assert len(results[code]) == 1


class TestAssembleInferenceFeaturesBatchSupplyDemandGap:
    """SPEC-ANALYZER-TRAIN-META-001 M7(REQ-TM-011, acceptance.md AC-TM-010/
    AC-TM-010b): `assemble_inference_features_batch()`가 종목별로 M2의
    `has_supply_demand_gap()`을 호출해 investor_trend 결측 + FROZEN 컬럼
    요구 조합을 `SkipReason.FEATURE_INSUFFICIENT`로 조기 분류해야 한다.

    `feature_columns` 키워드 인자가 생략되면(기본값 `None`) 이 가드는
    완전히 비활성화되고 기존 동작을 그대로 유지한다(REQ-TM-006 방향의
    회귀 최소화 원칙 — `TestAssembleInferenceFeaturesBatchPartialFailure`가
    이 회귀 가드다)."""

    def test_ac_tm_010_frozen_required_and_trend_empty_is_feature_insufficient(self):
        """AC-TM-010: `feature_columns`에 FROZEN 수급 컬럼이 포함돼 있고
        해당 종목의 investor_trend가 비어 있으면, 실제 피처 조립을
        시도하지 않고 매핑 값을 `SkipReason.FEATURE_INSUFFICIENT`로
        기록해야 한다(shall)."""
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()

        with (
            patch(
                "analyzer.inference.features.fetch_daily_ohlcv", return_value=_ohlcv("OVS1", dates)
            ),
            patch(
                "analyzer.inference.features.fetch_corporate_events", return_value=_empty_events()
            ),
            patch("analyzer.inference.features.fetch_investor_trend", return_value=_empty_trend()),
        ):
            results = assemble_inference_features_batch(
                engine,
                calendar,
                ["OVS1"],
                as_of_date,
                feature_columns=["foreign_net_ratio", "ROC_60"],
            )

        assert results["OVS1"] is SkipReason.FEATURE_INSUFFICIENT

    def test_ac_tm_010b_frozen_not_required_proceeds_to_normal_assembly(self):
        """AC-TM-010b 취지(회귀 가드): `feature_columns`에 FROZEN 컬럼이
        없으면(예: 사이드카가 해외 유니버스에 맞춰 이미 FROZEN을 제외한
        경우) investor_trend 결측과 무관하게 정상 조립 경로로 진행해야
        한다(shall not — 갭 가드가 이 경로를 가로채지 않는다)."""
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()

        with (
            patch(
                "analyzer.inference.features.fetch_daily_ohlcv", return_value=_ohlcv("OVS2", dates)
            ),
            patch(
                "analyzer.inference.features.fetch_corporate_events", return_value=_empty_events()
            ),
            patch("analyzer.inference.features.fetch_investor_trend", return_value=_empty_trend()),
        ):
            results = assemble_inference_features_batch(
                engine,
                calendar,
                ["OVS2"],
                as_of_date,
                feature_columns=["ROC_60", "KMID"],
            )

        assert isinstance(results["OVS2"], pd.DataFrame)
        assert len(results["OVS2"]) == 1

    def test_trend_present_proceeds_to_normal_assembly_even_with_frozen_columns(self):
        """FROZEN 컬럼이 필요해도 investor_trend가 실제로 존재하면 갭이
        아니므로(shall not) 정상 조립 경로로 진행해야 한다 — 도메스틱
        종목의 일반적인 케이스."""
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()

        with (
            patch(
                "analyzer.inference.features.fetch_daily_ohlcv", return_value=_ohlcv("DOM1", dates)
            ),
            patch(
                "analyzer.inference.features.fetch_corporate_events", return_value=_empty_events()
            ),
            patch(
                "analyzer.inference.features.fetch_investor_trend",
                return_value=_trend("DOM1", dates),
            ),
        ):
            results = assemble_inference_features_batch(
                engine,
                calendar,
                ["DOM1"],
                as_of_date,
                feature_columns=["foreign_net_ratio", "ROC_60"],
            )

        assert isinstance(results["DOM1"], pd.DataFrame)
        assert len(results["DOM1"]) == 1
        assert "foreign_net_ratio" in results["DOM1"].columns

    def test_feature_columns_omitted_keeps_gate_disabled_regardless_of_trend(self):
        """`feature_columns`가 생략되면(기본값 `None`) investor_trend가
        비어 있어도 갭 가드가 아예 비활성화돼 기존 동작(정상 조립 시도)을
        그대로 유지해야 한다(shall not skip) — REQ-TM-006 방향의 회귀
        최소화 원칙과 동일 취지."""
        dates = _weekdays(date(2026, 1, 1), date(2026, 4, 1))
        as_of_date = dates[-1]
        calendar = _calendar(date(2025, 12, 1), date(2026, 4, 10))
        engine = MagicMock()

        with (
            patch(
                "analyzer.inference.features.fetch_daily_ohlcv", return_value=_ohlcv("OVS3", dates)
            ),
            patch(
                "analyzer.inference.features.fetch_corporate_events", return_value=_empty_events()
            ),
            patch("analyzer.inference.features.fetch_investor_trend", return_value=_empty_trend()),
        ):
            results = assemble_inference_features_batch(engine, calendar, ["OVS3"], as_of_date)

        assert isinstance(results["OVS3"], pd.DataFrame)
        assert len(results["OVS3"]) == 1


class TestResolveFeatureColumns:
    """REQ-AIF-071(AC-AIF-013): 챔피언의 `.meta.json.feature_columns`를
    읽고, 사이드카가 없으면 FEATURE_REGISTRY 전체로 폴백한다."""

    def test_reads_feature_columns_from_meta_json_sidecar(self, tmp_path):
        model_path = tmp_path / "domestic_20_xgboost_2026-08-19.json"
        model_path.write_bytes(b"dummy")
        sidecar_path = model_path.with_suffix(model_path.suffix + ".meta.json")
        sidecar_path.write_text(
            json.dumps({"feature_columns": ["KMID", "ROC_5", "foreign_net_ratio"]}),
            encoding="utf-8",
        )

        result = resolve_feature_columns(model_path)

        assert result == ["KMID", "ROC_5", "foreign_net_ratio"]

    def test_falls_back_to_feature_registry_when_sidecar_missing(self, tmp_path):
        """AC-AIF-013 worked example: `.meta.json`이 부재한 합성 챔피언
        디렉토리 — 예외로 실패하지 않고(shall not) FEATURE_REGISTRY
        40개 전체가 사용돼야 한다(shall)."""
        model_path = tmp_path / "domestic_20_xgboost_2026-08-19.json"
        model_path.write_bytes(b"dummy")

        result = resolve_feature_columns(model_path)

        assert set(result) == set(FEATURE_REGISTRY)
        assert len(result) == 40


class TestHasSupplyDemandGap:
    """SPEC-ANALYZER-TRAIN-META-001 M2(REQ-TM-007/008, plan.md §E M2,
    research.md §6-3): 단일종목 스코어링 경로에도 `inference/sweep.py::
    assemble_sweep_feature_matrix()`와 동일한 조기 판별(FROZEN 컬럼 요구
    + investor_trend 결측)을 제공하는 얇은 판별 함수.

    `assemble_inference_features()`의 기존 반환 계약(`DataFrame | None`,
    M5)은 무수정 — 이 함수는 호출자가 그 계약과는 별개로 사전에 확인하는
    용도다(판별만 하고 라우팅은 호출자 책임)."""

    def test_frozen_required_and_trend_empty_returns_true(self):
        """해외 종목 시나리오(research.md §6-3) — FROZEN 수급 피처가
        feature_columns에 포함되어 있는데 investor_trend가 비어 있으면
        갭이 있다고 판별해야 한다(shall)."""
        result = has_supply_demand_gap(["foreign_net_ratio", "ROC_60"], _empty_trend())

        assert result is True

    def test_trend_present_returns_false(self):
        """FROZEN 컬럼이 필요해도 investor_trend가 존재하면 갭이 아니다
        (shall not) — 도메스틱 종목처럼 정상적으로 수급 데이터가 있는
        경우 조기 스킵을 걸지 않아야 한다."""
        dates = _weekdays(date(2026, 1, 1), date(2026, 1, 10))
        result = has_supply_demand_gap(["foreign_net_ratio"], _trend("A1", dates))

        assert result is False

    def test_frozen_not_required_and_trend_empty_returns_false(self):
        """FROZEN 컬럼이 애초에 feature_columns에 없으면(예: 사이드카가
        해외 유니버스에 맞춰 FROZEN 컬럼을 이미 제외한 경우) investor_trend가
        비어 있어도 갭이 아니다(shall not) — 스윕 경로(`_frozen_columns`
        비어있으면 investor_trend 조회 자체를 건너뛰는 동작)와 동일한
        판별 결과를 내야 한다."""
        result = has_supply_demand_gap(["ROC_60", "KMID"], _empty_trend())

        assert result is False


class TestFeatureAssemblyRegistryGuard:
    """AC-AIF-012: `inference/features.py`만 임포트한 새 Python 프로세스에서
    `analyzer.data.adjustment.HANDLER_REGISTRY`를 조회하면 SPLIT/DIVIDEND
    키가 모두 존재해야 한다(TRAIN-001 AC-AT-003과 동일 가드, 신규
    진입점에서 재검증)."""

    def test_ac_aif_012_handler_registry_populated_after_features_import_fresh_process(self):
        script = (
            "import analyzer.inference.features\n"
            "from analyzer.data.adjustment import HANDLER_REGISTRY\n"
            "assert HANDLER_REGISTRY, 'HANDLER_REGISTRY must not be empty'\n"
            "assert 'SPLIT' in HANDLER_REGISTRY, 'SPLIT handler missing'\n"
            "assert 'DIVIDEND' in HANDLER_REGISTRY, 'DIVIDEND handler missing'\n"
            "print('OK')\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "OK"

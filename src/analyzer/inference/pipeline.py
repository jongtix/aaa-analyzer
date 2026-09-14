"""추론 파이프라인 오케스트레이션 계층 (SPEC-ANALYZER-PIPELINE-001,
REQ-APL-100~107/110/111/121, design.md §1/§5).

SPEC-ANALYZER-INFER-001(M1~M9)이 남긴 9개 완성 순수 모듈(`resolution.py`/
`predict.py`/`scoring.py`/`features.py`/`boundaries_store.py`/`writer.py`/
`sweep.py`/`publish.py`/`metrics.py`)을 실제 종단간 흐름으로 조립한다 —
이 모듈은 그 함수들의 시그니처·내부 로직을 전혀 수정하지 않는다(PRESERVE).

이 모듈이 신설하는 것은 오케스트레이션 결정뿐이다: 어떤 순서로, 어떤 예외
경계로, 어떤 리소스 수명으로 호출하는가(design.md §1).
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

from analyzer.common.logging import get_logger
from analyzer.data.config import get_db_config
from analyzer.data.repository import (
    build_engine,
    fetch_market_calendar,
    fetch_stock_universe,
)
from analyzer.inference.boundaries_store import load_grade_boundaries
from analyzer.inference.config import get_inference_config
from analyzer.inference.features import assemble_inference_features
from analyzer.inference.metrics import InferenceMetrics
from analyzer.inference.outcome import InferenceOutcome
from analyzer.inference.predict import predict_point_models, predict_quantile_models
from analyzer.inference.publish import publish_trading_signal
from analyzer.inference.redis_client import build_redis_client
from analyzer.inference.resolution import (
    ServingPlan,
    SkipReason,
    compute_score_columns,
    resolve_latest_quantile_manifest,
    resolve_serving_targets,
)
from analyzer.inference.scoring import resolve_confidence_for_stock
from analyzer.inference.sweep import sweep_and_write_price_bands
from analyzer.inference.writer import (
    InsertOutcome,
    TradingSignalRow,
    format_model_version,
    insert_trading_signal,
)
from analyzer.training.models import HORIZONS

logger = get_logger(__name__)

MARKET_CALENDAR_CODE: dict[str, str] = {"domestic": "KRX", "overseas": "NYSE"}
"""(REQ-APL-121) `training/train.py`의 비공개 모듈 상수
(`_MARKET_CALENDAR_CODE_OVERRIDE`)를 임포트하지 않고 `inference/` 패키지
내부에 독립 정의한다 — `data/repository.py`가 이미 `training/`·`inference/`
양쪽에서 의존받고 있어 역방향 임포트는 순환 의존을 만든다(research.md §7).
`trade_date.py`/`data/repository.py`가 이미 동일 매핑을 독립적으로 중복
정의한 선례를 따른다."""


def resolve_calendar_code(market: str) -> str:
    """`market` 토큰을 `market_calendar.calendar_code`로 해석한다(REQ-APL-121)."""
    return MARKET_CALENDAR_CODE[market]


def resolve_model_version(serving_plan: ServingPlan) -> str:
    """(REQ-APL-110) 앙상블 조합은 두 알고리즘의 `trained_date` 중 더 최근
    값(`max()`)을, 단독 전략은 그 알고리즘의 `trained_date`를 그대로
    사용해 `model_version`을 산출한다.

    `writer.format_model_version()`(무수정)을 그대로 위임 호출한다 — 이
    함수는 순수하게 "어떤 trained_date를 넘길지"만 결정하는 얇은 레이어다.
    """
    if serving_plan.active_strategy == "ensemble":
        trained_date = max(m.trained_date for m in serving_plan.manifests.values())
        return format_model_version(
            serving_plan.market, serving_plan.horizon, "ensemble", trained_date
        )
    algorithm = serving_plan.active_strategy
    trained_date = serving_plan.manifests[algorithm].trained_date
    return format_model_version(serving_plan.market, serving_plan.horizon, algorithm, trained_date)


def _filter_universe(universe) -> list[tuple[int, str]]:  # type: ignore[no-untyped-def]
    """(REQ-APL-107) `fetch_stock_universe()`의 원본 결과를 `grade IN ('A',
    'B')` AND `delisted_at IS NULL`(상장 유지)로 필터링한다 — 함수 자체는
    필터링하지 않고 원본 컬럼만 반환하므로 호출부가 명시적으로 적용해야
    한다."""
    filtered = universe[universe["grade"].isin(("A", "B")) & universe["delisted_at"].isna()]
    return list(zip(filtered["stock_id"], filtered["stock_code"], strict=True))


class _SkippedStock(Exception):
    """종목 1건 처리를 조기 종료시키는 내부 시그널(design.md §1) — 스킵
    사유를 실어 나른다. `run_market_inference()` 밖으로 전파되지 않는다."""

    def __init__(self, reason: SkipReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


def run_market_inference(
    market: str,
    *,
    trace_id: str,
    models_root: Path,
    trade_date: date,
    calendar_code: str | None = None,
) -> InferenceOutcome:
    """`market`의 전 horizon 조합에 대해 실제 추론을 수행한다(REQ-APL-100).

    `resolve_serving_targets()`/`resolve_latest_quantile_manifest()`로 조합
    단위 서빙 대상을 해석하고, 조합이 유효하면 유니버스의 각 종목에 대해
    피처 조립 → 예측 → score/등급/confidence 산출 → INSERT → 발행 → 밴드
    스윕을 종목 단위 예외 경계로 감싸 실행한다(REQ-APL-101/102/104/105/107).

    `trade_date`는 이 함수 진입 전에 `trade_date.resolve_trade_date(engine,
    market)`로 산출돼 전달된다(design.md §1, REQ-AIF-021 계승) — 이 함수
    자체는 거래일을 산출하지 않는다.
    """
    resolved_calendar_code = calendar_code or resolve_calendar_code(market)

    engine = build_engine(get_db_config())
    redis_client = build_redis_client(get_inference_config())
    metrics = InferenceMetrics()  # REQ-APL-134: 프로세스당 정확히 1회
    start = time.monotonic()

    processed = 0
    skipped_combinations = 0
    partial_failures = 0

    try:
        boundaries = load_grade_boundaries()
        calendar = fetch_market_calendar(engine, resolved_calendar_code)
        raw_universe = fetch_stock_universe(engine, market)  # REQ-APL-107: 시장당 1회
        universe = _filter_universe(raw_universe)

        for horizon in HORIZONS:
            serving_plan = resolve_serving_targets(models_root, market, horizon)
            if isinstance(serving_plan, SkipReason):
                metrics.record_skip(market=market, horizon=horizon, reason=serving_plan)
                skipped_combinations += 1
                continue

            quantile_manifest = resolve_latest_quantile_manifest(models_root, market, horizon)
            if quantile_manifest is None:
                metrics.record_skip(
                    market=market, horizon=horizon, reason=SkipReason.QUANTILE_MISSING
                )
                skipped_combinations += 1
                continue

            if not universe:
                logger.info(
                    "유니버스가 비어 있어 처리 대상이 없다 market=%s horizon=%d", market, horizon
                )
                continue

            model_version = resolve_model_version(serving_plan)
            combo_had_stock_skip = False

            for stock_id, stock_code in universe:
                try:
                    features = assemble_inference_features(engine, calendar, stock_code, trade_date)
                    if features is None:
                        raise _SkippedStock(SkipReason.FEATURE_INSUFFICIENT)

                    predictions = predict_point_models(serving_plan, features)
                    score_cols = compute_score_columns(serving_plan.active_strategy, predictions)
                    p10, p90 = predict_quantile_models(quantile_manifest, features)
                    confidence = resolve_confidence_for_stock(
                        quantile_manifest, score_cols.score, p10, p90
                    )
                    if isinstance(confidence, SkipReason):
                        raise _SkippedStock(confidence)

                    signal_class = boundaries.classify(market, horizon, [score_cols.score])[0]
                    row = TradingSignalRow(
                        stock_id=stock_id,
                        trade_date=trade_date,
                        horizon=horizon,
                        score=score_cols.score,
                        p10=p10,
                        p90=p90,
                        lgbm_score=score_cols.lgbm_score,
                        xgb_score=score_cols.xgb_score,
                        signal_class=str(signal_class),
                        confidence=confidence,
                        model_version=model_version,
                    )
                    insert_outcome = insert_trading_signal(engine, row)
                    # REQ-APL-104: 예외 없이 반환(INSERTED 또는 SKIPPED_DUPLICATE)
                    # 하면 항상 발행한다 — 반환값으로 발행 여부를 분기하지 않는다.
                    assert insert_outcome in (
                        InsertOutcome.INSERTED,
                        InsertOutcome.SKIPPED_DUPLICATE,
                    )
                    publish_trading_signal(
                        redis_client,
                        market=market,
                        symbol=stock_code,
                        horizon=horizon,
                        trade_date=trade_date,
                        trace_id=trace_id,
                        signal_class=str(signal_class),
                        score=score_cols.score,
                        confidence=confidence,
                    )
                    metrics.record_signal(
                        market=market, horizon=horizon, signal_class=str(signal_class)
                    )

                    sweep_result = sweep_and_write_price_bands(
                        engine=engine,
                        calendar=calendar,
                        boundaries_artifact=boundaries,
                        serving_plan=serving_plan,
                        models_root=models_root,
                        stock_id=stock_id,
                        stock_code=stock_code,
                        market=market,
                        horizon=horizon,
                        trade_date=trade_date,
                        model_version=model_version,
                    )
                    if isinstance(sweep_result, SkipReason):
                        # REQ-APL-105: 밴드 스윕 스킵은 이미 완료된 점 신호
                        # INSERT/발행을 되돌리지 않는다.
                        metrics.record_skip(market=market, horizon=horizon, reason=sweep_result)
                        combo_had_stock_skip = True

                except _SkippedStock as exc:
                    metrics.record_skip(market=market, horizon=horizon, reason=exc.reason)
                    combo_had_stock_skip = True
                except Exception:
                    # REQ-APL-102/103: 그 밖의 모든 예외는 이 종목 1건만
                    # 스킵시킨다 — 조합/시장 처리를 중단시키지 않는다
                    # (aaa-infra#163 봉쇄, REQ-APL-140).
                    logger.exception(
                        "추론 처리 중 예상치 못한 예외 stock=%s market=%s horizon=%d",
                        stock_code,
                        market,
                        horizon,
                    )
                    metrics.record_skip(
                        market=market, horizon=horizon, reason=SkipReason.UNEXPECTED_ERROR
                    )
                    combo_had_stock_skip = True

            if combo_had_stock_skip:
                partial_failures += 1
            else:
                processed += 1

        metrics.observe_cycle_duration(market=market, seconds=time.monotonic() - start)
    finally:
        engine.dispose()
        redis_client.close()

    return InferenceOutcome(
        processed=processed,
        skipped_combinations=skipped_combinations,
        partial_failures=partial_failures,
    )

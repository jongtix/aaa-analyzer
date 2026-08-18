"""학습 파이프라인 오케스트레이션 진입점 (SPEC-ANALYZER-TRAIN-001 M7).

design.md §4 "인터페이스 계약": 자매 SPEC(SPEC-ANALYZER-TRAIN-AUTOMATION-001)이
이 진입점을 블랙박스로 원격 호출한다. 이 SPEC은 그 진입점이 **종료코드로
성공/실패를 신호**한다는 것 이상은 규정하지 않는다 — 원격 완료 감지
로직 자체는 자매 SPEC(REQ-AT-130)이 소관한다.

**확정 계약 (자매 SPEC이 소비)**:
- 모듈 경로: `analyzer.training.train`
- 함수 진입점: `run_training_pipeline(...) -> TrainingPipelineResult`
  (프로그래밍 방식 호출 — Python 프로세스 내에서 직접 호출할 때 사용)
- CLI 진입점: `python -m analyzer.training.train --calendar-code KRX
  --cache-dir <path> --models-root <path> --data-as-of YYYY-MM-DD
  --feature-code-version <str>` — 성공 시 종료코드 `0`, 실패 시 `1`
  (원격 호출 시 사용, `main()` 함수가 `sys.exit()`로 신호).

`run_training_pipeline()`은 앞선 마일스톤이 이미 구현한 순수 함수/
클래스를 순서대로 호출한다 — trainer DB 조회(`db.py`, 이 모듈이 신설한
`fetch_stock_universe()`/`fetch_market_data()` 포함) → 데이터셋 조립
(`dataset.py`, 캐싱은 `cache.py`) → 분할(`split.py`) → 4개 시장×horizon
조합 학습(`models.py`) → 저장(`persistence.py`). 재구현하지 않는다.

`stock_grades`/`stocks` 조인 쿼리(`fetch_stock_universe`)는
`data/repository.py`에 없는 신규 SELECT다 — `repository.py`는 이
SPEC의 PRESERVE 대상(plan.md §D, `_DB_USER`→`_ANALYZER_DB_USER` 리네임
1줄만 허용된 예외)이므로, 이 신규 쿼리는 `training/` 소관인 이 모듈에
정의한다. `trainer` 계정(SELECT 전용)으로만 실행되며 DDL/DML을 발행하지
않는다(REQ-AT-012).
"""

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from analyzer.common.logging import get_logger
from analyzer.common.trace import set_trace_id
from analyzer.data.models import TradingCalendar
from analyzer.data.repository import (
    fetch_corporate_events,
    fetch_daily_ohlcv,
    fetch_investor_trend,
    fetch_market_calendar,
)
from analyzer.features.classification import FEATURE_REGISTRY
from analyzer.training import cache as cache_module
from analyzer.training import dataset as dataset_module
from analyzer.training import persistence as persistence_module
from analyzer.training.db import build_trainer_engine
from analyzer.training.models import HORIZONS, MARKETS, train_pooled_models

logger = get_logger(__name__)

_PROGRESS_BATCH_SIZE = 25
"""SPEC-ANALYZER-TRAIN-OBSV-001 REQ-ATO-015: 25종목마다 1회 진행 로그를
남긴다 — 사용자가 확정한 값이므로 코드 상수로 고정한다."""


_MARKET_TO_STOCKS_MARKET_CODES: dict[str, tuple[str, ...]] = {
    "domestic": ("KOSPI", "KOSDAQ"),
    "overseas": ("NYSE", "NASDAQ", "AMEX"),
}
"""`training/` 소관 시장 토큰("domestic"/"overseas")과 `stocks.market`
컬럼 값 사이의 매핑. `stocks.market`은 거래소 코드로 저장되며
(aaa-collector `Market` enum), `KRX`/`US`는 그 enum에 존재하긴 하지만
지수 종목(`asset_type=INDEX`, 코스피지수/S&P500 등) 전용 값이라 개별
종목(`asset_type=STOCK`) 유니버스에는 절대 나타나지 않는다(2026-08-13
NAS DB 실측: market='KRX'/'US' AND asset_type='STOCK' → 0행 — 이전
회귀 버그, TECHSPEC §648 언급은 컬럼 존재 자체이지 이 값 매핑의 근거가
아니었다)."""

_STOCK_UNIVERSE_QUERY = text(
    "SELECT s.symbol AS stock_code, g.grade, s.delisted_at "
    "FROM stocks s "
    "JOIN stock_grades g ON g.stock_id = s.id "
    "WHERE s.market IN :market_codes AND s.asset_type = 'STOCK'"
).bindparams(bindparam("market_codes", expanding=True))


@dataclass(frozen=True, slots=True)
class TrainingPipelineResult:
    """`run_training_pipeline()` 실행 결과 — CLI 종료코드 신호의 기반 데이터."""

    success: bool
    saved_model_paths: list[Path] = field(default_factory=list)
    error: str | None = None


def fetch_stock_universe(engine: Engine, market: str) -> pd.DataFrame:
    """`stock_grades`/`stocks`에서 후보 유니버스(등급+상장폐지 여부)를 조회한다.

    `market`은 "domestic"/"overseas"(이 SPEC의 시장 토큰) — 내부적으로
    `stocks.market`의 거래소 코드 집합(domestic=KOSPI/KOSDAQ,
    overseas=NYSE/NASDAQ/AMEX)으로 변환해 조회한다.
    """
    market_codes = _MARKET_TO_STOCKS_MARKET_CODES[market]
    return pd.read_sql(_STOCK_UNIVERSE_QUERY, engine, params={"market_codes": market_codes})


def fetch_market_data(
    engine: Engine, stocks: pd.DataFrame, data_as_of: date
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """유니버스 종목별로 원주가/이벤트/수급 데이터를 조회한다(`repository.py` 기존 함수 재사용).

    `data_as_of` 상한을 `fetch_daily_ohlcv(..., end_date=data_as_of)`로 강제해,
    조립된 데이터셋에 `data_as_of`보다 미래인 원주가 행이 섞이지 않게 한다
    (REQ-ATE-001/002/003/005/006 — 학습 데이터 경계 결함 수정).

    반환된 세 딕셔너리는 `dataset.assemble_dataset()`이 요구하는
    `ohlcv_by_stock`/`events_by_stock`/`investor_trend_by_stock` 스키마와
    정확히 일치한다.
    """
    ohlcv_by_stock: dict[str, pd.DataFrame] = {}
    events_by_stock: dict[str, pd.DataFrame] = {}
    investor_trend_by_stock: dict[str, pd.DataFrame] = {}

    progress_batch: list[str] = []
    for stock_code in stocks["stock_code"]:
        ohlcv_by_stock[stock_code] = fetch_daily_ohlcv(engine, stock_code, end_date=data_as_of)
        events_by_stock[stock_code] = fetch_corporate_events(engine, stock_code)
        investor_trend_by_stock[stock_code] = fetch_investor_trend(engine, stock_code)

        progress_batch.append(stock_code)
        if len(progress_batch) >= _PROGRESS_BATCH_SIZE:
            # REQ-ATO-015: 25종목마다 1회 진행 로그(해당 배치 종목코드 나열).
            logger.info("market data fetch progress batch=%s", progress_batch)
            progress_batch = []

    if progress_batch:
        logger.info("market data fetch progress batch=%s", progress_batch)

    return ohlcv_by_stock, events_by_stock, investor_trend_by_stock


def _assemble_market_dataset(
    engine: Engine,
    calendar: TradingCalendar,
    market: str,
    cache_dir: Path,
    data_as_of: date,
    feature_code_version: str,
) -> pd.DataFrame:
    """단일 시장의 데이터셋을 조회+조립하고 캐시를 경유한다(`db.py`→`dataset.py`→`cache.py`)."""
    stocks = fetch_stock_universe(engine, market)
    grade_counts = stocks["grade"].value_counts().to_dict() if not stocks.empty else {}
    # REQ-ATO-018: 시장별 시작 + 유니버스 크기(등급별 종목 수) 단계 전이 로그.
    logger.info(
        "market start market=%s universe_size=%d grade_counts=%s",
        market,
        len(stocks),
        grade_counts,
        extra={"stage_marker": True},
    )
    ohlcv_by_stock, events_by_stock, investor_trend_by_stock = fetch_market_data(
        engine, stocks, data_as_of
    )

    def _assemble() -> pd.DataFrame:
        return dataset_module.assemble_dataset(
            stocks=stocks,
            ohlcv_by_stock=ohlcv_by_stock,
            events_by_stock=events_by_stock,
            investor_trend_by_stock=investor_trend_by_stock,
            calendar=calendar,
            market=market,
        )

    # REQ-ATO-018: 데이터셋 캐시 히트/미스 단계 전이 로그. 실제 캐시 사용은
    # 여전히 assemble_dataset_cached()가 전담한다 — 여기서는 관측 목적으로만
    # 존재 여부를 한 번 더 확인한다(cache.py 내부 구현/시그니처는 건드리지 않음).
    cache_hit = (
        cache_module.load_cached_dataset(cache_dir, market, data_as_of, feature_code_version)
        is not None
    )
    logger.info(
        "dataset cache %s market=%s",
        "hit" if cache_hit else "miss",
        market,
        extra={"stage_marker": True},
    )
    assembled = cache_module.assemble_dataset_cached(
        cache_dir=cache_dir,
        market=market,
        data_as_of=data_as_of,
        feature_code_version=feature_code_version,
        assemble_fn=_assemble,
    )
    # REQ-ATO-018: 데이터셋 조립 완료 행수 단계 전이 로그.
    logger.info(
        "dataset assembly complete market=%s rows=%d",
        market,
        len(assembled),
        extra={"stage_marker": True},
    )
    return assembled


def _split_features_and_labels(
    assembled: pd.DataFrame, horizon: int
) -> tuple[list[str], pd.DataFrame, pd.Series]:
    """조립된 데이터셋에서 `horizon`의 유효 레이블 행만 골라 (feature 열, X, y)로 나눈다.

    피처 컬럼은 `FEATURE_REGISTRY`(REQ-ATE-074, design.md §5.2)와
    `assembled.columns`의 교집합으로 도출한다 — 원시 OHLCV 컬럼
    (open_price/high_price/low_price/close_price/volume)과 `stock_code`/
    `trade_date`는 `FEATURE_REGISTRY`에 등록되어 있지 않으므로 자동으로
    제외된다. 일부 조합(예: 해외 종목의 수급 피처 결측, REQ-AT-064)은
    40개 키 중 일부가 `assembled.columns`에 존재하지 않을 수 있으며,
    교집합이므로 그 경우도 자연스럽게 처리된다.
    """
    feature_columns = [c for c in assembled.columns if c in FEATURE_REGISTRY]
    label_column = f"label_D{horizon}"
    valid = assembled.dropna(subset=[label_column])
    x: pd.DataFrame = valid.loc[:, feature_columns]
    y: pd.Series = valid.loc[:, label_column]
    return feature_columns, x, y


_MARKET_CALENDAR_CODE_OVERRIDE: dict[str, str] = {"overseas": "NYSE"}
"""`overseas` 시장은 호출자가 넘긴 `calendar_code`(도메스틱 기본값 "KRX")를
공유하면 안 된다 — `market_calendar` 실측(2026-08-13): calendar_code='NYSE'의
최초 거래일(2007-08-20)이 `labels.config.DEFAULT_START_DATES["overseas"]`와
정확히 일치해, 해외용으로 이미 시딩돼 있었다. 이전 코드는 `calendar_code`를
domestic/overseas 양쪽에 그대로 재사용해 미국 종목의 T+H(REQ-AL-030)를 KRX
개장일 기준으로 계산하고, "KRX는 열고 미국은 휴장인 날"(추수감사절 등)을
종목별 거래정지(`analyze_halt()`)로 오판하는 회귀 버그가 있었다 — 에러 없이
레이블만 조용히 왜곡되는 유형이라 발견이 늦었다."""


def _resolve_algorithm(model_key: tuple) -> str:
    """`models.train_pooled_models()`가 반환하는 키에서 저장용 algorithm 문자열을 뽑는다."""
    tag = model_key[2]
    return "lightgbm" if tag == "lightgbm_quantile" else tag


def _quantile_model_filename(market: str, horizon: int, alpha: float, trained_date: date) -> str:
    """분위수 보조 모델의 파일명 — 포인트 LightGBM 모델과 동일한 algorithm
    세그먼트("lightgbm")·디렉토리를 공유하되, 파일명 세그먼트에 alpha 구분자를
    추가해 충돌을 피한다(REQ-ATE-007/008/010).

    `persistence.model_filename()`은 algorithm을 {"lightgbm","xgboost"} 두 키로만
    검증하므로(`persistence.py` 무수정 유지, plan.md §D), 여기서는 그 함수가 만드는
    포인트 모델용 이름을 그대로 얻은 뒤 alpha 태그만 사후 삽입한다 — 확장자는
    변경하지 않는다.
    """
    base = persistence_module.model_filename(market, horizon, "lightgbm", trained_date)
    stem, _, ext = base.rpartition(".")
    alpha_tag = f"q{round(alpha * 100):02d}"
    return f"{stem}_{alpha_tag}.{ext}"


def _save_quantile_model(
    model: lgb.LGBMRegressor,
    models_root: Path,
    market: str,
    horizon: int,
    alpha: float,
    trained_date: date,
) -> persistence_module.SavedModel:
    """분위수 보조 모델을 저장한다 — `persistence.save_model_native()`(algorithm="lightgbm")를
    임시 스테이징 디렉토리에서 호출해 SHA-256 라운드트립 검증(REQ-AT-092)을 그대로
    재사용한 뒤, 포인트 모델의 실경로를 절대 건드리지 않고 alpha 접미사가 붙은
    최종 파일명으로 옮긴다(REQ-ATE-007/008/009/010, AC-ATE-003).
    """
    models_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models_root) as staging:
        staged = persistence_module.save_model_native(
            model, Path(staging), market, horizon, "lightgbm", trained_date
        )
        final_dir = persistence_module.model_dir(models_root, market, horizon, "lightgbm")
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / _quantile_model_filename(market, horizon, alpha, trained_date)
        final_sidecar = final_path.with_suffix(final_path.suffix + ".sha256")
        shutil.move(str(staged.model_path), str(final_path))
        shutil.move(str(staged.sidecar_path), str(final_sidecar))
        return persistence_module.SavedModel(
            model_path=final_path, sidecar_path=final_sidecar, sha256=staged.sha256
        )


def run_training_pipeline(
    *,
    trainer_engine: Engine,
    calendar_code: str,
    cache_dir: Path,
    models_root: Path,
    data_as_of: date,
    feature_code_version: str,
    lgbm_params: Mapping[str, object] | None = None,
    xgb_params: Mapping[str, object] | None = None,
) -> TrainingPipelineResult:
    """학습 파이프라인 전체를 순서대로 실행한다(design.md §4 진입점 계약).

    각 시장(`training.models.MARKETS`)에 대해 **시장별** 캘린더로 데이터셋을
    조회·조립·캐싱하고(domestic=`calendar_code`, overseas=`NYSE` —
    `_MARKET_CALENDAR_CODE_OVERRIDE`), 시장×horizon 4개 조합에 대해 16개
    모델(8 포인트+8 분위수 보조)을 학습해 네이티브 포맷으로 저장한다.
    예외가 발생하면 잡아 `TrainingPipelineResult(success=False, error=...)`로
    반환한다 — CLI 진입점(`main()`)이 이를 종료코드로 변환한다.
    """
    try:
        data_by_combo: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for market in MARKETS:
            market_calendar_code = _MARKET_CALENDAR_CODE_OVERRIDE.get(market, calendar_code)
            calendar = fetch_market_calendar(trainer_engine, market_calendar_code)
            assembled = _assemble_market_dataset(
                trainer_engine, calendar, market, cache_dir, data_as_of, feature_code_version
            )
            for horizon in HORIZONS:
                _feature_columns, x, y = _split_features_and_labels(assembled, horizon)
                # REQ-ATO-018: horizon별 유효 레이블 행수 단계 전이 로그.
                logger.info(
                    "valid label rows market=%s horizon=%s rows=%d",
                    market,
                    horizon,
                    len(x),
                    extra={"stage_marker": True},
                )
                data_by_combo[(market, horizon)] = (x.to_numpy(), y.to_numpy())

        trained_models = train_pooled_models(
            data_by_combo, lgbm_params=lgbm_params, xgb_params=xgb_params
        )

        saved_paths: list[Path] = []
        for model_key, model in trained_models.items():
            market, horizon = model_key[0], model_key[1]
            tag = model_key[2]
            algorithm = _resolve_algorithm(model_key)
            if tag == "lightgbm_quantile":
                alpha = model_key[3]
                assert isinstance(model, lgb.LGBMRegressor)
                saved = _save_quantile_model(model, models_root, market, horizon, alpha, data_as_of)
            else:
                saved = persistence_module.save_model_native(
                    model, models_root, market, horizon, algorithm, data_as_of
                )
            # REQ-ATO-019: 모델 저장 경로 단계 전이 로그.
            logger.info(
                "model saved market=%s horizon=%s algorithm=%s path=%s",
                market,
                horizon,
                algorithm,
                saved.model_path,
                extra={"stage_marker": True},
            )
            saved_paths.append(saved.model_path)

        return TrainingPipelineResult(success=True, saved_model_paths=saved_paths)
    except Exception as exc:  # noqa: BLE001 — 오케스트레이터 최상위 캐치-올(종료코드 신호 계약)
        # REQ-ATO-020: TrainingPipelineResult.error 반환 타입(문자열)은 그대로
        # 유지하되, 전체 traceback은 로그로 남긴다(exc_info=True).
        logger.error("training pipeline failed: %s", exc, exc_info=True)
        return TrainingPipelineResult(success=False, error=str(exc))


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점 — 성공 시 `0`, 실패 시 `1`을 반환한다(design.md §4)."""
    parser = argparse.ArgumentParser(prog="analyzer.training.train")
    parser.add_argument("--calendar-code", default="KRX")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--data-as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--feature-code-version", required=True)
    args = parser.parse_args(argv)

    # REQ-ATO-012/013/014: NAS 오케스트레이터가 전달한 run_id를 trace_id로
    # 즉시 설정한다 — 누락되어도(fail-open) 학습 자체는 계속 진행된다
    # (get_trace_id()가 None을 반환하는 기존 계약과 정합, REQ-ATO-014 참조).
    run_id = os.environ.get("TRAIN_RUN_ID")
    if run_id:
        set_trace_id(run_id)

    engine = build_trainer_engine()

    result = run_training_pipeline(
        trainer_engine=engine,
        calendar_code=args.calendar_code,
        cache_dir=args.cache_dir,
        models_root=args.models_root,
        data_as_of=args.data_as_of,
        feature_code_version=args.feature_code_version,
    )
    if not result.success:
        print(f"학습 파이프라인 실패: {result.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

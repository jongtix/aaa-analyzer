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
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from analyzer.data.models import TradingCalendar
from analyzer.data.repository import (
    fetch_corporate_events,
    fetch_daily_ohlcv,
    fetch_investor_trend,
    fetch_market_calendar,
)
from analyzer.training import cache as cache_module
from analyzer.training import dataset as dataset_module
from analyzer.training import persistence as persistence_module
from analyzer.training.db import build_trainer_engine
from analyzer.training.models import HORIZONS, MARKETS, train_pooled_models

_MARKET_TO_STOCKS_MARKET_CODE: dict[str, str] = {"domestic": "KRX", "overseas": "US"}
"""`training/` 소관 시장 토큰("domestic"/"overseas")과 `stocks.market`
컬럼 값("KRX"/"US", TECHSPEC §648) 사이의 매핑 — 이 SPEC 신규 결정이
아니라 기존 `stocks`/`labels.config.DEFAULT_START_DATES` 관례를 그대로
잇는 변환이다."""

_STOCK_UNIVERSE_QUERY = text(
    "SELECT s.symbol AS stock_code, g.grade, s.delisted_at "
    "FROM stocks s "
    "JOIN stock_grades g ON g.stock_id = s.id "
    "WHERE s.market = :market_code AND s.asset_type = 'STOCK'"
)


@dataclass(frozen=True, slots=True)
class TrainingPipelineResult:
    """`run_training_pipeline()` 실행 결과 — CLI 종료코드 신호의 기반 데이터."""

    success: bool
    saved_model_paths: list[Path] = field(default_factory=list)
    error: str | None = None


def fetch_stock_universe(engine: Engine, market: str) -> pd.DataFrame:
    """`stock_grades`/`stocks`에서 후보 유니버스(등급+상장폐지 여부)를 조회한다.

    `market`은 "domestic"/"overseas"(이 SPEC의 시장 토큰) — 내부적으로
    `stocks.market`의 "KRX"/"US" 값으로 변환해 조회한다.
    """
    market_code = _MARKET_TO_STOCKS_MARKET_CODE[market]
    return pd.read_sql(_STOCK_UNIVERSE_QUERY, engine, params={"market_code": market_code})


def fetch_market_data(
    engine: Engine, stocks: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """유니버스 종목별로 원주가/이벤트/수급 데이터를 조회한다(`repository.py` 기존 함수 재사용).

    반환된 세 딕셔너리는 `dataset.assemble_dataset()`이 요구하는
    `ohlcv_by_stock`/`events_by_stock`/`investor_trend_by_stock` 스키마와
    정확히 일치한다.
    """
    ohlcv_by_stock: dict[str, pd.DataFrame] = {}
    events_by_stock: dict[str, pd.DataFrame] = {}
    investor_trend_by_stock: dict[str, pd.DataFrame] = {}

    for stock_code in stocks["stock_code"]:
        ohlcv_by_stock[stock_code] = fetch_daily_ohlcv(engine, stock_code)
        events_by_stock[stock_code] = fetch_corporate_events(engine, stock_code)
        investor_trend_by_stock[stock_code] = fetch_investor_trend(engine, stock_code)

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
    ohlcv_by_stock, events_by_stock, investor_trend_by_stock = fetch_market_data(engine, stocks)

    def _assemble() -> pd.DataFrame:
        return dataset_module.assemble_dataset(
            stocks=stocks,
            ohlcv_by_stock=ohlcv_by_stock,
            events_by_stock=events_by_stock,
            investor_trend_by_stock=investor_trend_by_stock,
            calendar=calendar,
            market=market,
        )

    return cache_module.assemble_dataset_cached(
        cache_dir=cache_dir,
        market=market,
        data_as_of=data_as_of,
        feature_code_version=feature_code_version,
        assemble_fn=_assemble,
    )


def _split_features_and_labels(
    assembled: pd.DataFrame, horizon: int
) -> tuple[list[str], pd.DataFrame, pd.Series]:
    """조립된 데이터셋에서 `horizon`의 유효 레이블 행만 골라 (feature 열, X, y)로 나눈다."""
    feature_columns = [
        c
        for c in assembled.columns
        if not c.startswith("label_") and c not in ("stock_code", "trade_date")
    ]
    label_column = f"label_D{horizon}"
    valid = assembled.dropna(subset=[label_column])
    x: pd.DataFrame = valid.loc[:, feature_columns]
    y: pd.Series = valid.loc[:, label_column]
    return feature_columns, x, y


def _resolve_algorithm(model_key: tuple) -> str:
    """`models.train_pooled_models()`가 반환하는 키에서 저장용 algorithm 문자열을 뽑는다."""
    tag = model_key[2]
    return "lightgbm" if tag == "lightgbm_quantile" else tag


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

    각 시장(`training.models.MARKETS`)에 대해 데이터셋을 조회·조립·캐싱하고,
    시장×horizon 4개 조합에 대해 16개 모델(8 포인트+8 분위수 보조)을
    학습해 네이티브 포맷으로 저장한다. 예외가 발생하면 잡아
    `TrainingPipelineResult(success=False, error=...)`로 반환한다 —
    CLI 진입점(`main()`)이 이를 종료코드로 변환한다.
    """
    try:
        calendar = fetch_market_calendar(trainer_engine, calendar_code)

        data_by_combo: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for market in MARKETS:
            assembled = _assemble_market_dataset(
                trainer_engine, calendar, market, cache_dir, data_as_of, feature_code_version
            )
            for horizon in HORIZONS:
                _feature_columns, x, y = _split_features_and_labels(assembled, horizon)
                data_by_combo[(market, horizon)] = (x.to_numpy(), y.to_numpy())

        trained_models = train_pooled_models(
            data_by_combo, lgbm_params=lgbm_params, xgb_params=xgb_params
        )

        saved_paths: list[Path] = []
        for model_key, model in trained_models.items():
            market, horizon = model_key[0], model_key[1]
            algorithm = _resolve_algorithm(model_key)
            saved = persistence_module.save_model_native(
                model, models_root, market, horizon, algorithm, data_as_of
            )
            saved_paths.append(saved.model_path)

        return TrainingPipelineResult(success=True, saved_model_paths=saved_paths)
    except Exception as exc:  # noqa: BLE001 — 오케스트레이터 최상위 캐치-올(종료코드 신호 계약)
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

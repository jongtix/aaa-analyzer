"""역사적 Walk-Forward 캠페인 오케스트레이션 + CLI 진입점 (SPEC-ANALYZER-TRAIN-EVAL-001 M3, 골격).

REQ-ATE-011/015: 시장(국내/해외) × horizon(D20/D60) × 알고리즘(LightGBM/
XGBoost) 8개 포인트 조합 각각에 대해 독립적으로 주간 폴드 시퀀스를
순회한다 — 동일 시장의 D20/D60 조합은 동일한 `initial_train_end`와
`train_end` 시퀀스를 공유한다(`weekly_stride_fold_index_bounds()`의
`train_end` 산식이 horizon에 의존하지 않으므로 자동 충족).

REQ-ATE-022: 시장당 정확히 1회만 `dataset.assemble_dataset()`을 호출한다
— `train.py`의 기존 조립 경로(`_assemble_market_dataset()`, DB 조회→
피처/레이블 계산→캐싱)를 재사용하며 재구현하지 않는다(REQ-ATE-014).

REQ-ATE-027/028/029/030: 캠페인 평가 구간(REQ-ATE-021) 이전의 초기 학습
이력에 `training.split.purged_walk_forward_split()`을 적용해 70/15/15
분할을 얻고, 학습+검증(85%) 부분만으로 조합(시장×horizon×algorithm)당
1회 Optuna 튜닝을 수행한다 — 산출된 하이퍼파라미터는 그 조합의 모든
주간 폴드 학습 호출에 동결·재사용된다(폴드마다 재튜닝하지 않음).

REQ-ATE-012/013/016: 폴드별로 `backtest.compute_backtest_metrics()`와
`ensemble.compute_ensemble_score()`/`compute_confidence()`를 호출한다
(재구현 금지, REQ-ATE-014) — confidence 캘리브레이션용 LightGBM 분위수
보조 모델(alpha=0.10/0.90)도 매 폴드 함께 학습하며 Optuna trial 객체를
참조하지 않는다.

REQ-ATE-037: 폴드별 모델 객체(포인트 2개 + 분위수 보조 2개)는 지표 계산
직후 폐기되며 디스크에 저장되지 않는다 — 이 모듈은 어떤 `persistence.py`
저장 함수도 호출하지 않는다(챔피언 최종 폴드 저장은 M5/M6 소관).

REQ-ATE-023/024/025/026/076: `python -m analyzer.training.campaign`
독립 CLI로 제공되며(`scheduler.py` cron 미등록), `train.py`의 인자 관례를
따르고, 이름 있는 12시간 타임아웃 상수 초과 시 경고 로그 후 현재 폴드
완료 후 정상 종료(graceful stop)하며, `stage_marker=true` 로그를 재사용한다.

**주의(plan.md §F M3, F15)**: 이 모듈은 골격이다 — 실제 지표 산출을
목적으로 하는 라이브/준-라이브 실행은 M4(피처 허용목록 확정, REQ-ATE-074)
랜딩 이후로 미룬다. 이 세션은 합성 축소 데이터로 단위 테스트만 수행하며,
실 DB 접속 캠페인 실행을 수행하지 않았다.
"""

import argparse
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sqlalchemy.engine import Engine

from analyzer.common.logging import get_logger
from analyzer.data.repository import fetch_market_calendar
from analyzer.labels.config import PURGE_GAP_TRADING_DAYS
from analyzer.training import train as train_module
from analyzer.training.backtest import BacktestMetrics, compute_backtest_metrics
from analyzer.training.db import build_trainer_engine
from analyzer.training.ensemble import compute_confidence, compute_ensemble_score
from analyzer.training.models import HORIZONS, MARKETS, QUANTILE_ALPHAS, PooledModel
from analyzer.training.panel_folds import (
    PanelFoldDateBounds,
    extract_global_trade_date_axis,
    map_index_bounds_to_dates,
    slice_panel_by_date_bounds,
    weekly_stride_fold_index_bounds,
)
from analyzer.training.split import purged_walk_forward_split
from analyzer.training.tuning import create_or_resume_study, report_fold_and_maybe_prune

logger = get_logger(__name__)

CAMPAIGN_OOS_EVALUATION_SPAN_YEARS: int = 10
"""REQ-ATE-021: 캠페인 표본외(OOS) 평가 구간 길이(연). 폴드 수는 이 값과
실제 가용 거래일 수에서 런타임에 유도되며 하드코딩된 상수가 아니다."""

CAMPAIGN_TIMEOUT_HOURS: float = 12.0
"""REQ-ATE-025: 캠페인 실행 상한 시간. 초과 시 REQ-ATE-076에 따라 경고
로그 후 현재 폴드 완료 후 정상 종료(graceful stop)한다."""

CAMPAIGN_OPTUNA_TRIALS: int = 50
"""REQ-ATE-029: 1회 초기 이력 튜닝의 trial 수."""

CAMPAIGN_VAL_SIZE_TRADING_DAYS: int = 5
"""REQ-ATE-018/021: 표본외(OOS) 검증 윈도우 크기(영업일 1주)."""

_TRADING_DAYS_PER_YEAR: int = 252
"""REQ-ATE-021/design.md §2A: 평가 구간 연수를 거래일수로 환산할 때 쓰는
근사 상수(실제 거래일수는 시장별 실측 캘린더에서 유도되므로 최대 ±2%
편차가 있을 수 있다 — AC-ATE-012)."""

POINT_ALGORITHMS: tuple[str, ...] = ("lightgbm", "xgboost")

POINT_COMBOS: tuple[tuple[str, int, str], ...] = tuple(
    (market, horizon, algorithm)
    for market in MARKETS
    for horizon in HORIZONS
    for algorithm in POINT_ALGORITHMS
)
"""REQ-ATE-011/015: 8개(2시장×2horizon×2algorithm) 포인트 조합 전체 목록."""

_DEFAULT_LGBM_VERBOSITY: dict[str, Any] = {"verbosity": -1}
_DEFAULT_XGB_VERBOSITY: dict[str, Any] = {"verbosity": 0}
"""모델 구성 시 조용한 로그 레벨을 기본 적용한다 — `models.py`의
`_DEFAULT_LGBM_PARAMS`/`_DEFAULT_XGB_PARAMS`와 동일한 관례를 이 모듈
안에서 독립적으로 유지한다(`models.py` 자체는 무수정, PRESERVE)."""


def _initial_train_end_index(
    n_dates: int, oos_span_years: int = CAMPAIGN_OOS_EVALUATION_SPAN_YEARS
) -> int:
    """전역 거래일 축에서 평가 구간(REQ-ATE-021)이 시작되는 인덱스를 계산한다.

    그 이전 구간(0..이 인덱스-1)이 "초기 학습 이력"이며, Optuna 1회 튜닝
    (REQ-ATE-027)이 소비하는 구간이다.
    """
    span_days = oos_span_years * _TRADING_DAYS_PER_YEAR
    return max(n_dates - span_days, 1)


def _resolve_evaluation_fold_count(n_dates: int, initial_train_end_idx: int, val_size: int) -> int:
    """REQ-ATE-021 계산식(`floor(평가구간 거래일수 ÷ val_size)`)으로 폴드 수를 유도한다.

    REQ-ATE-015는 동일 시장의 D20/D60 조합이 동일한 `train_end` 시퀀스를
    공유해야 한다고 요구하므로, 두 horizon 중 더 넓은 purge gap(D60=60
    영업일)이 마지막 폴드에서도 안전하게 수용되도록 보수적으로 gap을
    차감한 뒤 폴드 수를 계산한다 — 이렇게 하면 동일한 n_folds로 두
    horizon 모두 `weekly_stride_fold_index_bounds()`가 예외 없이 성공한다.
    """
    max_gap = max(PURGE_GAP_TRADING_DAYS.values())
    usable_span = n_dates - initial_train_end_idx - max_gap
    return max(usable_span // val_size, 0)


def resolve_tuning_split(
    panel: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    initial_history_end_idx: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """REQ-ATE-027: 초기 학습 이력 구간에 `purged_walk_forward_split()`을 적용해
    70/15/15 분할을 얻고, 학습+검증(85%) 부분만 반환한다(test 15%는 튜닝에
    전혀 사용하지 않음 — 누출 방지 버퍼, design.md §4).

    `purged_walk_forward_split()`은 "이미 시간순 정렬된 샘플"을 전제하므로,
    이 함수는 전역 거래일 축(`trade_dates`)의 초기-이력 부분집합에 대해서만
    호출하고, 그 정수 인덱스 경계를 `trade_date` 값으로 역매핑한 뒤
    `panel_folds.slice_panel_by_date_bounds()`로 값 비교 필터링한다(REQ-ATE-019와
    동일한 leakage-safe 원칙 재사용).
    """
    initial_dates_axis = trade_dates[:initial_history_end_idx]
    bounds = purged_walk_forward_split(len(initial_dates_axis), horizon)

    date_bounds = PanelFoldDateBounds(
        train_end=cast(pd.Timestamp, initial_dates_axis[bounds.train_end]),
        val_start=cast(pd.Timestamp, initial_dates_axis[bounds.val_start]),
        val_end=cast(pd.Timestamp, initial_dates_axis[bounds.val_end]),
    )
    return slice_panel_by_date_bounds(panel, date_bounds)


def _lgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
    }


def _xgb_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
    }


_SEARCH_SPACE_BY_ALGORITHM: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "lightgbm": _lgbm_search_space,
    "xgboost": _xgb_search_space,
}


def _fit_point_model(
    algorithm: str, params: Mapping[str, Any], x: np.ndarray, y: np.ndarray
) -> PooledModel:
    """동결된(또는 trial 제안) 하이퍼파라미터로 포인트 모델 1개를 학습한다."""
    if algorithm == "lightgbm":
        model: PooledModel = lgb.LGBMRegressor(**{**_DEFAULT_LGBM_VERBOSITY, **dict(params)})
    elif algorithm == "xgboost":
        model = xgb.XGBRegressor(**{**_DEFAULT_XGB_VERBOSITY, **dict(params)})
    else:
        raise ValueError(f"알 수 없는 algorithm: {algorithm}")
    model.fit(x, y)
    return model


def _make_tuning_objective(
    algorithm: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Callable[[optuna.Trial], float]:
    """REQ-ATE-027/029: trial마다 후보 하이퍼파라미터로 학습·검증하고, 검증
    Rank IC를 Optuna에 보고한다(스터디 `direction="minimize"`이므로 Rank
    IC를 최대화하려면 그 음수를 반환·보고한다).
    """
    search_space = _SEARCH_SPACE_BY_ALGORITHM[algorithm]

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        model = _fit_point_model(algorithm, params, x_train, y_train)
        preds = np.asarray(model.predict(x_val))
        neutral_confidences = np.full(len(preds), 0.5)
        metrics = compute_backtest_metrics(preds, y_val, neutral_confidences)
        report_fold_and_maybe_prune(trial, fold_index=0, fold_metric=-metrics.rank_ic)
        return -metrics.rank_ic

    return objective


def tune_initial_history_hyperparameters(
    panel: pd.DataFrame,
    market: str,
    horizon: int,
    algorithm: str,
    storage_dir: Path,
    trial_count: int = CAMPAIGN_OPTUNA_TRIALS,
    oos_span_years: int = CAMPAIGN_OOS_EVALUATION_SPAN_YEARS,
) -> dict[str, Any]:
    """REQ-ATE-027/028/029: (시장,horizon,algorithm) 조합당 정확히 1회 튜닝하고
    동결할 하이퍼파라미터 딕셔너리를 반환한다 — 평가 구간(REQ-ATE-021) 데이터는
    `resolve_tuning_split()`이 초기 이력으로만 슬라이싱하므로 이 함수에
    노출되지 않는다.
    """
    trade_dates = extract_global_trade_date_axis(panel)
    initial_history_end_idx = _initial_train_end_index(len(trade_dates), oos_span_years)
    train_df, val_df = resolve_tuning_split(panel, trade_dates, initial_history_end_idx, horizon)

    _, x_train, y_train = train_module._split_features_and_labels(train_df, horizon)
    _, x_val, y_val = train_module._split_features_and_labels(val_df, horizon)

    study = create_or_resume_study(
        storage_dir, market, horizon, study_name=f"campaign_{market}_{horizon}_{algorithm}"
    )
    objective = _make_tuning_objective(
        algorithm, x_train.to_numpy(), y_train.to_numpy(), x_val.to_numpy(), y_val.to_numpy()
    )
    study.optimize(objective, n_trials=trial_count)
    return dict(study.best_params)


def _compute_fold_ensemble(
    lgbm_preds: np.ndarray, xgb_preds: np.ndarray, p10_preds: np.ndarray, p90_preds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """폴드별 예측 배열에 대해 앙상블 score/confidence를 계산한다.

    `ensemble.compute_ensemble_score()`/`compute_confidence()`는 스칼라
    입력을 받는 순수 함수(REQ-AT-080~084)이므로 배열용으로 재구현하지
    않고(REQ-ATE-014) 원소 단위로 그대로 호출한다.
    """
    n = len(lgbm_preds)
    ensemble_scores = np.empty(n, dtype=float)
    confidences = np.empty(n, dtype=float)
    for i in range(n):
        result = compute_ensemble_score(float(lgbm_preds[i]), float(xgb_preds[i]))
        ensemble_scores[i] = result.score_ensemble
        try:
            confidences[i] = compute_confidence(
                result.score_ensemble, float(p10_preds[i]), float(p90_preds[i])
            )
        except ValueError:
            # 축퇴 분위수 분포(p10==p90) — confidence 무정보(0.5)로 처리.
            confidences[i] = 0.5
    return ensemble_scores, confidences


@dataclass(frozen=True, slots=True)
class MarketHorizonFoldRecord:
    """(시장,horizon) 단위 폴드 1개의 3개 스코어링 전략(lgbm/xgb/ensemble) 지표(REQ-ATE-038)."""

    fold_index: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp | None
    lightgbm_metrics: BacktestMetrics
    xgboost_metrics: BacktestMetrics
    ensemble_metrics: BacktestMetrics


def run_campaign_for_market_horizon(
    panel: pd.DataFrame,
    market: str,
    horizon: int,
    initial_train_end_idx: int,
    n_folds: int,
    frozen_params_by_algorithm: Mapping[str, Mapping[str, Any]],
    val_size: int = CAMPAIGN_VAL_SIZE_TRADING_DAYS,
    timeout_deadline: float | None = None,
) -> list[MarketHorizonFoldRecord]:
    """(시장,horizon) 조합의 주간 폴드를 순회하며 lgbm/xgb/ensemble 3개 스코어링
    전략의 지표를 계산한다(REQ-ATE-012/013/015/016/026/037/076).

    `frozen_params_by_algorithm`은 `tune_initial_history_hyperparameters()`가
    산출한 동결 하이퍼파라미터를 "lightgbm"/"xgboost" 키로 담은 매핑이며,
    이 함수가 순회하는 **모든** 폴드의 모델 학습 호출에 동일하게 전달된다
    (REQ-ATE-028, 폴드마다 재튜닝하지 않음).

    각 폴드의 모델 객체(포인트 2개 + 분위수 보조 2개)는 지표 계산 직후
    이 함수 스코프를 벗어나며 폐기된다 — 어떤 `persistence.py` 저장
    함수도 호출하지 않는다(REQ-ATE-037).
    """
    trade_dates = extract_global_trade_date_axis(panel)
    index_bounds_list = weekly_stride_fold_index_bounds(
        n_dates=len(trade_dates),
        horizon=horizon,
        initial_train_end=initial_train_end_idx,
        val_size=val_size,
        n_folds=n_folds,
    )

    lgbm_params = dict(frozen_params_by_algorithm["lightgbm"])
    xgb_params = dict(frozen_params_by_algorithm["xgboost"])

    records: list[MarketHorizonFoldRecord] = []
    for fold_index, index_bounds in enumerate(index_bounds_list):
        if timeout_deadline is not None and time.monotonic() > timeout_deadline:
            # REQ-ATE-076: 폴드 경계에서만 타임아웃을 점검한다 — 처리 중인
            # 폴드를 강제 중단하지 않고, 다음 폴드를 시작하지 않은 채 정상 종료한다.
            logger.warning(
                "campaign timeout exceeded, graceful stop market=%s horizon=%s fold=%d",
                market,
                horizon,
                fold_index,
                extra={"stage_marker": True},
            )
            break

        date_bounds = map_index_bounds_to_dates(index_bounds, trade_dates)
        train_df, val_df = slice_panel_by_date_bounds(panel, date_bounds)

        _, x_train, y_train = train_module._split_features_and_labels(train_df, horizon)
        _, x_val, y_val = train_module._split_features_and_labels(val_df, horizon)
        x_train_arr, y_train_arr = x_train.to_numpy(), y_train.to_numpy()
        x_val_arr, y_val_arr = x_val.to_numpy(), y_val.to_numpy()

        logger.info(
            "campaign fold start market=%s horizon=%s fold=%d",
            market,
            horizon,
            fold_index,
            extra={"stage_marker": True},
        )

        lgbm_model = _fit_point_model("lightgbm", lgbm_params, x_train_arr, y_train_arr)
        xgb_model = _fit_point_model("xgboost", xgb_params, x_train_arr, y_train_arr)

        # REQ-ATE-016: confidence 캘리브레이션용 분위수 보조 모델 — Optuna
        # trial 객체를 참조하지 않는다(동결 lgbm_params 재사용, REQ-AT-062 원칙 계승).
        q10_model = lgb.LGBMRegressor(
            **{
                **_DEFAULT_LGBM_VERBOSITY,
                **lgbm_params,
                "objective": "quantile",
                "alpha": QUANTILE_ALPHAS[0],
            }
        )
        q90_model = lgb.LGBMRegressor(
            **{
                **_DEFAULT_LGBM_VERBOSITY,
                **lgbm_params,
                "objective": "quantile",
                "alpha": QUANTILE_ALPHAS[1],
            }
        )
        q10_model.fit(x_train_arr, y_train_arr)
        q90_model.fit(x_train_arr, y_train_arr)

        lgbm_preds = np.asarray(lgbm_model.predict(x_val_arr))
        xgb_preds = np.asarray(xgb_model.predict(x_val_arr))
        p10_preds = np.asarray(q10_model.predict(x_val_arr))
        p90_preds = np.asarray(q90_model.predict(x_val_arr))

        ensemble_scores, confidences = _compute_fold_ensemble(
            lgbm_preds, xgb_preds, p10_preds, p90_preds
        )

        records.append(
            MarketHorizonFoldRecord(
                fold_index=fold_index,
                train_end=date_bounds.train_end,
                val_start=date_bounds.val_start,
                val_end=date_bounds.val_end,
                lightgbm_metrics=compute_backtest_metrics(lgbm_preds, y_val_arr, confidences),
                xgboost_metrics=compute_backtest_metrics(xgb_preds, y_val_arr, confidences),
                ensemble_metrics=compute_backtest_metrics(ensemble_scores, y_val_arr, confidences),
            )
        )

        logger.info(
            "campaign fold complete market=%s horizon=%s fold=%d",
            market,
            horizon,
            fold_index,
            extra={"stage_marker": True},
        )
        # REQ-ATE-037: lgbm_model/xgb_model/q10_model/q90_model은 이 반복이
        # 끝나며 스코프를 벗어나 폐기된다 — 디스크에 기록하지 않는다.

    return records


def _assemble_campaign_dataset(
    engine: Engine,
    market: str,
    cache_dir: Path,
    data_as_of: date,
    feature_code_version: str,
    calendar_code: str,
) -> pd.DataFrame:
    """REQ-ATE-022: 시장당 정확히 1회 데이터셋을 조립한다.

    `train.py`의 기존 조립 경로(`_assemble_market_dataset()` — DB 조회→
    `dataset.assemble_dataset()`→캐싱)를 그대로 재사용한다(REQ-ATE-006/014,
    재구현 금지) — 시장별 캘린더 override(`_MARKET_CALENDAR_CODE_OVERRIDE`)도
    `train.py`와 동일하게 적용한다.
    """
    market_calendar_code = train_module._MARKET_CALENDAR_CODE_OVERRIDE.get(market, calendar_code)
    calendar = fetch_market_calendar(engine, market_calendar_code)
    return train_module._assemble_market_dataset(
        engine, calendar, market, cache_dir, data_as_of, feature_code_version
    )


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """`run_walk_forward_campaign()` 실행 결과 — CLI 종료코드 신호의 기반 데이터."""

    success: bool
    records_by_market_horizon: dict[tuple[str, int], list[MarketHorizonFoldRecord]] = field(
        default_factory=dict
    )
    frozen_params: dict[tuple[str, int, str], dict[str, Any]] = field(default_factory=dict)
    errors_by_market_horizon: dict[tuple[str, int], str] = field(default_factory=dict)
    error: str | None = None


def run_walk_forward_campaign(
    *,
    trainer_engine: Engine,
    calendar_code: str,
    cache_dir: Path,
    data_as_of: date,
    feature_code_version: str,
    optuna_storage_dir: Path,
    oos_span_years: int = CAMPAIGN_OOS_EVALUATION_SPAN_YEARS,
    optuna_trials: int = CAMPAIGN_OPTUNA_TRIALS,
    val_size: int = CAMPAIGN_VAL_SIZE_TRADING_DAYS,
    timeout_hours: float = CAMPAIGN_TIMEOUT_HOURS,
) -> CampaignResult:
    """캠페인 전체를 순서대로 실행한다(design.md §2 진입점).

    시장당 1회 데이터셋 조립(REQ-ATE-022) → 8개 포인트 조합당 1회 초기
    이력 튜닝(REQ-ATE-027) → (시장,horizon) 4개 조합 각각의 주간 폴드
    순회(REQ-ATE-011/015). 한 (시장,horizon) 조합의 예외는
    `errors_by_market_horizon`에 기록되고 나머지 조합 순회는 계속된다
    (AC-ATE-008/§C 엣지케이스1 — 단일 조합 실패가 전체 캠페인을 중단시키지 않음).
    """
    try:
        deadline = time.monotonic() + timeout_hours * 3600

        panels: dict[str, pd.DataFrame] = {}
        for market in MARKETS:
            panels[market] = _assemble_campaign_dataset(
                trainer_engine, market, cache_dir, data_as_of, feature_code_version, calendar_code
            )

        frozen_params: dict[tuple[str, int, str], dict[str, Any]] = {}
        for market, horizon, algorithm in POINT_COMBOS:
            frozen_params[(market, horizon, algorithm)] = tune_initial_history_hyperparameters(
                panels[market],
                market,
                horizon,
                algorithm,
                optuna_storage_dir,
                trial_count=optuna_trials,
                oos_span_years=oos_span_years,
            )

        records_by_market_horizon: dict[tuple[str, int], list[MarketHorizonFoldRecord]] = {}
        errors_by_market_horizon: dict[tuple[str, int], str] = {}
        for market in MARKETS:
            panel = panels[market]
            trade_dates = extract_global_trade_date_axis(panel)
            initial_train_end_idx = _initial_train_end_index(len(trade_dates), oos_span_years)
            n_folds = _resolve_evaluation_fold_count(
                len(trade_dates), initial_train_end_idx, val_size
            )
            for horizon in HORIZONS:
                try:
                    records_by_market_horizon[(market, horizon)] = run_campaign_for_market_horizon(
                        panel,
                        market,
                        horizon,
                        initial_train_end_idx,
                        n_folds,
                        frozen_params_by_algorithm={
                            "lightgbm": frozen_params[(market, horizon, "lightgbm")],
                            "xgboost": frozen_params[(market, horizon, "xgboost")],
                        },
                        val_size=val_size,
                        timeout_deadline=deadline,
                    )
                except Exception as exc:  # noqa: BLE001 — 조합 단위 격리(AC-ATE-008)
                    logger.error(
                        "campaign combo failed market=%s horizon=%s: %s",
                        market,
                        horizon,
                        exc,
                        exc_info=True,
                    )
                    errors_by_market_horizon[(market, horizon)] = str(exc)

        return CampaignResult(
            success=True,
            records_by_market_horizon=records_by_market_horizon,
            frozen_params=frozen_params,
            errors_by_market_horizon=errors_by_market_horizon,
        )
    except Exception as exc:  # noqa: BLE001 — 오케스트레이터 최상위 캐치-올(종료코드 신호 계약)
        logger.error("campaign failed: %s", exc, exc_info=True)
        return CampaignResult(success=False, error=str(exc))


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점 — 성공 시 `0`, 실패 시 `1`을 반환한다(REQ-ATE-023/024).

    `scheduler.py`의 `register_default_jobs()`에는 등록하지 않는다(REQ-ATE-023)
    — 사용자가 로컬(MacBook)에서 `python -m analyzer.training.campaign`으로
    직접 호출하는 독립 CLI다.
    """
    parser = argparse.ArgumentParser(prog="analyzer.training.campaign")
    parser.add_argument("--calendar-code", default="KRX")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--data-as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--feature-code-version", required=True)
    parser.add_argument("--optuna-storage-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    # `--models-root`는 챔피언 최종 폴드 아티팩트 저장 경로용으로 예약되어
    # 있다(M4/M5/M6 소관, REQ-ATE-031) — 이 골격 마일스톤은 아직 소비하지 않는다.

    engine = build_trainer_engine()

    result = run_walk_forward_campaign(
        trainer_engine=engine,
        calendar_code=args.calendar_code,
        cache_dir=args.cache_dir,
        data_as_of=args.data_as_of,
        feature_code_version=args.feature_code_version,
        optuna_storage_dir=args.optuna_storage_dir,
    )
    if not result.success:
        print(f"캠페인 실패: {result.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

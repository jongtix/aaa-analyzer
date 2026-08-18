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
from collections.abc import Callable, Mapping, Sequence
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
from analyzer.orchestration import activation as activation_module
from analyzer.training import campaign_metrics, stabilization
from analyzer.training import persistence as persistence_module
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


CAMPAIGN_TRAINED_DATE_DEFAULT_SOURCE = "data_as_of"
"""M6 Part 0: 챔피언 최종 재학습 아티팩트의 `trained_date`는 캠페인의
`data_as_of`(전체 이력 조립의 컷오프 날짜)를 그대로 사용한다 — 개별 폴드의
`val_end`가 아니라, 캠페인 실행 시점 자체를 아티팩트 버전으로 기록한다."""


def append_fold_records_to_jsonl(
    jsonl_dir: Path, market: str, horizon: int, records: Sequence[MarketHorizonFoldRecord]
) -> None:
    """M6 Part 0(REQ-ATE-038): (시장,horizon) 조합의 폴드 기록 전체를
    포인트(lgbm/xgb) + 앙상블 3개 JSONL 스트림에 append한다 —
    `campaign_metrics.append_fold_metrics()`(M4, 시그니처 무수정)를 그대로
    소비한다."""
    for record in records:
        campaign_metrics.append_fold_metrics(
            jsonl_dir,
            market,
            horizon,
            "lightgbm",
            record.fold_index,
            record.train_end,
            record.val_start,
            record.val_end,
            record.lightgbm_metrics,
        )
        campaign_metrics.append_fold_metrics(
            jsonl_dir,
            market,
            horizon,
            "xgboost",
            record.fold_index,
            record.train_end,
            record.val_start,
            record.val_end,
            record.xgboost_metrics,
        )
        campaign_metrics.append_fold_metrics(
            jsonl_dir,
            market,
            horizon,
            campaign_metrics.ENSEMBLE_ALGORITHM_TAG,
            record.fold_index,
            record.train_end,
            record.val_start,
            record.val_end,
            record.ensemble_metrics,
        )


def _final_fold_train_window(
    panel: pd.DataFrame,
    horizon: int,
    initial_train_end_idx: int,
    n_folds: int,
    val_size: int = CAMPAIGN_VAL_SIZE_TRADING_DAYS,
) -> pd.DataFrame:
    """M6 Part 0(REQ-ATE-030): 캠페인의 마지막(가장 최근 데이터를 포함하는)
    폴드의 학습 구간 데이터를 반환한다 — 챔피언 1차 배포 최종 재학습용."""
    trade_dates = extract_global_trade_date_axis(panel)
    index_bounds_list = weekly_stride_fold_index_bounds(
        n_dates=len(trade_dates),
        horizon=horizon,
        initial_train_end=initial_train_end_idx,
        val_size=val_size,
        n_folds=n_folds,
    )
    last_bounds = index_bounds_list[-1]
    date_bounds = map_index_bounds_to_dates(last_bounds, trade_dates)
    train_df, _val_df = slice_panel_by_date_bounds(panel, date_bounds)
    return train_df


def train_and_persist_champion_artifact(
    panel: pd.DataFrame,
    market: str,
    horizon: int,
    algorithm: str,
    initial_train_end_idx: int,
    n_folds: int,
    frozen_params: Mapping[str, Any],
    models_root: Path,
    trained_date: date,
    val_size: int = CAMPAIGN_VAL_SIZE_TRADING_DAYS,
) -> tuple[persistence_module.SavedModel, list[str], int]:
    """M6 Part 0(REQ-ATE-030): 챔피언으로 선정된 (시장,horizon,algorithm)
    조합을 캠페인이 검증한 동결 하이퍼파라미터로 마지막 폴드 학습 구간(최신
    데이터 포함) 재학습하고 `persistence.save_model_native()`로 저장한다
    — 캠페인이 검증한 것과 동일한 하이퍼파라미터로 배포한다(REQ-ATE-030,
    학습-서빙 불일치 방지). 폴드 순회 자체(`run_campaign_for_market_horizon`)는
    이 재학습된 모델을 저장하지 않는다(REQ-ATE-037 무관 — 이 함수는 별도의
    단일 최종 재학습이다).
    """
    train_df = _final_fold_train_window(panel, horizon, initial_train_end_idx, n_folds, val_size)
    feature_columns, x_train, y_train = train_module._split_features_and_labels(train_df, horizon)
    model = _fit_point_model(algorithm, frozen_params, x_train.to_numpy(), y_train.to_numpy())
    saved = persistence_module.save_model_native(
        model, models_root, market, horizon, algorithm, trained_date
    )
    return saved, feature_columns, len(x_train)


@dataclass(frozen=True, slots=True)
class ComboActivationOutcome:
    """M6 Part 0: (시장,horizon) 조합 1개의 안정화 판정 + 챔피언 선정 +
    (배포 가능 시) 활성화 결과를 담는다 — 캠페인 요약 리포트(REQ-ATE-034)와
    self-verification 근거로 함께 소비된다."""

    market: str
    horizon: int
    lightgbm_verdict: stabilization.ComboStabilizationVerdict
    xgboost_verdict: stabilization.ComboStabilizationVerdict
    champion_selection: stabilization.ChampionSelection
    persisted_algorithms: tuple[str, ...] = ()


def activate_market_horizon_combo(
    *,
    panel: pd.DataFrame,
    market: str,
    horizon: int,
    fold_records: Sequence[MarketHorizonFoldRecord],
    jsonl_dir: Path,
    models_root: Path,
    initial_train_end_idx: int,
    n_folds: int,
    frozen_params_by_algorithm: Mapping[str, Mapping[str, Any]],
    trained_date: date,
    val_size: int = CAMPAIGN_VAL_SIZE_TRADING_DAYS,
) -> ComboActivationOutcome:
    """M6 Part 0/Part 1 통합 배선: 폴드 기록 → 안정화 판정(GATE-1/2/3, M5) →
    챔피언 스코어링 전략 선정(F1) → (배포 가능 시) 최종 재학습+저장(REQ-ATE-030)
    + 사이드카(REQ-ATE-031/032/033) + 활성화 매니페스트(§2.9, REQ-ATE-052
    1차 배포 경로) + 스코어링 전략 매니페스트(REQ-ATE-050) 갱신까지 1회
    (시장,horizon) 조합 전체를 처리한다.
    """
    append_fold_records_to_jsonl(jsonl_dir, market, horizon, fold_records)

    lgbm_rank_ic = [r.lightgbm_metrics.rank_ic for r in fold_records]
    xgb_rank_ic = [r.xgboost_metrics.rank_ic for r in fold_records]
    ensemble_rank_ic = [r.ensemble_metrics.rank_ic for r in fold_records]

    lgbm_verdict = stabilization.evaluate_combo_stabilization(
        market, horizon, "lightgbm", lgbm_rank_ic
    )
    xgb_verdict = stabilization.evaluate_combo_stabilization(
        market, horizon, "xgboost", xgb_rank_ic
    )

    strategy_rolling_mean_rank_ic: dict[str, float] = {
        "lightgbm": stabilization.rolling_mean_rank_ic(lgbm_rank_ic) or float("-inf"),
        "xgboost": stabilization.rolling_mean_rank_ic(xgb_rank_ic) or float("-inf"),
        "ensemble": stabilization.rolling_mean_rank_ic(ensemble_rank_ic) or float("-inf"),
    }
    champion_selection = stabilization.select_champion_strategy(
        market, horizon, lgbm_verdict, xgb_verdict, strategy_rolling_mean_rank_ic
    )

    if champion_selection.deployment_prohibited:
        return ComboActivationOutcome(
            market=market,
            horizon=horizon,
            lightgbm_verdict=lgbm_verdict,
            xgboost_verdict=xgb_verdict,
            champion_selection=champion_selection,
            persisted_algorithms=(),
        )

    # REQ-ATE-046: 챔피언이 앙상블이면 lgbm+xgb 둘 다, 단독이면 그 알고리즘만
    # 활성화 매니페스트 대상으로 유지한다(반대편 미안정화 알고리즘은 제외).
    if champion_selection.champion_algorithm == "ensemble":
        algorithms_to_persist: tuple[str, ...] = ("lightgbm", "xgboost")
    else:
        assert champion_selection.champion_algorithm is not None
        algorithms_to_persist = (champion_selection.champion_algorithm,)

    verdict_by_algorithm = {"lightgbm": lgbm_verdict, "xgboost": xgb_verdict}
    aggregate_metrics_by_algorithm = {
        "lightgbm": campaign_metrics.compute_aggregate_metrics(lgbm_rank_ic),
        "xgboost": campaign_metrics.compute_aggregate_metrics(xgb_rank_ic),
    }

    persisted: list[str] = []
    for algorithm in algorithms_to_persist:
        saved, feature_columns, final_fold_row_count = train_and_persist_champion_artifact(
            panel,
            market,
            horizon,
            algorithm,
            initial_train_end_idx,
            n_folds,
            frozen_params_by_algorithm[algorithm],
            models_root,
            trained_date,
            val_size=val_size,
        )
        jsonl_relative_path = campaign_metrics.fold_metrics_jsonl_filename(
            market, horizon, algorithm
        )
        campaign_metrics.write_sidecar_metadata(
            saved.model_path,
            market=market,
            horizon=horizon,
            algorithm=algorithm,
            aggregate_metrics=aggregate_metrics_by_algorithm[algorithm],
            final_fold_train_row_count=final_fold_row_count,
            frozen_hyperparameters=frozen_params_by_algorithm[algorithm],
            feature_columns=feature_columns,
            fold_metrics_jsonl_relative_path=jsonl_relative_path,
        )
        this_verdict = verdict_by_algorithm[algorithm]
        activation_module.promote_activation_manifest(
            models_root,
            market=market,
            horizon=horizon,
            algorithm=algorithm,
            merged_to_active=True,
            gate_passed=True,
            trained_date=trained_date,
            sidecar_sha256=saved.sha256,
            promotion_basis={
                "path": "initial_deployment",
                "gate1_rolling_mean_rank_ic": this_verdict.gate1_rolling_mean_rank_ic,
                "gate2_mean_rank_ic": this_verdict.gate2_mean_rank_ic,
                "gate3_rolling_icir": this_verdict.gate3_rolling_icir,
            },
        )
        persisted.append(algorithm)

    activation_module.write_strategy_manifest(
        models_root,
        market=market,
        horizon=horizon,
        active_strategy=champion_selection.champion_algorithm,
        basis={"eligible_strategies": list(champion_selection.eligible_strategies)},
    )

    return ComboActivationOutcome(
        market=market,
        horizon=horizon,
        lightgbm_verdict=lgbm_verdict,
        xgboost_verdict=xgb_verdict,
        champion_selection=champion_selection,
        persisted_algorithms=tuple(persisted),
    )


def _combo_verdict_stub(
    outcome: ComboActivationOutcome,
) -> list[campaign_metrics.ComboGateVerdictStub]:
    """M6 Part 0(REQ-ATE-034): M4가 정의한 스텁 구조(시그니처 무수정)를 실제
    안정화 판정 데이터로 채워 캠페인 요약 리포트에 반영한다."""
    stubs: list[campaign_metrics.ComboGateVerdictStub] = []
    for algorithm, verdict in (
        ("lightgbm", outcome.lightgbm_verdict),
        ("xgboost", outcome.xgboost_verdict),
    ):
        stubs.append(
            campaign_metrics.ComboGateVerdictStub(
                market=outcome.market,
                horizon=outcome.horizon,
                algorithm=algorithm,
                gate_verdict="stabilized" if verdict.stabilized else "not_stabilized",
                supporting_metrics={
                    "gate1_passed": verdict.gate1_passed,
                    "gate2_passed": verdict.gate2_passed,
                    "gate3_passed": verdict.gate3_passed,
                    "gate1_rolling_mean_rank_ic": verdict.gate1_rolling_mean_rank_ic,
                    "gate2_mean_rank_ic": verdict.gate2_mean_rank_ic,
                    "gate3_rolling_icir": verdict.gate3_rolling_icir,
                },
            )
        )
    stubs.append(
        campaign_metrics.ComboGateVerdictStub(
            market=outcome.market,
            horizon=outcome.horizon,
            algorithm="champion_selection",
            gate_verdict=(
                "deployment_prohibited"
                if outcome.champion_selection.deployment_prohibited
                else f"champion={outcome.champion_selection.champion_algorithm}"
            ),
            supporting_metrics={
                "eligible_strategies": list(outcome.champion_selection.eligible_strategies),
                "persisted_algorithms": list(outcome.persisted_algorithms),
                "diagnostics": dict(outcome.champion_selection.diagnostics),
            },
        )
    )
    return stubs


@dataclass(frozen=True, slots=True)
class CampaignActivationResult:
    """`run_walk_forward_campaign_and_activate()`의 최종 결과 — 캠페인 실행
    결과(`CampaignResult`) + (시장,horizon) 조합별 활성화 결과."""

    campaign_result: CampaignResult
    activation_outcomes: dict[tuple[str, int], ComboActivationOutcome] = field(default_factory=dict)
    summary_report_path: Path | None = None


def run_walk_forward_campaign_and_activate(
    *,
    trainer_engine: Engine,
    calendar_code: str,
    cache_dir: Path,
    models_root: Path,
    data_as_of: date,
    feature_code_version: str,
    optuna_storage_dir: Path,
    summary_report_path: Path,
    oos_span_years: int = CAMPAIGN_OOS_EVALUATION_SPAN_YEARS,
    optuna_trials: int = CAMPAIGN_OPTUNA_TRIALS,
    val_size: int = CAMPAIGN_VAL_SIZE_TRADING_DAYS,
    timeout_hours: float = CAMPAIGN_TIMEOUT_HOURS,
) -> CampaignActivationResult:
    """M6 Part 0 최상위 진입점: `run_walk_forward_campaign()`(M3, 골격) 실행
    직후 각 (시장,horizon) 조합에 대해 `activate_market_horizon_combo()`를
    호출해 JSONL 영속화(M4) + 안정화 판정/챔피언 선정(M5) + 1차 배포
    활성화(M6 §2.9)까지 end-to-end로 연결한다. `panels`는 재구성하지 않고
    `run_walk_forward_campaign()` 내부에서 조립된 데이터를 재사용하기 위해,
    이 함수는 그 함수를 감싸는 대신 동일한 흐름을 인라인으로 재현한다
    (패널을 캠페인 실행 후에도 활성화 단계에서 재사용해야 하므로 —
    `run_walk_forward_campaign()`은 패널을 반환하지 않는다).
    """
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
    activation_outcomes: dict[tuple[str, int], ComboActivationOutcome] = {}
    all_stubs: list[campaign_metrics.ComboGateVerdictStub] = []

    for market in MARKETS:
        panel = panels[market]
        trade_dates = extract_global_trade_date_axis(panel)
        initial_train_end_idx = _initial_train_end_index(len(trade_dates), oos_span_years)
        n_folds = _resolve_evaluation_fold_count(len(trade_dates), initial_train_end_idx, val_size)
        for horizon in HORIZONS:
            try:
                records = run_campaign_for_market_horizon(
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
                records_by_market_horizon[(market, horizon)] = records

                outcome = activate_market_horizon_combo(
                    panel=panel,
                    market=market,
                    horizon=horizon,
                    fold_records=records,
                    jsonl_dir=models_root / market / str(horizon),
                    models_root=models_root,
                    initial_train_end_idx=initial_train_end_idx,
                    n_folds=n_folds,
                    frozen_params_by_algorithm={
                        "lightgbm": frozen_params[(market, horizon, "lightgbm")],
                        "xgboost": frozen_params[(market, horizon, "xgboost")],
                    },
                    trained_date=data_as_of,
                    val_size=val_size,
                )
                activation_outcomes[(market, horizon)] = outcome
                all_stubs.extend(_combo_verdict_stub(outcome))
            except Exception as exc:  # noqa: BLE001 — 조합 단위 격리(AC-ATE-008)
                logger.error(
                    "campaign combo failed market=%s horizon=%s: %s",
                    market,
                    horizon,
                    exc,
                    exc_info=True,
                )
                errors_by_market_horizon[(market, horizon)] = str(exc)

    campaign_metrics.write_campaign_summary_report(summary_report_path, all_stubs)

    campaign_result = CampaignResult(
        success=True,
        records_by_market_horizon=records_by_market_horizon,
        frozen_params=frozen_params,
        errors_by_market_horizon=errors_by_market_horizon,
    )
    return CampaignActivationResult(
        campaign_result=campaign_result,
        activation_outcomes=activation_outcomes,
        summary_report_path=summary_report_path,
    )


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
    parser.add_argument("--summary-report-path", type=Path, required=True)
    args = parser.parse_args(argv)

    engine = build_trainer_engine()

    activation_result = run_walk_forward_campaign_and_activate(
        trainer_engine=engine,
        calendar_code=args.calendar_code,
        cache_dir=args.cache_dir,
        models_root=args.models_root,
        data_as_of=args.data_as_of,
        feature_code_version=args.feature_code_version,
        optuna_storage_dir=args.optuna_storage_dir,
        summary_report_path=args.summary_report_path,
    )
    if not activation_result.campaign_result.success:
        print(f"캠페인 실패: {activation_result.campaign_result.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

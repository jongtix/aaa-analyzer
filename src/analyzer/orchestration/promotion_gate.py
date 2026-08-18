"""상시 챔피언/챌린저 오프라인 평가 + 승격/보류 판정 (SPEC-ANALYZER-TRAIN-EVAL-001
M6, design.md §8, v0.3.0 F2 정정).

REQ-ATE-056: 1차 배포 이후 매 주간 재학습이 산출하는 신규 모델을 챌린저로
취급한다 — `training.split.expanding_window_folds(n_folds=1)`로 [학습~
T-홀드아웃)/[T-홀드아웃~T) 경계를 산출하고, `training.models.
train_pooled_models()`의 `eval_data_by_combo`/`early_stopping_rounds`
(REQ-AT-105 훅 재사용)로 챌린저를 [학습~T-홀드아웃) 구간에만 학습시킨다 —
홀드아웃 구간은 챌린저의 가중치 갱신에 전혀 사용되지 않으며, 조기 종료와
표본외 평가 양쪽 목적으로 재사용된다(§B/design.md §8 "알려진 한계 —
잔여 낙관 편향"은 수용된 한계이지 결함이 아니다).

REQ-ATE-057: 챌린저의 홀드아웃 평가 지표가 산출되면, 현재 활성 챔피언
아티팩트를 `persistence.py`가 아니라 이 모듈이 직접 프레임워크 네이티브
로드 API(LightGBM `Booster(model_file=...)` / XGBoost `Booster.load_model()`)로
로드해 **동일한 홀드아웃 구간**에서 재채점한다 — 활성화 매니페스트에
보존된 챔피언의 과거 기록 지표(다른 시장 국면의 윈도우)와 비교하지
않는다(국면-드리프트 편향 회피).

REQ-ATE-058/059: 챌린저 재채점 지표가 챔피언 재채점 지표보다 이름 있는
임계값(`stabilization.PROMOTION_GATE_MIN_IMPROVEMENT`, REQ-ATE-061 단일
소스 원칙) 이상 우수하면 승격, 아니면 보류한다 — 판정은 (시장,horizon,
algorithm) 조합별로 독립적으로 수행된다.

REQ-ATE-060: 승격/보류 알림은 기존 경로(REQ-ATA-060/061 — Prometheus 게이지/
카운터를 vmalert가 소비해 텔레그램으로 트리거)를 그대로 재사용한다 — 이
모듈은 새 알림 채널을 설계하지 않는다. `evaluate_and_promote()`가 반환하는
`PromotionVerdict`를 `orchestration/runner.py`가 `TrainingMetrics.
record_success(outcome=...)` + 신규 Rank IC 게이지(REQ-ATE-066)로 발행하는
것 자체가 "알림 트리거 시그널"이다(별도 notify 함수를 새로 만들지 않음 —
research.md에서 재사용 가능한 기존 텔레그램/vmalert 직접 호출 함수를 찾지
못했으나, REQ-ATA-060/061이 이미 확립한 "Prometheus 게이지 발행 → vmalert
소비" 경로가 정확히 이 요구사항이 지목하는 기존 채널이다).

§C 엣지케이스 3: 1차 배포 이전(활성 챔피언 없음)에는 `champion_model_paths`에
해당 조합이 없으므로 `evaluate_promotion_gate()`가 그 조합을 건너뛴다 —
존재하지 않는 챔피언과의 비교를 시도하지 않는다(§2.9 1차 배포 경로로
라우팅하는 책임은 호출자에게 있다).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from analyzer.common.logging import get_logger
from analyzer.orchestration import activation as activation_module
from analyzer.training import persistence as persistence_module
from analyzer.training.backtest import BacktestMetrics, compute_backtest_metrics
from analyzer.training.models import HORIZONS, MARKETS, PooledModel, train_pooled_models
from analyzer.training.panel_folds import (
    PanelFoldIndexBounds,
    extract_global_trade_date_axis,
    map_index_bounds_to_dates,
    slice_panel_by_date_bounds,
)
from analyzer.training.split import expanding_window_folds
from analyzer.training.stabilization import PROMOTION_GATE_MIN_IMPROVEMENT
from analyzer.training.train import _split_features_and_labels

logger = get_logger(__name__)

PROMOTION_GATE_HOLDOUT_TRADING_DAYS: int = 20
"""REQ-ATE-056: 챌린저 홀드아웃 구간의 길이(영업일) — 조기 종료 라운드
선택과 표본외 평가 양쪽 목적으로 재사용된다. 잠정치(REVISABLE) — D20
horizon의 purge gap과 동일한 규모로 설정해 최소 표본을 보장한다."""

PROMOTION_GATE_EARLY_STOPPING_ROUNDS: int = 20
"""REQ-ATE-056: `train_pooled_models()`의 `early_stopping_rounds` 인자(REQ-AT-105
훅) — 챌린저 학습에만 적용되며, 캠페인(`campaign.py`)의 폴드별 학습에는
적용되지 않는다(캠페인은 조기 종료를 사용하지 않음, 별개 관심사)."""

_POINT_ALGORITHMS: tuple[str, ...] = ("lightgbm", "xgboost")


def resolve_challenger_holdout_index_bounds(
    n_samples: int, horizon: int, holdout_size: int = PROMOTION_GATE_HOLDOUT_TRADING_DAYS
) -> PanelFoldIndexBounds:
    """REQ-ATE-056: `expanding_window_folds(n_folds=1)`(TRAIN-001 REQ-AT-070/072,
    §2.10의 소규모 재사용 — 캠페인 본체의 주 폴드 생성기와는 무관, design.md
    §2A)로 [학습~T-홀드아웃)/[T-홀드아웃~T) 경계를 산출한다. `test_ratio=0.0`으로
    호출해 공유 고정 홀드아웃(테스트) 구간을 배제하고, 검증(val) 구간 자체를
    챌린저의 홀드아웃으로 사용한다.
    """
    bounds = expanding_window_folds(
        n_samples, horizon, n_folds=1, val_size=holdout_size, test_ratio=0.0
    )[0]
    return PanelFoldIndexBounds(
        train_end=bounds.train_end, val_start=bounds.val_start, val_end=bounds.val_end
    )


def build_pooled_train_holdout_data(
    panel_by_market: Mapping[str, pd.DataFrame],
    holdout_size: int = PROMOTION_GATE_HOLDOUT_TRADING_DAYS,
) -> tuple[
    dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
]:
    """REQ-ATE-056: 시장별 패널에 대해 (시장,horizon) 4개 조합 각각의
    [학습~T-홀드아웃)/[T-홀드아웃~T) 날짜-경계 분할을 구성한다.

    `panel_folds.py`의 leakage-safe 날짜 값 비교 필터링(REQ-ATE-019와
    동일한 원칙 재사용)으로 학습/홀드아웃을 분리하며, 홀드아웃 구간의
    데이터는 어떤 학습 호출의 `data_by_combo`에도 포함되지 않는다
    (in-sample 오염 없음, AC-ATE-041).
    """
    data_by_combo: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    eval_data_by_combo: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    for market in MARKETS:
        panel = panel_by_market[market]
        trade_dates = extract_global_trade_date_axis(panel)
        for horizon in HORIZONS:
            index_bounds = resolve_challenger_holdout_index_bounds(
                len(trade_dates), horizon, holdout_size
            )
            date_bounds = map_index_bounds_to_dates(index_bounds, trade_dates)
            train_df, holdout_df = slice_panel_by_date_bounds(panel, date_bounds)

            _, x_train, y_train = _split_features_and_labels(train_df, horizon)
            _, x_holdout, y_holdout = _split_features_and_labels(holdout_df, horizon)

            data_by_combo[(market, horizon)] = (x_train.to_numpy(), y_train.to_numpy())
            eval_data_by_combo[(market, horizon)] = (x_holdout.to_numpy(), y_holdout.to_numpy())

    return data_by_combo, eval_data_by_combo


def train_challenger_models(
    data_by_combo: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    eval_data_by_combo: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    *,
    lgbm_params: Mapping[str, Any] | None = None,
    xgb_params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = PROMOTION_GATE_EARLY_STOPPING_ROUNDS,
) -> dict[tuple, PooledModel]:
    """REQ-ATE-056: `train_pooled_models()`의 `eval_data_by_combo`/
    `early_stopping_rounds`(REQ-AT-105 훅)를 그대로 호출한다 — 재구현하지
    않는다. 홀드아웃 구간은 조기 종료 판단에만 관여하며 가중치 갱신에는
    사용되지 않는다(LightGBM/XGBoost 네이티브 조기 종료 메커니즘의 계약)."""
    return train_pooled_models(
        data_by_combo,
        lgbm_params=lgbm_params,
        xgb_params=xgb_params,
        eval_data_by_combo=eval_data_by_combo,
        early_stopping_rounds=early_stopping_rounds,
    )


def evaluate_models_on_holdout(
    models: Mapping[tuple, PooledModel],
    eval_data_by_combo: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[str, int, str], BacktestMetrics]:
    """챌린저 포인트 모델(lgbm/xgb, 분위수 보조 모델 제외)을 홀드아웃 구간에서
    `backtest.compute_backtest_metrics()`로 평가한다(REQ-ATE-056 3단계).
    confidence는 채점 목적상 중립값(0.5)으로 고정한다(§2.10은 confidence
    캘리브레이션이 아니라 Rank IC 승격 판정만을 다룬다).
    """
    results: dict[tuple[str, int, str], BacktestMetrics] = {}
    for model_key, model in models.items():
        market, horizon, tag = model_key[0], model_key[1], model_key[2]
        if tag not in _POINT_ALGORITHMS:
            continue  # 분위수 보조 모델은 승격 판정 대상이 아니다.
        x_holdout, y_holdout = eval_data_by_combo[(market, horizon)]
        preds = np.asarray(model.predict(x_holdout))
        neutral_confidences = np.full(len(preds), 0.5)
        results[(market, horizon, tag)] = compute_backtest_metrics(
            preds, y_holdout, neutral_confidences
        )
    return results


def load_champion_native(model_path: Path, algorithm: str) -> Any:
    """REQ-ATE-057: 현재 활성 챔피언 아티팩트를 프레임워크 네이티브 로드
    API로 직접 로드한다 — `persistence.py`는 저장 전용 책임을 유지하며
    이 로드에 관여하지 않는다."""
    if algorithm == "lightgbm":
        return lgb.Booster(model_file=str(model_path))
    if algorithm == "xgboost":
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        return booster
    raise ValueError(f"지원하지 않는 algorithm: {algorithm}")


def _predict_native(model: Any, algorithm: str, x: np.ndarray) -> np.ndarray:
    if algorithm == "lightgbm":
        return np.asarray(model.predict(x))
    if algorithm == "xgboost":
        return np.asarray(model.predict(xgb.DMatrix(x)))
    raise ValueError(f"지원하지 않는 algorithm: {algorithm}")


@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    """(시장,horizon,algorithm) 조합 1개의 승격/보류 판정 결과(REQ-ATE-057/058/059)."""

    market: str
    horizon: int
    algorithm: str
    promoted: bool
    challenger_rank_ic: float
    champion_rank_ic: float
    challenger_trained_date: date


def evaluate_promotion_gate(
    *,
    panel_by_market: Mapping[str, pd.DataFrame],
    champion_model_paths: Mapping[tuple[str, int, str], Path],
    challenger_trained_date: date,
    holdout_size: int = PROMOTION_GATE_HOLDOUT_TRADING_DAYS,
    early_stopping_rounds: int = PROMOTION_GATE_EARLY_STOPPING_ROUNDS,
    lgbm_params: Mapping[str, Any] | None = None,
    xgb_params: Mapping[str, Any] | None = None,
    threshold: float = PROMOTION_GATE_MIN_IMPROVEMENT,
) -> dict[tuple[str, int, str], PromotionVerdict]:
    """REQ-ATE-056/057/058/059: 챌린저 학습/홀드아웃 분리 → 챔피언 동일-윈도우
    재채점 → 조합별 독립 승격/보류 판정 전체 흐름을 실행한다.

    `champion_model_paths`에 없는 조합(활성 챔피언이 아직 없는 1차 배포
    이전 상태, §C 엣지케이스 3)은 판정 대상에서 제외된다 — 존재하지 않는
    챔피언과 비교를 시도하지 않는다.
    """
    data_by_combo, eval_data_by_combo = build_pooled_train_holdout_data(
        panel_by_market, holdout_size
    )
    challenger_models = train_challenger_models(
        data_by_combo,
        eval_data_by_combo,
        lgbm_params=lgbm_params,
        xgb_params=xgb_params,
        early_stopping_rounds=early_stopping_rounds,
    )
    challenger_metrics = evaluate_models_on_holdout(challenger_models, eval_data_by_combo)

    verdicts: dict[tuple[str, int, str], PromotionVerdict] = {}
    for (market, horizon, algorithm), challenger_bt in challenger_metrics.items():
        champion_path = champion_model_paths.get((market, horizon, algorithm))
        if champion_path is None:
            continue

        champion_model = load_champion_native(champion_path, algorithm)
        x_holdout, y_holdout = eval_data_by_combo[(market, horizon)]
        champion_preds = _predict_native(champion_model, algorithm, x_holdout)
        neutral_confidences = np.full(len(champion_preds), 0.5)
        champion_bt = compute_backtest_metrics(champion_preds, y_holdout, neutral_confidences)

        promoted = (challenger_bt.rank_ic - champion_bt.rank_ic) > threshold
        verdicts[(market, horizon, algorithm)] = PromotionVerdict(
            market=market,
            horizon=horizon,
            algorithm=algorithm,
            promoted=promoted,
            challenger_rank_ic=challenger_bt.rank_ic,
            champion_rank_ic=champion_bt.rank_ic,
            challenger_trained_date=challenger_trained_date,
        )
        logger.info(
            "promotion gate verdict market=%s horizon=%s algorithm=%s promoted=%s "
            "challenger_rank_ic=%.6f champion_rank_ic=%.6f",
            market,
            horizon,
            algorithm,
            promoted,
            challenger_bt.rank_ic,
            champion_bt.rank_ic,
            extra={"stage_marker": True},
        )

    return verdicts


def _read_sidecar_sha256(
    models_root: Path, market: str, horizon: int, algorithm: str, trained_date: date
) -> str:
    """챌린저 아티팩트(원격 학습+`promote_staging_to_active()`로 이미 active
    경로에 병합된 파일)의 `.sha256` 사이드카를 읽는다 — `save_model_native()`가
    저장 시점에 이미 기록한 값이므로 재계산하지 않는다."""
    model_path = persistence_module.model_dir(
        models_root, market, horizon, algorithm
    ) / persistence_module.model_filename(market, horizon, algorithm, trained_date)
    sidecar_path = model_path.with_suffix(model_path.suffix + ".sha256")
    return sidecar_path.read_text(encoding="utf-8").strip()


def evaluate_and_promote(
    *,
    models_root: Path,
    panel_by_market: Mapping[str, pd.DataFrame],
    champion_model_paths: Mapping[tuple[str, int, str], Path],
    challenger_trained_date: date,
    merged_to_active: bool,
    holdout_size: int = PROMOTION_GATE_HOLDOUT_TRADING_DAYS,
    early_stopping_rounds: int = PROMOTION_GATE_EARLY_STOPPING_ROUNDS,
    lgbm_params: Mapping[str, Any] | None = None,
    xgb_params: Mapping[str, Any] | None = None,
    threshold: float = PROMOTION_GATE_MIN_IMPROVEMENT,
) -> dict[tuple[str, int, str], PromotionVerdict]:
    """`evaluate_promotion_gate()` 판정 + 승격된 조합에 대해서만
    `activation.promote_activation_manifest()`를 호출한다(REQ-ATE-052 F3의
    §2.10 경로 — `merged_to_active`와 `gate_passed=verdict.promoted`를
    단일 게이트로 결합)."""
    verdicts = evaluate_promotion_gate(
        panel_by_market=panel_by_market,
        champion_model_paths=champion_model_paths,
        challenger_trained_date=challenger_trained_date,
        holdout_size=holdout_size,
        early_stopping_rounds=early_stopping_rounds,
        lgbm_params=lgbm_params,
        xgb_params=xgb_params,
        threshold=threshold,
    )
    for (market, horizon, algorithm), verdict in verdicts.items():
        if not verdict.promoted:
            continue
        sidecar_sha256 = _read_sidecar_sha256(
            models_root, market, horizon, algorithm, challenger_trained_date
        )
        activation_module.promote_activation_manifest(
            models_root,
            market=market,
            horizon=horizon,
            algorithm=algorithm,
            merged_to_active=merged_to_active,
            gate_passed=True,
            trained_date=challenger_trained_date,
            sidecar_sha256=sidecar_sha256,
            promotion_basis={
                "challenger_rank_ic": verdict.challenger_rank_ic,
                "champion_rank_ic": verdict.champion_rank_ic,
                "path": "standing_gate",
            },
        )
    return verdicts

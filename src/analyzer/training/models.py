"""16개 풀링 모델(8 포인트 + 8 분위수 보조) 학습 (SPEC-ANALYZER-TRAIN-001 M5).

REQ-AT-060: 시장(국내/해외) × horizon(D20/D60) × 알고리즘(LightGBM/XGBoost)
= 8개 포인트 회귀 모델을 학습한다 — objective는 L2 기본, `lgbm_params`/
`xgb_params`로 호출자가 override할 수 있다(§2.5 Huber/윈저라이징 결정을
이 경로로 주입 — 그 결정 자체의 확정은 `thick_tail.py` 하네스의 실데이터
실행 소관이며 이 모듈이 강제 채택하지 않는다).

REQ-AT-061: 시장 × horizon = 4개 조합 각각에 대해 LightGBM
`objective=quantile`(alpha=0.10, alpha=0.90) 분위수 보조 모델 총 8개를
학습한다 — 용도는 `ensemble.py`의 confidence 산출 전용이며, 등급·가격
밴드 산출에는 관여하지 않는다(TECHSPEC 954행).

REQ-AT-062: 분위수 보조 모델은 Optuna 튜닝 대상에서 제외한다(shall not)
— 해당 시장·horizon의 포인트 LightGBM 모델과 동일한 `lgbm_params`를
그대로 재사용한다(shall)(Optuna 통합 자체는 M6 소관, 이 모듈은 하이퍼
파라미터 재사용 계약만 담당).

REQ-AT-105(M6): `eval_data_by_combo`와 `early_stopping_rounds`가 함께
주어지면, LightGBM/XGBoost 각 프레임워크의 네이티브 조기 종료 메커니즘을
사용한다 — trial 자기 자신의 검증 지표 개선이 멈추면 그 trial의 부스팅
자체를 멈춘다. Optuna 프루닝(REQ-AT-104, `tuning.py` 소관)과는 다른
문제를 해결하며, 프루닝 채택 여부와 무관하게 적용된다. 8개 포인트
모델 + 8개 분위수 보조 모델 전부에 동일하게 적용한다.
"""

from collections.abc import Mapping
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from analyzer.common.logging import get_logger

logger = get_logger(__name__)

MARKETS: tuple[str, ...] = ("domestic", "overseas")

HORIZONS: tuple[int, ...] = (20, 60)
QUANTILE_ALPHAS: tuple[float, ...] = (0.10, 0.90)

PooledModel = lgb.LGBMRegressor | xgb.XGBRegressor
"""이 모듈이 반환하는 학습된 모델의 타입(포인트/분위수 보조 공통)."""

_DEFAULT_LGBM_PARAMS: dict[str, Any] = {"verbosity": -1}
_DEFAULT_XGB_PARAMS: dict[str, Any] = {"verbosity": 0}


def train_pooled_models(
    data_by_combo: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    lgbm_params: Mapping[str, Any] | None = None,
    xgb_params: Mapping[str, Any] | None = None,
    eval_data_by_combo: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray]] | None = None,
    early_stopping_rounds: int | None = None,
) -> dict[tuple, PooledModel]:
    """16개 풀링 모델(8 포인트 + 8 분위수 보조)을 전부 학습한다(REQ-AT-060/061/062/105).

    `data_by_combo`는 정확히 `MARKETS × HORIZONS` 4개 조합의 `(X, y)`
    학습 데이터를 가져야 한다 — 조합이 누락되거나 예상 밖 조합이 섞이면
    `ValueError`를 발생시킨다.

    반환 키:
    - 포인트 모델(8개): `(market, horizon, "lightgbm")` / `(market, horizon, "xgboost")`
    - 분위수 보조 모델(8개): `(market, horizon, "lightgbm_quantile", alpha)`

    분위수 보조 모델은 해당 조합의 포인트 LightGBM 모델과 동일한
    `lgbm_params`를 그대로 재사용한다(REQ-AT-062) — `objective`/`alpha`
    키만 quantile용 값으로 덮어쓴다.

    `eval_data_by_combo`와 `early_stopping_rounds`를 함께 지정하면 8개
    포인트 모델 + 8개 분위수 보조 모델 전부에 각 프레임워크의 네이티브
    조기 종료를 적용한다(REQ-AT-105) — 둘 중 하나라도 없으면 조기 종료
    없이 요청한 부스팅 라운드 수를 그대로 사용한다.
    """
    expected_combos = {(market, horizon) for market in MARKETS for horizon in HORIZONS}
    if set(data_by_combo.keys()) != expected_combos:
        raise ValueError(f"data_by_combo는 정확히 {sorted(expected_combos)} 조합을 가져야 한다")

    resolved_lgbm_params = {**_DEFAULT_LGBM_PARAMS, **dict(lgbm_params or {})}
    resolved_xgb_params = {**_DEFAULT_XGB_PARAMS, **dict(xgb_params or {})}

    models: dict[tuple, PooledModel] = {}
    for (market, horizon), (x, y) in data_by_combo.items():
        eval_data = (
            eval_data_by_combo.get((market, horizon)) if eval_data_by_combo is not None else None
        )
        use_early_stopping = eval_data is not None and early_stopping_rounds is not None
        effective_rounds = early_stopping_rounds if use_early_stopping else None

        # REQ-ATO-019: 모델 조합별(시장×horizon×algorithm) 학습 시작/완료 단계
        # 전이 로그.
        logger.info(
            "model training start market=%s horizon=%s algorithm=lightgbm",
            market,
            horizon,
            extra={"stage_marker": True},
        )
        lgbm_model = lgb.LGBMRegressor(**resolved_lgbm_params)
        _fit_lgbm(lgbm_model, x, y, eval_data, effective_rounds)
        models[(market, horizon, "lightgbm")] = lgbm_model
        logger.info(
            "model training complete market=%s horizon=%s algorithm=lightgbm",
            market,
            horizon,
            extra={"stage_marker": True},
        )

        logger.info(
            "model training start market=%s horizon=%s algorithm=xgboost",
            market,
            horizon,
            extra={"stage_marker": True},
        )
        xgb_extra = {"early_stopping_rounds": early_stopping_rounds} if use_early_stopping else {}
        xgb_model = xgb.XGBRegressor(**resolved_xgb_params, **xgb_extra)
        if use_early_stopping:
            xgb_model.fit(x, y, eval_set=[eval_data], verbose=False)
        else:
            xgb_model.fit(x, y)
        models[(market, horizon, "xgboost")] = xgb_model
        logger.info(
            "model training complete market=%s horizon=%s algorithm=xgboost",
            market,
            horizon,
            extra={"stage_marker": True},
        )

        for alpha in QUANTILE_ALPHAS:
            logger.info(
                "model training start market=%s horizon=%s algorithm=lightgbm_quantile alpha=%s",
                market,
                horizon,
                alpha,
                extra={"stage_marker": True},
            )
            quantile_params = {**resolved_lgbm_params, "objective": "quantile", "alpha": alpha}
            quantile_model = lgb.LGBMRegressor(**quantile_params)
            _fit_lgbm(quantile_model, x, y, eval_data, effective_rounds)
            models[(market, horizon, "lightgbm_quantile", alpha)] = quantile_model
            logger.info(
                "model training complete market=%s horizon=%s algorithm=lightgbm_quantile alpha=%s",
                market,
                horizon,
                alpha,
                extra={"stage_marker": True},
            )

    return models


def _fit_lgbm(
    model: lgb.LGBMRegressor,
    x: np.ndarray,
    y: np.ndarray,
    eval_data: tuple[np.ndarray, np.ndarray] | None,
    early_stopping_rounds: int | None,
) -> None:
    """LightGBM 학습을 조기 종료 유무에 따라 분기한다(REQ-AT-105)."""
    if eval_data is not None and early_stopping_rounds is not None:
        eval_x, eval_y = eval_data
        model.fit(
            x,
            y,
            eval_X=eval_x,
            eval_y=eval_y,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    else:
        model.fit(x, y)

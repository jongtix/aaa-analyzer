"""src/analyzer/training/models.py 16개 풀링 모델 학습 테스트 (SPEC-ANALYZER-TRAIN-001 M5).

REQ-AT-060(8개 포인트 모델: 시장×horizon×알고리즘)/REQ-AT-061(8개 분위수
보조 모델: 시장×horizon×alpha)/REQ-AT-062(분위수 보조 모델은 포인트
LightGBM 하이퍼파라미터 재사용, Optuna 튜닝 대상 제외)를 검증한다.
`len(models) == 16` assertion은 acceptance.md §F 추적표가 REQ-AT-060~064
예외 조항의 직접 검증 방법으로 명시한 방식이다.
"""

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from analyzer.training.models import HORIZONS, MARKETS, QUANTILE_ALPHAS, train_pooled_models


def _synthetic_data(seed: int, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = x @ np.array([0.02, -0.01, 0.015]) + rng.normal(scale=0.01, size=n)
    return x, y


def _full_data_by_combo() -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    return {
        (market, horizon): _synthetic_data(seed=hash((market, horizon)) % 1000)
        for market in MARKETS
        for horizon in HORIZONS
    }


class TestTrainPooledModels:
    def test_returns_exactly_16_models(self):
        """acceptance.md §F 추적표 명시: len(models) == 16 assertion으로 직접 검증."""
        models = train_pooled_models(_full_data_by_combo())

        assert len(models) == 16

    def test_returns_8_point_models_per_market_horizon_algorithm(self):
        """REQ-AT-060: 시장(2) × horizon(2) × 알고리즘(2) = 8개 포인트 모델."""
        models = train_pooled_models(_full_data_by_combo())

        point_keys = {
            (market, horizon, algo)
            for market in MARKETS
            for horizon in HORIZONS
            for algo in ("lightgbm", "xgboost")
        }
        assert point_keys.issubset(models.keys())
        assert len(point_keys) == 8

    def test_returns_8_quantile_models_per_market_horizon_alpha(self):
        """REQ-AT-061: 시장(2) × horizon(2) × alpha(0.10/0.90) = 8개 분위수 보조 모델."""
        models = train_pooled_models(_full_data_by_combo())

        quantile_keys = {
            (market, horizon, "lightgbm_quantile", alpha)
            for market in MARKETS
            for horizon in HORIZONS
            for alpha in QUANTILE_ALPHAS
        }
        assert quantile_keys.issubset(models.keys())
        assert len(quantile_keys) == 8

    def test_point_models_are_actually_fitted_and_predict(self):
        data = _full_data_by_combo()
        models = train_pooled_models(data)

        lgbm_model = models[("domestic", 20, "lightgbm")]
        xgb_model = models[("domestic", 20, "xgboost")]
        x, _y = data[("domestic", 20)]

        assert isinstance(lgbm_model, LGBMRegressor)
        assert isinstance(xgb_model, XGBRegressor)
        assert np.asarray(lgbm_model.predict(x)).shape[0] == len(x)
        assert np.asarray(xgb_model.predict(x)).shape[0] == len(x)

    def test_quantile_models_use_quantile_objective_with_correct_alpha(self):
        models = train_pooled_models(_full_data_by_combo())

        model_p10 = models[("domestic", 20, "lightgbm_quantile", 0.10)]
        model_p90 = models[("domestic", 20, "lightgbm_quantile", 0.90)]

        assert model_p10.get_params()["objective"] == "quantile"
        assert model_p10.get_params()["alpha"] == pytest.approx(0.10)
        assert model_p90.get_params()["alpha"] == pytest.approx(0.90)

    def test_quantile_models_reuse_point_lgbm_hyperparameters(self):
        """REQ-AT-062: 분위수 보조 모델은 포인트 LightGBM 하이퍼파라미터를 재사용한다."""
        custom_lgbm_params = {"n_estimators": 7, "max_depth": 2, "verbosity": -1}

        models = train_pooled_models(_full_data_by_combo(), lgbm_params=custom_lgbm_params)

        point_lgbm = models[("domestic", 20, "lightgbm")]
        quantile_lgbm = models[("domestic", 20, "lightgbm_quantile", 0.10)]

        assert point_lgbm.get_params()["n_estimators"] == 7
        assert point_lgbm.get_params()["max_depth"] == 2
        assert quantile_lgbm.get_params()["n_estimators"] == 7
        assert quantile_lgbm.get_params()["max_depth"] == 2

    def test_default_objective_is_l2(self):
        """REQ-AT-060: objective는 L2 기본."""
        models = train_pooled_models(_full_data_by_combo())

        lgbm_model = models[("domestic", 20, "lightgbm")]
        objective = lgbm_model.get_params().get("objective")
        assert objective is None or objective in ("regression", "l2")

    def test_caller_can_override_objective(self):
        """§2.5 Huber 채택 결정을 이 경로로 주입할 수 있어야 한다."""
        models = train_pooled_models(
            _full_data_by_combo(), lgbm_params={"objective": "huber", "verbosity": -1}
        )

        lgbm_model = models[("domestic", 20, "lightgbm")]
        assert lgbm_model.get_params()["objective"] == "huber"

    def test_rejects_incomplete_combo_set(self):
        data = _full_data_by_combo()
        del data[("domestic", 20)]

        with pytest.raises(ValueError, match="조합"):
            train_pooled_models(data)

    def test_rejects_unexpected_combo(self):
        data = _full_data_by_combo()
        data[("emerging", 20)] = _synthetic_data(seed=999)

        with pytest.raises(ValueError, match="조합"):
            train_pooled_models(data)

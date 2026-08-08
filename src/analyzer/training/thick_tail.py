"""Huber loss vs 타깃 윈저라이징 재현 가능 비교 하네스 (SPEC-ANALYZER-TRAIN-001 M4).

REQ-AT-050: L2 회귀 목적함수의 두꺼운 꼬리 강건성 확보를 위해, Huber loss
(`sklearn.linear_model.HuberRegressor`)와 타깃 윈저라이징(상하위 분위수
클리핑 + OLS)을 실데이터 백테스트로 비교한다(TECHSPEC §6, 953·995행이
이 SPEC에 명시적으로 위임).

REQ-AT-051: 이 비교는 aaa-infra#141(해외 배당 커버리지 결함)이 유발하는
허위 극단 음수 수익률 이상치 클래스를 함께 고려해야 한다 — 두 결정(두꺼운
꼬리 방법 선택, #141발 이상치 처리)을 독립적으로가 아니라 함께 평가한다.
호출자가 이상치를 포함한 `y`를 그대로 전달하면 두 전략 모두 동일한
이상치 분포에 대해 비교된다.

REQ-AT-052: 시계열 정보 누출 위험이 있는 합성 오버샘플링 계열 기법을
두꺼운 꼬리 대응 후보로 채택하지 않는다(shall not, acceptance.md §C
grep 가드 대상 — 금지 키워드 목록은 그 문서가 유일한 소스다) — 이
모듈은 그런 라이브러리를 임포트하지 않는다.

재현성(AC-AT-012): `random_state`가 고정되면 `train_test_split`/
`HuberRegressor`/`LinearRegression` 모두 결정론적이므로, 동일 입력+동일
`random_state`로 반복 호출 시 완전히 동일한 결과를 반환한다.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.model_selection import train_test_split


@dataclass(frozen=True, slots=True)
class ThickTailComparisonResult:
    """두 두꺼운 꼬리 대응 전략의 검증 MAE 비교 결과."""

    huber_val_mae: float
    winsorized_val_mae: float


def winsorize_targets(
    y: np.ndarray, lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> np.ndarray:
    """타깃 값을 `[lower_quantile, upper_quantile]` 분위수 범위로 클리핑한다.

    입력 배열을 변경하지 않고 새 배열을 반환한다.
    """
    lo, hi = np.quantile(y, [lower_quantile, upper_quantile])
    return np.clip(y, lo, hi)


def compare_thick_tail_strategies(
    x: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    test_size: float = 0.2,
    winsorize_lower: float = 0.01,
    winsorize_upper: float = 0.99,
    huber_epsilon: float = 1.35,
) -> ThickTailComparisonResult:
    """Huber 회귀와 (윈저라이징+OLS) 회귀를 동일 분할로 비교한다(REQ-AT-050/051).

    `random_state`가 고정되면 완전히 동일한 결과를 재현해야 한다(AC-AT-012).
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    huber = HuberRegressor(epsilon=huber_epsilon)
    huber.fit(x_train, y_train)
    huber_pred = huber.predict(x_val)
    huber_mae = float(np.mean(np.abs(y_val - huber_pred)))

    y_train_winsorized = winsorize_targets(np.asarray(y_train), winsorize_lower, winsorize_upper)
    winsorized_model = LinearRegression()
    winsorized_model.fit(x_train, y_train_winsorized)
    winsorized_pred = winsorized_model.predict(x_val)
    winsorized_mae = float(np.mean(np.abs(y_val - winsorized_pred)))

    return ThickTailComparisonResult(huber_val_mae=huber_mae, winsorized_val_mae=winsorized_mae)

"""`investor_trend` 기반 수급 피처(순매수 비율 + N일 누적 순매수).

SPEC-ANALYZER-FEATURE-001 REQ-AF-030~032(M3): 투자자별(외국인/기관/개인)
순매수 비율 3종 + `WINDOWS`(classification.py 공유 상수)의 각 윈도에 대한
N일 누적 순매수 3종 × 4윈도 = 12종, 총 15개 수급 피처를 계산한다.

`total_trading_value=0`일 때 pandas의 Series 나눗셈은 분자가 0이 아니면
`inf`/`-inf`를 반환한다(0-나눗셈 예외를 던지지 않음) — REQ-AF-032가 요구하는
NaN이 아니므로 `inf`/`-inf`를 명시적으로 NaN 치환한다(plan.md §B 리스크 3).

원본 컬럼(`trade_date` 등)을 보존한 채 신규 피처 컬럼만 추가해 반환하는
순수 함수다(REQ-AF-061).
"""

import pandas as pd

from analyzer.features.classification import WINDOWS

_INVESTOR_VALUE_COLUMNS: dict[str, str] = {
    "foreign": "foreign_net_value",
    "institution": "institution_net_value",
    "individual": "individual_net_value",
}


def compute_supply_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """`investor_trend` DataFrame을 입력으로 받아 15개 수급 피처를 추가해 반환한다."""
    result = df.copy()
    total_trading_value = df["total_trading_value"]

    for investor, value_column in _INVESTOR_VALUE_COLUMNS.items():
        net_value = df[value_column]

        ratio = net_value / total_trading_value
        result[f"{investor}_net_ratio"] = ratio.replace([float("inf"), float("-inf")], float("nan"))

        for window in WINDOWS:
            result[f"{investor}_net_cum_{window}"] = net_value.rolling(window=window).sum()

    return result

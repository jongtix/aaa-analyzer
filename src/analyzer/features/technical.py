"""가격·거래량 파생 기술적 지표 피처(pandas만 사용, REQ-AF-020).

SPEC-ANALYZER-FEATURE-001 REQ-AF-021~024(M2): KBAR 비율 5종(open_price
정규화) + `WINDOWS`(classification.py 공유 상수)의 각 윈도에 대한
ROC/MA/STD/RANK/CORR 5개 계열 = 25개 기술적 피처를 계산한다.

입력은 `adjust_prices()`(DATA-001 `adjustment.py`)의 출력 스키마
(`open_price`/`high_price`/`low_price`/`close_price`/`volume`)를 그대로
소비한다. 원본 컬럼(`trade_date` 등)을 보존한 채 신규 피처 컬럼만 추가해
반환하는 순수 함수다(REQ-AF-061).

`min_periods`를 재정의하지 않는다 — pandas rolling 기본값(`min_periods=w`)이
관측치 부족 시 자동으로 NaN을 반환하므로 REQ-AF-024(NaN 전파, 0/전방채움
금지)를 별도 로직 없이 자연히 충족한다(plan.md §B 리스크 4).

`CORR_{window}`(종가-거래량 롤링 피어슨 상관계수)는 거래정지(volume=0) 구간이
롤링 윈도우에 포함되면 종가·거래량 분산이 0에 수렴해 pandas가 0/0을 NaN이
아니라 `inf`/`-inf`로 반환한다(실측: 035900 2007-08-08 CORR_10=-inf,
supply_demand.py REQ-AF-032와 동일한 종류의 pandas 0-나눗셈 위험). NaN이
아닌 `inf`/`-inf`는 REQ-AF-024가 요구하는 결측 표현이 아니므로 명시적으로
NaN 치환한다.
"""

import pandas as pd

from analyzer.features.classification import WINDOWS


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """`adjust_prices()` 출력을 입력으로 받아 25개 기술적 피처를 추가해 반환한다."""
    result = df.copy()

    open_price = df["open_price"]
    high_price = df["high_price"]
    low_price = df["low_price"]
    close_price = df["close_price"]
    volume = df["volume"]

    body_high = pd.concat([open_price, close_price], axis=1).max(axis=1)
    body_low = pd.concat([open_price, close_price], axis=1).min(axis=1)

    result["KMID"] = (close_price - open_price) / open_price
    result["KLEN"] = (high_price - low_price) / open_price
    result["KUP"] = (high_price - body_high) / open_price
    result["KLOW"] = (body_low - low_price) / open_price
    result["KSFT"] = (2 * close_price - high_price - low_price) / open_price

    for window in WINDOWS:
        result[f"ROC_{window}"] = close_price / close_price.shift(window) - 1
        rolling_close = close_price.rolling(window=window)
        result[f"MA_{window}"] = rolling_close.mean() / close_price - 1
        result[f"STD_{window}"] = rolling_close.std() / close_price
        result[f"RANK_{window}"] = rolling_close.rank(pct=True)
        corr = close_price.rolling(window=window).corr(volume)
        result[f"CORR_{window}"] = corr.replace([float("inf"), float("-inf")], float("nan"))

    return result

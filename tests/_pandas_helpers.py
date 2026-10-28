"""pandas 타입 추론 변화에 영향받지 않도록 컬럼 타입을 좁히는 테스트 헬퍼."""

import pandas as pd


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """`df[name]`을 `Series`로 확정해 반환한다.

    pandas 3.1부터 `df["col"]`이 `Series | DataFrame` 합집합으로 추론되어 `.nunique()` 등의
    결과를 `assert`에 쓰면 pyright가 `Series.__bool__` 오류를 낸다. 한 번만 좁혀 둔다.
    """
    series = df[name]
    assert isinstance(series, pd.Series)
    return series

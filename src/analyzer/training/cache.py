"""로컬 디스크 Parquet 피처 캐시 (SPEC-ANALYZER-TRAIN-001 M2).

REQ-AT-030/031: 조립된 데이터셋(`dataset.assemble_dataset()` 출력)을
`features_{market}_{데이터기준일}_{피처코드버전}.parquet` 키로 MacBook
로컬 디스크에 `pyarrow` 엔진으로 캐싱한다 — TECHSPEC(2026-07-03 [D-6] 확정)
/ DATA-001 §3 / FEATURE-001 §4.6에 이미 명시된 캐시 키 관례를 그대로
따르며, 이 SPEC이 관례 자체를 재설계하지 않는다.

캐시 무효화 전략은 별도 TTL/명시적 invalidate API 없이 캐시 키 자체(피처
코드버전이 바뀌면 자동으로 새 키 → 캐시 미스)에 내장되어 있다(REQ-AT-031).
"""

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd


def cache_key(market: str, data_as_of: date, feature_code_version: str) -> str:
    """`features_{market}_{데이터기준일}_{피처코드버전}.parquet` 캐시 파일명을 조립한다.

    REQ-AT-030 캐시 키 관례를 그대로 따른다.
    """
    return f"features_{market}_{data_as_of.isoformat()}_{feature_code_version}.parquet"


def cache_path(cache_dir: Path, market: str, data_as_of: date, feature_code_version: str) -> Path:
    """캐시 디렉터리 아래 캐시 키 파일의 전체 경로를 반환한다."""
    return cache_dir / cache_key(market, data_as_of, feature_code_version)


def load_cached_dataset(
    cache_dir: Path, market: str, data_as_of: date, feature_code_version: str
) -> pd.DataFrame | None:
    """캐시 파일이 존재하면 로드해 반환하고, 없으면 `None`을 반환한다(REQ-AT-031)."""
    path = cache_path(cache_dir, market, data_as_of, feature_code_version)
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")


def save_dataset_to_cache(
    df: pd.DataFrame,
    cache_dir: Path,
    market: str,
    data_as_of: date,
    feature_code_version: str,
) -> Path:
    """조립된 데이터셋을 캐시 키 경로에 Parquet(`pyarrow` 엔진)로 저장한다(REQ-AT-030)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, market, data_as_of, feature_code_version)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def assemble_dataset_cached(
    *,
    cache_dir: Path,
    market: str,
    data_as_of: date,
    feature_code_version: str,
    assemble_fn: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """캐시를 경유해 데이터셋을 반환한다 — 히트 시 `assemble_fn`을 호출하지 않는다(REQ-AT-031).

    `assemble_fn`은 인자 없는 콜러블로, 실제 DB 조회 + `dataset.assemble_dataset()`
    호출을 감싸는 클로저다. 캐시 미스일 때만 호출되어 결과가 캐시에
    저장된다(AC-AT-004 — 동일 파라미터 재호출 시 DB 조회 함수가 재호출되지
    않아야 함).
    """
    cached = load_cached_dataset(cache_dir, market, data_as_of, feature_code_version)
    if cached is not None:
        return cached

    df = assemble_fn()
    save_dataset_to_cache(df, cache_dir, market, data_as_of, feature_code_version)
    return df

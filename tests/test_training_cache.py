"""src/analyzer/training/cache.py Parquet 피처 캐시 테스트 (SPEC-ANALYZER-TRAIN-001 M2).

REQ-AT-030(Parquet `pyarrow` 엔진 저장)/REQ-AT-031(캐시 히트 시 DB 재조회
회피)을 검증한다. AC-AT-004의 worked example(동일 파라미터 2회 호출 시
2번째 호출에서 DB 조회 함수가 호출되지 않아야 함)을 그대로 구현한다.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from analyzer.training.cache import (
    assemble_dataset_cached,
    cache_key,
    cache_path,
    load_cached_dataset,
    save_dataset_to_cache,
)


class TestCacheKey:
    """REQ-AT-030: 캐시 키 형식 `features_{market}_{데이터기준일}_{피처코드버전}.parquet`."""

    def test_formats_cache_key(self):
        key = cache_key("domestic", date(2026, 8, 8), "v1")

        assert key == "features_domestic_2026-08-08_v1.parquet"

    def test_cache_path_joins_dir_and_key(self, tmp_path: Path):
        path = cache_path(tmp_path, "overseas", date(2026, 8, 8), "v2")

        assert path == tmp_path / "features_overseas_2026-08-08_v2.parquet"


class TestSaveAndLoadCachedDataset:
    """REQ-AT-030/031: pyarrow 엔진 저장 + 존재 시 재사용."""

    def test_save_then_load_roundtrip(self, tmp_path: Path):
        df = pd.DataFrame({"trade_date": [date(2026, 8, 8)], "label_D20": [0.05]})

        saved_path = save_dataset_to_cache(df, tmp_path, "domestic", date(2026, 8, 8), "v1")
        loaded = load_cached_dataset(tmp_path, "domestic", date(2026, 8, 8), "v1")

        assert saved_path.exists()
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, df)

    def test_load_returns_none_when_cache_missing(self, tmp_path: Path):
        loaded = load_cached_dataset(tmp_path, "domestic", date(2026, 8, 8), "v1")

        assert loaded is None

    def test_save_creates_missing_cache_dir(self, tmp_path: Path):
        cache_dir = tmp_path / "nested" / "cache"
        df = pd.DataFrame({"trade_date": [date(2026, 8, 8)]})

        saved_path = save_dataset_to_cache(df, cache_dir, "domestic", date(2026, 8, 8), "v1")

        assert saved_path.exists()


class TestAssembleDatasetCached:
    """AC-AT-004: 동일 파라미터 2회 호출 시 2번째 호출에서 DB 조회 함수가 호출되지 않는다."""

    def test_second_call_does_not_invoke_assemble_fn(self, tmp_path: Path):
        db_query_mock = MagicMock(
            return_value=pd.DataFrame({"trade_date": [date(2026, 8, 8)], "label_D20": [0.1]})
        )

        first = assemble_dataset_cached(
            cache_dir=tmp_path,
            market="domestic",
            data_as_of=date(2026, 8, 8),
            feature_code_version="v1",
            assemble_fn=db_query_mock,
        )
        second = assemble_dataset_cached(
            cache_dir=tmp_path,
            market="domestic",
            data_as_of=date(2026, 8, 8),
            feature_code_version="v1",
            assemble_fn=db_query_mock,
        )

        assert db_query_mock.call_count == 1
        pd.testing.assert_frame_equal(first, second)

    def test_different_feature_code_version_is_a_cache_miss(self, tmp_path: Path):
        """REQ-AT-031: 캐시 무효화는 캐시 키(피처코드버전)에 내장된다."""
        db_query_mock = MagicMock(
            side_effect=[
                pd.DataFrame({"trade_date": [date(2026, 8, 8)], "label_D20": [0.1]}),
                pd.DataFrame({"trade_date": [date(2026, 8, 8)], "label_D20": [0.2]}),
            ]
        )

        assemble_dataset_cached(
            cache_dir=tmp_path,
            market="domestic",
            data_as_of=date(2026, 8, 8),
            feature_code_version="v1",
            assemble_fn=db_query_mock,
        )
        assemble_dataset_cached(
            cache_dir=tmp_path,
            market="domestic",
            data_as_of=date(2026, 8, 8),
            feature_code_version="v2",
            assemble_fn=db_query_mock,
        )

        assert db_query_mock.call_count == 2

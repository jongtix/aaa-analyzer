"""src/analyzer/training/panel_folds.py 날짜-경계 어댑터 테스트 (SPEC-ANALYZER-TRAIN-EVAL-001 M2).

REQ-ATE-017(전역 캘린더 축 추출)/REQ-ATE-018(신규 주간 스트라이드 산식,
expanding_window_folds() 미사용)/REQ-ATE-019(날짜값 비교 필터링)를
검증한다. AC-ATE-010(합성 다종목 패널, 상장지연/상장폐지 종목 포함)과
AC-ATE-011(인접 폴드 전진폭 = val_size, purge gap = horizon)의 worked
example을 구현한다.
"""

import pandas as pd
import pytest

from analyzer.labels.config import PURGE_GAP_TRADING_DAYS
from analyzer.training.panel_folds import (
    build_date_sorted_panel,
    extract_global_trade_date_axis,
    map_index_bounds_to_dates,
    slice_panel_by_date_bounds,
    slice_sorted_panel_by_date_bounds,
    weekly_stride_fold_index_bounds,
)
from analyzer.training.split import expanding_window_folds


def _make_synthetic_panel() -> pd.DataFrame:
    """AC-ATE-010의 합성 다종목 패널 — 종목 A(전체 기간), 종목 B(상장지연
    시뮬레이션, 중간부터 존재), 종목 C(상장폐지 시뮬레이션, 중간까지만 존재)."""
    full_dates = pd.bdate_range("2020-01-01", periods=400)

    stock_a = pd.DataFrame({"stock_code": "A", "trade_date": full_dates})
    stock_b = pd.DataFrame({"stock_code": "B", "trade_date": full_dates[100:]})
    stock_c = pd.DataFrame({"stock_code": "C", "trade_date": full_dates[:300]})

    panel = pd.concat([stock_a, stock_b, stock_c], ignore_index=True)
    panel["value"] = range(len(panel))
    return panel


def _to_object_dtype_date_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """`pd.read_sql(...)`이 `parse_dates` 없이 MySQL `DATE` 컬럼을 읽었을 때의
    패널을 재현한다 — `trade_date`가 `datetime64`가 아니라 파이썬
    `datetime.date` 객체를 담은 object dtype Series가 된다(pandas는 이
    경로에서 자동 변환하지 않는다).

    실 DB 조립 경로(`train.fetch_market_data()` → `dataset.assemble_dataset()`)가
    산출하는 dtype이며, 합성 패널(`pd.bdate_range`)만 쓰는 기존 테스트는
    이 dtype을 한 번도 통과시키지 않았다.
    """
    object_dtype_panel = panel.copy()
    object_dtype_panel["trade_date"] = pd.Series(
        [value.date() for value in panel["trade_date"]], dtype=object
    )
    return object_dtype_panel


class TestExtractGlobalTradeDateAxis:
    """REQ-ATE-017: 종목별로 연속 배치된 풀링 패널에서 전역 캘린더 축을 추출한다."""

    def test_axis_length_equals_distinct_trading_days_not_row_count(self):
        panel = _make_synthetic_panel()
        axis = extract_global_trade_date_axis(panel)

        assert len(axis) == 400
        assert len(axis) != len(panel)

    def test_axis_is_sorted_ascending(self):
        panel = _make_synthetic_panel()
        axis = extract_global_trade_date_axis(panel)

        assert list(axis) == sorted(axis)


class TestWeeklyStrideFoldIndexBounds:
    """REQ-ATE-018: 전진폭 = val_size만(gap 제외), 인접 폴드 검증 윈도우가 빈틈없이 맞닿음."""

    def test_stride_is_val_size_only_not_gap_plus_val_size(self):
        bounds = weekly_stride_fold_index_bounds(
            n_dates=1000, horizon=20, initial_train_end=200, val_size=5, n_folds=10
        )

        for i in range(1, len(bounds)):
            assert bounds[i].train_end - bounds[i - 1].train_end == 5

    def test_adjacent_validation_windows_tile_without_overlap_or_gap(self):
        """val_start[i+1] - val_end[i] == 0 (design.md §2A 타일링 불변식)."""
        bounds = weekly_stride_fold_index_bounds(
            n_dates=1000, horizon=20, initial_train_end=200, val_size=5, n_folds=10
        )

        for i in range(1, len(bounds)):
            assert bounds[i].val_start - bounds[i - 1].val_end == 0

    def test_each_fold_purge_gap_equals_horizon(self):
        horizon = 20
        gap = PURGE_GAP_TRADING_DAYS[horizon]
        bounds = weekly_stride_fold_index_bounds(
            n_dates=1000, horizon=horizon, initial_train_end=200, val_size=5, n_folds=10
        )

        for fold in bounds:
            assert fold.val_start - fold.train_end == gap

    def test_val_window_size_matches_val_size(self):
        bounds = weekly_stride_fold_index_bounds(
            n_dates=1000, horizon=20, initial_train_end=200, val_size=5, n_folds=10
        )

        for fold in bounds:
            assert fold.val_end - fold.val_start == 5

    def test_does_not_reuse_expanding_window_folds_stride_formula(self):
        """expanding_window_folds()의 전진폭(gap+val_size) 공식과 값이 달라야 한다."""
        horizon = 20
        gap = PURGE_GAP_TRADING_DAYS[horizon]
        val_size = 5
        n_folds = 10

        weekly_bounds = weekly_stride_fold_index_bounds(
            n_dates=1000,
            horizon=horizon,
            initial_train_end=200,
            val_size=val_size,
            n_folds=n_folds,
        )
        legacy_folds = expanding_window_folds(
            n_samples=1000, horizon=horizon, n_folds=n_folds, val_size=val_size
        )

        weekly_stride = weekly_bounds[1].train_end - weekly_bounds[0].train_end
        legacy_stride = legacy_folds[1].train_end - legacy_folds[0].train_end

        assert weekly_stride == val_size
        assert legacy_stride == gap + val_size
        assert weekly_stride != legacy_stride

    def test_raises_when_insufficient_dates_for_requested_folds(self):
        with pytest.raises(ValueError, match="거래일 수"):
            weekly_stride_fold_index_bounds(
                n_dates=50, horizon=20, initial_train_end=10, val_size=5, n_folds=20
            )

    def test_rejects_unsupported_horizon(self):
        with pytest.raises(KeyError):
            weekly_stride_fold_index_bounds(
                n_dates=1000, horizon=99, initial_train_end=200, val_size=5, n_folds=10
            )

    def test_rejects_n_folds_below_1(self):
        with pytest.raises(ValueError, match="n_folds"):
            weekly_stride_fold_index_bounds(
                n_dates=1000, horizon=20, initial_train_end=200, val_size=5, n_folds=0
            )

    def test_rejects_val_size_below_1(self):
        with pytest.raises(ValueError, match="val_size"):
            weekly_stride_fold_index_bounds(
                n_dates=1000, horizon=20, initial_train_end=200, val_size=0, n_folds=10
            )

    def test_rejects_initial_train_end_below_1(self):
        with pytest.raises(ValueError, match="initial_train_end"):
            weekly_stride_fold_index_bounds(
                n_dates=1000, horizon=20, initial_train_end=0, val_size=5, n_folds=10
            )


class TestMapIndexBoundsToDates:
    """map_index_bounds_to_dates()의 부족한 거래일 수 방어(REQ-ATE-019)."""

    def test_raises_when_train_end_or_val_start_out_of_axis_range(self):
        from analyzer.training.panel_folds import PanelFoldIndexBounds

        axis = pd.bdate_range("2020-01-01", periods=100)
        bounds = PanelFoldIndexBounds(train_end=90, val_start=120, val_end=125)

        with pytest.raises(ValueError, match="거래일 수"):
            map_index_bounds_to_dates(bounds, axis)


class TestDateBoundaryPanelFiltering:
    """AC-ATE-010: 날짜-경계 어댑터가 패널을 trade_date 값으로 필터링한다(위치 인덱스 아님)."""

    def test_delisted_and_late_listed_stocks_are_assigned_by_date_value(self):
        panel = _make_synthetic_panel()
        axis = extract_global_trade_date_axis(panel)
        horizon = 20

        index_bounds = weekly_stride_fold_index_bounds(
            n_dates=len(axis), horizon=horizon, initial_train_end=150, val_size=5, n_folds=1
        )[0]
        date_bounds = map_index_bounds_to_dates(index_bounds, axis)

        train_df, val_df = slice_panel_by_date_bounds(panel, date_bounds)

        # 종목 C(300일까지만 존재)는 train_end가 300일 미만이면 학습 구간에 등장해야 한다.
        assert "C" in train_df["stock_code"].unique()
        # 모든 부분집합의 행이 실제로 날짜 경계를 만족해야 한다(값 비교 확인).
        assert (train_df["trade_date"] < date_bounds.train_end).all()
        assert (val_df["trade_date"] >= date_bounds.val_start).all()
        if date_bounds.val_end is not None:
            assert (val_df["trade_date"] < date_bounds.val_end).all()

    def test_train_and_val_subsets_do_not_overlap_per_stock(self):
        panel = _make_synthetic_panel()
        axis = extract_global_trade_date_axis(panel)

        index_bounds = weekly_stride_fold_index_bounds(
            n_dates=len(axis), horizon=20, initial_train_end=150, val_size=5, n_folds=1
        )[0]
        date_bounds = map_index_bounds_to_dates(index_bounds, axis)
        train_df, val_df = slice_panel_by_date_bounds(panel, date_bounds)

        merged = train_df.merge(val_df, on=["stock_code", "trade_date"], how="inner")
        assert merged.empty

    def test_purge_gap_boundary_is_half_open_no_1day_overlap(self):
        """plan.md §E F16: [train_end, val_start) 반개구간 — 경계 1일이라도 중첩되면 FAIL."""
        panel = _make_synthetic_panel()
        axis = extract_global_trade_date_axis(panel)
        horizon = 20
        gap = PURGE_GAP_TRADING_DAYS[horizon]

        index_bounds = weekly_stride_fold_index_bounds(
            n_dates=len(axis), horizon=horizon, initial_train_end=150, val_size=5, n_folds=1
        )[0]
        date_bounds = map_index_bounds_to_dates(index_bounds, axis)
        train_df, _ = slice_panel_by_date_bounds(panel, date_bounds)

        last_train_date = train_df["trade_date"].max()
        purge_window = axis[(axis > last_train_date) & (axis < date_bounds.val_start)]

        assert len(purge_window) == gap
        assert date_bounds.val_start > last_train_date

    def test_last_fold_val_end_is_none_when_it_reaches_axis_end(self):
        panel = _make_synthetic_panel()
        axis = extract_global_trade_date_axis(panel)
        horizon = 20
        gap = PURGE_GAP_TRADING_DAYS[horizon]
        val_size = 5

        # initial_train_end를 마지막 폴드의 val_end가 정확히 len(axis)에 닿도록 구성.
        initial_train_end = len(axis) - gap - val_size
        index_bounds = weekly_stride_fold_index_bounds(
            n_dates=len(axis),
            horizon=horizon,
            initial_train_end=initial_train_end,
            val_size=val_size,
            n_folds=1,
        )[0]
        date_bounds = map_index_bounds_to_dates(index_bounds, axis)

        assert date_bounds.val_end is None

        _, val_df = slice_panel_by_date_bounds(panel, date_bounds)
        assert (val_df["trade_date"] >= date_bounds.val_start).all()


class TestObjectDtypeDateColumnRegression:
    """실 DB 조립 패널(object dtype `datetime.date`)과 `pd.Timestamp` 경계의 비교 회귀.

    `extract_global_trade_date_axis()`는 `pd.DatetimeIndex`(=`pd.Timestamp`)
    축을 파생시키는 반면, 패널의 `trade_date` 컬럼 자체는 `pd.read_sql(...)`이
    `parse_dates` 없이 읽은 object dtype `datetime.date`로 남는다 — 두 슬라이싱
    함수는 이 두 타입을 직접 비교하므로 실 캠페인 첫 실행에서 결정적으로
    깨졌다(`TypeError: Cannot compare Timestamp with datetime.date`,
    `TypeError: '>' not supported between instances of 'datetime.datetime'
    and 'datetime.date'`). 슬라이싱 함수는 호출부 dtype과 무관하게 동작해야 한다.
    """

    def test_mask_slicing_accepts_object_dtype_date_column(self):
        panel = _to_object_dtype_date_panel(_make_synthetic_panel())
        axis = extract_global_trade_date_axis(panel)

        index_bounds = weekly_stride_fold_index_bounds(
            n_dates=len(axis), horizon=20, initial_train_end=150, val_size=5, n_folds=1
        )[0]
        date_bounds = map_index_bounds_to_dates(index_bounds, axis)

        train_df, val_df = slice_panel_by_date_bounds(panel, date_bounds)

        assert not train_df.empty
        assert not val_df.empty
        assert (pd.to_datetime(train_df["trade_date"]) < date_bounds.train_end).all()
        assert (pd.to_datetime(val_df["trade_date"]) >= date_bounds.val_start).all()
        assert date_bounds.val_end is not None
        assert (pd.to_datetime(val_df["trade_date"]) < date_bounds.val_end).all()

    def test_sorted_cache_slicing_accepts_object_dtype_date_column(self):
        """`build_date_sorted_panel()` + `slice_sorted_panel_by_date_bounds()`
        이진 탐색 경로도 동일한 행 집합을 산출해야 한다(캠페인 폴드 루프 경로)."""
        panel = _to_object_dtype_date_panel(_make_synthetic_panel())
        axis = extract_global_trade_date_axis(panel)

        index_bounds = weekly_stride_fold_index_bounds(
            n_dates=len(axis), horizon=20, initial_train_end=150, val_size=5, n_folds=1
        )[0]
        date_bounds = map_index_bounds_to_dates(index_bounds, axis)

        sorted_panel, sorted_dates = build_date_sorted_panel(panel)
        train_df, val_df = slice_sorted_panel_by_date_bounds(
            sorted_panel, sorted_dates, date_bounds
        )
        expected_train_df, expected_val_df = slice_panel_by_date_bounds(panel, date_bounds)

        assert not train_df.empty
        assert not val_df.empty
        assert set(train_df["value"]) == set(expected_train_df["value"])
        assert set(val_df["value"]) == set(expected_val_df["value"])

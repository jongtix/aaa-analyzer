"""src/analyzer/labels/config.py 파라미터 상수 테스트 (SPEC-ANALYZER-LABEL-001 M1).

REQ-AL-010(HORIZONS)/REQ-AL-011(시장별 학습 시작일)/REQ-AL-061(purge gap
상수)을 검증한다. AC-AL-009(purge gap 상수 정합성)의 대상이다.
"""

from datetime import date

from analyzer.labels.config import (
    DEFAULT_START_DATES,
    HORIZONS,
    PURGE_GAP_TRADING_DAYS,
)


class TestHorizons:
    def test_horizons_is_20_and_60(self):
        assert HORIZONS == (20, 60)


class TestDefaultStartDates:
    def test_domestic_start_date(self):
        assert DEFAULT_START_DATES["domestic"] == date(2005, 1, 1)

    def test_overseas_start_date(self):
        assert DEFAULT_START_DATES["overseas"] == date(2007, 8, 20)


class TestAcAl009PurgeGapConsistency:
    """AC-AL-009: horizon별 purge gap이 HORIZONS와 정확히 일치해야 한다."""

    def test_purge_gap_keys_match_horizons(self):
        assert set(PURGE_GAP_TRADING_DAYS.keys()) == set(HORIZONS)

    def test_purge_gap_20_equals_20(self):
        assert PURGE_GAP_TRADING_DAYS[20] == 20

    def test_purge_gap_60_equals_60(self):
        assert PURGE_GAP_TRADING_DAYS[60] == 60

    def test_purge_gap_value_equals_horizon_length_for_every_horizon(self):
        """TECHSPEC §6.3: purge gap = horizon 길이."""
        for horizon in HORIZONS:
            assert PURGE_GAP_TRADING_DAYS[horizon] == horizon

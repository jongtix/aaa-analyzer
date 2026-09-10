"""`repository.fetch_stock_universe()`(`stock_id` 포함판) 테스트
(SPEC-ANALYZER-INFER-001 M5, plan.md §B 리스크 7).

`training/train.py`의 동명 함수(`stock_id` 미포함, dataset 조립 경로
전용)는 이 SPEC에서 무수정이다 — 이 함수는 `data/repository.py`에 신규
추가된 별개 함수로, INSERT 경로(REQ-AIF-100)가 FK로 필요로 하는
`stock_id`를 포함해 반환한다.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from analyzer.data.repository import fetch_stock_universe


class TestFetchStockUniverseWithStockId:
    """§B 리스크 7: 후보 유니버스를 `stock_id` FK 포함 스키마로 조회한다."""

    def test_returns_stock_id_column_for_domestic(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(
            {
                "stock_id": [1, 2],
                "stock_code": ["005930", "000660"],
                "grade": ["A", "B"],
                "delisted_at": [None, None],
            }
        )

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            result = fetch_stock_universe(engine, "domestic")

        assert "stock_id" in result.columns
        assert "stock_code" in result.columns
        assert "grade" in result.columns
        assert "delisted_at" in result.columns
        _, kwargs = mock_read_sql.call_args
        assert kwargs["params"]["market_codes"] == ("KOSPI", "KOSDAQ")

    def test_overseas_market_uses_nyse_nasdaq_amex_codes(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(columns=["stock_id", "stock_code", "grade", "delisted_at"])

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            fetch_stock_universe(engine, "overseas")

        _, kwargs = mock_read_sql.call_args
        assert kwargs["params"]["market_codes"] == ("NYSE", "NASDAQ", "AMEX")


class TestNoWriteOperationsStillHolds:
    """REQ-AD-011/plan.md §D: 신규 함수 추가 후에도 `repository.py`는
    SELECT 전용이어야 한다(기존 회귀 가드와 동일 취지, 이 신규 함수
    자체를 대상으로 재확인)."""

    def test_new_function_source_has_no_write_keywords(self):
        import inspect

        source = inspect.getsource(fetch_stock_universe)
        forbidden = [
            "INSERT INTO",
            "UPDATE ",
            "DELETE FROM",
            "DROP TABLE",
            "ALTER TABLE",
            "TRUNCATE",
        ]
        for keyword in forbidden:
            assert keyword not in source, f"forbidden DML/DDL keyword found: {keyword}"

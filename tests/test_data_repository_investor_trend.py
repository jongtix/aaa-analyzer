"""src/analyzer/data/repository.py investor_trend 읽기 계층 테스트
(SPEC-ANALYZER-FEATURE-001 M4).

REQ-AF-010(SELECT 전용 fetcher)/REQ-AF-011(청크 단위 읽기, AC-AF-014)/
REQ-AF-051(DATE 전용 조인·정렬 키, AC-AF-012)을 검증한다. 실 DB 접속 없이
`pd.read_sql`을 모킹한 경량 단위 테스트다(DATA-001 M5 패턴 재사용).
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from analyzer.data.repository import fetch_investor_trend, iter_investor_trend_by_stock


class TestFetchInvestorTrend:
    """REQ-AF-010: investor_trend에 대한 SELECT 전용 접근, stocks JOIN."""

    def test_query_joins_stocks_for_stock_code(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(
            columns=[
                "trade_date",
                "foreign_net_value",
                "institution_net_value",
                "individual_net_value",
                "total_trading_value",
            ]
        )

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            result = fetch_investor_trend(engine, "005930")

        call_args = mock_read_sql.call_args
        sql, kwargs = call_args.args[0], call_args.kwargs
        sql_text = str(sql)
        assert "investor_trend" in sql_text
        assert "stocks" in sql_text
        assert kwargs["params"] == {"stock_code": "005930"}
        assert isinstance(result, pd.DataFrame)

    def test_blank_stock_code_raises_value_error(self):
        import pytest

        engine = MagicMock()
        with pytest.raises(ValueError, match="stock_code"):
            fetch_investor_trend(engine, "")


class TestFetchInvestorTrendDateOnlyKeys:
    """AC-AF-012 (REQ-AF-051): created_at/updated_at을 조인·정렬 기준으로 사용 금지."""

    def test_query_never_references_created_or_updated_at(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(columns=["trade_date"])

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            fetch_investor_trend(engine, "005930")

        sql_text = str(mock_read_sql.call_args.args[0])
        assert "created_at" not in sql_text
        assert "updated_at" not in sql_text
        assert "trade_date" in sql_text


class TestIterInvestorTrendByStock:
    """AC-AF-014 (REQ-AF-010/011): 종목 단위 청크 읽기, 시장 전체 단일 호출 금지."""

    def test_each_call_targets_exactly_one_stock_code(self):
        engine = MagicMock()
        stock_codes = ["005930", "000660", "035420"]

        with patch("analyzer.data.repository.fetch_investor_trend") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(columns=["trade_date"])

            chunks = list(iter_investor_trend_by_stock(engine, stock_codes))

        assert len(chunks) == len(stock_codes)
        assert mock_fetch.call_count == len(stock_codes)
        called_stock_codes = [call.args[1] for call in mock_fetch.call_args_list]
        assert called_stock_codes == stock_codes

    def test_never_loads_all_stocks_in_a_single_dataframe(self):
        engine = MagicMock()
        stock_codes = [f"CODE{i}" for i in range(5)]

        with patch("analyzer.data.repository.fetch_investor_trend") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({"trade_date": ["2026-01-02"]})

            result = iter_investor_trend_by_stock(engine, stock_codes)

            assert mock_fetch.call_count == 0

            consumed = list(result)

        assert len(consumed) == len(stock_codes)
        assert mock_fetch.call_count == len(stock_codes)


class TestNoWriteOperations:
    """plan.md §D: analyzer는 DDL/쓰기 권한이 없다 — SELECT만 수행한다."""

    def test_new_query_never_uses_forbidden_dml_keywords(self):
        import inspect

        import analyzer.data.repository as repo_module

        source = inspect.getsource(repo_module)
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

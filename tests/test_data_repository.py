"""src/analyzer/data/repository.py DB 읽기 계층 테스트 (SPEC-ANALYZER-DATA-001 M5).

REQ-AD-010(SQLAlchemy 엔진 구성)/REQ-AD-011(SELECT 전용 접근)/REQ-AD-012
(market_calendar 캘린더 소스)/REQ-AD-013(청크 단위 읽기, AC-AD-008)을 검증한다.
실 DB 접속 없이 `pd.read_sql`을 모킹한 경량 단위 테스트다 — 통합 테스트(M6)와는
별개다.
"""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from analyzer.data.config import DbConfig
from analyzer.data.models import TradingCalendar
from analyzer.data.repository import (
    build_engine,
    fetch_corporate_events,
    fetch_daily_ohlcv,
    fetch_market_calendar,
    iter_daily_ohlcv_by_stock,
)


class TestBuildEngine:
    """REQ-AD-010: 커넥션 풀링을 지원하는 DB 엔진 구성."""

    def test_builds_engine_from_db_config(self):
        config = DbConfig(host="aaa-mysql", port=3306, database="aaa", password="secret-value")

        with patch("analyzer.data.repository.create_engine") as mock_create_engine:
            mock_create_engine.return_value = MagicMock()

            build_engine(config)

        assert mock_create_engine.call_count == 1
        (url,), kwargs = mock_create_engine.call_args
        assert url == "mysql+pymysql://analyzer:secret-value@aaa-mysql:3306/aaa"
        assert kwargs.get("pool_pre_ping") is True


class TestFetchMarketCalendar:
    """REQ-AD-012: market_calendar를 거래일 캘린더 소스로 사용."""

    def test_returns_trading_calendar_with_open_days_only(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(
            {
                "calendar_code": ["KRX", "KRX", "KRX"],
                "cal_date": [date(2023, 12, 27), date(2023, 12, 28), date(2023, 12, 30)],
                "is_open": [True, True, False],
            }
        )

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            calendar = fetch_market_calendar(engine, "KRX")

        assert isinstance(calendar, TradingCalendar)
        assert calendar.calendar_code == "KRX"
        assert calendar.trading_days == frozenset({date(2023, 12, 27), date(2023, 12, 28)})
        _, kwargs = mock_read_sql.call_args
        assert kwargs["params"] == {"calendar_code": "KRX"}

    def test_query_selects_only_market_calendar_columns(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(columns=["calendar_code", "cal_date", "is_open"])

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            fetch_market_calendar(engine, "NYSE")

        call_args = mock_read_sql.call_args
        sql, kwargs = call_args.args[0], call_args.kwargs
        sql_text = str(sql)
        assert "market_calendar" in sql_text
        assert "calendar_code" in sql_text
        assert kwargs["params"] == {"calendar_code": "NYSE"}


class TestFetchDailyOhlcv:
    """REQ-AD-011: daily_ohlcv에 대한 SELECT 전용 접근, stocks JOIN을 통한 stock_code 해소."""

    def test_query_joins_stocks_for_stock_code(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(
            columns=[
                "stock_code",
                "trade_date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
            ]
        )

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            result = fetch_daily_ohlcv(engine, "005930")

        call_args = mock_read_sql.call_args
        sql, kwargs = call_args.args[0], call_args.kwargs
        sql_text = str(sql)
        assert "daily_ohlcv" in sql_text
        assert "stocks" in sql_text
        assert kwargs["params"] == {"stock_code": "005930"}
        assert isinstance(result, pd.DataFrame)

    def test_optional_date_range_narrows_query_and_params(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(columns=["stock_code", "trade_date"])

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            fetch_daily_ohlcv(
                engine, "AAPL", start_date=date(2020, 1, 1), end_date=date(2020, 12, 31)
            )

        call_args = mock_read_sql.call_args
        sql, kwargs = call_args.args[0], call_args.kwargs
        sql_text = str(sql)
        assert "trade_date" in sql_text
        assert kwargs["params"] == {
            "stock_code": "AAPL",
            "start_date": date(2020, 1, 1),
            "end_date": date(2020, 12, 31),
        }

    def test_no_date_range_omits_date_params(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(columns=["stock_code", "trade_date"])

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            fetch_daily_ohlcv(engine, "AAPL")

        _, kwargs = mock_read_sql.call_args
        assert kwargs["params"] == {"stock_code": "AAPL"}

    def test_price_columns_cast_from_decimal_to_float64(self):
        """MySQL DECIMAL(18,4) 컬럼은 pymysql이 `decimal.Decimal`(object dtype)로
        반환한다 — `compute_technical_features()` 등 하류 피처 계산이 float 연산을
        가정하므로, 여기서 float64로 캐스팅하지 않으면
        `TypeError: unsupported operand type(s) for /: 'float' and 'decimal.Decimal'`로
        즉시 크래시한다(2026-08-13 실측 재현, SPEC-ANALYZER-TRAIN-AUTOMATION-001 조사)."""
        engine = MagicMock()
        mock_df = pd.DataFrame(
            {
                "stock_code": ["005930"],
                "trade_date": [date(2026, 1, 2)],
                "open_price": [Decimal("100.0000")],
                "high_price": [Decimal("101.0000")],
                "low_price": [Decimal("99.0000")],
                "close_price": [Decimal("100.5000")],
                "volume": [1000],
            }
        )

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df):
            result = fetch_daily_ohlcv(engine, "005930")

        for column in ("open_price", "high_price", "low_price", "close_price"):
            assert result[column].dtype == "float64"
        assert result["open_price"].iloc[0] == 100.0


class TestFetchCorporateEvents:
    """REQ-AD-011: corporate_events에 대한 SELECT 전용 접근."""

    def test_query_joins_stocks_for_stock_code(self):
        engine = MagicMock()
        mock_df = pd.DataFrame(
            columns=[
                "stock_code",
                "event_type",
                "event_date",
                "stock_rate",
                "cash_amount",
                "event_subtype",
                "ex_dividend_date",
                "currency_code",
            ]
        )

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df) as mock_read_sql:
            result = fetch_corporate_events(engine, "005930")

        call_args = mock_read_sql.call_args
        sql, kwargs = call_args.args[0], call_args.kwargs
        sql_text = str(sql)
        assert "corporate_events" in sql_text
        assert "stocks" in sql_text
        assert kwargs["params"] == {"stock_code": "005930"}
        assert isinstance(result, pd.DataFrame)

    def test_cash_amount_and_stock_rate_cast_from_decimal_to_float64(self):
        """`cash_amount`(DECIMAL(15,5))/`stock_rate`(DECIMAL(12,4))도 daily_ohlcv
        가격 컬럼과 동일한 pymysql Decimal 반환 문제를 가진다(nullable — 배당
        이벤트는 stock_rate가, 분할 이벤트는 cash_amount가 NULL)."""
        engine = MagicMock()
        mock_df = pd.DataFrame(
            {
                "stock_code": ["005930"],
                "event_type": ["DIVIDEND"],
                "event_date": [date(2026, 1, 2)],
                "stock_rate": [None],
                "cash_amount": [Decimal("500.00000")],
                "event_subtype": ["일반배당"],
                "ex_dividend_date": [date(2026, 1, 1)],
                "currency_code": ["KRW"],
            }
        )

        with patch("analyzer.data.repository.pd.read_sql", return_value=mock_df):
            result = fetch_corporate_events(engine, "005930")

        assert result["cash_amount"].dtype == "float64"
        assert result["stock_rate"].dtype == "float64"
        assert result["cash_amount"].iloc[0] == 500.0
        assert pd.isna(result["stock_rate"].iloc[0])


class TestIterDailyOhlcvByStock:
    """AC-AD-008 (REQ-AD-013): 시장 전체 조회는 종목 단위 청크로만 읽어야 한다."""

    def test_each_call_targets_exactly_one_stock_code(self):
        engine = MagicMock()
        stock_codes = ["005930", "000660", "AAPL", "MSFT"]

        with patch("analyzer.data.repository.fetch_daily_ohlcv") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(columns=["stock_code", "trade_date"])

            chunks = list(iter_daily_ohlcv_by_stock(engine, stock_codes))

        assert len(chunks) == len(stock_codes)
        assert mock_fetch.call_count == len(stock_codes)
        called_stock_codes = [call.args[1] for call in mock_fetch.call_args_list]
        assert called_stock_codes == stock_codes
        # 시장 전체를 아우르는 단일 호출이 관측되어서는 안 된다 — 각 호출은
        # 정확히 1개 종목코드만을 인자로 받는다.
        for call in mock_fetch.call_args_list:
            assert isinstance(call.args[1], str)

    def test_propagates_optional_date_range_to_each_chunk_call(self):
        engine = MagicMock()
        stock_codes = ["005930", "AAPL"]
        start = date(2020, 1, 1)
        end = date(2020, 12, 31)

        with patch("analyzer.data.repository.fetch_daily_ohlcv") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(columns=["stock_code", "trade_date"])

            list(iter_daily_ohlcv_by_stock(engine, stock_codes, start_date=start, end_date=end))

        for call in mock_fetch.call_args_list:
            assert call.kwargs == {"start_date": start, "end_date": end}

    def test_never_loads_all_stocks_in_a_single_dataframe(self):
        """전체 시장을 단일 인메모리 자료구조로 적재해서는 안 된다(shall not)."""
        engine = MagicMock()
        stock_codes = [f"CODE{i}" for i in range(5)]

        with patch("analyzer.data.repository.fetch_daily_ohlcv") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {"stock_code": ["x"], "trade_date": [date(2020, 1, 1)]}
            )

            result = iter_daily_ohlcv_by_stock(engine, stock_codes)

            # generator이므로 fetch_daily_ohlcv는 아직 호출되지 않았어야 한다
            # (지연 평가 — 전체를 미리 메모리에 모으지 않음).
            assert mock_fetch.call_count == 0

            consumed = []
            for chunk in result:
                consumed.append(chunk)

        assert len(consumed) == len(stock_codes)
        assert mock_fetch.call_count == len(stock_codes)


class TestNoWriteOperations:
    """REQ-AD-011/plan.md §D: analyzer는 DDL/쓰기 권한이 없다 — SELECT만 수행한다."""

    def test_repository_module_never_uses_forbidden_dml_keywords(self):
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


class TestFetchDailyOhlcvRequiresStockCode:
    def test_blank_stock_code_raises_value_error(self):
        engine = MagicMock()
        with pytest.raises(ValueError, match="stock_code"):
            fetch_daily_ohlcv(engine, "")


class TestFetchCorporateEventsRequiresStockCode:
    def test_blank_stock_code_raises_value_error(self):
        engine = MagicMock()
        with pytest.raises(ValueError, match="stock_code"):
            fetch_corporate_events(engine, "")

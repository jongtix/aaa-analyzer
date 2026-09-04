"""`trade_date` 자체 산출 (SPEC-ANALYZER-INFER-001 M1, REQ-AIF-021).

`stream:daily:complete` 이벤트는 `market`/`attempted`/`succeeded`/`skipped`
4개 필드만 담고 `trade_date`를 포함하지 않는다(collector `DailyCompletePublisher`
계약, 변경 불가). 따라서 analyzer는 시장별 `daily_ohlcv` 최신 거래일을 자체
DB 조회로 산출한다.

해외 시장의 최신 확정 일봉은 ET 기준 D-1 지연이 정상이므로, 산출은 **오직 DB
조회 결과**로만 이뤄진다 — 이 모듈은 벽시계(KST 오늘 날짜)를 참조하지 않는다.

이벤트의 `attempted`/`succeeded`/`skipped` 값은 완전성 판정 근거로 쓰이지
않는다(REQ-AIF-021): 휴장일에도 이벤트가 발행되고 catch-up 재실행으로 하루
2회 도착할 수 있으므로, (market, trade_date) 단위 최종 멱등 방어선은 DB
INSERT 계층의 UNIQUE 키 스킵(REQ-AIF-100)이다.
"""

from datetime import date

from sqlalchemy import Date, bindparam, text
from sqlalchemy.engine import Engine

MARKET_TO_STOCKS_MARKET_CODES: dict[str, tuple[str, ...]] = {
    "domestic": ("KOSPI", "KOSDAQ"),
    "overseas": ("NYSE", "NASDAQ", "AMEX"),
}
"""analyzer 시장 토큰("domestic"/"overseas")과 `stocks.market` 거래소 코드의
매핑 — `training/train.py`가 NAS 실측으로 확립한 매핑을 그대로 계승한다.
`KRX`/`US`는 지수 종목(`asset_type='INDEX'`) 전용 값이라 개별 종목 유니버스에
나타나지 않으므로 포함하지 않는다."""

_LATEST_TRADE_DATE_QUERY = (
    text(
        "SELECT MAX(d.trade_date) AS max_trade_date "
        "FROM daily_ohlcv d "
        "JOIN stocks s ON s.id = d.stock_id "
        "WHERE s.market IN :market_codes AND s.asset_type = 'STOCK'"
    )
    .bindparams(bindparam("market_codes", expanding=True))
    .columns(max_trade_date=Date)
)


def resolve_trade_date(engine: Engine, market: str) -> date | None:
    """`market`의 최신 확정 거래일을 `daily_ohlcv`에서 산출한다.

    해당 시장에 일봉이 한 행도 없으면 `None`을 반환한다 — 호출부는 이를
    "추론할 대상이 없음"으로 처리한다(자식 프로세스를 스폰하지 않는다).
    """
    market_codes = MARKET_TO_STOCKS_MARKET_CODES.get(market)
    if market_codes is None:
        raise ValueError(f"지원하지 않는 시장 토큰: {market}")

    with engine.connect() as conn:
        return conn.execute(
            _LATEST_TRADE_DATE_QUERY, {"market_codes": list(market_codes)}
        ).scalar_one_or_none()

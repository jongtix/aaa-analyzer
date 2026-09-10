"""`trading_signals` INSERT-ONLY 기록 + `model_version` 포맷 조립
(SPEC-ANALYZER-INFER-001 M5, REQ-AIF-100, design.md §4).

`analyzer` 계정은 `trading_signals`에 대해 INSERT 권한만 보유한다(UPDATE
없음 → ODKU 불가, TECHSPEC §4 접근 보안) — 동일 (stock_id, trade_date,
horizon) 키 재삽입 시도는 UNIQUE 키 위반(`IntegrityError`)이 정상적인
"이미 처리됨" 신호이며, 이를 캐치해 스킵으로 처리한다(재추론 재실행에
대한 유일한 방어선, AC-AIF-017).

이 모듈은 score/confidence 등 값 자체를 계산하지 않는다 — 그 값은
`inference/resolution.py`(G3 score 컬럼)·`inference/scoring.py`(G4
confidence)가 이미 산출한 결과를 `TradingSignalRow`로 조립해 전달받는다
(관심사 분리, AC-AIF-017 worked example은 이미 계산된 값을 입력으로
가정한다).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

_KST = ZoneInfo("Asia/Seoul")
"""CLAUDE.md 공통 규칙: 모든 타임스탬프는 KST(Asia/Seoul) 기준(`orchestration/
scheduler.py`·`orchestration/staleness.py`와 동일 관례)."""

_INSERT_TRADING_SIGNAL_SQL = text(
    "INSERT INTO trading_signals "
    "(stock_id, trade_date, horizon, score, p10, p90, lgbm_score, xgb_score, "
    "signal_class, confidence, regime_suppressed, model_version, created_at, updated_at) "
    "VALUES (:stock_id, :trade_date, :horizon, :score, :p10, :p90, :lgbm_score, "
    ":xgb_score, :signal_class, :confidence, :regime_suppressed, :model_version, "
    ":created_at, :created_at)"
)
"""V43(SPEC-ANALYZER-SCHEMA-001) 컬럼 계약. `created_at`/`updated_at`은
DDL DEFAULT가 없어 애플리케이션 레벨로 기록해야 하며(V43 코멘트), INSERT-ONLY
시맨틱상 실질 갱신은 발생하지 않으므로(REQ-ASCH-009) 두 컬럼에 동일한
삽입 시각을 채운다. `lgbm_score`/`p10`/`p90`은 V48 마이그레이션(2026-09-10
NAS 라이브 적용 확인)으로 NULL을 허용한다 — `xgb_score`는 V48이 다루지
않은 별개 컬럼이며 여전히 NOT NULL이다(§ Residual-risk)."""


class InsertOutcome:
    """`insert_trading_signal()` 반환값 — 단순 문자열 상수(테스트/향후
    `InferenceMetrics`(M8) 소비용, 새 예외 타입을 도입하지 않는다)."""

    INSERTED = "inserted"
    SKIPPED_DUPLICATE = "skipped_duplicate"


def format_model_version(market: str, horizon: int, algo: str, trained_date: date) -> str:
    """`model_version` 자유 텍스트 포맷(REQ-AIF-100): `{market}_{horizon}_{algo}_{trained_date}`.

    `training/persistence.model_filename()`과 동일한 세그먼트 순서(확장자만
    없음) — FK가 아니다(SCHEMA-001 관례). 앙상블 조합은 `algo="ensemble"`
    (REQ-AIF-040), 단독 전략은 그 알고리즘명(REQ-AIF-041)을 사용한다.
    """
    return f"{market}_{horizon}_{algo}_{trained_date.isoformat()}"


def format_horizon_label(horizon: int) -> str:
    """`trading_signals.horizon`(VARCHAR(3)) 저장 형식 — "D20"/"D60".

    `format_model_version()`의 순수 정수 표기와는 별개 표현이다 — DB
    컬럼 코멘트("추론 기간 — D20/D60")가 요구하는 형식으로만 변환한다.
    """
    return f"D{horizon}"


@dataclass(frozen=True, slots=True)
class TradingSignalRow:
    """`trading_signals` 1행 — INSERT 대상 컬럼 전체(REQ-AIF-040/041/090/100).

    `lgbm_score`/`xgb_score` 중 하나는 단독 전략 조합(G3, REQ-AIF-041)에서
    `None`일 수 있다 — `resolution.compute_score_columns()`가 이미 그
    규칙을 적용해 산출한 값을 그대로 전달받는다(재계산하지 않는다).
    """

    stock_id: int
    trade_date: date
    horizon: int
    score: float
    p10: float
    p90: float
    lgbm_score: float | None
    xgb_score: float | None
    signal_class: str
    confidence: float
    model_version: str
    regime_suppressed: bool = False
    """REQ-AIF-090: 체제 판정 로직이 아직 구현되지 않았으므로 항상 FALSE로
    고정 기록한다(REGIME-OBSV-001 소관, 훅 지점만 남겨둔다)."""


def insert_trading_signal(engine: Engine, row: TradingSignalRow) -> str:
    """`trading_signals`에 INSERT-ONLY로 기록한다(REQ-AIF-100, AC-AIF-017).

    UNIQUE(stock_id, trade_date, horizon) 위반은 "이미 처리됨" 정상 신호로
    캐치-스킵한다(재추론 재실행에 대한 유일한 방어선) — 프로세스는
    크래시하지 않는다. 반환값은 `InsertOutcome.INSERTED` 또는
    `InsertOutcome.SKIPPED_DUPLICATE`.
    """
    inserted_at = datetime.now(_KST)
    with engine.begin() as conn:
        try:
            conn.execute(
                _INSERT_TRADING_SIGNAL_SQL,
                {
                    "stock_id": row.stock_id,
                    "trade_date": row.trade_date,
                    "horizon": format_horizon_label(row.horizon),
                    "score": row.score,
                    "p10": row.p10,
                    "p90": row.p90,
                    "lgbm_score": row.lgbm_score,
                    "xgb_score": row.xgb_score,
                    "signal_class": row.signal_class,
                    "confidence": row.confidence,
                    "regime_suppressed": row.regime_suppressed,
                    "model_version": row.model_version,
                    "created_at": inserted_at,
                },
            )
        except IntegrityError:
            return InsertOutcome.SKIPPED_DUPLICATE
    return InsertOutcome.INSERTED

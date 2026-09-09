"""등급 경계 실측 산출 CLI — 실데이터 실현 수익률로 경계표 + δ를 역산해 저장소 JSON을 갱신한다
(SPEC-ANALYZER-INFER-001 M4, REQ-AIF-080/081/111, AC-AIF-014).

REQ-AIF-080: TECHSPEC §6.1의 합성 데이터 초안표 대신, 백필 완료 후 실현
수익률(`label_D20`/`label_D60`, LABEL-001 데이터셋 조립 경로)로 시장×horizon
4개 조합의 경계값을 `training/boundaries.py`의 분위수 역산 함수(TRAIN-001,
PRESERVE)로 새로 계산한다. 목표 등급 비율은 사용자 확정값
(`DEFAULT_TARGET_RATIOS`)을 호출자로서 주입한다 — `boundaries.py` 자체는
어떤 비율도 하드코딩하지 않는다(REQ-AT-041).

REQ-AIF-111 서브 Task: 4개 경계표가 확정된 **이후에** 인접 경계 최소 간격의
10%로 δ 잠정값을 유도해 같은 JSON에 `provisional: true`로 함께 기록한다 —
NOTIFIER-FILTER-001이 합동 확정값으로 이 값만 교체하면 전환된다.

이 CLI는 반기 재검토(plan.md 운영 절차) 시점에 재실행되는 1회성 산출
경로다 — 추론 프로세스는 이 CLI를 호출하지 않으며 `inference/
boundaries_store.py`로 JSON을 읽기만 한다(REQ-AIF-081, AC-AIF-015).

DB 접근은 `training/db.py::build_trainer_engine()`(trainer 계정, SELECT
전용)만 사용하며, 데이터 조회·조립은 `train._assemble_market_dataset()`
(캘린더 override 포함, `campaign.py`와 동일 재사용 패턴)을 그대로 호출한다 —
새 SQL을 작성하지 않는다. `trainer` 계정의 동시 접속 상한(3) 때문에 시장은
단일 엔진으로 순차 조회한다.

CLI: `python -m analyzer.training.boundaries_cli --cache-dir <path>
--data-as-of YYYY-MM-DD --feature-code-version <str> [--calendar-code KRX]
[--output <path>]` — 성공 시 종료코드 `0`, 실패 시 `1`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy.engine import Engine

from analyzer.common.logging import get_logger
from analyzer.data.repository import fetch_market_calendar
from analyzer.inference.boundaries_store import (
    BOUNDARY_KEYS,
    DEFAULT_GRADE_MARGIN_DELTA_RATIO,
    default_artifact_path,
    derive_grade_margin_delta,
)
from analyzer.training import train as train_module
from analyzer.training.boundaries import infer_grade_boundaries_all_combinations
from analyzer.training.db import build_trainer_engine
from analyzer.training.models import HORIZONS, MARKETS

logger = get_logger(__name__)

DEFAULT_TARGET_RATIOS: dict[str, float] = {
    "STRONG_SELL": 0.05,
    "SELL": 0.20,
    "HOLD": 0.50,
    "BUY": 0.20,
    "STRONG_BUY": 0.05,
}
"""목표 등급 비율(2026-09 사용자 확정, 4개 조합에 동일 적용). 경계값 산출의
입력이지 산출물이 아니므로 코드 상수로 둔다 — 산출된 경계값 수치는 JSON
저장소에만 존재한다(REQ-AIF-080)."""

SCHEMA_VERSION = 1

_KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True, slots=True)
class RealizedReturnsCollection:
    """조합별 실현 수익률 시리즈 + 비밀 아닌 메타데이터(행 수, 실제 사용된 최신 거래일)."""

    returns_by_combination: dict[tuple[str, int], pd.Series]
    row_counts: dict[tuple[str, int], int]
    data_as_of: date


def _extract_realized_returns(assembled: pd.DataFrame, horizon: int) -> pd.Series:
    """`label_D{h}`가 유효하고 `label_D{h}_exclude_reason`이 비어 있는 행만 남긴다."""
    label_column = f"label_D{horizon}"
    exclude_column = f"{label_column}_exclude_reason"
    mask = assembled[label_column].notna()
    if exclude_column in assembled.columns:
        mask &= assembled[exclude_column].isna()
    return assembled.loc[mask, label_column].astype(float).reset_index(drop=True)


def collect_realized_returns(
    engine: Engine,
    *,
    calendar_code: str,
    cache_dir: Path,
    data_as_of: date,
    feature_code_version: str,
) -> RealizedReturnsCollection:
    """시장별로 순차 조회·조립해 4개 조합의 실현 수익률 분포를 모은다(REQ-AIF-080).

    `train._assemble_market_dataset()`을 시장당 정확히 1회 호출한다 — 해외
    시장의 캘린더 override(`_MARKET_CALENDAR_CODE_OVERRIDE`)는 `train.py`/
    `campaign.py`와 동일하게 적용한다.
    """
    returns_by_combination: dict[tuple[str, int], pd.Series] = {}
    row_counts: dict[tuple[str, int], int] = {}
    latest_trade_dates: list[date] = []

    for market in MARKETS:
        market_calendar_code = train_module._MARKET_CALENDAR_CODE_OVERRIDE.get(
            market, calendar_code
        )
        calendar = fetch_market_calendar(engine, market_calendar_code)
        assembled = train_module._assemble_market_dataset(
            engine, calendar, market, cache_dir, data_as_of, feature_code_version
        )
        if not assembled.empty:
            latest = pd.to_datetime(assembled["trade_date"]).max()
            latest_trade_dates.append(latest.date())

        for horizon in HORIZONS:
            returns = _extract_realized_returns(assembled, horizon)
            if returns.empty:
                raise ValueError(f"({market}, D{horizon}) 조합의 실현 수익률 행이 0건이다")
            returns_by_combination[(market, horizon)] = returns
            row_counts[(market, horizon)] = int(len(returns))
            logger.info(
                "realized returns collected market=%s horizon=%d rows=%d",
                market,
                horizon,
                len(returns),
                extra={"stage_marker": True},
            )

    return RealizedReturnsCollection(
        returns_by_combination=returns_by_combination,
        row_counts=row_counts,
        data_as_of=max(latest_trade_dates),
    )


def build_artifact_payload(
    returns_by_combination: Mapping[tuple[str, int], pd.Series],
    *,
    row_counts: Mapping[tuple[str, int], int],
    target_ratios: Mapping[str, float],
    data_as_of: date,
    generated_at: str,
    delta_ratio: float = DEFAULT_GRADE_MARGIN_DELTA_RATIO,
) -> dict[str, Any]:
    """경계표 역산 → δ 유도 순으로 저장소 JSON 페이로드를 만든다(순수 함수).

    경계표는 `infer_grade_boundaries_all_combinations()`가 조합별 독립
    분위수 역산으로 산출하고, δ는 그 결과가 모두 확정된 뒤에만 유도한다
    (REQ-AIF-111 순서 의존성).
    """
    boundaries_by_combination = infer_grade_boundaries_all_combinations(
        returns_by_combination, target_ratios
    )
    delta, min_gap = derive_grade_margin_delta(boundaries_by_combination, ratio=delta_ratio)

    combinations = [
        {
            "market": market,
            "horizon": horizon,
            "row_count": int(row_counts[(market, horizon)]),
            "boundaries": {key: float(boundaries[key]) for key in BOUNDARY_KEYS},
        }
        for (market, horizon), boundaries in sorted(boundaries_by_combination.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "generated_by": "python -m analyzer.training.boundaries_cli (SPEC-ANALYZER-INFER-001 M4)",
        "data_as_of": data_as_of.isoformat(),
        "source": (
            "실데이터 실현 수익률(label_D20/label_D60, exclude_reason 없는 행) — "
            "training/boundaries.py 분위수 역산(REQ-AIF-080)"
        ),
        "target_ratios": {str(k): float(v) for k, v in target_ratios.items()},
        "combinations": combinations,
        "grade_margin_delta": {
            "value": float(delta),
            "provisional": True,
            "derivation": (
                f"{delta_ratio:g} × min(인접 경계 간격) over 4개 조합 전체 (REQ-AIF-111 잠정값)"
            ),
            "min_adjacent_gap": float(min_gap),
            "replace_by": (
                "SPEC-NOTIFIER-FILTER-001 합동 확정값으로 이 value만 교체 — 구조 변경 불요"
            ),
        },
    }


def write_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    """페이로드를 UTF-8 JSON(들여쓰기 2, 키 순서 유지)으로 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyzer.training.boundaries_cli")
    parser.add_argument("--calendar-code", default="KRX")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--data-as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--feature-code-version", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_artifact_path(),
        help="저장소 JSON 경로(기본: 패키지 동봉 analyzer/inference/grade_boundaries.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점 — 성공 시 `0`, 실패 시 `1`. 산출 수치는 로그·stdout에 출력하지 않는다."""
    args = build_parser().parse_args(argv)
    try:
        engine = build_trainer_engine()
        collected = collect_realized_returns(
            engine,
            calendar_code=args.calendar_code,
            cache_dir=args.cache_dir,
            data_as_of=args.data_as_of,
            feature_code_version=args.feature_code_version,
        )
        payload = build_artifact_payload(
            collected.returns_by_combination,
            row_counts=collected.row_counts,
            target_ratios=DEFAULT_TARGET_RATIOS,
            data_as_of=collected.data_as_of,
            generated_at=datetime.now(_KST).isoformat(timespec="seconds"),
        )
        write_artifact(args.output, payload)
    except Exception as exc:  # noqa: BLE001 — CLI 경계: 종료코드로 실패를 신호한다.
        print(f"등급 경계 산출 실패: {exc}", file=sys.stderr)
        return 1

    logger.info(
        "grade boundaries artifact written path=%s data_as_of=%s row_counts=%s",
        args.output,
        collected.data_as_of.isoformat(),
        {f"{m}/D{h}": n for (m, h), n in sorted(collected.row_counts.items())},
        extra={"stage_marker": True},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

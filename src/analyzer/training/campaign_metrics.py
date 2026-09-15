"""캠페인 메트릭/메타데이터 영속화 — 사이드카 JSON + 폴드별 JSONL + 요약 리포트
(SPEC-ANALYZER-TRAIN-EVAL-001 M4, design.md §7).

REQ-ATE-031/032: 챔피언 후보 최종(최근-데이터) 폴드 학습이 완료된 시점에만
`{model_path}.meta.json` 사이드카 1개를 기록한다 — XGBoost 네이티브 `.json`
확장자와 충돌하지 않도록 `model_path.suffix + ".meta.json"` 이중 접미를
채택한다(`persistence.py`의 기존 `.sha256` 사이드카 명명 관례와 동일).

REQ-ATE-033: 사이드카는 캠페인 전체 집계(평균/표준편차/ICIR), 최종 폴드
학습 행수, 동결 하이퍼파라미터, 실제 사용된 피처 컬럼 목록(design.md
§5.3 — `_split_features_and_labels()`가 산출한 교집합 이후의 최종
목록)을 포함하며, 폴드별 지표 시계열은 인라인 포함하지 않고 JSONL
파일의 상대경로만 참조한다.

REQ-ATE-037/038(F12): 개별 폴드(최대 약 500개)의 모델 아티팩트는 이
모듈이 전혀 소비하지 않는다(campaign.py가 이미 폐기) — 이 모듈은 각
폴드의 이미 계산된 `backtest.BacktestMetrics` 값만 입력받아 조합당
정확히 1개의 append-only JSONL 파일에 1줄씩 기록한다. 8개 포인트
조합(시장×horizon×algorithm) + 4개 (시장,horizon) 앙상블 의사조합
(algorithm 의사값 `"ensemble"`) = 총 12개 스트림.

REQ-ATE-034/039: 캠페인 요약 리포트는 8개 포인트 조합 각각의 게이트
판정 근거를 포함한다 — 이 마일스톤(M4) 시점에는 안정화 게이트
(`stabilization.py`, M5)가 아직 구현되지 않았으므로, 이 모듈은 M5가
채울 게이트 판정 스텁 구조만 정의한다(순환 의존 없음, plan.md §F M4).
리포트는 인접 폴드가 통계적으로 독립적인 표본이 아니라는 주의사항
(REQ-ATE-039, AC-ATE-026)을 항상 포함한다.

REQ-ATE-035/036: 이 모듈은 MySQL 스키마를 전혀 도입하지 않으며(파일시스템
전용), 사이드카/JSONL/리포트 모두 텍스트 기반 포맷(JSON/JSONL/Markdown)만
사용한다 — Parquet/pickle 등 이진 직렬화를 사용하지 않는다.
"""

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from analyzer.training.backtest import BacktestMetrics

ENSEMBLE_ALGORITHM_TAG: str = "ensemble"
"""REQ-ATE-038(F12): (시장,horizon) 앙상블 의사조합의 algorithm 필드 값."""

FOLD_JSONL_ALGORITHM_TAGS: tuple[str, ...] = ("lightgbm", "xgboost", ENSEMBLE_ALGORITHM_TAG)


def sidecar_path_for(model_path: Path) -> Path:
    """사이드카 경로 — `persistence.py`의 `.sha256` 관례와 동일한 이중 접미
    (`model_path.suffix + ".meta.json"`)로 XGBoost 네이티브 `.json`
    확장자와의 충돌을 회피한다(REQ-ATE-032, design.md §7)."""
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def fold_metrics_jsonl_filename(market: str, horizon: int, algorithm: str) -> str:
    """조합당 단일 JSONL 파일명 관례(REQ-ATE-038):
    `{market}_{horizon}_{algorithm|"ensemble"}_campaign_folds.jsonl`."""
    return f"{market}_{horizon}_{algorithm}_campaign_folds.jsonl"


@dataclass(frozen=True, slots=True)
class CampaignAggregateMetrics:
    """캠페인 전체 폴드에 걸친 Rank IC 집계(REQ-ATE-033) — 평균/표준편차/ICIR."""

    mean_rank_ic: float
    stddev_rank_ic: float
    icir: float


def compute_aggregate_metrics(rank_ic_values: Sequence[float]) -> CampaignAggregateMetrics:
    """폴드별 Rank IC 시계열로부터 캠페인 전체 집계를 계산한다.

    표본이 2개 미만이면 표준편차/ICIR을 0.0으로 취급한다(0으로 나누기 회피
    — 이 함수는 순수 집계 계산이며 GATE 판정 로직(M5, stabilization.py)이
    아니다).
    """
    if not rank_ic_values:
        return CampaignAggregateMetrics(mean_rank_ic=0.0, stddev_rank_ic=0.0, icir=0.0)
    mean = statistics.fmean(rank_ic_values)
    stddev = statistics.pstdev(rank_ic_values) if len(rank_ic_values) > 1 else 0.0
    icir = mean / stddev if stddev else 0.0
    return CampaignAggregateMetrics(mean_rank_ic=mean, stddev_rank_ic=stddev, icir=icir)


def write_sidecar_metadata(
    model_path: Path,
    *,
    market: str,
    horizon: int,
    algorithm: str,
    feature_columns: Sequence[str],
    aggregate_metrics: CampaignAggregateMetrics | None = None,
    final_fold_train_row_count: int | None = None,
    frozen_hyperparameters: Mapping[str, Any] | None = None,
    fold_metrics_jsonl_relative_path: str | None = None,
) -> Path:
    """사이드카 메타데이터 JSON을 기록한다(REQ-ATE-031/032/033,
    SPEC-ANALYZER-TRAIN-META-001 REQ-TM-001/004/005). 캠페인 경로(폴드
    집계가 존재)에서는 챔피언 후보 최종 폴드 학습 완료 시점에 호출되며,
    폴드별 시계열 전체는 인라인 포함하지 않고 동반 JSONL 파일의
    상대경로만 `fold_metrics_jsonl` 필드로 참조한다.

    캠페인 외 컨텍스트(주간 재학습, 단일종목 경로)에서는
    `aggregate_metrics`/`final_fold_train_row_count`/
    `frozen_hyperparameters`/`fold_metrics_jsonl_relative_path`가 존재하지
    않을 수 있다 — `None`(기본값)이면 해당 키를 payload에서 완전히
    생략한다(가짜 값으로 채우지 않는다, REQ-TM-004/B2). 기존
    `read_frozen_hyperparameters()`/`resolve_feature_columns()` 소비자는
    `.get()`/특정 키 조회만 사용하므로 다른 키의 부재는 영향을 주지
    않는다(REQ-TM-006 회귀 가드 대상 — 캠페인 호출부는 인자를 그대로
    채워 넘기므로 동작 무변경).
    """
    payload: dict[str, Any] = {
        "market": market,
        "horizon": horizon,
        "algorithm": algorithm,
        "feature_columns": list(feature_columns),
    }
    if aggregate_metrics is not None:
        payload["aggregate_metrics"] = {
            "mean_rank_ic": aggregate_metrics.mean_rank_ic,
            "stddev_rank_ic": aggregate_metrics.stddev_rank_ic,
            "icir": aggregate_metrics.icir,
        }
    if final_fold_train_row_count is not None:
        payload["final_fold_train_row_count"] = final_fold_train_row_count
    if frozen_hyperparameters is not None:
        payload["frozen_hyperparameters"] = dict(frozen_hyperparameters)
    if fold_metrics_jsonl_relative_path is not None:
        payload["fold_metrics_jsonl"] = fold_metrics_jsonl_relative_path
    sidecar_path = sidecar_path_for(model_path)
    sidecar_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar_path


def _isoformat(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.date().isoformat() if hasattr(value, "date") else str(value)


def append_fold_metrics(
    jsonl_dir: Path,
    market: str,
    horizon: int,
    algorithm: str,
    fold_index: int,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp | None,
    metrics: BacktestMetrics,
) -> Path:
    """조합당 단일 append-only JSONL 파일에 폴드 1개의 지표 1줄을 append한다
    (REQ-ATE-037/038) — 개별 폴드 사이드카 파일을 생성하지 않는다.
    """
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / fold_metrics_jsonl_filename(market, horizon, algorithm)
    record: dict[str, Any] = {
        "fold_index": fold_index,
        "train_end": _isoformat(train_end),
        "val_start": _isoformat(val_start),
        "val_end": _isoformat(val_end),
        "hit_rate": metrics.hit_rate,
        "pearson_ic": metrics.pearson_ic,
        "rank_ic": metrics.rank_ic,
        "precision": metrics.precision,
        "sharpe_ratio": metrics.sharpe_ratio,
        "max_drawdown": metrics.max_drawdown,
        "confidence_calibration": metrics.confidence_calibration,
    }
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")
    return jsonl_path


_ADJACENT_FOLD_CORRELATION_CAVEAT: str = (
    "인접 폴드는 통계적으로 독립적인 표본이 아니다 — 주간 케이던스에서 "
    "인접 확장 폴드(i, i+1)의 학습 구간은 검증 윈도우 크기(5거래일)만큼만 "
    "차이 나며 약 99.9% 중첩된다. 아래 롤링 집계(GATE-1/GATE-3)의 표준편차 "
    "계산에서 실제 유효 독립 표본 수는 관측된 폴드 수보다 작을 수 있다 "
    "(design.md §2C, REQ-ATE-039)."
)


@dataclass(frozen=True, slots=True)
class ComboGateVerdictStub:
    """(시장,horizon,algorithm) 조합 1개의 게이트 판정 근거 — M5
    (`stabilization.py`)가 아직 구현되지 않은 이 마일스톤 시점에는
    미평가 플레이스홀더로만 채워진다(순환 의존 회피, plan.md §F M4).
    """

    market: str
    horizon: int
    algorithm: str
    gate_verdict: str = "not_yet_evaluated"
    supporting_metrics: Mapping[str, Any] = field(default_factory=dict)


def write_campaign_summary_report(
    report_path: Path,
    combo_verdicts: Sequence[ComboGateVerdictStub],
) -> Path:
    """캠페인 요약 리포트를 기록한다(REQ-ATE-034) — 8개 포인트 조합 각각의
    게이트 판정 근거 + 인접 폴드 상관성 주의사항(REQ-ATE-039)을 포함한다.
    """
    lines: list[str] = ["# 캠페인 요약 리포트", "", _ADJACENT_FOLD_CORRELATION_CAVEAT, ""]
    for verdict in combo_verdicts:
        lines.append(f"## {verdict.market} / D{verdict.horizon} / {verdict.algorithm}")
        lines.append(f"- 게이트 판정: {verdict.gate_verdict}")
        for key, value in verdict.supporting_metrics.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

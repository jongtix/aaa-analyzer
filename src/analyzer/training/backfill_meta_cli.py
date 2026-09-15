"""이미 활성화된 프로덕션 모델의 `.meta.json` 사이드카 운영 백필 CLI
(SPEC-ANALYZER-TRAIN-META-001 M6, REQ-TM-009).

대상: `aaa-infra#163`이 보고한, 코드 배포(M3/M4) 이전에 이미 standing_gate
프로모션으로 활성화된 overseas D60 챔피언(xgboost, trained_date 2026-09-05)과
D20 분위수 쌍(q10/q90, lightgbm, trained_date 2026-09-05) — 다음 자연
주간 재학습 주기를 기다리지 않고 즉시 `.meta.json`을 재생성해
`resolve_feature_columns()`가 `FEATURE_REGISTRY` 전체(40개, FROZEN 수급
피처 15개 포함)로 폴백하는 것을 막는다(spec.md §0).

**안전성 근거(재학습·재훈련 없이 사이드카만 재생성해도 안전한 이유)**:
`_split_features_and_labels()`(`train.py`)가 산출하는 `feature_columns`는
`assembled.columns`와 `FEATURE_REGISTRY`의 교집합이며, 컬럼 존재 여부는
`dataset.py::assemble_dataset()`가 각 종목의 `investor_trend_by_stock`이
`None`이거나 비어 있으면 그 종목에 대해 FROZEN 수급 피처 병합을 완전히
건너뛰는 데서 결정된다 — 이는 특정 `trained_date`의 데이터 가용성이
아니라 **시장 자체의 구조적 특성**이다: overseas 유니버스는 investor_trend
데이터가 전혀 없으므로(aaa-infra#163), 어떤 시점의 어떤 패널을 조립해도
FROZEN 컬럼은 결코 병합되지 않는다. 따라서 "현재" 기준으로 재계산한
overseas feature_columns는 실제 학습 시점(2026-09-05)에 계산됐을 값과
동일하다 — 이 모듈의 `resolve_backfill_feature_columns()`가 이 불변식을
DB 접속 없이(합성 OHLCV 프레임으로 `compute_technical_features()`의
실제 컬럼 생성 순서를 재현) 검증하고 재현한다.

**컬럼 순서가 정확성에 중요한 이유**: `predict.py::_select_feature_columns()`는
`feature_columns` 리스트 순서 그대로 `feature_row.loc[:, feature_columns]`를
선택하고, LightGBM/XGBoost의 `predict()`는 위치 기반이다 — 잘못된 순서로
사이드카를 기록하면 조용한 오예측을 유발한다. `FEATURE_REGISTRY`의 딕셔너리
삽입 순서(통계량 우선: ROC_5,ROC_10,...,MA_5,...)는 실제 학습 시
`compute_technical_features()`가 만드는 순서(윈도 우선: KMID..KSFT,
ROC_5,MA_5,STD_5,RANK_5,CORR_5, ROC_10,...)와 다르므로, 이 모듈은 반드시
`compute_technical_features()`를 직접 호출해 실제 컬럼 순서를 재현한다.

**운영 안전 장치**:
- 기본은 dry-run이다 — `--apply`를 명시해야 실제로 파일을 기록한다.
- 이미 사이드카가 있는 모델은 스킵한다(idempotent — 재실행 안전).
- 모델 바이너리 파일 자체는 존재 여부만 확인하며 절대 열거나 수정하지
  않는다(읽기 전용 접근).
- `.env*` 파일이나 자격증명을 전혀 읽지 않으며, DB/네트워크/SSH 호출이
  전혀 없다 — 로컬 파일 경로만 조작한다(오케스트레이터가 이 스크립트를
  NAS에 별도로 전달·실행하는 절차를 소유한다, 이 스크립트는 그 실행
  메커니즘 자체를 다루지 않는다).
- overseas가 아닌 market은 `NotImplementedError`로 명시적으로 거부한다
  (도메스틱은 investor_trend 데이터가 실제 존재하므로, 합성 프레임만으로
  feature_columns를 안전하게 재현할 수 없다 — 잘못된 사이드카를 기록하는
  사고를 방지한다).

CLI: `python -m analyzer.training.backfill_meta_cli MODEL_PATH [MODEL_PATH ...]
[--apply]` — 성공(오류 없음) 시 종료코드 `0`, 하나라도 오류가 있으면 `1`.
스킵(사이드카 기존 존재)은 오류가 아니다.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analyzer.features.classification import FEATURE_REGISTRY, FeatureClass, classify_feature
from analyzer.features.technical import compute_technical_features
from analyzer.training import campaign_metrics

_SUPPORTED_MARKETS_FOR_SYNTHETIC_RESOLUTION: frozenset[str] = frozenset({"overseas"})
"""이 모듈이 합성 프레임만으로 안전하게 feature_columns를 재현할 수 있는
market 집합 — overseas만 해당한다(investor_trend 구조적 결측, 위 모듈
docstring 참조). domestic 등 그 밖의 market은 실제 investor_trend 데이터
유무에 따라 결과가 달라질 수 있으므로 이 스크립트의 범위 밖이다."""

_SYNTHETIC_OHLCV_ROWS = 65
"""`WINDOWS`(classification.py, 최대 60) 롤링 계산이 구조적으로 성립하도록
충분한 행 수 — 값 자체는 사이드카에 기록되지 않으며(컬럼 이름·순서만
필요), 실제 시세 데이터가 전혀 아니다."""

_MODEL_FILENAME_RE = re.compile(
    r"^(?P<market>[a-z]+)_(?P<horizon>\d+)_(?P<algorithm>lightgbm|xgboost)_"
    r"(?P<trained_date>\d{4}-\d{2}-\d{2})(?:_q(?P<alpha_tag>\d{2}))?\.(?P<ext>json|txt)$"
)
"""`persistence.py::model_filename()`/`quantile_model_filename()` 명명 관례와
정확히 대응한다: `{market}_{horizon}_{algorithm}_{trained_date}[_q{NN}].{ext}`."""


def _synthetic_ohlcv_frame(n_rows: int = _SYNTHETIC_OHLCV_ROWS) -> pd.DataFrame:
    """DB 접속 없이 `compute_technical_features()`의 실제 컬럼 생성 순서를
    재현하기 위한 최소 합성 OHLCV 프레임 — 값은 임의의 단조 증가 시퀀스이며
    사이드카에 기록되지 않는다(feature_columns 이름·순서만 필요)."""
    idx = np.arange(n_rows, dtype=float)
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2020-01-01", periods=n_rows, freq="B"),
            "open_price": 100.0 + idx,
            "high_price": 101.0 + idx,
            "low_price": 99.0 + idx,
            "close_price": 100.5 + idx,
            "volume": 1_000.0 + idx,
        }
    )


def resolve_backfill_feature_columns(market: str) -> list[str]:
    """`train.py::_split_features_and_labels()`가 그 market에 대해 산출할
    `feature_columns`를 DB 접속 없이 재현한다(REQ-TM-009).

    `overseas` 외의 market은 `NotImplementedError`를 발생시킨다 — 모듈
    docstring의 안전성 근거 참조.
    """
    if market not in _SUPPORTED_MARKETS_FOR_SYNTHETIC_RESOLUTION:
        raise NotImplementedError(
            f"이 백필 스크립트는 market={sorted(_SUPPORTED_MARKETS_FOR_SYNTHETIC_RESOLUTION)}만 "
            f"지원한다(REQ-TM-009 범위) — '{market}'은 지원 대상 밖이다. investor_trend 실데이터를 "
            "포함하는 일반 재학습 경로(train.py::run_training_pipeline())를 사용할 것."
        )
    technical = compute_technical_features(_synthetic_ohlcv_frame())
    feature_columns = [c for c in technical.columns if c in FEATURE_REGISTRY]

    frozen = [c for c in feature_columns if classify_feature(c) == FeatureClass.FROZEN]
    if frozen:
        raise AssertionError(
            "안전성 불변식 위반 — overseas feature_columns에 FROZEN 컬럼이 포함되었다: "
            f"{frozen}. 백필을 중단한다(REQ-TM-009 안전 가드) — investor_trend 병합 로직이 "
            "변경되었을 가능성이 있으니 이 스크립트를 재검토할 것."
        )
    return feature_columns


@dataclass(frozen=True, slots=True)
class ParsedModelFilename:
    """모델 파일명에서 파싱한 (market, horizon, algorithm) — `persistence.py`
    명명 관례 기반."""

    market: str
    horizon: int
    algorithm: str
    trained_date: str
    is_quantile: bool


def parse_model_filename(model_path: Path) -> ParsedModelFilename:
    """`persistence.py::model_filename()`/`quantile_model_filename()` 명명
    관례로부터 (market, horizon, algorithm)을 역파싱한다.

    관례와 일치하지 않으면 `ValueError`를 발생시킨다 — market/horizon/
    algorithm을 안전하게 추론할 수 없는 파일에 대해 잘못된 사이드카를
    기록하는 사고를 방지한다.
    """
    match = _MODEL_FILENAME_RE.match(model_path.name)
    if match is None:
        raise ValueError(
            f"{model_path.name!r}이(가) 알려진 모델 파일명 관례"
            "({market}_{horizon}_{algorithm}_{trained_date}[_q{NN}].{ext})와 일치하지 않는다 — "
            "market/horizon/algorithm을 안전하게 추론할 수 없다."
        )
    return ParsedModelFilename(
        market=match.group("market"),
        horizon=int(match.group("horizon")),
        algorithm=match.group("algorithm"),
        trained_date=match.group("trained_date"),
        is_quantile=match.group("alpha_tag") is not None,
    )


@dataclass(frozen=True, slots=True)
class BackfillPlan:
    """모델 파일 1개에 대한 백필 계획 — dry-run/apply 양쪽에서 공유하는
    순수 계산 결과(부수효과 없음)."""

    model_path: Path
    sidecar_path: Path
    market: str
    horizon: int
    algorithm: str
    feature_columns: tuple[str, ...]
    action: str  # "write" | "skip_sidecar_exists" | "skip_model_missing"


def plan_backfill(model_path: Path) -> BackfillPlan:
    """모델 파일 1개에 대해 무엇을 할지 결정한다 — 파일시스템에 아무것도
    쓰지 않는 순수 계획 단계(부수효과 없음, dry-run/apply 공유)."""
    sidecar_path = campaign_metrics.sidecar_path_for(model_path)

    if not model_path.is_file():
        return BackfillPlan(
            model_path=model_path,
            sidecar_path=sidecar_path,
            market="",
            horizon=0,
            algorithm="",
            feature_columns=(),
            action="skip_model_missing",
        )

    if sidecar_path.exists():
        return BackfillPlan(
            model_path=model_path,
            sidecar_path=sidecar_path,
            market="",
            horizon=0,
            algorithm="",
            feature_columns=(),
            action="skip_sidecar_exists",
        )

    parsed = parse_model_filename(model_path)
    feature_columns = tuple(resolve_backfill_feature_columns(parsed.market))
    return BackfillPlan(
        model_path=model_path,
        sidecar_path=sidecar_path,
        market=parsed.market,
        horizon=parsed.horizon,
        algorithm=parsed.algorithm,
        feature_columns=feature_columns,
        action="write",
    )


def apply_backfill(plan: BackfillPlan) -> Path:
    """계획을 실제로 실행해 `.meta.json` 사이드카를 기록한다 — 모델
    바이너리 파일은 절대 열거나 수정하지 않는다(축소 스키마, M1: `frozen_
    hyperparameters`/`aggregate_metrics`/`final_fold_train_row_count`는
    생략 — 과거 실행 로그가 보존되지 않은 백필 컨텍스트이므로, 크래시
    방지 목적에는 `feature_columns`만으로 충분하다, research.md §8)."""
    if plan.action != "write":
        raise ValueError(
            f"apply_backfill()은 action='write' 계획에만 호출할 수 있다: {plan.action}"
        )
    return campaign_metrics.write_sidecar_metadata(
        plan.model_path,
        market=plan.market,
        horizon=plan.horizon,
        algorithm=plan.algorithm,
        feature_columns=plan.feature_columns,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyzer.training.backfill_meta_cli",
        description=(
            "이미 활성화된 프로덕션 모델의 .meta.json 사이드카를 재생성한다"
            "(SPEC-ANALYZER-TRAIN-META-001 REQ-TM-009). 기본은 dry-run이다."
        ),
    )
    parser.add_argument(
        "model_paths",
        nargs="+",
        type=Path,
        help="사이드카를 백필할 모델 파일 절대/상대 경로 (1개 이상)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 .meta.json을 기록한다 — 생략 시 dry-run(무엇을 할지만 출력)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점 — 오류 없이 완료(스킵 포함)하면 `0`, 하나라도 오류가
    있으면 `1`. 스킵(사이드카 기존 존재)은 오류가 아니다."""
    args = build_parser().parse_args(argv)
    exit_code = 0

    for raw_path in args.model_paths:
        model_path = raw_path.resolve()
        try:
            plan = plan_backfill(model_path)
        except (ValueError, NotImplementedError, AssertionError) as exc:
            print(f"[ERROR] {model_path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        if plan.action == "skip_model_missing":
            print(f"[ERROR] {model_path}: 모델 파일이 존재하지 않는다 — 스킵", file=sys.stderr)
            exit_code = 1
            continue

        if plan.action == "skip_sidecar_exists":
            print(f"[SKIP] {plan.sidecar_path}: 이미 존재함(idempotent) — 재실행해도 무변경")
            continue

        # plan.action == "write"
        summary = (
            f"market={plan.market} horizon={plan.horizon} algorithm={plan.algorithm} "
            f"feature_columns={len(plan.feature_columns)}개"
        )
        if not args.apply:
            print(f"[DRY-RUN] would write {plan.sidecar_path} ({summary}) — --apply로 재실행할 것")
            continue

        written = apply_backfill(plan)
        print(f"[APPLIED] wrote {written} ({summary})")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

"""모델 정체(staleness) 감지 (SPEC-ANALYZER-TRAIN-AUTOMATION-001 §2.7, REQ-ATA-072/083).

활성 모델 파일명 관례 `{market}_{horizon}_{algorithm}_{trained_date}`(TRAIN-001이
확립, `training/persistence.py` `model_filename()`)에서 `trained_date`를 파싱해
정체를 감지한다(REQ-ATA-072) — 어떤 활성 모델이든 마지막 성공 재학습 후
`threshold_days`(기본 4주=28일)를 초과하면 정체로 표시한다. 이 감지는
REQ-ATA-083이 등록하는 전용 일일 cron 잡을 통해 주기적으로 트리거된다(§2.8,
`scheduler.py`).

`training/persistence.py`는 이 SPEC의 PRESERVE 대상이며 재정의하지 않는다 — 이
모듈은 파일명 관례를 독립적으로(중복) 파싱할 뿐, `persistence.py`의 함수를
호출하거나 그 내부 구현에 의존하지 않는다.
"""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_MODEL_FILENAME_PATTERN = re.compile(
    r"^(?P<market>[a-z]+)_(?P<horizon>\d+)_(?P<algorithm>[a-z]+)_"
    r"(?P<trained_date>\d{4}-\d{2}-\d{2})\.\w+$"
)


@dataclass(frozen=True, slots=True)
class ModelStalenessInfo:
    """(market, horizon, algorithm) 조합 1개에 대한 정체 판정 결과(AC-ATA-007)."""

    market: str
    horizon: int
    algorithm: str
    most_recent_trained_date: date
    is_stale: bool


def _parse_model_filename(filename: str) -> tuple[str, int, str, date] | None:
    """`{market}_{horizon}_{algorithm}_{trained_date}.{ext}`를 파싱한다.

    관례와 맞지 않는 파일(사이드카 `.sha256`, README 등)은 `None`을 반환한다.
    """
    match = _MODEL_FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    return (
        match.group("market"),
        int(match.group("horizon")),
        match.group("algorithm"),
        date.fromisoformat(match.group("trained_date")),
    )


def detect_stale_models(
    models_root: Path,
    *,
    threshold_days: int = 28,
    as_of: date | None = None,
) -> list[ModelStalenessInfo]:
    """REQ-ATA-072: `models_root` 하위를 스캔해 (market, horizon, algorithm) 조합별
    가장 최근 `trained_date`가 `threshold_days`를 초과하면 정체로 판정한다.

    동일 조합에 여러 활성 버전(최근 12개 보존 정책)이 존재해도 가장 최근
    `trained_date` 하나만 기준으로 판정한다 — 오래된 버전이 섞여 있어도 최신
    버전이 신선하면 정체가 아니다.
    """
    reference_date = as_of if as_of is not None else date.today()
    latest_by_combo: dict[tuple[str, int, str], date] = {}

    for path in models_root.rglob("*"):
        if not path.is_file():
            continue
        parsed = _parse_model_filename(path.name)
        if parsed is None:
            continue
        market, horizon, algorithm, trained_date = parsed
        key = (market, horizon, algorithm)
        if key not in latest_by_combo or trained_date > latest_by_combo[key]:
            latest_by_combo[key] = trained_date

    results: list[ModelStalenessInfo] = []
    for (market, horizon, algorithm), latest_date in latest_by_combo.items():
        age_days = (reference_date - latest_date).days
        results.append(
            ModelStalenessInfo(
                market=market,
                horizon=horizon,
                algorithm=algorithm,
                most_recent_trained_date=latest_date,
                is_stale=age_days > threshold_days,
            )
        )
    return results

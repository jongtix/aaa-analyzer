"""모델 정체(staleness) 감지 (SPEC-ANALYZER-TRAIN-AUTOMATION-001 §2.7, REQ-ATA-072/083).

활성 모델 파일명 관례 `{market}_{horizon}_{algorithm}_{trained_date}`(TRAIN-001이
확립, `training/persistence.py` `model_filename()`)에서 `trained_date`를 파싱해
정체를 감지한다(REQ-ATA-072) — 어떤 활성 모델이든 마지막 성공 재학습 후
`threshold_days`(기본 4주=28일)를 초과하면 정체로 표시한다. 이 감지는
REQ-ATA-083이 등록하는 전용 일일 cron 잡을 통해 주기적으로 트리거된다(§2.8,
`scheduler.py`).

`training/persistence.py`는 이 SPEC의 PRESERVE 대상이며 재정의하지 않는다 — 이
모듈은 파일명 관례를 독립적으로(중복) 파싱할 뿐, `persistence.py`의 함수를
호출하거나 그 내부 구현에 의존하지 않는다. 단, REQ-ATD-008(M4)이 도입한 확장자
allowlist는 `persistence.py`의 `_NATIVE_EXTENSION` 값 집합을 참조만 한다(DRY,
§B 리스크 5) — `persistence.py`의 함수 호출이나 내부 로직 의존은 여전히 없다.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from analyzer.training.persistence import _NATIVE_EXTENSION

_KST = ZoneInfo("Asia/Seoul")

_ALLOWED_MODEL_EXTENSIONS = sorted(set(_NATIVE_EXTENSION.values()))
"""REQ-ATD-008(§B 리스크 5): `training/persistence.py`의 `_NATIVE_EXTENSION`
값 집합(`{"txt", "json"}`)에서 동적으로 파생한다 — 신규 알고리즘 도입 시
`_NATIVE_EXTENSION`에 확장자가 추가되면 이 allowlist도 자동으로 갱신되는
단일 소스(DRY) 방식을 택한다(하드코딩 독립 상수 대신). `persistence.py`는
이 SPEC의 PRESERVE 대상이며 참조만 하고 재정의하지 않는다."""

_MODEL_FILENAME_PATTERN = re.compile(
    r"^(?P<market>[a-z]+)_(?P<horizon>\d+)_(?P<algorithm>[a-z]+)_"
    r"(?P<trained_date>\d{4}-\d{2}-\d{2})"
    r"\.(?:" + "|".join(re.escape(ext) for ext in _ALLOWED_MODEL_EXTENSIONS) + r")$"
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
    reference_date = as_of if as_of is not None else datetime.now(_KST).date()
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

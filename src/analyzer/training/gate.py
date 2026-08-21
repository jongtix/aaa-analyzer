"""맥 측 게이트 CLI — 챔피언 경로 해석 + 동결 하이퍼파라미터 리더 + verdict
직렬화 계약 (SPEC-ANALYZER-TRAIN-GATE-001 M1/M2, REQ-ATG-008/009/010).

`orchestration/promotion_gate.py`의 `evaluate_and_promote()`는 판정 로직이
완결되어 있으나(REQ-ATE-055~061), 그 입력 생산자(챔피언 경로 해석기, 동결
하이퍼파라미터 리더)가 프로덕션 코드에 전무하다 — 이 모듈이 최초로 배선한다.

verdict 직렬화/역직렬화 계약은 NAS 측 어댑터(`orchestration/gate_adapter.py`,
M4)와 이 모듈이 공유한다 — 양측이 같은 코드로 stdout JSON 라운드트립을
수행한다(REQ-ATG-008e).
"""

import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from analyzer.common.logging import get_logger
from analyzer.orchestration import activation as activation_module
from analyzer.orchestration.promotion_gate import PromotionVerdict
from analyzer.training import campaign_metrics as campaign_metrics_module
from analyzer.training import persistence as persistence_module
from analyzer.training.models import HORIZONS, MARKETS

logger = get_logger(__name__)

_POINT_ALGORITHMS: tuple[str, ...] = ("lightgbm", "xgboost")


def resolve_champion_model_paths(
    models_root: Path, algorithms: tuple[str, ...] = _POINT_ALGORITHMS
) -> dict[tuple[str, int, str], Path]:
    """REQ-ATG-009: `read_activation_manifest()` + `persistence.model_dir()`/
    `model_filename()` 조합으로 활성 챔피언 아티팩트 경로 매핑을 산출한다.

    활성화 매니페스트가 없는 조합은 매핑에서 제외한다 —
    `promotion_gate.py:246-248`의 기존 스킵 동작을 보존하는 회귀 가드다.
    """
    paths: dict[tuple[str, int, str], Path] = {}
    for market in MARKETS:
        for horizon in HORIZONS:
            for algorithm in algorithms:
                manifest = activation_module.read_activation_manifest(
                    models_root, market, horizon, algorithm
                )
                if manifest is None:
                    continue
                model_path = persistence_module.model_dir(
                    models_root, market, horizon, algorithm
                ) / persistence_module.model_filename(
                    market, horizon, algorithm, manifest.trained_date
                )
                paths[(market, horizon, algorithm)] = model_path
    return paths


def _list_active_trained_dates(
    models_root: Path, market: str, horizon: int, algorithm: str
) -> list[date]:
    """(market,horizon,algorithm) 조합의 active 경로에 남아있는 trained_date
    목록을 파일명 관례에서 파싱한다 — `persistence.py`는 무수정이며(PRESERVE),
    `staleness.py`와 동일하게 파일명 관례를 독립적으로 파싱한다."""
    directory = persistence_module.model_dir(models_root, market, horizon, algorithm)
    if not directory.exists():
        return []
    dates: list[date] = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix == ".sha256":
            continue
        trained_date_str = path.stem.rsplit("_", 1)[-1]
        try:
            dates.append(date.fromisoformat(trained_date_str))
        except ValueError:
            continue
    return dates


def warn_dangling_champions(
    models_root: Path, champion_model_paths: Mapping[tuple[str, int, str], Path]
) -> None:
    """REQ-ATG-009: 매니페스트가 가리키는 trained_date가 2단계 보존 정책에
    의해 아카이브로 이동된 조합에 대해 `detect_dangling_manifest()`(ATE
    REQ-ATE-054 — 현재 호출자 0건) 경유 경고 로그를 발행한다."""
    for market, horizon, algorithm in champion_model_paths:
        manifest = activation_module.read_activation_manifest(
            models_root, market, horizon, algorithm
        )
        if manifest is None:
            continue
        active_dates = _list_active_trained_dates(models_root, market, horizon, algorithm)
        if activation_module.detect_dangling_manifest(manifest, active_dates):
            logger.warning(
                "dangling activation manifest market=%s horizon=%s algorithm=%s trained_date=%s",
                market,
                horizon,
                algorithm,
                manifest.trained_date,
            )


def read_frozen_hyperparameters(
    models_root: Path, market: str, horizon: int, algorithm: str
) -> Mapping[str, Any] | None:
    """REQ-ATG-010: 챔피언 아티팩트 `.meta.json` 사이드카에서
    `frozen_hyperparameters`를 조합별로 읽는다.

    챔피언 매니페스트/사이드카/필드 중 어느 하나라도 부재하면 경고 로그를
    발행하고 `None`을 반환한다(호출자는 라이브러리 기본값으로 폴백) —
    게이트 전체를 실패시키지 않는다.
    """
    manifest = activation_module.read_activation_manifest(models_root, market, horizon, algorithm)
    if manifest is None:
        return None
    model_path = persistence_module.model_dir(
        models_root, market, horizon, algorithm
    ) / persistence_module.model_filename(market, horizon, algorithm, manifest.trained_date)
    sidecar_path = campaign_metrics_module.sidecar_path_for(model_path)
    if not sidecar_path.exists():
        logger.warning(
            "frozen hyperparameters sidecar missing market=%s horizon=%s algorithm=%s path=%s",
            market,
            horizon,
            algorithm,
            sidecar_path,
        )
        return None
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_hyperparameters")
    if not frozen:
        logger.warning(
            "frozen_hyperparameters field missing or empty market=%s horizon=%s algorithm=%s",
            market,
            horizon,
            algorithm,
        )
        return None
    return frozen


def serialize_verdicts(verdicts: Mapping[tuple[str, int, str], PromotionVerdict]) -> str:
    """REQ-ATG-008e: `PromotionVerdict` 매핑을 stdout 단일 JSON 문서로
    직렬화한다 — NAS 측 어댑터와 공유하는 계약."""
    payload = [
        {
            "market": verdict.market,
            "horizon": verdict.horizon,
            "algorithm": verdict.algorithm,
            "promoted": verdict.promoted,
            "challenger_rank_ic": verdict.challenger_rank_ic,
            "champion_rank_ic": verdict.champion_rank_ic,
            "challenger_trained_date": verdict.challenger_trained_date.isoformat(),
        }
        for verdict in verdicts.values()
    ]
    return json.dumps(payload, ensure_ascii=False)


def deserialize_verdicts(raw: str) -> dict[tuple[str, int, str], PromotionVerdict]:
    """`serialize_verdicts()`의 역함수 — NAS 측 어댑터가 stdout에서 받은
    JSON 문서를 `PromotionVerdict` 매핑으로 복원한다."""
    payload = json.loads(raw)
    verdicts: dict[tuple[str, int, str], PromotionVerdict] = {}
    for item in payload:
        verdict = PromotionVerdict(
            market=item["market"],
            horizon=item["horizon"],
            algorithm=item["algorithm"],
            promoted=item["promoted"],
            challenger_rank_ic=item["challenger_rank_ic"],
            champion_rank_ic=item["champion_rank_ic"],
            challenger_trained_date=date.fromisoformat(item["challenger_trained_date"]),
        )
        verdicts[(verdict.market, verdict.horizon, verdict.algorithm)] = verdict
    return verdicts

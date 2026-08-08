"""Optuna 하이퍼파라미터 튜닝 — 재개 가능 스터디 + 폴드 단위 프루닝 (SPEC-ANALYZER-TRAIN-001 M6).

REQ-AT-100: 영속적 storage 백엔드(SQLite RDBStorage)와
`load_if_exists=True`로 재개 가능한 단일 머신 순차 튜닝 스터디를
구현한다.

REQ-AT-101: `optuna-integration` 패키지를 의존성으로 추가하지 않는다
(4.9.0 릴리스가 Python 3.14 지원을 선언하지 않음, §4.5 근거) — 폴드
단위 프루닝을 Optuna 코어 API(`trial.report()`/`trial.should_prune()`)
만으로 직접 구현한다.

REQ-AT-102: 튜닝 대상은 포인트 모델(8개)에 한정한다 — `models.py`의
분위수 보조 모델 학습 경로는 Optuna `trial` 객체를 전혀 참조하지
않는다(REQ-AT-062, 코드 리뷰/grep으로 확인 가능).

REQ-AT-104: WFV 폴드가 완료될 때마다(부스팅 라운드 단위가 아님)
`trial.report()`/`trial.should_prune()`을 직접 호출한다.
`MedianPruner(n_warmup_steps=2)`로 최소 2개 폴드가 완료되기 전에는
프루닝을 허용하지 않는다 — 시장 국면 전반에는 일반화하지만 초기
폴드에서만 저조한 하이퍼파라미터 영역을 조기 폐기하는 위험을
회피한다.

plan.md §B 리스크3: Optuna SQLite storage 파일이 여러 (시장, horizon)
조합의 학습 프로세스에 동시 접근되면 쓰기 잠금 경합이 발생할 수
있다 — 조합별로 별도 storage 파일(`optuna_{market}_{horizon}.db`)을
사용해 회피한다.
"""

from pathlib import Path

import optuna

_N_WARMUP_STEPS = 2
"""REQ-AT-104: 최소 2개 폴드 완료 이전에는 프루닝을 허용하지 않는다."""


def storage_url_for_combo(storage_dir: Path, market: str, horizon: int) -> str:
    """조합(시장×horizon)별 별도 SQLite storage 파일 경로를 반환한다(plan.md §B 리스크3)."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = storage_dir / f"optuna_{market}_{horizon}.db"
    return f"sqlite:///{db_path}"


def create_or_resume_study(
    storage_dir: Path,
    market: str,
    horizon: int,
    study_name: str | None = None,
    direction: str = "minimize",
) -> optuna.Study:
    """재개 가능한(`load_if_exists=True`) Optuna 스터디를 생성/로드한다(REQ-AT-100, AC-AT-010).

    프루너는 `MedianPruner(n_warmup_steps=2)`로 고정한다(REQ-AT-104) —
    `optuna-integration`의 콜백 기반 프루닝에 의존하지 않는다(REQ-AT-101).
    """
    resolved_study_name = study_name or f"{market}_{horizon}"
    storage_url = storage_url_for_combo(storage_dir, market, horizon)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=_N_WARMUP_STEPS)
    return optuna.create_study(
        study_name=resolved_study_name,
        storage=storage_url,
        direction=direction,
        load_if_exists=True,
        pruner=pruner,
    )


def report_fold_and_maybe_prune(trial: optuna.Trial, fold_index: int, fold_metric: float) -> None:
    """WFV 폴드가 완료될 때마다(폴드당 정확히 1회) 호출한다(REQ-AT-104).

    부스팅 라운드 단위로 호출해서는 안 된다 — 호출자(WFV 루프)가 폴드
    완료 시점에만 이 함수를 호출할 책임을 진다. `trial.report()`로
    `fold_index`를 step으로 삼아 폴드별 검증 지표를 보고하고,
    `trial.should_prune()`이 `True`를 반환하면 `optuna.TrialPruned`를
    발생시킨다. warm-up 유예(최소 2개 폴드) 자체는 스터디 생성 시
    설정한 `MedianPruner(n_warmup_steps=2)`가 담당하며, 이 함수는 별도
    warm-up 로직을 갖지 않는다.
    """
    trial.report(fold_metric, step=fold_index)
    if trial.should_prune():
        raise optuna.TrialPruned(f"fold {fold_index}에서 프루닝됨(trial={trial.number})")

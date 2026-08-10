"""src/analyzer/training/tuning.py Optuna 튜닝 테스트 (SPEC-ANALYZER-TRAIN-001 M6).

REQ-AT-100(재개 가능한 storage 스터디)/REQ-AT-101(optuna-integration 미채택)/
REQ-AT-102(포인트 모델 전용)/REQ-AT-104(폴드 단위 프루닝)를 검증한다.
AC-AT-010의 worked example(SQLite RDBStorage, 5개 trial 기록된 기존
스터디를 재로드해 이어서 진행)을 그대로 구현한다.
"""

import subprocess
from pathlib import Path

import optuna

from analyzer.training.tuning import (
    create_or_resume_study,
    report_fold_and_maybe_prune,
    storage_url_for_combo,
)


class TestStorageUrlForCombo:
    """plan.md §B 리스크3: 조합별 별도 storage 파일로 SQLite 쓰기 잠금 경합 회피."""

    def test_different_combos_get_different_storage_files(self, tmp_path: Path):
        url_domestic_20 = storage_url_for_combo(tmp_path, "domestic", 20)
        url_domestic_60 = storage_url_for_combo(tmp_path, "domestic", 60)
        url_overseas_20 = storage_url_for_combo(tmp_path, "overseas", 20)

        assert url_domestic_20 != url_domestic_60
        assert url_domestic_20 != url_overseas_20

    def test_storage_url_is_sqlite_scheme(self, tmp_path: Path):
        url = storage_url_for_combo(tmp_path, "domestic", 20)

        assert url.startswith("sqlite:///")
        assert "domestic" in url
        assert "20" in url


class TestCreateOrResumeStudy:
    """AC-AT-010: 재개 가능한(load_if_exists=True) 스터디."""

    def test_ac_at_010_resumes_existing_study_with_continued_trial_numbers(self, tmp_path: Path):
        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2

        study1 = create_or_resume_study(tmp_path, "domestic", 20)
        study1.optimize(objective, n_trials=5)
        assert len(study1.trials) == 5

        study2 = create_or_resume_study(tmp_path, "domestic", 20)
        study2.optimize(objective, n_trials=3)

        assert len(study2.trials) == 8
        trial_numbers = sorted(t.number for t in study2.trials)
        assert trial_numbers == list(range(8))

    def test_new_study_starts_empty(self, tmp_path: Path):
        study = create_or_resume_study(tmp_path, "domestic", 20)

        assert len(study.trials) == 0

    def test_study_uses_median_pruner_with_warmup_2(self, tmp_path: Path):
        study = create_or_resume_study(tmp_path, "domestic", 20)

        assert isinstance(study.pruner, optuna.pruners.MedianPruner)
        assert study.pruner._n_warmup_steps == 2


class TestReportFoldAndMaybePrune:
    """REQ-AT-104: WFV 폴드 완료 시점에만 호출(부스팅 라운드 단위 아님)."""

    def test_called_exactly_once_per_fold_not_per_boosting_round(self, tmp_path: Path):
        study = create_or_resume_study(tmp_path, "domestic", 20)
        call_log: list[int] = []

        def objective(trial: optuna.Trial) -> float:
            for fold_index in range(4):
                call_log.append(fold_index)
                report_fold_and_maybe_prune(trial, fold_index, fold_metric=1.0 - fold_index * 0.1)
            return 0.5

        study.optimize(objective, n_trials=1)

        assert call_log == [0, 1, 2, 3]

    def test_first_trial_of_fresh_study_completes_without_comparison_baseline(self, tmp_path: Path):
        """§B 경계 사례: 신규 study의 첫 trial(비교 대상 없음)에서도 정상 진행.

        `MedianPruner`는 비교할 완료 trial이 없으면 프루닝을 발동시키지 않아야
        한다 — warm-up 없이 곧바로 trial #0에서 호출되는 첫 케이스를 검증한다
        (기존 테스트들은 모두 5-trial warm-up 이후의 trial을 대상으로 했음).
        """
        study = create_or_resume_study(tmp_path, "domestic", 20)
        fold_calls: list[int] = []

        def first_trial_objective(trial: optuna.Trial) -> float:
            for fold_index in range(4):
                fold_calls.append(fold_index)
                report_fold_and_maybe_prune(trial, fold_index, fold_metric=1.0)
            return 1.0

        study.optimize(first_trial_objective, n_trials=1)

        assert fold_calls == [0, 1, 2, 3]
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE

    def test_ac_at_010_quantile_training_function_never_touches_optuna_trial(self):
        """AC-AT-010 별도 확인: 분위수 보조 모델 학습 함수가 Optuna trial을 전혀 참조하지 않는다."""
        result = subprocess.run(
            ["grep", "-n", "optuna", "src/analyzer/training/models.py"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, f"models.py가 optuna를 참조함:\n{result.stdout}"

    def test_prunes_after_warmup_when_trial_is_clearly_worse(self, tmp_path: Path):
        study = create_or_resume_study(tmp_path, "domestic", 20)

        def good_objective(trial: optuna.Trial) -> float:
            for fold_index in range(4):
                report_fold_and_maybe_prune(trial, fold_index, fold_metric=0.1)
            return 0.1

        study.optimize(good_objective, n_trials=5)

        def bad_objective(trial: optuna.Trial) -> float:
            for fold_index in range(4):
                report_fold_and_maybe_prune(trial, fold_index, fold_metric=100.0)
            return 100.0

        study.optimize(bad_objective, n_trials=1)

        assert study.trials[-1].state == optuna.trial.TrialState.PRUNED

    def test_does_not_prune_before_warmup_steps_complete(self, tmp_path: Path):
        """MedianPruner(n_warmup_steps=2) — 최소 2개 폴드 완료 전에는 프루닝하지 않는다."""
        study = create_or_resume_study(tmp_path, "domestic", 20)
        completed_fold_indices: list[int] = []

        def good_objective(trial: optuna.Trial) -> float:
            for fold_index in range(4):
                report_fold_and_maybe_prune(trial, fold_index, fold_metric=0.1)
            return 0.1

        study.optimize(good_objective, n_trials=5)

        def bad_objective(trial: optuna.Trial) -> float:
            for fold_index in range(4):
                report_fold_and_maybe_prune(trial, fold_index, fold_metric=100.0)
                completed_fold_indices.append(fold_index)
            return 100.0

        study.optimize(bad_objective, n_trials=1)

        assert study.trials[-1].state == optuna.trial.TrialState.PRUNED
        # 프루닝이 발동했다면 그 이전(0, 1)까지는 정상적으로 완료 기록되어야 한다
        # (n_warmup_steps=2 이전에는 프루닝되지 않음을 간접 확인).
        assert 0 in completed_fold_indices
        assert 1 in completed_fold_indices
        # 완료된 폴드 수가 4 미만이면 도중에 프루닝되어 루프가 중단된 것이다.
        assert len(completed_fold_indices) < 4

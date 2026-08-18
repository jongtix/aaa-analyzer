"""src/analyzer/training/stabilization.py 테스트 (SPEC-ANALYZER-TRAIN-EVAL-001 M5).

GATE-1/2/3 경계 케이스, 단일 게이트 실패 시 미안정화, 부분 안정화
시나리오(F1 핵심 검증 — LightGBM만 안정화 시 앙상블 미자격), 챔피언
선정이 롤링 평균 Rank IC 최댓값 기준으로 정확히 동작함을 검증한다.
"""

from analyzer.training.stabilization import (
    CAMPAIGN_GATE1_ROLLING_WEEKS,
    CAMPAIGN_GATE3_ICIR_FLOOR,
    CAMPAIGN_GATE3_ROLLING_WEEKS,
    ChampionSelection,
    ComboStabilizationVerdict,
    evaluate_combo_stabilization,
    gate1_passed,
    gate2_passed,
    gate3_passed,
    select_champion_strategy,
)


def _verdict(
    *,
    market: str = "domestic",
    horizon: int = 20,
    algorithm: str = "lightgbm",
    stabilized: bool,
) -> ComboStabilizationVerdict:
    """select_champion_strategy 테스트용 최소 verdict 헬퍼."""
    return ComboStabilizationVerdict(
        market=market,
        horizon=horizon,
        algorithm=algorithm,
        stabilized=stabilized,
        gate1_passed=stabilized,
        gate2_passed=stabilized,
        gate3_passed=stabilized,
        gate1_rolling_mean_rank_ic=0.05 if stabilized else -0.01,
        gate2_mean_rank_ic=0.05 if stabilized else -0.01,
        gate3_rolling_icir=0.2 if stabilized else 0.0,
    )


class TestGate1RollingMeanRankIc:
    def test_window_not_yet_met_fails(self):
        values = [0.05] * (CAMPAIGN_GATE1_ROLLING_WEEKS - 1)
        assert gate1_passed(values) is False

    def test_exactly_at_window_boundary_with_positive_mean_passes(self):
        values = [0.05] * CAMPAIGN_GATE1_ROLLING_WEEKS
        assert gate1_passed(values) is True

    def test_exactly_at_window_boundary_with_negative_mean_fails(self):
        values = [-0.05] * CAMPAIGN_GATE1_ROLLING_WEEKS
        assert gate1_passed(values) is False

    def test_only_trailing_window_considered_not_full_history(self):
        # 과거(초반) 구간은 전부 음수이지만, 최근 rolling_weeks 구간이
        # 전부 양수면 GATE-1은 통과해야 한다(트레일링 윈도우 원칙).
        values = [-0.1] * 100 + [0.05] * CAMPAIGN_GATE1_ROLLING_WEEKS
        assert gate1_passed(values) is True

    def test_single_fold_never_used_as_pass_condition(self):
        # 단일 폴드(윈도우 미만 관측치)는 그 값이 아무리 커도 통과할 수
        # 없다 — REQ-ATE-039 하드 불변조건.
        assert gate1_passed([1.0]) is False


class TestGate2CampaignWideMeanRankIc:
    def test_empty_series_fails(self):
        assert gate2_passed([]) is False

    def test_positive_overall_mean_passes(self):
        assert gate2_passed([0.05, -0.01, 0.02]) is True

    def test_nonpositive_overall_mean_fails(self):
        assert gate2_passed([0.01, -0.02, 0.005]) is False

    def test_uses_full_history_not_only_recent_window(self):
        # GATE-2는 롤링이 아니라 캠페인 전체 평균이므로, 최근 값이
        # 전부 음수여도 전체 평균이 양수면 통과해야 한다.
        values = [0.5] + [-0.01] * 20
        assert gate2_passed(values) is True


class TestGate3RollingIcir:
    def test_window_not_yet_met_fails(self):
        values = [0.05] * (CAMPAIGN_GATE3_ROLLING_WEEKS - 1)
        assert gate3_passed(values) is False

    def test_zero_stddev_treated_as_zero_icir_and_fails(self):
        values = [0.05] * CAMPAIGN_GATE3_ROLLING_WEEKS
        assert gate3_passed(values) is False

    def test_high_mean_low_variance_exceeds_floor_and_passes(self):
        values = [0.09] * 26 + [0.11] * 26
        assert gate3_passed(values) is True

    def test_exactly_at_icir_floor_boundary_fails(self):
        # icir_floor는 초과(>)만 통과이며 등호는 실패해야 한다.
        values = [0.1] * CAMPAIGN_GATE3_ROLLING_WEEKS
        # stddev=0 -> icir=0.0, floor(0.10)보다 작으므로 실패
        assert gate3_passed(values, icir_floor=CAMPAIGN_GATE3_ICIR_FLOOR) is False


class TestComboStabilizationSingleGateFailure:
    def _all_gates_pass_values(self) -> list[float]:
        return [0.09] * 26 + [0.11] * 26

    def test_all_three_gates_pass_stabilized(self):
        values = self._all_gates_pass_values()
        verdict = evaluate_combo_stabilization("domestic", 20, "lightgbm", values)
        assert verdict.gate1_passed is True
        assert verdict.gate2_passed is True
        assert verdict.gate3_passed is True
        assert verdict.stabilized is True

    def test_gate1_failure_alone_causes_not_stabilized(self):
        # GATE-2/3는 통과하지만 최근 윈도우 평균이 음수라 GATE-1만 실패.
        values = [0.2] * 40 + [-0.05] * CAMPAIGN_GATE1_ROLLING_WEEKS
        verdict = evaluate_combo_stabilization("domestic", 20, "lightgbm", values)
        assert verdict.gate1_passed is False
        assert verdict.gate2_passed is True
        assert verdict.stabilized is False

    def test_gate2_failure_alone_causes_not_stabilized(self):
        # 최근 구간은 GATE-1/3을 만족시키도록 강한 양수지만, 과거 구간의
        # 큰 음수값이 전체 평균(GATE-2)을 음수로 끌어내린다.
        values = [-5.0] * 5 + [0.09] * 26 + [0.11] * 26
        verdict = evaluate_combo_stabilization("domestic", 20, "lightgbm", values)
        assert verdict.gate2_passed is False
        assert verdict.gate1_passed is True
        assert verdict.gate3_passed is True
        assert verdict.stabilized is False

    def test_gate3_failure_alone_causes_not_stabilized(self):
        # 최근 52주 구간의 평균은 양수(GATE-1/2 통과)이지만 변동성이 커
        # ICIR(0.05/0.85≈0.059)이 floor(0.10)를 넘지 못한다.
        values = [0.9, -0.8] * (CAMPAIGN_GATE3_ROLLING_WEEKS // 2)
        verdict = evaluate_combo_stabilization("domestic", 20, "lightgbm", values)
        assert verdict.gate1_passed is True
        assert verdict.gate2_passed is True
        assert verdict.gate3_passed is False
        assert verdict.stabilized is False


class TestChampionSelectionPartialStabilizationF1:
    """F1 핵심 검증: LightGBM만 안정화 시 LightGBM 단독 전략만 자격을
    얻고 앙상블은 자격을 얻지 못한다."""

    def test_lightgbm_stabilized_xgboost_not_lightgbm_eligible(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=False)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.05, "xgboost": -0.02, "ensemble": 0.03},
        )
        assert "lightgbm" in result.eligible_strategies

    def test_lightgbm_stabilized_xgboost_not_ensemble_not_eligible(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=False)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.05, "xgboost": -0.02, "ensemble": 0.03},
        )
        assert "ensemble" not in result.eligible_strategies

    def test_lightgbm_stabilized_xgboost_not_champion_is_lightgbm(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=False)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.05, "xgboost": -0.02, "ensemble": 0.03},
        )
        assert result.champion_algorithm == "lightgbm"

    def test_lightgbm_champion_xgboost_artifact_excluded(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=False)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.05, "xgboost": -0.02, "ensemble": 0.03},
        )
        assert result.excluded_artifact_algorithms == ("xgboost",)
        assert result.deployment_prohibited is False


class TestChampionSelectionXgboostOnlyStabilized:
    def test_xgboost_stabilized_lightgbm_not_champion_is_xgboost(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=False)
        xgb = _verdict(algorithm="xgboost", stabilized=True)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": -0.02, "xgboost": 0.06, "ensemble": 0.03},
        )
        assert result.champion_algorithm == "xgboost"
        assert result.eligible_strategies == ("xgboost",)

    def test_xgboost_champion_lightgbm_artifact_excluded(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=False)
        xgb = _verdict(algorithm="xgboost", stabilized=True)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": -0.02, "xgboost": 0.06, "ensemble": 0.03},
        )
        assert result.excluded_artifact_algorithms == ("lightgbm",)
        assert result.deployment_prohibited is False


class TestChampionSelectionBothStabilized:
    def test_both_stabilized_ensemble_eligible(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=True)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.05, "xgboost": 0.04, "ensemble": 0.08},
        )
        assert set(result.eligible_strategies) == {"lightgbm", "xgboost", "ensemble"}

    def test_maximum_rolling_mean_rank_ic_selected_as_champion(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=True)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.05, "xgboost": 0.04, "ensemble": 0.08},
        )
        assert result.champion_algorithm == "ensemble"

    def test_ensemble_champion_keeps_both_artifacts(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=True)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.05, "xgboost": 0.04, "ensemble": 0.08},
        )
        assert result.excluded_artifact_algorithms == ()

    def test_both_stabilized_but_single_algorithm_wins_no_exclusion(self):
        # 둘 다 안정화됐지만 앙상블보다 lightgbm 단독의 롤링 평균이 더
        # 높은 경우 — 챔피언은 단독이어도 반대편이 안정화 상태이므로
        # 아티팩트 제외 대상이 아니다.
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=True)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.10, "xgboost": 0.04, "ensemble": 0.06},
        )
        assert result.champion_algorithm == "lightgbm"
        assert result.excluded_artifact_algorithms == ()


class TestChampionSelectionFullProhibitionRequAte047:
    def test_neither_stabilized_deployment_prohibited(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=False)
        xgb = _verdict(algorithm="xgboost", stabilized=False)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": -0.02, "xgboost": -0.03, "ensemble": -0.01},
        )
        assert result.deployment_prohibited is True
        assert result.champion_algorithm is None
        assert result.eligible_strategies == ()

    def test_neither_stabilized_diagnostics_identify_failed_gates_and_values(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=False)
        xgb = _verdict(algorithm="xgboost", stabilized=False)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": -0.02, "xgboost": -0.03, "ensemble": -0.01},
        )
        assert "lightgbm" in result.diagnostics
        assert "xgboost" in result.diagnostics
        lgbm_diag = result.diagnostics["lightgbm"]
        assert lgbm_diag["failed_gates"] != ()
        assert lgbm_diag["gate1_rolling_mean_rank_ic"] == lgbm.gate1_rolling_mean_rank_ic

    def test_exactly_one_stabilized_is_not_full_prohibition_case(self):
        # 정확히 하나만 안정화된 경우는 REQ-ATE-047의 적용 대상이 아니다
        # — 부분 배포(단독 전략) 자격을 얻어야 한다(전면 금지와 구분).
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=False)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.05, "xgboost": -0.02, "ensemble": 0.03},
        )
        assert result.deployment_prohibited is False
        assert result.champion_algorithm is not None


class TestChampionSelectionReturnType:
    def test_returns_champion_selection_dataclass(self):
        lgbm = _verdict(algorithm="lightgbm", stabilized=True)
        xgb = _verdict(algorithm="xgboost", stabilized=True)
        result = select_champion_strategy(
            "domestic",
            20,
            lgbm,
            xgb,
            {"lightgbm": 0.05, "xgboost": 0.04, "ensemble": 0.08},
        )
        assert isinstance(result, ChampionSelection)
        assert result.market == "domestic"
        assert result.horizon == 20

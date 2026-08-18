"""Prometheus 계측 (SPEC-ANALYZER-TRAIN-AUTOMATION-001 §2.7, REQ-ATA-070/071).

`prometheus_client`를 사용해 학습 실행의 성공/실패 메트릭과 모델 정체(staleness)
메트릭을 발행한다(REQ-ATA-070). 이 SPEC의 analyzer 측 책임은 올바른 메트릭
발행에 한정된다(REQ-ATA-071) — 이 메트릭을 소비하는 vmalert 알람 규칙 YAML은
`aaa-infra` 레포 소관이며 이 모듈이 작성하지 않는다.

메트릭 이름/레이블 계약(aaa-infra vmalert 규칙 후속 작업이 소비, REQ-ATA-071/
acceptance.md §D):

- ``aaa_analyzer_training_run_total{stage, outcome}`` (Counter) — 학습 실행
  결과 카운터. ``stage``: ``wol|ssh|training|timeout|success``.
  ``outcome``: ``success|failure``.
- ``aaa_analyzer_training_run_last_success_timestamp_seconds{market, horizon,
  algorithm}`` (Gauge) — 마지막 성공 학습 실행의 Unix epoch 초. 정체
  (staleness) 감지의 입력.
- ``aaa_analyzer_model_stale{market, horizon, algorithm}`` (Gauge, 0/1) —
  모델 정체 여부.
"""

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge

TRAINING_RUN_TOTAL_NAME = "aaa_analyzer_training_run_total"
LAST_SUCCESS_TIMESTAMP_NAME = "aaa_analyzer_training_run_last_success_timestamp_seconds"
MODEL_STALE_NAME = "aaa_analyzer_model_stale"
RANK_IC_NAME = "aaa_analyzer_training_rank_ic"
"""REQ-ATE-066(M6): (시장,horizon,algorithm) 조합별 Rank IC(§2.8/§2.10 1차
평가 지표) 게이지 이름 — 기존 `aaa_analyzer_training_run_*` 명명 관례를
따른다. 이 게이지를 소비하는 vmalert 알람 규칙은 이 SPEC의 책임이 아니다
(REQ-ATE-066 shall not, REQ-ATA-071 원칙 계승)."""


class TrainingMetrics:
    """analyzer 프로세스의 Prometheus 레지스트리에 바인딩된 계측 묶음.

    기본적으로 `prometheus_client`의 전역(default) 레지스트리를 사용하지만,
    테스트에서는 격리된 `CollectorRegistry()`를 주입해 전역 상태 오염과
    "Duplicated timeseries" 재등록 오류를 피한다.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        target_registry: CollectorRegistry = registry if registry is not None else REGISTRY

        self.training_run_total = Counter(
            TRAINING_RUN_TOTAL_NAME,
            "학습 실행 결과 카운터(stage/outcome 레이블)",
            ["stage", "outcome"],
            registry=target_registry,
        )
        self.last_success_timestamp = Gauge(
            LAST_SUCCESS_TIMESTAMP_NAME,
            "마지막 성공 학습 실행 Unix epoch 초",
            ["market", "horizon", "algorithm"],
            registry=target_registry,
        )
        self.model_stale = Gauge(
            MODEL_STALE_NAME,
            "모델 정체(staleness) 여부 (0=정상, 1=정체)",
            ["market", "horizon", "algorithm"],
            registry=target_registry,
        )
        self.rank_ic = Gauge(
            RANK_IC_NAME,
            "조합별 Rank IC(1차 평가 지표, §2.8/§2.10)",
            ["market", "horizon", "algorithm"],
            registry=target_registry,
        )

    def record_success(
        self,
        *,
        market: str,
        horizon: int,
        algorithm: str,
        timestamp: float,
        outcome: str = "success",
    ) -> None:
        """AC-ATA-001 + REQ-ATE-064/065(M6): 성공 경로 카운터를 `outcome`
        레이블로 구분해 증가시킨다 — `outcome="success"`(승격)일 때만 마지막
        성공 시각 게이지를 갱신한다. `outcome="held-back"`(보류)은 카운터만
        증가시키고 게이지는 갱신하지 않는다(보류된 조합은 기존 챔피언이
        계속 서빙 중이므로 "마지막 성공"을 갱신할 근거가 없다). 기존
        호출자(1차 배포 이전, `outcome` 미지정)는 `outcome="success"` 기본값으로
        하위 호환된다."""
        self.training_run_total.labels(stage="success", outcome=outcome).inc()
        if outcome == "success":
            self.last_success_timestamp.labels(
                market=market, horizon=str(horizon), algorithm=algorithm
            ).set(timestamp)

    def record_rank_ic(self, *, market: str, horizon: int, algorithm: str, rank_ic: float) -> None:
        """REQ-ATE-066: (시장,horizon,algorithm) 조합별 Rank IC 게이지를 갱신한다."""
        self.rank_ic.labels(market=market, horizon=str(horizon), algorithm=algorithm).set(rank_ic)

    def record_failure(self, *, stage: str) -> None:
        """REQ-ATA-060/061: WoL/SSH/학습스크립트/타임아웃 4종 실패를 동일한
        카운터에 `stage` 레이블로 구분해 기록한다 — 유형별 별도 처리 로직을
        분기하지 않는다."""
        self.training_run_total.labels(stage=stage, outcome="failure").inc()

    def record_staleness(
        self, *, market: str, horizon: int, algorithm: str, is_stale: bool
    ) -> None:
        """AC-ATA-007(REQ-ATA-072): (market, horizon, algorithm) 조합의 정체 여부를
        게이지로 발행한다."""
        self.model_stale.labels(market=market, horizon=str(horizon), algorithm=algorithm).set(
            1.0 if is_stale else 0.0
        )

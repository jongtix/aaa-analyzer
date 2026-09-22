"""추론 Prometheus 계측 (SPEC-ANALYZER-INFER-001 M8, REQ-AIF-130, design.md §8).

`orchestration/metrics.py`의 `TrainingMetrics` 구조(레이블 규약, 주입 가능한
레지스트리, 얇은 record_* 편의 메서드)를 그대로 계승한다 — REQ-AIF-130이
명시적으로 요구하는 계승이다. 이 SPEC의 analyzer 측 책임은 올바른 메트릭
발행에 한정되며, 이 메트릭을 소비하는 vmalert 알람 규칙 YAML은 `aaa-infra`
레포 소관이다(REQ-ATA-071 원칙 계승, REQ-AIF-142).

메트릭 이름 결정(B-INFER-1) — design.md §8은 `analyzer_inference_*`로 표기했고
AC-AIF-021도 그 이름으로 `/metrics` 조회를 검증하지만, 실제 코드베이스 관례는
`orchestration/metrics.py`의 `aaa_analyzer_training_*` 접두사다. REQ-AIF-130이
"`TrainingMetrics` 패턴을 계승"하라고 명시하므로 **코드베이스 관례를 따라**
`aaa_analyzer_inference_*`로 명명한다. design.md §8의 이름은 부분 문자열로
그대로 보존되므로(`aaa_` + `analyzer_inference_skip_total`) AC-AIF-021의
`/metrics` 문자열 존재 검증은 그대로 충족된다.

메트릭 이름/레이블 계약:

- ``aaa_analyzer_inference_skip_total{market, horizon, reason}`` (Counter) —
  조합/종목 단위 스킵 카운터. ``reason``은 `resolution.SkipReason`의 6개 값
  (``no_manifest``/``sha_mismatch``/``quantile_missing``/
  ``feature_insufficient``/``degenerate_quantile``/``manifest_race``).
- ``aaa_analyzer_inference_signals_total{market, horizon, signal_class}``
  (Counter) — 신호 생성 카운터. ``signal_class``는 등급 5클래스
  (`training/boundaries.GRADE_ORDER`).
- ``aaa_analyzer_inference_cycle_duration_seconds{market}`` (Histogram) —
  추론 사이클 소요 시간.
- ``aaa_analyzer_inference_last_cycle_seconds{market}`` (Gauge) —
  시장별 가장 최근 성공 완료된 추론 사이클의 UTC epoch 초
  (SPEC-OBSV-ANALYZER-DEADMAN-001 REQ-DMR-001). `multiprocess_mode="max"`로
  등록한다 — 이 게이지 값은 단조 증가하는 UTC epoch 초이므로, 재기동으로
  `PROMETHEUS_MULTIPROC_DIR`의 잔존 pid db 파일이 정리된 뒤에도 (a) Redis
  warm-start 기준선(부모 자신의 pid db 파일)과 (b) 그 이후 자식 프로세스가
  기록한 실제 완료 시각 중 항상 더 최근 값(=최댓값)이 병합 결과로 노출된다
  (REQ-DMR-005). `aaa-analyzer/src/analyzer/inference/last_cycle.py`가 이
  게이지의 write(자식)/warm-start(부모) 양쪽 배선을 소유한다.
"""

from __future__ import annotations

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

from analyzer.inference.resolution import SkipReason

INFERENCE_SKIP_TOTAL_NAME = "aaa_analyzer_inference_skip_total"
INFERENCE_SIGNALS_TOTAL_NAME = "aaa_analyzer_inference_signals_total"
INFERENCE_CYCLE_DURATION_NAME = "aaa_analyzer_inference_cycle_duration_seconds"
INFERENCE_LAST_CYCLE_NAME = "aaa_analyzer_inference_last_cycle_seconds"
"""SPEC-OBSV-ANALYZER-DEADMAN-001 REQ-DMR-001 — 데드맨 스위치 재설계가
`absent_over_time`을 대체하는 재기동-생존 신호로 도입한 게이지 이름."""

CYCLE_DURATION_BUCKETS: tuple[float, ...] = (
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    900.0,
    1200.0,
    1800.0,
    2700.0,
    float("inf"),
)
"""`prometheus_client` 기본 버킷(최대 10초)은 이 메트릭에 쓸 수 없다 — M7
실측 사이클이 domestic 745.88초 / overseas 306.86초로 전부 최상단 버킷에
몰려 분포 정보가 사라진다. 분 단위로 재구성하되, 상단은 REQ-AIF-142 vmalert
임계(30분 = 1800초)와 `InferenceConfig.stream_claim_idle_seconds`(1800초)를
경계로 포함해 "임계 초과" 여부가 버킷 하나로 판정되게 한다."""

_VALID_SKIP_REASONS: frozenset[str] = frozenset(member.value for member in SkipReason)
"""REQ-AIF-130 (a)가 확정한 6개 사유 레이블 — `resolution.SkipReason`이 유일한
어휘 출처다(이 모듈은 새 문자열을 정의하지 않는다)."""


class InferenceMetrics:
    """추론 자식 프로세스의 Prometheus 레지스트리에 바인딩된 계측 묶음.

    기본적으로 `prometheus_client` 전역(default) 레지스트리를 사용하지만,
    테스트에서는 격리된 `CollectorRegistry()`를 주입해 전역 상태 오염과
    "Duplicated timeseries" 재등록 오류를 피한다(`TrainingMetrics` 동일 관례).
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        target_registry: CollectorRegistry = registry if registry is not None else REGISTRY

        self.inference_skip_total = Counter(
            INFERENCE_SKIP_TOTAL_NAME,
            "추론 스킵 카운터(market/horizon/reason 레이블)",
            ["market", "horizon", "reason"],
            registry=target_registry,
        )
        self.inference_signals_total = Counter(
            INFERENCE_SIGNALS_TOTAL_NAME,
            "추론 신호 생성 카운터(market/horizon/signal_class 레이블)",
            ["market", "horizon", "signal_class"],
            registry=target_registry,
        )
        self.inference_cycle_duration = Histogram(
            INFERENCE_CYCLE_DURATION_NAME,
            "시장별 추론 사이클 소요 시간(초)",
            ["market"],
            buckets=CYCLE_DURATION_BUCKETS,
            registry=target_registry,
        )
        self.inference_last_cycle = Gauge(
            INFERENCE_LAST_CYCLE_NAME,
            "시장별 가장 최근 성공 완료된 추론 사이클의 UTC epoch 초",
            ["market"],
            registry=target_registry,
            multiprocess_mode="max",
        )

    def record_skip(self, *, market: str, horizon: int, reason: SkipReason | str) -> None:
        """REQ-AIF-130 (a): 스킵 1건을 사유 레이블로 구분해 기록한다.

        `reason`은 `SkipReason` 멤버 또는 그 값과 동일한 문자열만 허용한다 —
        오타 레이블이 조용히 새 시계열을 만들어 vmalert 규칙(REQ-AIF-142)을
        무력화하는 것을 막기 위해 어휘 밖 값은 `ValueError`로 거부한다.
        """
        reason_label = str(reason)
        if reason_label not in _VALID_SKIP_REASONS:
            raise ValueError(
                f"알 수 없는 스킵 사유 레이블: {reason_label!r} "
                f"(허용: {sorted(_VALID_SKIP_REASONS)})"
            )
        self.inference_skip_total.labels(
            market=market, horizon=str(horizon), reason=reason_label
        ).inc()

    def record_signal(self, *, market: str, horizon: int, signal_class: str) -> None:
        """REQ-AIF-130 (b): 신호 1건을 등급 레이블로 구분해 기록한다."""
        self.inference_signals_total.labels(
            market=market, horizon=str(horizon), signal_class=signal_class
        ).inc()

    def observe_cycle_duration(self, *, market: str, seconds: float) -> None:
        """REQ-AIF-130 (c): 시장 1회 추론 사이클 소요 시간을 관측한다."""
        self.inference_cycle_duration.labels(market=market).observe(seconds)

    def record_last_cycle(self, *, market: str, epoch_seconds: float) -> None:
        """REQ-DMR-001: 시장별 마지막 성공 완료 사이클의 UTC epoch 초를
        게이지에 기록한다. 호출부(`inference/last_cycle.py`)가 write(자식
        프로세스 사이클 완료)와 warm-start(부모 프로세스 부팅) 양쪽에서
        이 메서드를 공유한다 — 값의 출처만 다를 뿐 기록 방식은 동일하다."""
        self.inference_last_cycle.labels(market=market).set(epoch_seconds)

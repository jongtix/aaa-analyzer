# aaa-analyzer

AAA(Algorithmic Alpha Advisor) Phase 2 ML 분석 서비스. 시장 데이터 기반 피처 계산·레이블링·학습·추론을 담당하는 Python/FastAPI 서비스.

> 본 레포의 현재 상태는 SPEC-ANALYZER-FOUNDATION-001(스캐폴딩 + 프로세스 골격 + CI/CD 기반) +
> SPEC-ANALYZER-SCHEMA-001(DB 스키마 정합) + SPEC-ANALYZER-DATA-001(DB 읽기 계층 + SPLIT/DIVIDEND
> 가격 조정 엔진) + SPEC-ANALYZER-FEATURE-001(기술적 지표 25종 + 수급 피처 15종 + 가격파생/동결 분류
> 레지스트리) + SPEC-ANALYZER-LABEL-001(실현 수익률 기반 연속 레이블 생성 — T+H as-of 가격 조정,
> 거래정지·상장폐지 NaN 처리, purge gap) + SPEC-ANALYZER-TRAIN-001(FEATURE-001/LABEL-001 산출물로
> 데이터셋을 조립하고 purged expanding-window walk-forward 검증을 수행, LightGBM+XGBoost ×
> 시장 2종 × 예측기간 2종의 pooled 모델 16개(quantile 보조 모델 8개 포함) + Optuna 하이퍼파라미터
> 튜닝을 학습해 앙상블 점수·신뢰도를 산출하고 SHA-256 무결성 검증과 2단계 보존 정책으로 모델을
> 영속화하는 학습 코어) + SPEC-ANALYZER-TRAIN-AUTOMATION-001(WoL 매직패킷 기동 + SSH 원격
> 디스패치 + MySQL 터널 + APScheduler cron 스케줄링(주간/월간 학습 + 일일 모델 정체 감지) +
> Prometheus 계측 + 통합 실패 처리 경로로 TRAIN-001 학습 스크립트를 원격 자동화) 범위로 한정된다.
> 추론 로직은 후속 SPEC(INFER-001 등)에서 구현된다.

## Stack

- Python 3.14 (표준 GIL 빌드, free-threaded 아님)
- FastAPI + uvicorn
- uv (패키지/가상환경 관리)
- APScheduler 3.x, SQLAlchemy(동기) + PyMySQL, prometheus-client

## Package Layout

```
src/analyzer/
├── common/         # 구조화 로깅, Trace ID 등 공통 유틸
├── data/           # DB 읽기 계층 + SPLIT/DIVIDEND 가격 조정 엔진(as-of point-in-time)
├── features/       # 기술적 지표(KBAR+ROC/MA/STD/RANK/CORR) + 수급 피처 + PRICE_DERIVED/FROZEN 분류 레지스트리
├── labels/         # 학습 레이블(타깃) 생성 — T+H as-of 가격 조정 기반 실현 수익률, 거래정지/가용범위 NaN 처리, purge gap
├── training/        # 모델 학습 코어 — 데이터셋 조립, walk-forward 검증, LightGBM+XGBoost 학습, Optuna 튜닝, 앙상블 스코어링, SHA-256 무결성 검증 모델 영속화(진입점: python -m analyzer.training.train)
├── inference/       # 완결형 자식 CLI 추론 진입점
├── orchestration/   # WoL 매직패킷 송신 + SSH 원격 디스패치(MySQL 터널) + SchedulerRegistry cron 확장 + 모델 정체(staleness) 감지 + Prometheus 계측 + 통합 실패 처리 경로(진입점: analyzer.orchestration.runner)
└── api/             # FastAPI 부모 프로세스 (/health, /metrics)
```

## Setup

```bash
uv sync
```

## Run

```bash
# 부모 프로세스 (FastAPI: /health, /metrics + 로거·컨슈머/스케줄러 배선)
uv run python -m analyzer.api.main

# 자식 CLI (완결형 추론 진입점 골격)
uv run python -m analyzer.inference --market domestic
```

## Test

```bash
uv run pytest                       # 전체 (통합 포함)
uv run pytest -m "not integration"  # 단위만 (pre-push와 동일)
uv run pytest -m integration        # 통합만
```

`@pytest.mark.integration`은 DB/Redis 등 외부 의존성이 필요한 테스트에 사용한다. 미등록 마커는
`--strict-markers`에 의해 차단된다.

## Quality Gates

```bash
uv run ruff check
uv run ruff format --check
uv run pyright
```

Git 훅 설치:

```bash
./scripts/install-hooks.sh
```

## Time Zone / Scheduling

모든 타임스탬프는 KST(Asia/Seoul) 기준이다. APScheduler 잡은 cron 트리거만 사용한다(TECHSPEC 공통 규칙).

## Docker

```bash
docker build -t aaa-analyzer:test .
```

비루트 UID 1005로 실행(collector UID 1004와 비충돌). read-only 파일시스템·`cap_drop: [ALL]` 호환.

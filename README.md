# aaa-analyzer

AAA(Algorithmic Alpha Advisor) Phase 2 ML 분석 서비스. 시장 데이터 기반 피처 계산·레이블링·학습·추론을 담당하는 Python/FastAPI 서비스.

> 본 레포의 현재 상태는 SPEC-ANALYZER-FOUNDATION-001(스캐폴딩 + 프로세스 골격 + CI/CD 기반) 범위로 한정된다.
> 실제 추론/피처/레이블/학습 로직은 후속 SPEC(DATA-001/FEATURE-001/LABEL-001/INFER-001/TRAIN-001 등)에서 구현된다.

## Stack

- Python 3.14 (표준 GIL 빌드, free-threaded 아님)
- FastAPI + uvicorn
- uv (패키지/가상환경 관리)
- APScheduler 3.x, SQLAlchemy(동기) + PyMySQL, prometheus-client

## Package Layout

```
src/analyzer/
├── common/         # 구조화 로깅, Trace ID 등 공통 유틸
├── data/           # 데이터 접근 계층 (후속 SPEC)
├── features/       # 피처 계산 (후속 SPEC)
├── labels/         # 레이블 생성 (후속 SPEC)
├── training/        # 모델 학습 (후속 SPEC)
├── inference/       # 완결형 자식 CLI 추론 진입점
├── orchestration/   # 스트림 컨슈머·스케줄러 자리(모듈 경계만)
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

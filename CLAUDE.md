# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Service Overview

AAA(Algorithmic Alpha Advisor) Phase 2 ML 분석 서비스. 시장 데이터 기반 피처 계산·레이블링·학습·추론을
담당한다. 최종 목표(후속 SPEC 소관): `stream:daily:complete` 구독 → 시장별 모델 Lazy Load → A·B등급
배치 추론 → `trading_signals` INSERT + `stream:signal:*` 발행.

현재 레포 상태는 SPEC-ANALYZER-FOUNDATION-001(스캐폴딩 + 프로세스 골격 + CI/CD 기반) 범위로
한정된다. 실제 추론/피처/레이블/학습 로직은 없다.

## Tech Stack

- Python 3.14 (표준 GIL 빌드, free-threaded(3.14t) 아님 — lightgbm/xgboost cp314t 휠 미제공)
- FastAPI + uvicorn, uv(패키지/가상환경 관리)
- APScheduler 3.x(cron 트리거만), 동기 SQLAlchemy + PyMySQL, prometheus-client

## Process Model

- **상주 부모**: 단일 asyncio 프로세스. FastAPI(`/health`, `/metrics`) + 스트림 컨슈머/APScheduler
  **자리**(`orchestration/consumer.py`·`orchestration/scheduler.py` — 실제 구독/잡 로직은
  후속 SPEC INFER-001/TRAIN-001 소관).
- **완결형 자식**: `python -m analyzer.inference --market <market>` — 인자 파싱 후 exit 0(현재는
  predict 로직 없음). 자식 프로세스 종료로 OS 메모리 확정 반환(500MB 예산 구조적 보장) + 장애 격리.

## Build & Run

```bash
uv sync

# 부모 프로세스
uv run python -m analyzer.api.main

# 자식 CLI
uv run python -m analyzer.inference --market domestic
```

## Test Marker Convention (SPEC-ANALYZER-FOUNDATION-001)

DB/Redis 등 외부 의존성이 필요한 통합 테스트는 도입하는 첫 커밋부터 `@pytest.mark.integration`으로
표시한다(collector TESTLAYER-001 사후 치료를 예방하는 선제 컨벤션).

```bash
uv run pytest                       # 전체 (통합 포함) — CI(release.yml)와 동일
uv run pytest -m "not integration"  # 단위만 — pre-push와 동일
uv run pytest -m integration        # 통합만 (현재 0개 — 정상. 마커 필터가 0개 선택 시
                                     # pytest는 exit code 5(NO_TESTS_COLLECTED)를 반환한다.
                                     # 이는 실패가 아니라 pytest 표준 관례다.)
```

- `--strict-markers`가 미등록 마커 사용을 collection 단계에서 차단한다.
- pre-push는 `-m "not integration"`만 실행(컨테이너 기동 없음, < 90초 목표). CI(release.yml)는
  전체 실행 + 85% 커버리지 게이트(`--cov-fail-under=85`).

## Time Zone / Scheduling

모든 타임스탬프는 KST(Asia/Seoul) 기준(`zoneinfo.ZoneInfo`). `@Scheduled`에 대응하는 Python
등가물인 APScheduler 잡은 **cron 트리거만** 사용한다(TECHSPEC 공통 규칙 — collector의
`fixedDelay` 금지 규칙과 동일 취지, Redis Streams 이벤트 구독은 이 규칙 적용 대상 아님).

## Git Hooks

```bash
./scripts/install-hooks.sh   # core.hooksPath=scripts 설정
```

- `pre-commit`: `ruff check` + `ruff format --check`.
- `pre-push`: `pyright` + `pytest -m "not integration"`(단위 전용, 컨테이너 기동 없음).
- 4층(CI, `release.yml`)에서 전체 게이트(lock 무결성 + ruff + pyright + pytest 전체 + 85% 커버리지)를
  최종 재확인한다.

## DB / Migration (ADR-016)

analyzer는 DDL이 없다. `trading_signals` 스키마 마이그레이션은 collector 레포 Flyway가 소유한다.
본 레포는 DB에 접속하지 않는 골격 단계이며, 실제 DB 접속은 INFER-001(추론)·TRAIN-001(학습)에서
도입된다.

## Docker (ADR-032)

`python:3.14-slim`(glibc, ADR-012 alpine 원칙의 명시적 예외 — ML 휠 manylinux 호환 목적) digest
핀. 비루트 UID **1005**(collector=1004와 비충돌). read-only fs·`cap_drop: [ALL]` 호환(쓰기 경로
미가정, stdout 전용 로깅). HEALTHCHECK는 curl/wget이 없는 slim 이미지 특성상 Python
`urllib.request` 기반 프로브를 사용한다.

## CI/CD

`main` 푸시 → Release(python-semantic-release, `pyproject.toml [project.version]` 네이티브
갱신) → Docker(GHCR 3-tag 빌드/푸시) → Deploy(NAS self-hosted, `pull → up -d --wait analyzer`).
analyzer는 DDL이 없으므로(ADR-016) collector의 마이그레이션 체크·조건부 롤백 분기가 없다 — 실패
시 무조건 이전 digest로 롤백 + Telegram 알림.

## Project Documents

- 프로젝트 전체 문서: `aaa-infra/docs/` — 상위 `aaa/CLAUDE.md` 참고
- SPEC 문서: `aaa/.moai/specs/SPEC-ANALYZER-FOUNDATION-001/`

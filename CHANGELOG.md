# CHANGELOG


## v0.1.0 (2026-07-04)

### Other

- 🎉 init: empty initial commit
  ([`6e7c7e4`](https://github.com/jongtix/aaa-analyzer/commit/6e7c7e4a2cc5ba941196a7e60b88fb4e1cc97604))

Co-Authored-By: Claude Code <noreply@anthropic.com>

- 👷 ci(analyzer): CI/CD 4층 게이트 + Release→Docker→Deploy 파이프라인
  ([`edfbe6a`](https://github.com/jongtix/aaa-analyzer/commit/edfbe6a259f117e167f8589596d0a4bcac8ef85a))

release.yml: test 잡(uv sync --locked → ruff check/format → pyright → pytest 전체+커버리지 85% 게이트) →
  release 잡(needs:test, python-semantic-release, pyproject.toml [project.version] 네이티브 갱신, PyPI
  미배포). docker.yml: Release workflow_run 성공 시 GHCR 3-tag(v/latest/sha) 빌드·푸시 (linux/amd64,
  collector와 동일 패턴 이식). deploy.yml: Docker workflow_run 성공 시 NAS self-hosted pull→up -d --wait
  analyzer. B4 확정 반영 — analyzer는 DDL 없음(ADR-016)이므로 마이그레이션 체크 분기 제거, 실패 시 무조건 롤백+Telegram 알림.
  dependabot.yml: uv 에코시스템 weekly.

commit_parser="emoji" 내장 파서를 uvx 에페메럴 실행으로 실제 레포 커밋 이력에 dry-run 검증: ✨ feat→minor, 🔧 chore→no_release
  확인 (collector .releaserc.js 매핑과 일치, B3/R2 잔여 확인 항목 해소).

SPEC: SPEC-ANALYZER-FOUNDATION-001

- 📝 docs(analyzer): 서비스 CLAUDE.md + 마커 컨벤션 스모크 테스트
  ([`e55bd5b`](https://github.com/jongtix/aaa-analyzer/commit/e55bd5b2d952eee1f3a83c85c48ece5b04567e67))

CLAUDE.md: 서비스 개요, 프로세스 모델, 빌드/실행, 테스트 마커 컨벤션, KST/APScheduler cron 규칙, Docker/CI-CD 요약.

tests/test_pytest_marker_convention.py: `-m integration` 0개 선택 성공 종료 (pytest 표준 관례상 exit code
  5=NO_TESTS_COLLECTED, 실패 아님), `-m "not integration"`은 단위 테스트 정상 실행, --strict-markers가 미등록 마커 사용을
  collection 단계에서 차단함을 검증.

디버깅 메모: 최초 구현은 스크래치 테스트 파일을 OS temp(tmp_path/tempfile)에 생성했으나, 이미 실행 중인 외부(outer) pytest 프로세스 안에서 레포
  밖 경로를 대상으로 재귀적으로 pytest를 기동하면 conftest 조상 탐색이 레포와 무관한 거대한 디렉토리 트리를 훑어 사실상 무한 대기가 발생함을 실측 확인. 스크래치
  파일을 레포 내부(.gitignore 처리)에 생성하도록 수정해 해결.

SPEC: SPEC-ANALYZER-FOUNDATION-001

- 🔧 chore(analyzer): Docker 하드닝 (ADR-032)
  ([`719651a`](https://github.com/jongtix/aaa-analyzer/commit/719651a558c95a7d0e4947de3272bb54a9ebc2b3))

python:3.14-slim(3.14.6-slim-trixie) linux/amd64 digest 핀. multi-stage 빌드(uv sync --locked
  --no-dev). 비루트 UID 1005(collector=1004와 비충돌, Debian groupadd/useradd). read-only/cap_drop 호환(쓰기 경로
  미가정, stdout 전용 로깅). HEALTHCHECK는 curl/wget 없는 slim 이미지 특성상 urllib 기반 Python 프로브 사용. .dockerignore로
  빌드 컨텍스트 최소화.

docker build -t aaa-analyzer:test . 성공 확인(로컬 arm64 호스트에서 amd64 digest 강제 사용 — QEMU 에뮬레이션 경고는 예상된
  동작).

SPEC: SPEC-ANALYZER-FOUNDATION-001

- 🔧 chore(analyzer): uv 프로젝트 스캐폴딩 + 8개 서브패키지 골격
  ([`56ff1ef`](https://github.com/jongtix/aaa-analyzer/commit/56ff1ef07659903c10376ff8b101d58c0a0ea336))

Python 3.14 표준 빌드(free-threaded 아님) 대상 pyproject.toml 초기화.
  common/data/features/labels/training/inference/orchestration/api 8개 서브패키지 디렉토리 생성. 런타임
  의존성(fastapi/uvicorn/pymysql/ sqlalchemy/apscheduler 3.x/prometheus-client)+개발 도구(ruff/pyright/
  pytest/pytest-cov) uv.lock 고정. asyncmy·APScheduler 4.x 미포함 확인. pytest integration 마커 +
  --strict-markers + --cov-fail-under=85 배선.

SPEC: SPEC-ANALYZER-FOUNDATION-001

- 🔧 chore(analyzer): 로컬 Git 훅 (pre-commit/pre-push, 2·3층)
  ([`96f3dbe`](https://github.com/jongtix/aaa-analyzer/commit/96f3dbeab29b1ac3748177aee9ac8dd48d833f89))

scripts/pre-commit: ruff check + ruff format --check. scripts/pre-push: pyright + pytest -m "not
  integration"(단위 전용, --no-cov), 컨테이너 기동 없음, 90초 watchdog. 실측 벽시계 ~1.6초(목표 <90초 크게 하회).
  scripts/install-hooks.sh: core.hooksPath=scripts 설정(pre-commit 프레임워크 도입 없이 단순 셸 훅 유지 — 과설계 회피).

SPEC: SPEC-ANALYZER-FOUNDATION-001

### ✨

- ✨ feat(analyzer): FastAPI 부모 프로세스 /health·/metrics + orchestration 자리
  ([`2e0bf43`](https://github.com/jongtix/aaa-analyzer/commit/2e0bf4354f89ef288a5e20be7e7b4811e0100d50))

api/app.py: FastAPI 앱 팩토리, GET /health({"status":"ok"}), GET /metrics (prometheus_client
  generate_latest + CONTENT_TYPE_LATEST). api/main.py: asyncio 엔트리포인트 — orchestration 자리 배선 후
  uvicorn 서빙. orchestration/consumer.py: StreamConsumer 구조적 자리(구독 로직 없음, INFER-001 소관).
  orchestration/scheduler.py: SchedulerRegistry 빈 등록부(잡 등록 로직 없음).

dev 의존성에 httpx 추가(starlette TestClient 구동에 필요, 런타임 미포함).

RED-GREEN: TestClient 기반 스펙 테스트 선작성 후 최소 구현.

SPEC: SPEC-ANALYZER-FOUNDATION-001

- ✨ feat(analyzer): 구조화 JSON 로깅 + Trace ID 유틸
  ([`6d8248a`](https://github.com/jongtix/aaa-analyzer/commit/6d8248ac1b7bfb62fb2b6d42fd9f7aa9e6bf9ed9))

contextvars 기반 Trace ID 발급/조회/명시적 설정/복원(new_trace_id/ get_trace_id/set_trace_id/reset_trace_id).
  JSON 포매터가 KST(Asia/Seoul) 타임스탬프와 활성 Trace ID를 로그 레코드에 자동 주입. get_logger는 이름별로 1회만 핸들러를
  구성(idempotent).

RED-GREEN: 11개 스펙 테스트 선작성 후 최소 구현.

SPEC: SPEC-ANALYZER-FOUNDATION-001

- ✨ feat(analyzer): 완결형 자식 CLI 진입점 골격 (inference)
  ([`e5367ef`](https://github.com/jongtix/aaa-analyzer/commit/e5367ef627f98915465f8344918b65693035ab2b))

python -m analyzer.inference --market <market>: argparse로 --market 필수 인자 파싱 후 로그 1줄 남기고 exit 0.
  predict/모델 로드 없음(INFER-001 소관). --market 누락 시 argparse 기본 동작으로 non-zero 종료.

RED-GREEN: parse_args/main 단위 테스트 + subprocess 실제 모듈 호출 테스트 선작성 후 최소 구현.

SPEC: SPEC-ANALYZER-FOUNDATION-001

### 🐛

- 🐛 fix(analyzer): python-semantic-release build_command 타입 오류 수정
  ([`e84fe80`](https://github.com/jongtix/aaa-analyzer/commit/e84fe8092c5f781767ac60c99c55116bb86fc8e1))

[tool.semantic_release] build_command = false(불리언)가 pydantic RawConfig 스키마상 문자열 타입이 아니라 CI Release
  잡이 즉시 실패했다 (PyPI 배포 없음 → 빌드 스텝 불필요, 실제 CI 실행 후 발견). build_command = ""(빈 문자열, 스텝 없음)으로 수정. uvx
  semantic-release version --print 로컬 검증: 1.0.0 정상 계산 확인.

SPEC: SPEC-ANALYZER-FOUNDATION-001

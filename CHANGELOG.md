# CHANGELOG


## Unreleased

### ✨

- 일일 모델 정체 감지 cron 활성화 (SPEC-ANALYZER-TRAIN-STALENESS-001 M1/M3/M4, REQ-ATD-*)
  - 신규 필수 환경변수 `TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT` — 컨테이너 내부 활성 모델 마운트 경로. `AutomationConfig.container_models_root` 필드 + `_REQUIRED_ENV_VARS` 추가로 기존 기동 fail-fast 경로에 편입(REQ-ATD-003). NAS 호스트 측 `:ro` 마운트와 권한은 aaa-infra 몫
  - `api/main.py`에 `daily-staleness-check` 잡 개별 등록 — `detect_stale_models()` 호출 → 성공 시 `TrainingMetrics.record_staleness_batch()`, 실패 시 `record_failure(stage="staleness_scan")` 후 재발생하는 콜백 클로저 배선. GATE-001이 확립한 프로세스 싱글턴 `TrainingMetrics`를 재사용하며 콜백 내부에서 재생성하지 않는다. 등록 잡은 주간+일일 정확히 2건(월간은 여전히 미등록) (REQ-ATD-005/007/010)
  - `record_staleness_batch()` 신설 — 스캔마다 `aaa_analyzer_model_stale` 게이지 패밀리를 clear한 뒤 이번 결과만 재기록해, 삭제된 (market, horizon) 조합의 값이 영구 잔존하지 않게 한다. 기존 `record_staleness()` 시그니처는 무수정(REQ-ATD-009)
  - `daily_staleness_check_trigger()` 발화 시각 07:00 → **04:00 KST** 변경 — 월간 예정 시각·주간 실행창과의 3중 충돌 회피. 잡 ID `daily-staleness-check`는 무수정(REQ-ATD-006)
  - `_MODEL_FILENAME_PATTERN` 확장자 그룹을 `\.\w+`에서 `training/persistence.py`의 `_NATIVE_EXTENSION` 기반 동적 allowlist로 강화 — `.meta.json` 등 사이드카 파일을 명시적으로 배제한다. 확장자 집합의 단일 소스를 유지해 신규 알고리즘 도입 시 이중 갱신이 필요 없다(REQ-ATD-008)
  - 정체 판정 로직(`detect_stale_models()` 본체)과 임계값(기본 28일)은 무수정 — 이 SPEC은 배선 SPEC이다
  - 프로덕션 활성화는 별도 운영자 작업 대기 중: NAS `.env.analyzer`에 신규 환경변수 기입 → `docker-compose.yml` 적용 → `init-nas.sh` 재실행 **후에만** 신규 이미지 배포(순서 위반 시 `MissingConfigError` 크래시루프)
- CI/CD 룰셋 강화 (SPEC-INFRA-CICD-002)
  - `main` 브랜치 룰셋(`main-protection`) 신설 — 선형 히스토리 강제, 강제 푸시/삭제 차단, `status-check` 상태 체크 필수
  - `release.yml`의 test job에 `pull_request` 트리거 추가 — PR에서 머지 전 실제 CI 검증
  - GitHub App(`aaa-ci-release-bot`)이 `actions/create-github-app-token`으로 보호된 `main`을 우회해 릴리스 태그/커밋을 푸시(룰셋 `bypass_actors`에 유일하게 등재), 사람은 PR 경로만 허용
  - `docker.yml` 트리거를 `workflow_run: ["Release"]`에서 `push: tags: ['v*']`로 변경 — `workflow_run` 3단 체인(GitHub 문서상 깊이 제한)을 2단으로 축소, App이 푸시한 태그로도 안정적으로 빌드 발화. 중복 태그 탐색용 2중 체크아웃 로직 제거
  - `deploy.yml`/`release.yml`에 `concurrency` 그룹 추가 — 배포/릴리스 중복 실행 방지
  - `dependabot-auto-merge.yml` 신규 — non-major Dependabot PR을 CI 통과 후 자동 머지(`dependabot/fetch-metadata` + `gh pr merge --auto --rebase`), `dependabot.yml`에 3일 쿨다운 추가
  - `tag-protection` 룰셋 신설(`refs/tags/v*`) — 릴리스 태그 삭제·재태그 차단
  - 체크아웃 스텝에 `persist-credentials: false` 추가(푸시가 필요 없는 스텝 한정)
  - 릴리스 커밋백(commit-back) 메커니즘 제거 — `pyproject.toml`의 python-semantic-release 설정을 `commit: false, push: true`로 변경(더 이상 봇이 `pyproject.toml`/`uv.lock`을 재작성하지 않음). 버전은 Docker 빌드 시점에 `uv version --no-sync "${VERSION}"`으로 주입. `pyproject.toml`의 정적 버전 필드는 이제 비활성 placeholder(코드에서 미참조)

### 🐛

- fix(ci): `deploy.yml`의 `workflow_run.head_branch == 'main'` 게이트가 태그 트리거 Docker 실행 시 `head_branch`가 태그명으로 보고되는 것을 놓쳐 M5 적용 후 모든 릴리스에서 Deploy가 조용히 스킵되던 결함 수정 — `startsWith(github.event.workflow_run.head_branch, 'v')` 조건으로 교체. v0.14.1 배포로 라이브 검증 완료


## v0.13.1 (2026-08-25)

### 🐛

- 🐛 fix(ci): 배포 롤백의 컨테이너명 불일치로 자동 롤백 무력화 수정
  ([`eae6d6f`](https://github.com/jongtix/aaa-analyzer/commit/eae6d6f))
- 🐛 fix(docker): 런타임 이미지에 libgomp1 설치 — lightgbm import 크래시 해결
  ([`d937cb1`](https://github.com/jongtix/aaa-analyzer/commit/d937cb1))

v0.13.0 배포(NAS)가 `container aaa-analyzer is unhealthy`로 실패하며 크래시루프 상태로 노출됐다. 두 결함을
같은 배포 시도에서 함께 발견했다: (1) lightgbm의 Linux wheel이 `libgomp.so.1`(GNU OpenMP)에 동적 링크돼
있으나 wheel 안에 번들하지 않는 알려진 upstream 제약([microsoft/LightGBM#4484](https://github.com/microsoft/LightGBM/issues/4484)) —
이 SPEC의 M5에서 처음으로 `main.py` 기동 경로가 `gate_adapter→promotion_gate→lightgbm`을 즉시 import하게
되며 실전에 노출됐다(xgboost는 자체 libgomp를 정적 번들해 지금까지 문제가 드러나지 않았음). (2)
`deploy.yml`의 "Save current image digest" 단계가 컨테이너명을 오탈자(`analyzer`, 실제는 `aaa-analyzer`)로
조회해 `prev_digest`가 항상 빈 문자열이 되어 자동 롤백 조건이 상시 거짓이었다 — 이번이 실전에서 처음 걸린
경로였다. 긴급 수동 SSH 롤백(v0.12.2)으로 서비스를 복구한 뒤 두 결함을 근본 수정하고 재배포해 정상화를
확인했다(`docker inspect` healthy, `"orchestration wired (jobs=1)"` 로그, `/metrics` 노출, VM 스크랩 `up`).

SPEC: SPEC-ANALYZER-TRAIN-GATE-001


## v0.13.0 (2026-08-25)

### ✨

- ✨ feat(SPEC-ANALYZER-TRAIN-GATE-001): 주간 챌린저 게이트 배선 + cron 활성화 — 맥 원격 게이트 실행 + 관측 경로 개통
  ([`138fd95`](https://github.com/jongtix/aaa-analyzer/commit/138fd95)..
  [`58f1ae6`](https://github.com/jongtix/aaa-analyzer/commit/58f1ae6))

TRAIN-EVAL-001이 도입한 오프라인 챔피언/챌린저 게이트를 실제 주간 자동 실행 경로에 배선한다 — M1
`gate.py` 순수함수부(챔피언 경로 해석 + 동결 파라미터 리더 + verdict 직렬화), M2 게이트 CLI 본체
(`run_gate` + `main`), M3 주간 학습 CLI 동결 파라미터 주입, M4 NAS 측 게이트 어댑터 + `E-1` + `manual_run`
확장, M5 `main.py` 기동 배선 + 스케줄러 안전장치를 순서대로 구현한다.

배선 완료 후 발견된 5건(2 Critical + 3 High)의 리뷰 지적 — `params_from_active_meta` 프로덕션 배선 누락,
게이트 실패가 구조화 로그 경로를 우회하는 결함, `run_gate()` 조합별 예외 격리 부재, `TrainingMetrics`
싱글턴 반복 발화, `record_success` 미발행 검증 부재 — 를 같은 릴리즈에서 수정한다.

신규 MySQL 스키마·env var·마이그레이션 없음. 신규 파일: `orchestration/gate.py`,
`orchestration/gate_cli.py`, `orchestration/gate_adapter.py`.

SPEC: SPEC-ANALYZER-TRAIN-GATE-001


## v0.12.0 (2026-08-18)

### ✨

- ✨ feat(SPEC-ANALYZER-TRAIN-EVAL-001): 역사적 Walk-Forward 캠페인 평가 + 안정화 게이트 + 상시 챔피언/챌린저 게이트 도입
  ([`f6823a9`](https://github.com/jongtix/aaa-analyzer/commit/f6823a99a02ecaf224eee78b592af33076b92716)..
  [`61f5c7d`](https://github.com/jongtix/aaa-analyzer/commit/61f5c7d50ea0ba81eaa4aa69085e2bf18974668b))

배포 전 다중 폴드 표본외 성능 검증과 배포 후 상시 게이트를 도입한다 — 프로덕션 동작 자체는 변경하지 않으며,
`SPEC-ANALYZER-TRAIN-001`이 구현했지만 호출자가 없던(orphaned) `split.py`/`backtest.py`/`ensemble.py`의
순수 함수를 실제로 배선한 오프라인 검증/배포 계층이다.

- `data_as_of` 상한이 실제 학습 데이터 조회를 제한하지 않던 결함과 분위수 보조 모델 8개의 파일명 충돌 결함을
  수정한다(REQ-ATE-001~010).
- 국내 2005-01-01/해외 2007-08-20부터 현재까지 주간(weekly) 표본외 윈도우로 확장 폴드를 구성해 시장×horizon×
  algorithm 8개 포인트 조합 각각의 표본외 성능(7개 백테스트 지표)을 측정하는 역사적 Walk-Forward 캠페인을
  신설한다(신규 CLI `python -m analyzer.training.campaign`, cron 미등록, REQ-ATE-011~038).
- 폴드 지표 시계열에 롤링 집계 기반 기계적 안정화 게이트(GATE-1/2/3)를 적용하고, 통과한 조합에 대해 LightGBM/
  XGBoost/앙상블 스코어링 전략 중 챔피언을 선정한다(REQ-ATE-039~047).
- 증거 기반 1차 배포(활성화 매니페스트 + 롤백 가능 스킴)와, 이후 주간 재학습을 챌린저로 취급하는 오프라인
  챔피언/챌린저 상시 게이트를 도입한다. `record_success()`를 조합 단위로 수정하고 Prometheus 모델 품질
  Rank IC 게이지를 추가한다(REQ-ATE-048~076).

신규 MySQL 스키마·env var·마이그레이션 없음. Breaking API change 없음(`TrainingPipelineResult`는 additive
확장만). 신규 파일: `training/panel_folds.py`, `training/campaign.py`, `training/campaign_metrics.py`,
`training/stabilization.py`, `orchestration/activation.py`, `orchestration/promotion_gate.py`.

SPEC: SPEC-ANALYZER-TRAIN-EVAL-001


## v0.11.0 (2026-08-16)

### ✨

- ✨ feat(SPEC-ANALYZER-TRAIN-OBSV-001): M1 SSH 채널 드레인 루프 + 자체 타임아웃 재설계
  ([`2756ccb`](https://github.com/jongtix/aaa-analyzer/commit/2756ccb28bd86c205211abcd79c46c38fe81356f))

exec_command()를 폴링 드레인 루프로 전면 재구현해 SSH 채널 버퍼 포화로 인한 원격 프로세스 write() 블로킹(15시간 데드락
  실측)을 구조적으로 방지한다(REQ-ATO-001/003, 보편 적용). recv_exit_status()가 settimeout() 값을 준수하지 않는 실측
  근본원인에 대응해, 읽기 루프 자체가 time.monotonic() 데드라인을 추적해 타임아웃을 강제한다(REQ-ATO-009/011). 완료
  판정은 여전히 exit_status_ready() 종료코드 획득 경로 단독이며, 스트림 EOF만으로는 완료를 추론하지 않는다(REQ-ATO-026).

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

- ✨ feat(SPEC-ANALYZER-TRAIN-OBSV-001): M2 트레이너 파일 sink + 저볼륨 릴레이 배선
  ([`2f7e33c`](https://github.com/jongtix/aaa-analyzer/commit/2f7e33c7d7d1d70276b7615755397e5689205a25))

AutomationConfig에 trainer_log_base_dir 필드(필수 env var, 기본값 없음)를 추가한다(REQ-ATO-005). 원격 디스패치 명령에
  trainer_<run_id>.log tee 리다이렉션을 배선해 원격 학습 CLI stdout/stderr 전체를 원문 영속 기록하며, 마운트 확인
  게이트 이후에만 시작된다(REQ-ATO-004/008). NAS 측 릴레이는 stage_marker:true JSON 필드 기반 저볼륨 요약만 전달해
  트레이너 파일과의 이중 적재를 방지한다(REQ-ATO-002/007).

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

- ✨ feat(SPEC-ANALYZER-TRAIN-OBSV-001): M3 run_id → trace_id 전파
  ([`afcdaa5`](https://github.com/jongtix/aaa-analyzer/commit/afcdaa5411292610b8e49338c72fa1f7b0c7e254))

NAS 오케스트레이터가 발급한 run_id를 TRAIN_RUN_ID env var로 원격 학습 CLI에 전달하고(shlex.quote 이스케이프,
  REQ-ATO-012/014), train.py main()이 기존 trace_id 유틸리티(set_trace_id())로 즉시 설정한다(REQ-ATO-013) — NAS
  오케스트레이션 로그와 원격 trainer 로그를 동일 trace_id로 상관 조회 가능하게 한다. env var 부재 시 fail-open.

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

- ✨ feat(SPEC-ANALYZER-TRAIN-OBSV-001): M4 파이프라인 진행 로그 + 배치 집계 + traceback 보존
  ([`bef9502`](https://github.com/jongtix/aaa-analyzer/commit/bef95029028866e56044c2b648ae40ac4d435b76))

학습 파이프라인에 시장별 시작·유니버스 크기·데이터셋 캐시 히트/미스·조립 완료 행수·horizon별 유효 레이블 행수·16개
  모델 조합 학습 시작/완료+저장 경로 단계 전이 로그를 추가한다(REQ-ATO-018/019). 종목별 조회·데이터셋 조립 루프는
  25종목마다 1회, 배당 스킵 경고는 25건마다 1회 집계 로그로 전환한다(REQ-ATO-015/016/017). 실패 시 전체 traceback을
  로그로 남기되 TrainingPipelineResult.error 반환 타입(문자열)은 그대로 유지한다(REQ-ATO-020). 오케스트레이터에
  WoL·SSH 연결·디스패치 시작·원격 종료코드·프로모션 결과 단계 전이 로그를 추가한다(REQ-ATO-021).

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

- ✨ feat(SPEC-ANALYZER-TRAIN-OBSV-001): M5 프레임워크 로거 라우팅 + 실패 처리 경로 로거 교체
  ([`c507465`](https://github.com/jongtix/aaa-analyzer/commit/c50746517998a46a4577d3b5df275f0f23b1a4d7))

orchestration/failure.py의 로거를 raw 표준 로거에서 기존 구조화 JSON 로거(analyzer.common.logging.get_logger)로
  교체해 평문 stderr 유출을 제거한다(REQ-ATO-022). LightGBM/XGBoost verbosity를 완전 무음에서 Error/Warning·warning
  레벨로 낮추고, LightGBM 네이티브 로그는 lgb.register_logger()로 analyzer 구조화 로거에 라우팅한다(REQ-ATO-023/024).
  XGBoost는 공식 로거 라우팅 API가 없어 M2의 원격 셸 리다이렉션(tee)을 통해 트레이너 파일로 stderr가 합류한다
  (REQ-ATO-025) — vector 파싱 보존 여부는 M7 라이브 검증에서 확인한다.

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

### ✅

- ✅ test(SPEC-ANALYZER-TRAIN-OBSV-001): M6 stderr 채널 드레인 커버리지 보강
  ([`e6dfa2a`](https://github.com/jongtix/aaa-analyzer/commit/e6dfa2aa3405065a9e03402159c18d66fa1bdc9c))

exec_command()의 stderr 드레인 분기(recv_stderr_ready/recv_stderr)에 대한 명시적 회귀 테스트를 추가한다
  (REQ-ATO-001) — stdout뿐 아니라 stderr도 동일하게 소비·릴레이됨을 검증. M1~M6 통합 검증: pytest 438건 전부 통과,
  커버리지 97.01%, ruff/pyright 클린, aaa-infra 레포 무변경, REQ-ATO-001~030 30개 결번 없음 확인.

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

- ✅ test(SPEC-ANALYZER-TRAIN-OBSV-001): 채널 버퍼 포화 회귀 테스트가 실 구현을 검증하도록 재작성
  ([`7e9be7d`](https://github.com/jongtix/aaa-analyzer/commit/7e9be7df167c92eeacf5c762d4ba16871e06b70f))

TestChannelBufferSaturationPrevention이 자체 for 루프를 재구현한 페이크(_FakeBufferedChannelConnection)의 점유량
  로직만 검증해 읽기 루프 자체를 제거해도 실패하지 않던 결함(AC-ATO-001/002 회귀 가드 무력화, sync-auditor FAIL F1)을
  수정한다. paramiko 채널의 window 기반 흐름 제어를 모사하는 _FlowControlledChannel로 교체해
  ParamikoSshConnection.exec_command()(실 구현)를 직접 검증한다.

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

### 🐛

- 🐛 fix(analyzer): 원격 학습 로거를 analyzer 표준 구조화 로거로 교체
  ([`2ca1e4f`](https://github.com/jongtix/aaa-analyzer/commit/2ca1e4fc3a8100c8f0a1c543494133a33a84c070))

training/dataset.py와 data/dividend_adjustment.py가 raw stdlib logging.getLogger()를 사용해 프로덕션에서
  REQ-ATO-016 진행 로그가 핸들러 부재로 소실되고 배당 스킵 경고가 평문으로 새어나가던 결함(sync-auditor FAIL F2)을
  수정한다. analyzer.common.logging.get_logger()로 통일한다.

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

- 🐛 fix(analyzer): NAS 오케스트레이터 단계 전이 로그의 trace_id를 run_id로 설정
  ([`109df30`](https://github.com/jongtix/aaa-analyzer/commit/109df30b52a35ed553186f728855a31167d1b82b))

execute_scheduled_training_run()이 시작 시점에 set_trace_id(run_id)를 호출하지 않아 릴레이·단계 전이 로그의
  trace_id 필드가 run_id를 반영하지 못하던 AC-ATO-008 결함(sync-auditor FAIL F4)을 수정한다. 함수 종료 시 finally에서
  토큰을 복원해 값이 무관한 컨텍스트로 새어나가지 않게 한다.

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

- 🐛 fix(analyzer): 원격 디스패치 명령에 트레이너 로그 디렉터리 mkdir -p 추가
  ([`e30af93`](https://github.com/jongtix/aaa-analyzer/commit/e30af9307ca0636a507454a288160ffb629e0787))

build_remote_dispatch_command()의 tee 대상 트레이너 로그 디렉터리가 사전에 생성되지 않아 디렉터리 부재 시 tee 자체가
  실패할 수 있던 결함(sync-auditor FAIL F3)을 수정한다. 마운트 게이트 뒤·tee 앞에 mkdir -p를 추가한다.

SPEC: SPEC-ANALYZER-TRAIN-OBSV-001

M1~M6 unit 레벨 구현 + sync-auditor binding FAIL 4건(F1/F2/F4 critical, F3 major) 후속 수정을 포함해 main에 머지·
  v0.11.0으로 릴리스했다. AC-ATO-001~021 중 19건 PASS, 2건(AC-ATO-006 물리 저장소 동일성, AC-ATO-016 vector 파싱
  보존)은 네트워크 접근이 필요한 M7 라이브 검증으로 위임된 PASS-WITH-DEBT 상태다 — M7은 이 SPEC의 M1~M6 완료를
  블로킹하지 않는 후속 DoD 항목이며 아직 미착수다.

M7 라이브 검증 완료(2026-08-16): 실제 원격 학습 재실행(run_id=81c88e4be3734c1a94d0520a4252625b, 16개 모델
  전부 exit_code=0 + staging→active 승격)으로 AC-ATO-006(트레이너 파일 물리 저장소 동일성)·AC-ATO-016(XGBoost
  stderr vector 파싱 보존)을 라이브 확인해 PASS로 전환했다. AC-ATO-001~021 전건 PASS.

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

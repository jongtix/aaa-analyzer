# CHANGELOG


## Unreleased

### ✨

- 모델 영속화 경로 전면에 `.meta.json` 사이드카 배선 — standing-gate/주간 재학습/분위수 모델 커버리지 결함 수정 (SPEC-ANALYZER-TRAIN-META-001 M1-M7b/M6a/M6b, REQ-TM-001~011, [aaa-infra#163](https://github.com/jongtix/aaa-infra/issues/163))
  - **근본원인**: `.meta.json` 사이드카 WRITER가 캠페인 배포 경로(`campaign.py::activate_market_horizon_combo()`)에만 배선되어 있었고, 주간 재학습 저장 경로(`train.py::_persist_trained_models()`)에는 배선된 적이 없었다 — `promotion_gate.py::evaluate_and_promote()`는 모델 파일을 저장하지 않는 계층이라 수정 지점이 아니었다(추정 귀속 TRAIN-GATE-001/TRAIN-AUTOMATION-001은 방향은 맞으나 부정확했음이 research.md로 확인). 분위수 모델(q10/q90)은 **어떤 저장 경로에서도** `.meta.json`을 받은 적이 없는 별도의 구조적 공백이었다
  - M1: 축소 사이드카 스키마 확정, `campaign_metrics.write_sidecar_metadata()` 키워드 전용 인자로 재정의(REQ-TM-004/005/005b)
  - M2: `inference/features.py::has_supply_demand_gap()` 신설 — 단일종목 수급 데이터 결측 판별 함수(REQ-TM-007/008)
  - M3(**근본 수정 핵심**): `train.py::_persist_trained_models()`/`run_training_pipeline()`에 사이드카 기록 배선 — 주간 재학습 저장 경로의 포인트 모델(lightgbm/xgboost) 및 분위수 모델(q10/q90) 전부에 `.meta.json` 기록(REQ-TM-001/002)
  - M4: `campaign.py::train_and_persist_champion_quantiles()`에 사이드카 배선 — 캠페인 분위수 저장 경로도 커버(REQ-TM-003). 분위수 모델은 포인트 모델과 달리 폴드별 백테스트 평가를 거치지 않으므로 `aggregate_metrics` 등 포인트 모델 전용 필드를 포함하지 않는 축소 스키마 변형을 채택 — `feature_columns`는 동일 윈도 인자로 호출되는 `_split_features_and_labels()`의 결정론적 결과를 재사용(독립 재계산 아님)
  - M5/M7/M7b(REQ-TM-011, 실행 중 스코프 정정): `predict.py::predict_point_models()`/`predict_quantile_models()`에 옵트인 `investor_trend` 파라미터 추가, `assemble_inference_features_batch()`에 옵트인 `feature_columns` 키워드 인자 추가 — `has_supply_demand_gap()`(M2) 호출로 수급 데이터 결측 종목을 `SkipReason.FEATURE_INSUFFICIENT`로 정밀 분류. 이미 병합된 `SPEC-ANALYZER-PIPELINE-001`의 `pipeline.py::run_market_inference()` 호출부에 `feature_columns=_union_feature_columns(serving_plan)` 한 줄 추가(사용자 승인)로 실제 프로덕션 경로까지 실배선 완료 — SPEC-ANALYZER-PIPELINE-001의 REQ-APL-140(일반 예외 경계, `SkipReason.UNEXPECTED_ERROR`)은 이 결함의 증상을 봉쇄하는 안전망으로 남고, REQ-TM-011이 그 증상의 정밀 근본 분류를 프로덕션 경로 레벨에서 완성한다(양자는 구분되는 별개 조치)
  - M6a: `TestReqTm006CampaignPointModelSidecarSchemaLock` 신설 — 캠페인 포인트 모델 사이드카 payload 전체 키 집합(`==` 엄격 비교)과 각 값의 구조적 타입을 고정하는 REQ-TM-006 회귀 가드(코드 수정 없음, 순수 회귀 고정)
  - M6b(REQ-TM-009/AC-TM-009): 운영 백필 CLI `src/analyzer/training/backfill_meta_cli.py` 신규 — dry-run 기본, `--apply` 필요, 모델 바이너리 무수정, 멱등성. NAS 실측(`docker exec aaa-analyzer` + `activation_manifest.json` 대조)으로 대상 3개 파일(`overseas/60/xgboost` D60 챔피언, `overseas/20/lightgbm` D20 분위수 q10/q90) 확정, dry-run 성공(feature_columns=25개 확인) — **`--apply` 실제 적용은 이 SPEC의 PR 병합 + CI/CD NAS 배포 이후 별도 운영 절차로 수행**(현재 프로덕션에는 아직 M1의 시그니처 확장이 배포되지 않아 `--apply` 시도가 `TypeError`로 안전하게 실패함을 확인, 부작용 없음)
  - 검증: `uv run pytest -m "not integration"` 1011 passed / 커버리지 97.63%(게이트 85%), `ruff check`/`ruff format --check`/`pyright` 전부 clean, `grep -rn 'AskUserQuestion' src/analyzer/` 0건
  - **알려진 갭**: 60 수평선 lightgbm 분위수 모델(q10/q90)도 전체 trained_date에서 `.meta.json` 결손 상태임을 NAS 실측 중 추가 발견 — 이 SPEC의 REQ-TM-009/AC-TM-009 범위는 D20 분위수+D60 챔피언 3개 파일로 명시적으로 한정되며, 60 분위수는 M3/M4 코드 수정 이후의 신규 주간 재학습·캠페인 사이클에서 자연히 해소된다(급한 크래시 위험 아님, 스코프 확장 없이 다음 자연 주기에 맡김)
- 추론 서버 — 스트림 컨슈머 + 자식 프로세스 추론 + trading_signals/signal_price_bands 기록 (SPEC-ANALYZER-INFER-001 M1-M8, REQ-AIF-010~142)
  - M1: 프로세스 모델 + 부모↔자식 IPC 계약 + Redis Streams 컨슈머 배선(REQ-AIF-010/020/021) — `inference/{__main__,spawn,outcome,config,trade_date,lock,redis_client}.py` 신규, `orchestration/consumer.py`(`StreamConsumer.start()` 실장), `api/main.py` 기동 배선 확장, `pyproject.toml` `redis` 의존성 추가. 자식은 종료코드(0/1/2)만을 IPC 계약면으로 관찰, stdout은 로그 상관관계용으로만 릴레이(JSON 파싱 금지)
  - M2: 매니페스트 기반 모델 해석 — Lazy Load + SHA-256 무결성 검증 + G2(매니페스트 부재 조합 스킵)/G3(단독 전략 챔피언 시 반대쪽 score 컬럼 NULL) 정책(REQ-AIF-030/031/032/040/041) — `inference/resolution.py`(`resolve_serving_targets()`, `compute_score_columns()`) 신규
  - M3: 분위수 서빙 모델 선택 정책(G4) + Confidence 가드(REQ-AIF-050/051) — `resolve_latest_quantile_manifest()`(trained_date 무관 최신 q10/q90 선택), `inference/scoring.py`(`resolve_confidence_for_stock()` — `compute_confidence()` 호출부 `try/except ValueError` 가드)
  - M4: 등급 경계 실측(G1) + δ(PROMOTE/DEMOTE 마진) 잠정값 유도 + 저장소/분류 배선(REQ-AIF-080/081/111) — `inference/boundaries_store.py`(패키지 동봉 JSON `grade_boundaries.json`, `shift_boundaries()`, `derive_grade_margin_delta()`, `classify_grades()`), `training/boundaries_cli.py`(실측 산출 CLI) 신규. 시장×horizon 4개 조합 경계값을 실데이터(백필 완료 후 실현 수익률)로 역산
  - M5: 피처 조립 + 예측 실행 + `trading_signals` INSERT 경로(REQ-AIF-040/041/050/070/071/100) — `inference/features.py`(61거래일 룩백 조립기), `inference/predict.py`(포인트/분위수 모델 `.predict()` 실행), `data/repository.py`에 `stock_id` 반환 `fetch_stock_universe()` 신규 추가, `inference/writer.py`(`trading_signals` INSERT-ONLY + UNIQUE 스킵)
  - M6: 밴드 스윕 + `signal_price_bands` INSERT + 학습 잡 레이스 방어(REQ-AIF-060/110/111) — `inference/sweep.py`(그리드 생성 + 가상 일봉 재계산 + 인접 밴드 병합), `resolution.py`에 `SkipReason.MANIFEST_RACE`/`detect_manifest_race()` 추가, `writer.py`에 `insert_signal_price_bands()`(단일 트랜잭션 사전 SELECT + band_seq 일괄 INSERT) 추가
  - M7: 실측 확정 — 해외 밴드 그리드 범위(±15%→±21.5%), 병합 임계(1% 유지), 가상 일봉 규칙(동결 유지), 지연 목표·XAUTOCLAIM idle 임계(600초→1800초 상향, REQ-AIF-020/110/111/142) — NAS 프로덕션 실측 근거(domestic 745.88초/overseas 306.86초 사이클 소요)
  - M8: `stream:signal:{market}` 발행(REQ-AIF-120) + `InferenceMetrics` Prometheus 계측 3종(REQ-AIF-130) — `inference/publish.py`(7키 XADD, `MAXLEN ~500`), `inference/metrics.py`(`aaa_analyzer_inference_{skip,signals}_total`/`aaa_analyzer_inference_cycle_duration_seconds`)
  - **알려진 갭**: `__main__.py`의 end-to-end 파이프라인 배선(피처 조립 → 예측 → 등급 분류 → INSERT → 발행 조립)은 이 SPEC 범위 밖 — M1~M8이 납품한 것은 각 함수가 계약대로 독립 검증된 **미배선 단위 모듈**이며, 후속 미작성 SPEC이 실제 파이프라인 배선을 담당한다. overseas D60/quantile 조합의 `.meta.json` 사이드카 부재로 인한 실서비스 스코어링 경로 크래시 결함([aaa-infra#163](https://github.com/jongtix/aaa-infra/issues/163))도 이 SPEC의 코드로 해소되지 않았다(TRAIN-GATE-001/TRAIN-AUTOMATION-001 소관 추정)
- 추론 파이프라인 실배선 — `__main__.py` 종단간 조립 + 크로스-프로세스 메트릭 가시성 (SPEC-ANALYZER-PIPELINE-001 M1-M6, REQ-APL-100~150)
  - `inference/pipeline.py` 신규 — SPEC-ANALYZER-INFER-001(M1~M9)이 남긴 9개 완성 순수 모듈(`resolution`/`predict`/`scoring`/`features`/`boundaries_store`/`writer`/`sweep`/`publish`/`metrics`)을 실제 종단간 흐름으로 조립하는 오케스트레이션 계층. `__main__.py`의 `run_market_inference()`가 이제 "처리한 조합 없음"만 반환하던 M1 스캐폴딩 대신 `pipeline.run_market_inference()`를 호출한다 — 기존 exit code 0/1/2 계약(`resolve_exit_code()`)은 무수정(REQ-APL-100)
  - 조합 단위 스킵(NO_MANIFEST/SHA_MISMATCH/QUANTILE_MISSING)과 종목 단위 예외 경계(FEATURE_INSUFFICIENT/DEGENERATE_QUANTILE/신규 `SkipReason.UNEXPECTED_ERROR`)를 분리 — 종목 1건의 임의 예외가 같은 조합의 다른 종목이나 다른 조합/시장 처리를 중단시키지 않는다(REQ-APL-101/102/103)
  - `insert_trading_signal()`이 예외 없이 반환(`INSERTED`/`SKIPPED_DUPLICATE`)하면 항상 `publish_trading_signal()`을 호출 — 반환값으로 발행 여부를 분기하는 판정 함수를 두지 않는다(REQ-APL-104). 점 신호 INSERT+발행 완료 후 `sweep_and_write_price_bands()`(밴드 스윕)가 스킵돼도 이미 커밋된 신호는 롤백하지 않는다(REQ-APL-105)
  - `fetch_stock_universe()` 원본 결과에 `grade IN ('A','B')` AND `delisted_at IS NULL` 필터를 호출부(파이프라인)에서 적용 — (시장,horizon) 조합마다 반복하지 않고 시장당 1회만 조회·필터링(REQ-APL-107)
  - 앙상블 조합의 `model_version`은 두 알고리즘(lightgbm/xgboost) `trained_date` 중 `max()`(더 최근 값)로 산출하는 `resolve_model_version()` 신설, 단독 전략은 기존과 동일(REQ-APL-110). `orchestration/consumer.py`가 이벤트 수신 시점에 생성한 `trace_id`를 `spawn.py`의 `--trace-id` CLI 인자로 자식에 전파해 `stream:signal:{market}` 발행까지 상관관계 유지(부재 시 자식이 자체 생성, 하위호환)(REQ-APL-111)
  - `inference/config.py`에 `InferenceConfig.container_models_root` 필드 추가 — `orchestration/config.py`의 `AutomationConfig`가 이미 소비하는 동일 환경변수 `TRAIN_AUTOMATION_CONTAINER_MODELS_ROOT`를 재사용, `AutomationConfig` 전체를 임포트/요구하지 않는다(REQ-APL-120). `market_calendar.calendar_code` 매핑(domestic=KRX/overseas=NYSE)을 `inference/pipeline.py` 내부에 독립 정의 — `training/train.py`의 비공개 심볼을 임포트하지 않는다(REQ-APL-121)
  - Prometheus 공식 멀티프로세스 모드 도입 — `api/app.py`의 `GET /metrics`는 `PROMETHEUS_MULTIPROC_DIR` 설정 시 `multiprocess.MultiProcessCollector`를 전용 `CollectorRegistry()`에 등록해 자식 프로세스가 발행한 메트릭을 부모의 `/metrics`에서 실제로 관측 가능하게 한다(REQ-APL-130/131). `orchestration/metrics.py`의 `TrainingMetrics` Gauge 2종(`last_success_timestamp`/`model_stale`)에 `multiprocess_mode="max"` 명시 — 멀티프로세스 모드 도입이 기존 노출 형태(pid별 중복 라벨 없음)를 바꾸지 않도록 보존(REQ-APL-132)
  - `spawn.spawn_inference_child()`가 자식 프로세스 종료 직후 `prometheus_client.multiprocess.mark_process_dead(process.pid)` 호출 — 자식마다 남긴 pid별 메트릭 덤프 파일을 정리(REQ-APL-133). `InferenceMetrics()`는 자식 프로세스 1회 실행당 정확히 1회만 생성(REQ-APL-134), 전체 사이클 소요시간을 `time.monotonic()`으로 측정해 `observe_cycle_duration()`을 정확히 1회 호출(REQ-APL-150)
  - aaa-infra#163(overseas D60 챔피언/분위수 모델 `.meta.json` 사이드카 부재로 인한 실서비스 스코어링 크래시 위험) 봉쇄 — 파이프라인의 일반 예외 경계가 이 증상의 `ValueError`를 `SkipReason.UNEXPECTED_ERROR`로 자동 흡수해 자식 프로세스가 크래시하지 않는다(REQ-APL-140). 근본원인(`.meta.json` WRITER 미배선) 수정은 `SPEC-ANALYZER-TRAIN-META-001` 소관 — 이 SPEC은 이슈를 봉쇄만 하고 닫지 않는다
  - aaa-infra `docker-compose.yml`의 `analyzer` 서비스에 `PROMETHEUS_MULTIPROC_DIR` 환경변수 추가(기존 `tmpfs: [/tmp]` 재사용, 신규 볼륨 없음) — 부모/자식 프로세스가 동일 디렉토리를 공유해야 `MultiProcessCollector`가 합산 가능(REQ-APL-130)
  - `run_market_inference()` 진입 후 실제로 처리할 조합이 확인된 시점에 `assemble_inference_features_batch()`를 시장당 1회만 지연 호출해 캐싱 — horizon(20/60)마다 유니버스를 재순회하며 종목당 DB 왕복 3회를 중복 실행하던 문제를 제거(표준 코드 리뷰 W1 발견 사항 반영, 동작 무변경 순수 성능 리팩터)
  - 검증: `pytest -q` 968 passed / 커버리지 97.60%, `ruff check`/`ruff format --check`/`pyright` 전부 clean, `docker compose config --no-interpolate -q`(aaa-infra) exit 0, `AnalyzerInferenceDeadman` vmalert 룰 메트릭명 정합성 회귀 가드 양쪽 레포에 추가(AC-APL-160)
  - **알려진 갭**: 실서비스(NAS) 배포·라이브 트래픽 검증은 미수행(로컬 워크트리 + 합성 데이터 검증 범위). 추론 데드맨 알람 라우팅 활성화(`alertmanager.yml` null 라우팅 제거)는 별도 후속 SPEC 소관 — 이 SPEC의 완료가 그 선행 조건
- 로그 디렉토리 무한 증가 방지 — 트레이너 로그 보존 sweep + 로그 총량 상한 패리티 문서화 (SPEC-OBSV-LOGS-003 M1-M3, REQ-001~010)
  - `orchestration/log_retention.py` 신설 — `trainer_{run_id}.log` 파일군에 개수 기반 보존 정책 적용(기본 최신 10개 유지, mtime 내림차순). 라이브 NAS 실측상 `/var/log/aaa-analyzer`가 74MB까지 증가해 있었고 그중 `trainer_*.log` 2개가 각 38MB·18일 이상 방치 상태였다 — 셸 `tee`로 기록되는 파일이라 Python `logging` 회전 대상이 아니었고 어디에도 정리 로직이 없었다(REQ-001/003)
  - sweep의 glob 대상은 `TRAIN_AUTOMATION_TRAINER_LOG_BASE_DIR`가 **아니라** analyzer 컨테이너 자신의 로컬 로그 디렉토리(`LOG_PATH`, 기본값 `/var/log/aaa-analyzer`) — 전자는 맥북 SMB 마운트 경로 문자열이며 원격 SSH 명령 문자열 구성에만 쓰인다. 컨테이너 내부에서 그 값을 그대로 `Path().glob()`했다면 존재하지 않는 macOS 경로를 대상으로 빈 이터레이터를 반환해 프로덕션에서만 조용히 무동작했을 결함(plan-audit iteration 1 D1에서 정정)
  - 현재 진행 중인 run의 `trainer_{run_id}.log`는 가장 오래되어도 삭제 대상에서 제외되며 보존 정원도 잠식하지 않는다. 비-트레이너 파일(`aaa-analyzer.log*`)은 glob 패턴 불일치로 자연 제외(REQ-003/004)
  - 삭제 실패는 파일 단위로 건너뛰고 로그만 남긴다 — dispatch 실패로 전파되지 않는다(REQ-005, SPEC-OBSV-LOGS-002 fail-open 원칙 계승)
  - sweep 호출은 `runner.py`/`monthly_dispatch.py`의 디스패치 완료 경로(`finally`)에 배선 — 두 호출부가 동일 함수를 재사용한다(REQ-002/006). `ssh_dispatch.py`의 두 빌더 함수는 원격 셸 명령 **문자열만 조립**하고 디스패치를 실행하지 않으므로 배선 지점이 될 수 없었다(계획 대비 문서화된 편차, 근거는 progress.md §E.2)
  - 신규 환경변수 `TRAIN_AUTOMATION_TRAINER_LOG_RETENTION_COUNT`(`.env.example` 문서화) — 미설정 시 기본값 10, 비정수·음수는 경고 로그 후 기본값으로 대체
  - `common/logging.py` docstring에 collector/notifier `total-size-cap` 대비 패리티 판단 기록(REQ-007/008) — `RotatingFileHandler(maxBytes=10 MiB, backupCount=5)`의 결정론적 상한(≈60 MiB)이 **디스크 사용량 무한 증가 방지라는 의도 하나에 대해서만** 동등하며, logback `max-history: 30`이 확보하는 사고 조사용 30일 보존 창과의 동등성은 함의하지 않는다는 범위 한정 + 보존 기간 정성적 추정(단일 시점 스냅샷 한계 명시) + 재검토 조건을 병기. 회전 상수 자체는 SPEC-OBSV-LOGS-002 확정값에서 무변경(순수 docstring 추가)
  - ADR-011이 "Phase 2에서 별도 ADR로 결정한다"고 남긴 미이행 문구를 aaa-infra 측 ADR-035 신설로 이행(별도 레포 변경)
  - 검증은 단위 테스트 수준(`LOG_PATH`를 임시 디렉토리로 오버라이드) — NAS 컨테이너 실배포 후 실제 `trainer_*.log` 삭제 관측은 미수행
- 월간 Optuna 재튜닝 cron 활성화 (SPEC-ANALYZER-TRAIN-TUNING-001 M1-M9, REQ-ATT-*)
  - `monthly-optuna-tuning` cron 잡 신규 등록(매월 1일 06:00 KST) — `SchedulerRegistry.registered_jobs()`가 이제 `weekly-full-retrain`/`daily-staleness-check`/`monthly-optuna-tuning` 정확히 3건을 반환한다. `register_default_jobs()` 호출은 0건, `register_cron_job()` 개별 호출 3회로 전환(REQ-ATT-002/003/004)
  - 원격 실행 벡터는 기존 주간 학습 스크립트(`analyzer.training.train`)가 아닌 캠페인 CLI(`python -m analyzer.training.campaign`)를 재사용 — `ssh_dispatch.build_remote_dispatch_command()`에서 골격을 추출해 신규 캠페인 디스패치 함수를 파생시켰다(바이트 동일성 회귀 테스트로 기존 주간 경로 무변경 확인, REQ-ATT-005/006/007)
  - 캠페인 CLI에 `--n-trials` 선택 인자 추가 — 월간 cron은 이 값을 프로덕션 트라이얼 수로 전달, 생략 시 기존 기본값 유지(REQ-ATT-008)
  - 월간 전용 설정 3종(`monthly_optuna_tuning_trigger()` day=1/hour=6/minute=0/Asia-Seoul, active_models_root/staging_models_root 구분 디스패치, max_instances=1/coalesce/misfire_grace_time 명시)을 GATE-001이 확립한 `SchedulerRegistry`/`TrainingMetrics` 싱글턴 패턴으로 재사용(REQ-ATT-010/011/012)
  - 조합별(시장×horizon×algorithm 8개) 보존 정책 프로덕션 배선 지점 추가 — `persistence.apply_retention_policy()` 시그니처는 무수정, 소환 지점만 신설(REQ-ATT-017)
  - 운영자 수동 롤백 CLI(`python -m analyzer.training.rollback`) 신설 — `--algorithm {lightgbm,xgboost}` 인자에 `choices` 제약을 부여해 오타 시 raw traceback 대신 argparse 표준 에러로 안내(REQ-ATT-018/019/020/024, review finding W3 수정 포함)
  - 모델 버전 열거 헬퍼 추가 — 롤백 대상 후보 나열에 사용(REQ-ATT-016)
  - 월간 캠페인 후처리(보존 정책 적용) 실패 경로에 `run_id` 상관관계 로깅 추가 — 캠페인 자체 성공/보존 정책만 실패라는 구분을 유지한 채 관측성을 강화(review finding W1 수정)
  - `/moai review` 4관점 팬아웃 + sync-auditor 종합에서 Critical 0건, Warning 4건(W1/W3는 코드 수정 완료, W2는 SPEC 문서상 스케줄링 겹침 서술 정정, W4는 SSH 비밀번호 `ps` 노출 — 리뷰어 권고에 따라 침습도 대비 심각도가 낮아 별도 후속 과제로 이연, byte-identical 회귀 게이트 리스크 회피)
  - 프로덕션 활성화는 별도 운영자 작업 대기 중 — NAS 측 실배포·운영 검증 미수행(TRAIN-STALENESS-001/TRAIN-GATE-001과 동일한 후속 절차)
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

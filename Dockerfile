# === Build stage ===
# digest 고정: python:3.14-slim(3.14.6-slim-trixie)의 linux/amd64 플랫폼 전용 digest.
# 버전 갱신 시 `docker manifest inspect python:3.14-slim --verbose`로 재확인.
FROM python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4 AS build
WORKDIR /analyzer

# 릴리스 태그 버전 주입(REQ-020/AC-023): PSR이 commit: false로 동작해 pyproject.toml의
# version은 정적으로 남으므로, 빌드타임에 태그 버전을 pyproject.toml에 반영한다.
# 미지정 시 0.0.0(로컬 빌드 기본값).
ARG VERSION=0.0.0

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# 의존성 레이어 캐시용 — src 변경 시 재다운로드 방지
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# 소스 복사 및 프로젝트 자체 설치(런타임 코드 포함, dev 그룹 제외)
COPY src/ src/
COPY README.md ./
# 프로젝트 자체 설치 직전에 버전을 덮어써야 .venv에 설치되는 dist-info 버전이 태그와 일치한다.
# pyproject.toml만 sed로 고치면 안 된다 — uv.lock이 루트 패키지 버전을 함께 기록하므로
# 아래 `uv sync --locked`가 lockfile 불일치로 exit 1이 된다(실측 재현). `uv version --no-sync`는
# pyproject.toml과 uv.lock의 루트 버전을 함께 갱신하고 의존성 해석 결과는 건드리지 않는다.
RUN uv version --no-sync "${VERSION}"
RUN uv sync --locked --no-dev

# === Runtime stage ===
# digest 재핀(2026-08-12): python:3.14-slim(3.14.6-slim-trixie) linux/amd64 최신 digest로 갱신.
# 재확인: `docker buildx imagetools inspect python:3.14-slim`
# apt-get upgrade 생략(의도적, REQ-CVE-018 RETIRED — 재현성 우선 정책): 빌드마다 배포판
# 최신 패치를 얹는 대신 base 이미지 재핀만으로 바닥을 고정한다. 상류 base 이미지 재빌드
# 지연으로 재핀 후에도 fix-available CRITICAL/HIGH가 잔존하는 경우(예: util-linux 계열
# CVE-2026-53615 — Debian trixie에 수정판 2.41.5-0+deb13u1은 이미 게시됐으나 이 이미지가
# 아직 반영하지 못한 상태)는 만료일 명시 `.trivyignore` 항목으로 처리한다
# (SPEC-INFRA-CVE-SCAN-001 v0.8.1 REQ-CVE-016/022).
FROM python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4

# 시스템 pip 제거(REQ-CVE-021 후속): 이 이미지는 uv만 사용하고 시스템 pip을 런타임에
# 전혀 호출하지 않는다. base 이미지가 내장한 pip 26.2.1이 벤더링한
# msgpack==1.1.2/setuptools==70.3.0(vendor.txt)이 Trivy HIGH CVE 게이트를 유발했다
# (GHSA-6v7p-g79w-8964, CVE-2025-47273) — 사용되지 않는 코드이므로 제거가 근본 해결이다.
RUN rm -rf /usr/local/lib/python3.14/ensurepip \
    /usr/local/lib/python3.14/site-packages/pip* \
    /usr/local/bin/pip*

# 비루트 유저 생성(UID 1005 — collector UID 1004와 비충돌, Debian 계열 groupadd/useradd)
RUN groupadd -g 1005 analyzer \
    && useradd -u 1005 -g analyzer --no-create-home --shell /usr/sbin/nologin analyzer

WORKDIR /analyzer
COPY --chown=analyzer:analyzer --from=build /analyzer/.venv /analyzer/.venv
COPY --chown=analyzer:analyzer --from=build /analyzer/src /analyzer/src

ENV PATH="/analyzer/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul

USER analyzer
EXPOSE 8000

# 헬스체크: python:3.14-slim(Debian)은 curl/wget 미포함 — 표준 라이브러리로 HTTP GET 수행.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]

ENTRYPOINT ["python", "-m", "analyzer.api.main"]

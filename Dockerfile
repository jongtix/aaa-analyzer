# === Build stage ===
# digest 고정: python:3.14-slim(3.14.6-slim-trixie)의 linux/amd64 플랫폼 전용 digest.
# 버전 갱신 시 `docker manifest inspect python:3.14-slim --verbose`로 재확인.
FROM python:3.14-slim@sha256:d6e0850f13fda0e2305d4c3c1c2f7930fe1042d34ddd958e49bba6ef685d0bb2 AS build
WORKDIR /analyzer

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# 의존성 레이어 캐시용 — src 변경 시 재다운로드 방지
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# 소스 복사 및 프로젝트 자체 설치(런타임 코드 포함, dev 그룹 제외)
COPY src/ src/
COPY README.md ./
RUN uv sync --locked --no-dev

# === Runtime stage ===
# digest 재핀(2026-08-12): python:3.14-slim(3.14.6-slim-trixie) linux/amd64 최신 digest로 갱신.
# 재확인: `docker buildx imagetools inspect python:3.14-slim`
# apt-get upgrade 생략(의도적): 재핀 후 Trivy 재스캔 결과 잔여 CRITICAL/HIGH가 전부
# Debian trixie 상류에 fix 자체가 없는 unfixed 상태(perl-base/util-linux/ncurses 등 시스템
# 유틸리티) — apt-get upgrade는 저장소에 패치가 있어야 의미가 있어 여기선 무력하다.
# 게이트는 --ignore-unfixed로 이 잔여를 정책적으로 처리한다(REQ-CVE-021).
FROM python:3.14-slim@sha256:d6e0850f13fda0e2305d4c3c1c2f7930fe1042d34ddd958e49bba6ef685d0bb2

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

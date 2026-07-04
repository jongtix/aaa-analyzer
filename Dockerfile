# === Build stage ===
# digest 고정: python:3.14-slim(3.14.6-slim-trixie)의 linux/amd64 플랫폼 전용 digest.
# 버전 갱신 시 `docker manifest inspect python:3.14-slim --verbose`로 재확인.
FROM python:3.14-slim@sha256:8e4071294d046d45b31a902e02a8560a45c351898513b66ec659ca39fd30d170 AS build
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
FROM python:3.14-slim@sha256:8e4071294d046d45b31a902e02a8560a45c351898513b66ec659ca39fd30d170

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

ENTRYPOINT ["python", "-m", "uvicorn", "analyzer.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

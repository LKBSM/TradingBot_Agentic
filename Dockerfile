# Root-level Dockerfile — shim to infrastructure/Dockerfile.
#
# Many container tooling (docker build, docker-compose at root, CI/CD
# providers) defaults to looking for ./Dockerfile. This file re-declares
# the build from the canonical Dockerfile in infrastructure/, so:
#
#   docker build -t smart-sentinel:latest .
#
# works from the repo root without needing -f infrastructure/Dockerfile.
#
# Maintenance: keep this file in lock-step with infrastructure/Dockerfile.
# This is intentional duplication; refactor to one if Docker introduces
# proper #include support.

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS production

LABEL maintainer="Smart Sentinel AI"
LABEL version="1.0.0-institutional"
LABEL description="Smart Sentinel AI — Market Intelligence (Docker-canonical, post Sprint 0-7)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app \
    TZ=UTC

RUN groupadd -r sentinel && useradd -r -g sentinel sentinel

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=sentinel:sentinel . .

RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R sentinel:sentinel /app

USER sentinel

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "src.intelligence.main"]

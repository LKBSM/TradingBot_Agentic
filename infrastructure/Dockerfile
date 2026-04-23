# =============================================================================
# SMART SENTINEL AI — Production Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
#
# Build: docker build -t smart-sentinel:latest -f infrastructure/Dockerfile .
# Run:   docker run -d --name sentinel -p 8000:8000 \
#          -e ANTHROPIC_API_KEY=sk-... \
#          -e SENTINEL_TESTING_MODE=1 \
#          smart-sentinel:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

# Labels
LABEL maintainer="Smart Sentinel AI"
LABEL version="1.0.0"
LABEL description="Smart Sentinel AI — Market Intelligence"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=UTC

# Create non-root user for security
RUN groupadd -r sentinel && \
    useradd -r -g sentinel sentinel

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=sentinel:sentinel . .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R sentinel:sentinel /app

# Switch to non-root user
USER sentinel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
# 8000: REST API
EXPOSE 8000

# Default command — Smart Sentinel AI entry point
CMD ["python", "-m", "src.intelligence.main"]

# -----------------------------------------------------------------------------
# Stage 3: Development (optional)
# -----------------------------------------------------------------------------
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    ipython

USER sentinel

# Override command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

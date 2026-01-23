# =============================================================================
# TRADING BOT DOCKERFILE - Production Image
# =============================================================================
# Multi-stage build for optimized production image
#
# Build: docker build -t trading-bot:latest .
# Run:   docker run -d --name trading-bot -p 8080:8080 trading-bot:latest
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
LABEL maintainer="Trading Bot Team"
LABEL version="3.0.0"
LABEL description="Agentic Trading Bot - Sprint 3"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=UTC

# Create non-root user for security
RUN groupadd -r tradingbot && \
    useradd -r -g tradingbot tradingbot

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
COPY --chown=tradingbot:tradingbot . .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R tradingbot:tradingbot /app

# Switch to non-root user
USER tradingbot

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
# 8080: REST API
# 9090: Metrics (Prometheus)
EXPOSE 8080 9090

# Default command
CMD ["python", "-m", "src.main"]

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

USER tradingbot

# Override command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

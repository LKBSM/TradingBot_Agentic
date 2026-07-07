# syntax=docker/dockerfile:1
# =============================================================================
# MIA Markets — Frontend (Next.js 15) image for 24/7 operation.
#
# Uses Next's standalone output (see next.config.js `output: 'standalone'`) so
# the runtime image ships only the traced deps + a tiny node server.
#
# The browser talks ONLY to this frontend (same-origin). `/api/*` is proxied
# server-side to the backend via the Next rewrite; the proxy target is baked at
# build time from NEXT_PUBLIC_API_BASE (compose passes http://backend:8000).
# NEXT_PUBLIC_BETA_LOCKDOWN is inlined so the client gate fails closed in beta.
# =============================================================================

# ---- deps ------------------------------------------------------------------
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
# The project has a vitest/vite peer-dep conflict — legacy-peer-deps is required
# (documented in project CI notes).
RUN npm ci --legacy-peer-deps --no-audit --no-fund

# ---- builder ---------------------------------------------------------------
FROM node:22-slim AS builder
WORKDIR /app

# Build-time public config (inlined into the client bundle by Next).
ARG NEXT_PUBLIC_API_BASE=http://backend:8000
ARG NEXT_PUBLIC_BETA_LOCKDOWN=1
ARG NEXT_PUBLIC_LIVE_TICK=1
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE} \
    NEXT_PUBLIC_BETA_LOCKDOWN=${NEXT_PUBLIC_BETA_LOCKDOWN} \
    NEXT_PUBLIC_LIVE_TICK=${NEXT_PUBLIC_LIVE_TICK} \
    NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ---- runtime ---------------------------------------------------------------
FROM node:22-slim AS runtime
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# curl for the HEALTHCHECK; drop root.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10002 webapp

# Standalone server + static assets + public dir.
COPY --from=builder --chown=webapp:webapp /app/.next/standalone ./
COPY --from=builder --chown=webapp:webapp /app/.next/static ./.next/static
COPY --from=builder --chown=webapp:webapp /app/public ./public

USER webapp
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:3000/ || exit 1

# server.js is emitted by Next standalone at the bundle root.
CMD ["node", "server.js"]

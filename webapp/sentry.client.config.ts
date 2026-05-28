/**
 * DG-033 — Sentry browser SDK init for the M.I.A. webapp.
 *
 * No-op when ``NEXT_PUBLIC_SENTRY_DSN`` is unset, matching the backend
 * behaviour in ``src/performance/observability.py::init_sentry``.
 * That way the dev loop (and CI builds without a DSN) stay fully
 * offline — no network calls, no errors.
 */
import * as Sentry from '@sentry/nextjs';

const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;

if (dsn) {
  Sentry.init({
    dsn,
    environment: process.env.NEXT_PUBLIC_SENTRY_ENV ?? 'dev',
    release: process.env.NEXT_PUBLIC_SENTRY_RELEASE,
    tracesSampleRate: Number(process.env.NEXT_PUBLIC_SENTRY_TRACES_RATE ?? '0.05'),
    replaysSessionSampleRate: 0, // no session replay V1 — privacy + cost
    replaysOnErrorSampleRate: 0.1,
    // Strip PII at source: API keys can land in fetch error bodies.
    beforeSend(event) {
      const serialized = JSON.stringify(event);
      const suspicious = [
        'ANTHROPIC_API_KEY=',
        'TELEGRAM_BOT_TOKEN=',
        'sk_live_',
        'sk_test_',
      ];
      if (suspicious.some((s) => serialized.includes(s))) {
        return null;
      }
      return event;
    },
  });
}

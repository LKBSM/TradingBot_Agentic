/**
 * DG-033 — Sentry edge-runtime SDK init (middleware + edge API routes).
 * No-op when ``SENTRY_DSN`` is unset.
 */
import * as Sentry from '@sentry/nextjs';

const dsn = process.env.SENTRY_DSN;

if (dsn) {
  Sentry.init({
    dsn,
    environment: process.env.SENTRY_ENV ?? process.env.NODE_ENV ?? 'dev',
    release: process.env.SENTRY_RELEASE,
    tracesSampleRate: Number(process.env.SENTRY_TRACES_RATE ?? '0.05'),
  });
}

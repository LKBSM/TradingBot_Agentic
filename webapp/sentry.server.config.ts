/**
 * DG-033 — Sentry server-side SDK init (Next.js API routes + RSC).
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
    // PII scrub mirror — keep behaviour identical on server + client.
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

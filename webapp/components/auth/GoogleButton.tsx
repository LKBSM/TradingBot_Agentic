'use client';

import { useTranslations } from 'next-intl';
import * as React from 'react';
import { GOOGLE_START_URL, fetchGoogleEnabled } from '@/lib/auth/google';

/**
 * "Continue with Google" button. Renders NOTHING unless the backend reports
 * Google sign-in is configured (env-gated) — so a deploy without Google simply
 * never shows it. Clicking navigates to the server start endpoint (a full-page
 * redirect into Google's consent screen), not a fetch.
 */
export function GoogleButton() {
  const t = useTranslations('auth.google');
  const [enabled, setEnabled] = React.useState(false);

  React.useEffect(() => {
    let active = true;
    fetchGoogleEnabled().then((ok) => {
      if (active) setEnabled(ok);
    });
    return () => {
      active = false;
    };
  }, []);

  if (!enabled) return null;

  return (
    <a
      href={GOOGLE_START_URL}
      className="flex w-full items-center justify-center gap-2 rounded-md border border-border/60 bg-background px-4 py-2 text-sm font-medium text-foreground transition hover:bg-muted/40"
    >
      <svg viewBox="0 0 18 18" width="18" height="18" aria-hidden>
        <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62Z" />
        <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.02-3.7H.92v2.33A9 9 0 0 0 9 18Z" />
        <path fill="#FBBC05" d="M3.98 10.72a5.4 5.4 0 0 1 0-3.44V4.95H.92a9 9 0 0 0 0 8.1l3.06-2.33Z" />
        <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.47.9 11.43 0 9 0A9 9 0 0 0 .92 4.95l3.06 2.33C4.68 5.16 6.66 3.58 9 3.58Z" />
      </svg>
      {t('button')}
    </a>
  );
}

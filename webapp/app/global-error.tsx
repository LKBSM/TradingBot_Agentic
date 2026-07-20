'use client';

/**
 * Root error boundary (UI-02, last-resort). Catches errors thrown in the locale
 * layout itself — the one place `[locale]/error.tsx` can't reach. It replaces
 * the whole document, so it must render its own <html>/<body> and cannot rely
 * on the theme provider or app fonts. Deliberately dependency-free and inline-
 * styled so it renders even when the app shell is broken. FR-only.
 */

import * as React from 'react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  React.useEffect(() => {
    console.error('Global error boundary caught:', error);
  }, [error]);

  return (
    <html lang="fr">
      <body
        style={{
          margin: 0,
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily:
            'system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
          background: '#0a0f1c',
          color: '#e8edf7',
          padding: '2rem',
        }}
      >
        <div style={{ maxWidth: 480, textAlign: 'center' }}>
          <h1 style={{ fontSize: '1.5rem', marginBottom: '0.75rem' }}>
            Une erreur est survenue
          </h1>
          <p style={{ opacity: 0.75, marginBottom: '1.5rem', lineHeight: 1.6 }}>
            L&apos;application a rencontré un problème inattendu. Recharge la page
            pour réessayer.
          </p>
          <button
            onClick={() => reset()}
            style={{
              cursor: 'pointer',
              border: 'none',
              borderRadius: 8,
              padding: '0.6rem 1.25rem',
              fontSize: '0.95rem',
              fontWeight: 600,
              background: '#c9a227',
              color: '#0a0f1c',
            }}
          >
            Réessayer
          </button>
        </div>
      </body>
    </html>
  );
}

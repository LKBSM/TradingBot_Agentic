'use client';

import * as React from 'react';
import { renderLegalMarkdown } from '@/lib/legal/render-markdown';

/**
 * Fetches the canonical CGU markdown from the backend
 * (`GET /api/v1/legal/conditions`, via the same-origin /api/* rewrite) and
 * renders it TEL QUEL. The document version comes from the `X-Document-Version`
 * response header so the page never hard-codes a date that could drift from the
 * source file.
 */
export function ConditionsDocument() {
  const [markdown, setMarkdown] = React.useState<string | null>(null);
  const [version, setVersion] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await fetch('/api/v1/legal/conditions', {
          headers: { accept: 'text/markdown' },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        if (!active) return;
        setVersion(res.headers.get('X-Document-Version'));
        setMarkdown(text);
      } catch {
        if (active) {
          setError(
            'Le document des Conditions est momentanément indisponible. Réessayez plus tard.',
          );
        }
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return (
      <p
        role="alert"
        className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
      >
        {error}
      </p>
    );
  }

  if (markdown === null) {
    return <p className="text-sm text-muted-foreground">Chargement du document…</p>;
  }

  return (
    <article className="space-y-1">
      {version && (
        <p className="mb-6 text-xs uppercase tracking-wider text-muted-foreground">
          Version {version}
        </p>
      )}
      {renderLegalMarkdown(markdown)}
    </article>
  );
}

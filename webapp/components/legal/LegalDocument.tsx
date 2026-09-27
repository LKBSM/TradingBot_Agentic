'use client';

import * as React from 'react';
import { useLocale, useTranslations } from 'next-intl';
import { renderLegalMarkdown } from '@/lib/legal/render-markdown';

export type LegalDocKind = 'terms' | 'privacy';

/** Backend path serving each document (LEG-1 — one source, served verbatim). */
const ENDPOINT: Record<LegalDocKind, string> = {
  terms: '/api/v1/legal/conditions',
  privacy: '/api/v1/legal/privacy',
};

/**
 * Renders a legal document fetched from the backend (via the same-origin
 * `/api/*` rewrite) TEL QUEL.
 *
 * LEG-1 — the text lives in exactly one place, `docs/legal/*.{fr,en,es}.md`, and
 * this component never rewrites it. Two consequences worth keeping:
 *
 *  - the UI locale is passed as `?lang=`, so a reader gets the document in the
 *    language they are reading the site in (or English, which the document
 *    itself discloses, for a locale we do not publish a legal text in);
 *  - the version comes from the `X-Document-Version` response header, never
 *    hard-coded here, so the displayed version can never drift from the version
 *    stamped on a customer's consent.
 */
export function LegalDocument({ doc }: { doc: LegalDocKind }) {
  const t = useTranslations('legal');
  const locale = useLocale();
  const [markdown, setMarkdown] = React.useState<string | null>(null);
  const [version, setVersion] = React.useState<string | null>(null);
  const [hasError, setHasError] = React.useState(false);

  React.useEffect(() => {
    let active = true;
    setMarkdown(null);
    setHasError(false);
    (async () => {
      try {
        const res = await fetch(
          `${ENDPOINT[doc]}?lang=${encodeURIComponent(locale)}`,
          { headers: { accept: 'text/markdown' } },
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        if (!active) return;
        setVersion(res.headers.get('X-Document-Version'));
        setMarkdown(text);
      } catch {
        if (active) setHasError(true);
      }
    })();
    return () => {
      active = false;
    };
  }, [doc, locale]);

  // LEG-2 — land on the clause a deep link asks for. The document is fetched
  // AFTER hydration, so by the time `#remboursement` has a target the browser has
  // long since given up on the hash: nothing scrolls unless we do it ourselves.
  // This is what makes the footer link and the one next to the payment button
  // arrive at « Résiliation et remboursement » instead of at the top of the page.
  React.useEffect(() => {
    if (markdown === null) return;
    const jump = () => {
      const id = decodeURIComponent(window.location.hash.slice(1));
      if (!id) return;
      const target = document.getElementById(id);
      // `scrollIntoView` is missing in jsdom — guarded so tests exercise the
      // lookup rather than crash on the scroll.
      if (typeof target?.scrollIntoView === 'function') {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    };
    // Next frame: the blocks have to be in the DOM before we can find one.
    const raf = requestAnimationFrame(jump);
    window.addEventListener('hashchange', jump);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('hashchange', jump);
    };
  }, [markdown]);

  if (hasError) {
    return (
      <p
        role="alert"
        className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
      >
        {t('document.error')}
      </p>
    );
  }

  if (markdown === null) {
    return <p className="text-sm text-muted-foreground">{t('document.loading')}</p>;
  }

  return (
    <article className="space-y-1">
      {version && (
        <p className="mb-6 text-xs uppercase tracking-wider text-muted-foreground">
          {t('document.versionLabel', { version })}
        </p>
      )}
      {renderLegalMarkdown(markdown)}
    </article>
  );
}

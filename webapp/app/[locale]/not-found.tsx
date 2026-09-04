/**
 * Localized 404 (NAV-09). Renders inside the locale layout (Nav + Footer +
 * theme + fonts), so an unknown URL no longer drops to Next's bare, unstyled
 * default page. Also catches `notFound()` from the layout (unsupported locale).
 *
 * I18N-1: fully translated (fr/en/es) — the whole screen must be in the active
 * language, a 404 included. The regulatory notice reuses the canonical
 * `connexion.trust` disclaimer so the wording stays a single source.
 */

import Link from 'next/link';
import { getTranslations } from 'next-intl/server';
import { Button } from '@/components/ui/button';

export default async function NotFound() {
  const t = await getTranslations('notFound');
  const tTrust = await getTranslations('connexion');
  return (
    <div className="container-prose flex min-h-[60vh] flex-col items-center justify-center gap-6 py-16 text-center">
      <div className="space-y-3">
        <p className="text-sm font-medium text-muted-foreground">{t('code')}</p>
        <h1 className="text-2xl font-semibold text-foreground">{t('heading')}</h1>
        <p className="text-muted-foreground">{t('body')}</p>
      </div>
      <Button asChild>
        <Link href="/">{t('back')}</Link>
      </Button>
      {/* Regulatory notice — one per page, never zero. Canonical disclaimer. */}
      <p className="max-w-md text-xs text-muted-foreground/70">{tTrust('trust')}</p>
    </div>
  );
}

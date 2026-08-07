import Link from 'next/link';
import { getTranslations } from 'next-intl/server';
import { localizeHref } from '@/lib/i18n/href';
import { cn } from '@/lib/utils';

/**
 * SC-2 — entry point between the two scanner modes.
 *
 *   · "conditions" → the manual palette scanner (`/scanner`), and
 *   · "describe"   → the conversational scanner (`/scanner/decrire`).
 *
 * The conversational mode is an ADDITIONAL entry to the same read-only scan,
 * never a replacement (see the palette page's doc-comment). This segmented
 * control makes it discoverable: it sits at the top of both surfaces, the
 * active mode is highlighted and inert (`aria-current`), the other is a
 * locale-preserving link. Pure server component — two links, no client JS.
 */
export async function ScannerModeToggle({
  locale,
  active,
}: {
  locale: string;
  active: 'conditions' | 'describe';
}) {
  const t = await getTranslations({ locale, namespace: 'scanner' });

  const segment =
    'inline-flex min-h-[44px] items-center justify-center rounded-full px-4 py-1.5 text-sm font-medium transition-colors xl:min-h-0';
  const activeCls = 'bg-background text-foreground shadow-sm';
  const inactiveCls = 'text-muted-foreground hover:text-foreground';

  return (
    <div
      className="inline-flex items-center rounded-full border border-border/70 bg-muted/40 p-1"
      role="group"
      aria-label={t('mode.groupLabel')}
    >
      {active === 'conditions' ? (
        <span className={cn(segment, activeCls)} aria-current="page">
          {t('mode.conditions')}
        </span>
      ) : (
        <Link href={localizeHref('/scanner', locale)} className={cn(segment, inactiveCls)}>
          {t('mode.conditions')}
        </Link>
      )}

      {active === 'describe' ? (
        <span className={cn(segment, activeCls)} aria-current="page">
          {t('mode.describe')}
        </span>
      ) : (
        <Link href={localizeHref('/scanner/decrire', locale)} className={cn(segment, inactiveCls)}>
          {t('mode.describe')}
        </Link>
      )}
    </div>
  );
}

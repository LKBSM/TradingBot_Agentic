import { useTranslations } from 'next-intl';
import { ShieldCheck } from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * Compact disclaimer line shown under a market-reading hero. Single source of
 * truth (Chantier 5.C) — previously duplicated across `insight/` and
 * `market-reading/`. The wording is a placeholder pending the legal terminal.
 *
 * Two variants:
 *   · 'hero'  — the full line (early-access mention + legal disclaimer), used on
 *               the text-only landing samples where no separate badge is shown.
 *   · 'chart' — the PERSISTENT legal mention only ("educational · neither signal
 *               nor advice"), used just under the /app chart. The temporary
 *               "early access" status is carried separately by EarlyAccessBadge
 *               so the compliance line stays clean and never varies.
 *
 * LEGAL-PENDING: do not ship this wording to production. Reviewers grep for
 * "LEGAL-PENDING" before each release tag.
 */
export function DisclaimerStub({
  className,
  variant = 'hero',
}: {
  className?: string;
  variant?: 'hero' | 'chart';
}) {
  const t = useTranslations('legal');
  return (
    <p
      className={cn(
        'text-[11px] italic leading-relaxed text-muted-foreground/80',
        className,
      )}
      data-legal-pending="hero-disclaimer"
    >
      {/* LEGAL-PENDING: disclaimer — to be replaced with the wording finalised
          by the legal terminal. */}
      {t(variant === 'chart' ? 'disclaimer.chart' : 'disclaimer.hero')}
    </p>
  );
}

/**
 * Discreet "Accès anticipé" badge — the TEMPORARY product-status marker, kept
 * separate from the persistent legal disclaimer (DisclaimerStub) so the two are
 * never conflated. Mirrors the footer badge styling.
 */
export function EarlyAccessBadge({ className }: { className?: string }) {
  const t = useTranslations('legal');
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border border-sentinel-bull/40 bg-sentinel-bull/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-sentinel-bull',
        className,
      )}
    >
      <ShieldCheck className="h-2.5 w-2.5" aria-hidden />
      {t('earlyAccessBadge')}
    </span>
  );
}

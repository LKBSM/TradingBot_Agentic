import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ShieldCheck, AlertTriangle, ArrowRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

/**
 * Section 5 — « Honnêteté » (Engagement public).
 *
 * Pleine largeur, fond sombre subtil. C'est le moat de MIA : on assume
 * publiquement notre posture — un outil de compréhension, pas une promesse
 * de gain.
 *
 * Post-pivot 2026-05-27 (Chantier 5.C) : retrait des chiffres de backtest
 * (Profit Factor, Deflated Sharpe Ratio, PBO) et de la colonne « edge
 * mesurable » — ces métriques relevaient du positionnement pré-pivot
 * « système de trading » et n'ont pas leur place sur un indicateur descriptif
 * (edge_claim=false, niveau 1.5 strict).
 *
 * Conservé : la citation « Engagement public » (lock 2) + les colonnes
 * « Ce que nous ne ferons jamais » et « Ce que nous faisons aujourd'hui ».
 *
 * Cette section EXISTE parce qu'elle constitue la posture d'honnêteté
 * publique et le différenciateur "fini de mentir" face aux fournisseurs
 * de signaux qui annoncent 90 % de win-rate.
 */
export function HonestConfidenceSection() {
  const t = useTranslations('landing.honesty');
  return (
    <section
      id="honnetete"
      aria-labelledby="honest-confidence-title"
      className="bg-foreground/[0.02] py-16 sm:py-24 dark:bg-foreground/[0.03]"
    >
      <div className="container-wide">
        <header className="mb-10 max-w-3xl">
          <Badge
            variant="outline"
            className="mb-3 border-sentinel-warn/40 text-[11px] uppercase tracking-wider text-sentinel-warn"
          >
            <ShieldCheck className="mr-1 h-3 w-3" aria-hidden />
            {t('badge')}
          </Badge>
          <h2
            id="honest-confidence-title"
            className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl lg:text-4xl"
          >
            {t('title')}
          </h2>
          <p className="mt-4 max-w-2xl text-pretty text-muted-foreground">
            {t('subtitle')}
          </p>
        </header>

        {/* Citation imposée (lock 2) */}
        <figure className="max-w-3xl">
          <blockquote className="border-l-2 border-sentinel-warn pl-5 text-balance text-base italic text-foreground sm:text-lg">
            {t('quote')}
          </blockquote>
          <figcaption className="mt-3 text-xs text-muted-foreground">
            {t('quoteCaption')}
          </figcaption>
        </figure>

        {/* Que faisons-nous alors */}
        <div className="mt-12 grid gap-6 lg:grid-cols-2">
          <ValueColumn
            icon={<AlertTriangle className="h-5 w-5" aria-hidden />}
            title={t('neverTitle')}
            items={[t('never1'), t('never2'), t('never3'), t('never4')]}
            tone="bad"
          />
          <ValueColumn
            icon={<ShieldCheck className="h-5 w-5" aria-hidden />}
            title={t('todayTitle')}
            items={[t('today1'), t('today2'), t('today3'), t('today4')]}
            tone="good"
          />
        </div>

        <div className="mt-8">
          <Link
            href="/methodology"
            className="inline-flex items-center gap-1.5 text-sm font-medium text-foreground underline-offset-4 hover:underline"
          >
            {t('cta')}
            <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
        </div>
      </div>
    </section>
  );
}

interface ValueColumnProps {
  icon: React.ReactNode;
  title: string;
  items: ReadonlyArray<string>;
  tone: 'bad' | 'good';
}

function ValueColumn({ icon, title, items, tone }: ValueColumnProps) {
  const iconClasses = {
    bad: 'text-sentinel-bear',
    good: 'text-sentinel-bull',
  } as const;
  const bulletClasses = {
    bad: 'bg-sentinel-bear',
    good: 'bg-sentinel-bull',
  } as const;

  return (
    <div>
      <h3 className="flex items-center gap-2 text-sm font-semibold tracking-tight">
        <span className={iconClasses[tone]}>{icon}</span>
        <span>{title}</span>
      </h3>
      <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
        {items.map((item) => (
          <li key={item} className="flex items-start gap-2">
            <span
              className={cn(
                'mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full',
                bulletClasses[tone],
              )}
              aria-hidden
            />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

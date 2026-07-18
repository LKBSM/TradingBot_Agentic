'use client';

import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { Check, Circle, CircleSlash } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { buildAppHref } from '@/lib/conditions/app-link';
import type { ComboMatch } from '@/lib/conditions/types';
import { biasGlyph, biasTone, instrumentLabel, toneTextClass } from './labels';
import { useScannerLabels } from './use-scanner-labels';

/**
 * One scan result, laid out as a full-width horizontal row:
 *   [ marché / TF ]  [ conditions remplies & non remplies ]  [ contexte + Analyser ]
 *
 * Shows which conditions are met AND which are unmet (transparency — "2 de tes
 * 3"), and the FULL context (including what goes against). No quality score,
 * no ranking. "Analyser" sends the user to look at the chart themselves.
 */
export function ComboCard({
  match,
  locale,
  now,
}: {
  match: ComboMatch;
  locale: string;
  /** Injected ticking clock (epoch ms) so the age stays honest; defaults to now. */
  now?: number;
}) {
  const t = useTranslations('scanner.combo');
  const { bias, phase, age } = useScannerLabels();
  const ctx = match.context;
  const readingAge = age(match.candle_close_ts, now ?? Date.now());

  // Heads-up factuel : on ne garde que les actus à IMPACT HAUT (les
  // moyennes/faibles sont du bruit). Aucune direction, aucune prédiction —
  // juste un signalement de volatilité programmée. Détection inchangée : on
  // filtre uniquement à l'affichage.
  const importantNewsCount = ctx.news_upcoming.filter(
    (n) => n.impact === 'high',
  ).length;

  return (
    <Card className="overflow-hidden">
      <div className="flex flex-col gap-4 p-4 md:flex-row md:items-stretch">
        {/* Left — market / timeframe */}
        <div className="flex flex-row items-center gap-3 md:w-44 md:shrink-0 md:flex-col md:items-start md:justify-center md:gap-1 md:border-r md:border-border/60 md:pr-4">
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold">{instrumentLabel(match.instrument)}</span>
            <Badge variant="outline">{match.timeframe}</Badge>
          </div>
          <div className="flex flex-col text-xs text-muted-foreground">
            <span>{t('conditionsPresent', { met: match.met_count, total: match.total })}</span>
            {match.close_price != null && <span>{t('price', { price: match.close_price })}</span>}
            {readingAge && <span>{t('reading', { age: readingAge })}</span>}
          </div>
        </div>

        {/* Middle — conditions met / unmet (transparent both ways) */}
        <ul className="min-w-0 flex-1 space-y-1.5">
          {match.conditions_met.map((c) => (
            <li key={`met-${c.type}`} className="flex items-start gap-2 text-sm">
              <Check className="mt-0.5 h-4 w-4 shrink-0 text-sentinel-bull" aria-hidden />
              <span>
                <span className="font-medium">{c.label}</span>
                <span className="text-muted-foreground"> — {c.detail}</span>
              </span>
            </li>
          ))}
          {match.conditions_unmet.map((c) => {
            // Data gap (a sibling timeframe not read yet) reads differently from
            // a genuine "not met" — never let the client mistake one for the other.
            const unavailable = c.available === false;
            return (
              <li key={`unmet-${c.type}`} className="flex items-start gap-2 text-sm">
                {unavailable ? (
                  <CircleSlash className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
                ) : (
                  <Circle className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
                )}
                <span className="text-muted-foreground">
                  <span className="font-medium">{c.label}</span>
                  {unavailable && (
                    <Badge variant="outline" className="mx-1 align-middle text-[10px]">
                      {t('unavailable')}
                    </Badge>
                  )}{' '}
                  — {c.detail}
                </span>
              </li>
            );
          })}
        </ul>

        {/* Right — full context + Analyser */}
        <div className="space-y-2 md:w-72 md:shrink-0 md:border-l md:border-border/60 md:pl-4">
          <p className="text-xs font-medium text-foreground">{t('fullContext')}</p>
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge variant={biasTone(ctx.trend)}>{t('trend', { label: bias(ctx.trend) })}</Badge>
            <Badge variant="neutral">{phase(ctx.market_phase)}</Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            MTF{' '}
            {(['h4', 'h1', 'm15'] as const).map((k) => {
              // Each timeframe's OWN trend (authoritative) — falls back to the
              // legacy mtf_confluence only for older payloads.
              const mtfTrend = ctx.mtf_trends?.[k] ?? ctx.mtf_confluence?.[k];
              return (
                <span key={k} title={`${k.toUpperCase()} ${bias(mtfTrend)}`}>
                  {k.toUpperCase()}
                  <span className={`mx-0.5 ${toneTextClass(mtfTrend)}`}>{biasGlyph(mtfTrend)}</span>
                </span>
              );
            })}
          </p>
          <p className="text-xs text-muted-foreground">
            {t('obFvg', { ob: ctx.active_order_blocks, fvg: ctx.active_fair_value_gaps })}
            {ctx.bos ? t('bosSuffix', { dir: bias(ctx.bos.direction) }) : ''}
          </p>
          {importantNewsCount > 0 && (
            <p className="text-xs text-sentinel-warn">
              {t('importantNews', { count: importantNewsCount })}
            </p>
          )}
          <Button asChild variant="outline" size="sm" className="mt-1 w-full">
            <Link href={buildAppHref(locale, match)}>{t('analyse')} →</Link>
          </Button>
        </div>
      </div>

      <p className="border-t border-border/60 px-4 py-2 text-[11px] leading-snug text-muted-foreground">
        {t('disclaimer')}
      </p>
    </Card>
  );
}

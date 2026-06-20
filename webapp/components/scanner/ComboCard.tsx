'use client';

import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { buildAppHref } from '@/lib/conditions/app-link';
import type { ComboMatch } from '@/lib/conditions/types';
import {
  biasGlyph,
  biasLabel,
  biasTone,
  instrumentLabel,
  phaseLabel,
  relativeAge,
  toneTextClass,
} from './labels';

/**
 * One scan result, laid out as a full-width horizontal row:
 *   [ marché / TF ]  [ conditions remplies & non remplies ]  [ contexte + Analyser ]
 *
 * Shows which conditions are met AND which are unmet (transparency — "2 de tes
 * 3"), and the FULL context (including what goes against). No quality score,
 * no ranking. "Analyser" sends the user to look at the chart themselves.
 */
export function ComboCard({ match, locale }: { match: ComboMatch; locale: string }) {
  const ctx = match.context;
  const age = relativeAge(match.candle_close_ts);

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
            <span>
              {match.met_count} de tes {match.total} condition{match.total > 1 ? 's' : ''} présente
              {match.met_count > 1 ? 's' : ''}
            </span>
            {match.close_price != null && <span>Prix {match.close_price}</span>}
            {age && <span>lecture {age}</span>}
          </div>
        </div>

        {/* Middle — conditions met / unmet (transparent both ways) */}
        <ul className="min-w-0 flex-1 space-y-1.5">
          {match.conditions_met.map((c) => (
            <li key={`met-${c.type}`} className="flex items-start gap-2 text-sm">
              <span aria-hidden className="mt-0.5 text-sentinel-bull">✓</span>
              <span>
                <span className="font-medium">{c.label}</span>
                <span className="text-muted-foreground"> — {c.detail}</span>
              </span>
            </li>
          ))}
          {match.conditions_unmet.map((c) => (
            <li key={`unmet-${c.type}`} className="flex items-start gap-2 text-sm">
              <span aria-hidden className="mt-0.5 text-muted-foreground">○</span>
              <span className="text-muted-foreground">
                <span className="font-medium">{c.label}</span> — {c.detail}
              </span>
            </li>
          ))}
        </ul>

        {/* Right — full context + Analyser */}
        <div className="space-y-2 md:w-72 md:shrink-0 md:border-l md:border-border/60 md:pl-4">
          <p className="text-xs font-medium text-foreground">Contexte complet</p>
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge variant={biasTone(ctx.trend)}>Tendance : {biasLabel(ctx.trend)}</Badge>
            <Badge variant="neutral">{phaseLabel(ctx.market_phase)}</Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            MTF{' '}
            {(['h4', 'h1', 'm15'] as const).map((k) => (
              <span key={k} title={`${k.toUpperCase()} ${biasLabel(ctx.mtf_confluence?.[k])}`}>
                {k.toUpperCase()}
                <span className={`mx-0.5 ${toneTextClass(ctx.mtf_confluence?.[k])}`}>
                  {biasGlyph(ctx.mtf_confluence?.[k])}
                </span>
              </span>
            ))}
          </p>
          <p className="text-xs text-muted-foreground">
            {ctx.active_order_blocks} OB · {ctx.active_fair_value_gaps} FVG actifs
            {ctx.bos ? ` · BOS ${biasLabel(ctx.bos.direction)}` : ''}
          </p>
          {ctx.news_upcoming.length > 0 && (
            <p className="text-xs text-sentinel-warn">
              {ctx.news_upcoming.length} actu(s) à venir — à garder en tête.
            </p>
          )}
          <Button asChild variant="outline" size="sm" className="mt-1 w-full">
            <Link href={buildAppHref(locale, match)}>Analyser →</Link>
          </Button>
        </div>
      </div>

      <p className="border-t border-border/60 px-4 py-2 text-[11px] leading-snug text-muted-foreground">
        Faits structurels observés au présent. Le scanner ne prédit rien et ne
        recommande rien — à toi le jugement.
      </p>
    </Card>
  );
}

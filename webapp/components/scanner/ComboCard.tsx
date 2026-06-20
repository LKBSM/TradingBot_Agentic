'use client';

import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
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
 * One scan result. Shows the market/timeframe, which conditions are met AND
 * which are unmet (transparency — "2 de tes 3"), and the FULL context (including
 * what goes against). No quality score, no ranking. The "Analyser" button sends
 * the user to look at the chart themselves.
 */
export function ComboCard({ match, locale }: { match: ComboMatch; locale: string }) {
  const ctx = match.context;
  const age = relativeAge(match.candle_close_ts);

  return (
    <Card className="overflow-hidden">
      <CardHeader className="gap-2 pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-base font-semibold">{instrumentLabel(match.instrument)}</span>
            <Badge variant="outline">{match.timeframe}</Badge>
          </div>
          <span className="text-xs text-muted-foreground">
            {match.met_count} de tes {match.total} condition{match.total > 1 ? 's' : ''} présente
            {match.met_count > 1 ? 's' : ''}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
          {match.close_price != null && <span>Prix {match.close_price}</span>}
          {age && <span>· lecture {age}</span>}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Conditions met / unmet — transparent both ways */}
        <ul className="space-y-1.5">
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

        {/* Full context — including what goes against */}
        <div className="rounded-lg border border-border/60 bg-muted/30 p-3 text-xs">
          <p className="mb-2 font-medium text-foreground">Contexte complet</p>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={biasTone(ctx.trend)}>Tendance : {biasLabel(ctx.trend)}</Badge>
            <Badge variant="neutral">Phase : {phaseLabel(ctx.market_phase)}</Badge>
            <span className="text-muted-foreground">
              MTF {' '}
              {(['h4', 'h1', 'm15'] as const).map((k) => (
                <span key={k} title={`${k.toUpperCase()} ${biasLabel(ctx.mtf_confluence?.[k])}`}>
                  {k.toUpperCase()}
                  <span className={`mx-0.5 ${toneTextClass(ctx.mtf_confluence?.[k])}`}>
                    {biasGlyph(ctx.mtf_confluence?.[k])}
                  </span>
                </span>
              ))}
            </span>
          </div>
          <p className="mt-2 text-muted-foreground">
            {ctx.active_order_blocks} OB actif{ctx.active_order_blocks > 1 ? 's' : ''} ·{' '}
            {ctx.active_fair_value_gaps} FVG non comblé
            {ctx.active_fair_value_gaps > 1 ? 's' : ''}
            {ctx.bos ? ` · BOS ${biasLabel(ctx.bos.direction)} (${ctx.bos.validation_status ?? '—'})` : ''}
          </p>
          {ctx.news_upcoming.length > 0 && (
            <p className="mt-1 text-sentinel-warn">
              {ctx.news_upcoming.length} actu(s) à venir — à garder en tête.
            </p>
          )}
        </div>

        <p className="text-[11px] leading-snug text-muted-foreground">
          Faits structurels observés au présent. Le scanner ne prédit rien et ne
          recommande rien — à toi le jugement.
        </p>

        <Button asChild variant="outline" size="sm">
          <Link href={buildAppHref(locale, match)}>Analyser →</Link>
        </Button>
      </CardContent>
    </Card>
  );
}

'use client';

import * as React from 'react';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  PHASE_OPTIONS,
  TREND_OPTIONS,
  VOLATILITY_OPTIONS,
  paletteEntry,
} from '@/lib/conditions/palette';
import type { ConditionsConfig, ConditionsScanResponse, ScanCondition } from '@/lib/conditions/types';
import { freshnessLabel } from '@/lib/conditions/candle-clock';
import { useNow } from '@/lib/conditions/use-now';
import { ComboCard } from './ComboCard';
import { AutoRefreshToggle } from './AutoRefreshToggle';
import { instrumentLabel } from './labels';

/** Re-render the freshness/age labels twice a minute (no re-fetch). */
const CLOCK_TICK_MS = 30_000;

/** Compact human description of a condition's chosen parameter, if any. */
function conditionParam(c: ScanCondition): string {
  if (c.trend) return TREND_OPTIONS.find((o) => o.value === c.trend)?.label ?? c.trend;
  if (c.phase) return PHASE_OPTIONS.find((o) => o.value === c.phase)?.label ?? c.phase;
  if (c.volatility)
    return VOLATILITY_OPTIONS.find((o) => o.value === c.volatility)?.label ?? c.volatility;
  if (c.direction && c.direction !== 'any')
    return c.direction === 'bullish' ? 'haussier' : 'baissier';
  return '';
}

/**
 * Results view. A neutral, descriptive list — never a ranking.
 *  · "Conditions présentes maintenant" = combos satisfying the full AND/OR
 *    logic ON A CURRENT reading (fresh/aging). A combo that satisfies it only
 *    on a STALE reading is held back into its own "lecture plus ancienne"
 *    section — we never assert an aged reading as "présent maintenant".
 *  · "Correspondances partielles" = combos meeting ≥1 condition (transparency).
 *  · Combos meeting nothing, and combos without a reading yet, are listed
 *    plainly so coverage is honest (no silent truncation).
 */
export function ScanResults({
  response,
  config,
  locale,
  onEdit,
  onRefresh,
  isRefreshing,
  autoRefreshEnabled,
  onToggleAutoRefresh,
}: {
  response: ConditionsScanResponse;
  config: ConditionsConfig;
  locale: string;
  onEdit(): void;
  onRefresh(): void;
  isRefreshing: boolean;
  autoRefreshEnabled: boolean;
  onToggleAutoRefresh(next: boolean): void;
}) {
  // A full match on a STALE reading (server-reported freshness) is held back
  // from the "maintenant" section so we never assert an aged reading as present.
  // Auto-refresh self-heals most staleness; this is the honest fallback when a
  // reading hasn't been regenerated yet (cold open, weekend, quiet market).
  const isStale = (m: (typeof response.matches)[number]) => m.freshness === 'stale';
  const full = response.matches.filter((m) => m.matched && !isStale(m));
  const staleFull = response.matches.filter((m) => m.matched && isStale(m));
  const partial = response.matches.filter((m) => !m.matched && m.met_count > 0);
  const none = response.matches.filter((m) => m.met_count === 0);

  // Ticking clock so "dernière analyse il y a X" / per-combo ages stay honest
  // while the page is open — without triggering any network request.
  const now = useNow(CLOCK_TICK_MS);
  const freshness = freshnessLabel(response.as_of, now);

  return (
    <div className="space-y-6">
      <header className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-lg font-semibold">Conditions présentes maintenant</h1>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" onClick={onRefresh} disabled={isRefreshing}>
              {isRefreshing ? 'Scan…' : 'Relancer le scan'}
            </Button>
            <Button size="sm" variant="ghost" onClick={onEdit}>
              Modifier mes conditions
            </Button>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 text-xs text-muted-foreground">
          <span aria-live="polite" data-testid="scan-freshness">
            {isRefreshing
              ? 'Analyse en cours…'
              : freshness
                ? `Dernière analyse : ${freshness}`
                : 'Dernière analyse : —'}
          </span>
          <AutoRefreshToggle enabled={autoRefreshEnabled} onChange={onToggleAutoRefresh} />
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span>Tes conditions ({config.logic === 'AND' ? 'toutes' : 'au moins une'})&nbsp;:</span>
          {config.conditions.map((c) => {
            const param = conditionParam(c);
            return (
              <Badge key={c.type} variant="secondary">
                {paletteEntry(c.type)?.label ?? c.type}
                {param ? ` · ${param}` : ''}
              </Badge>
            );
          })}
        </div>
      </header>

      {/* Full matches — on current readings only */}
      <section className="space-y-3">
        {full.length === 0 ? (
          staleFull.length === 0 && (
            <p className="rounded-lg border border-dashed border-border/70 p-4 text-sm text-muted-foreground">
              Aucun marché ne réunit actuellement{' '}
              {config.logic === 'AND' ? 'toutes tes conditions' : 'au moins une de tes conditions'}.
              Les correspondances partielles ci-dessous montrent ce qui est présent ailleurs.
            </p>
          )
        ) : (
          <div className="flex flex-col gap-4">
            {full.map((m) => (
              <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
            ))}
          </div>
        )}
      </section>

      {/* Full matches on a STALE reading — held back from "maintenant" */}
      {staleFull.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-sm font-semibold text-foreground">
            Sur une lecture plus ancienne — à rafraîchir
          </h2>
          <p className="text-xs text-muted-foreground">
            Ces marchés réunissent tes conditions, mais sur une lecture qui date
            (week-end ou marché calme). Le scan se rafraîchit à la prochaine
            clôture de bougie, ou relance-le pour la mettre à jour avant de
            t&apos;y fier.
          </p>
          <div className="flex flex-col gap-4 pt-1">
            {staleFull.map((m) => (
              <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
            ))}
          </div>
        </section>
      )}

      {/* Partial matches — transparency */}
      {partial.length > 0 && (
        <Accordion type="single" collapsible>
          <AccordionItem value="partial">
            <AccordionTrigger className="text-sm">
              Correspondances partielles ({partial.length})
            </AccordionTrigger>
            <AccordionContent>
              <div className="flex flex-col gap-4 pt-2">
                {partial.map((m) => (
                  <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {/* Honest coverage footer */}
      {(none.length > 0 || response.unavailable.length > 0) && (
        <footer className="space-y-1 border-t border-border/60 pt-4 text-xs text-muted-foreground">
          {none.length > 0 && (
            <p>
              Aucune de tes conditions présente :{' '}
              {none.map((m) => `${instrumentLabel(m.instrument)} ${m.timeframe}`).join(' · ')}.
            </p>
          )}
          {response.unavailable.length > 0 && (
            <p>
              Lecture pas encore générée :{' '}
              {response.unavailable
                .map((u) => `${instrumentLabel(u.instrument)} ${u.timeframe}`)
                .join(' · ')}
              .
            </p>
          )}
          <p>
            {response.scanned} combos analysés en lecture seule sur les dernières
            lectures déjà produites.
          </p>
        </footer>
      )}
    </div>
  );
}

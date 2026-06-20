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
import { paletteEntry } from '@/lib/conditions/palette';
import type { ConditionsConfig, ConditionsScanResponse } from '@/lib/conditions/types';
import { ComboCard } from './ComboCard';
import { instrumentLabel } from './labels';

/**
 * Results view. A neutral, descriptive list — never a ranking.
 *  · "Conditions présentes" = combos that satisfy the full AND/OR logic.
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
}: {
  response: ConditionsScanResponse;
  config: ConditionsConfig;
  locale: string;
  onEdit(): void;
  onRefresh(): void;
  isRefreshing: boolean;
}) {
  const full = response.matches.filter((m) => m.matched);
  const partial = response.matches.filter((m) => !m.matched && m.met_count > 0);
  const none = response.matches.filter((m) => m.met_count === 0);

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
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span>Tes conditions ({config.logic === 'AND' ? 'toutes' : 'au moins une'})&nbsp;:</span>
          {config.conditions.map((c) => (
            <Badge key={c.type} variant="secondary">
              {paletteEntry(c.type)?.label ?? c.type}
              {c.direction !== 'any' ? ` · ${c.direction === 'bullish' ? 'haussier' : 'baissier'}` : ''}
            </Badge>
          ))}
        </div>
      </header>

      {/* Full matches */}
      <section className="space-y-3">
        {full.length === 0 ? (
          <p className="rounded-lg border border-dashed border-border/70 p-4 text-sm text-muted-foreground">
            Aucun marché ne réunit actuellement{' '}
            {config.logic === 'AND' ? 'toutes tes conditions' : 'au moins une de tes conditions'}.
            Les correspondances partielles ci-dessous montrent ce qui est présent ailleurs.
          </p>
        ) : (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {full.map((m) => (
              <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} />
            ))}
          </div>
        )}
      </section>

      {/* Partial matches — transparency */}
      {partial.length > 0 && (
        <Accordion type="single" collapsible>
          <AccordionItem value="partial">
            <AccordionTrigger className="text-sm">
              Correspondances partielles ({partial.length})
            </AccordionTrigger>
            <AccordionContent>
              <div className="grid grid-cols-1 gap-4 pt-2 lg:grid-cols-2">
                {partial.map((m) => (
                  <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} />
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

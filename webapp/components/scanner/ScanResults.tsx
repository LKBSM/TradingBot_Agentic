'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import type { ConditionsConfig, ConditionsScanResponse, ScanCondition } from '@/lib/conditions/types';
import { paletteEntry } from '@/lib/conditions/palette';
import { useNow } from '@/lib/conditions/use-now';
import { ComboCard } from './ComboCard';
import { AutoRefreshToggle } from './AutoRefreshToggle';
import { instrumentLabel } from './labels';
import { useScannerLabels } from './use-scanner-labels';

/** Re-render the freshness/age labels twice a minute (no re-fetch). */
const CLOCK_TICK_MS = 30_000;

/** Capitalise a locale-agnostic enum value to build an ICU key suffix. */
function cap(v: string): string {
  return v.charAt(0).toUpperCase() + v.slice(1);
}

/**
 * Results view (UI-2 terminal reference). A neutral, descriptive grid — never a
 * ranking, never a score.
 *  · "Conditions présentes maintenant" = combos satisfying the full AND/OR
 *    logic ON A CURRENT reading (fresh/aging). A combo that satisfies it only
 *    on a STALE reading is held back into its own "lecture plus ancienne"
 *    section — we never assert an aged reading as "présent maintenant".
 *  · "Correspondances partielles" = combos meeting ≥1 condition (transparency).
 *  · Combos meeting nothing are listed plainly so coverage is honest.
 *  · Combos WITHOUT a fresh reading render as a degraded `.combo.na` card —
 *    honest "Non disponible", never fabricated.
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
  const t = useTranslations('scanner');
  const { age } = useScannerLabels();

  // Condition label: translated when a key exists, else the palette's own FR
  // label (covers conditions added after the i18n pass — deferred).
  const plabel = (type: ScanCondition['type']): string =>
    t.has(`palette.${type}_label`) ? t(`palette.${type}_label`) : paletteEntry(type)?.label ?? type;

  /** Compact human description of a condition's chosen parameter, if any. */
  const conditionParam = (c: ScanCondition): string => {
    if (c.trend) return t(`options.trend${cap(c.trend)}`);
    if (c.phase) return t(`options.phase${cap(c.phase)}`);
    if (c.volatility) return t(`options.volatility${cap(c.volatility)}`);
    if (c.direction && c.direction !== 'any') return t(`options.direction${cap(c.direction)}`);
    return '';
  };

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
  const freshness = age(response.as_of, now);

  // Compact results meta: "6 combos · il y a 3 min · aucun classement". Present
  // tense, honest — the "aucun classement" note makes the no-ranking promise
  // visible on screen.
  const resultsMeta = t('resultsMeta', {
    count: response.scanned,
    when: isRefreshing
      ? t('results.analysisInProgress')
      : freshness ?? t('results.lastAnalysisNone'),
  });

  return (
    <div className="pagewrap">
      {/* Page head — title, live badge, refresh */}
      <div className="pghead">
        <div>
          <h1>{t('page.title')}</h1>
          <div className="sub">{t('page.subtitle')}</div>
        </div>
        <span className="hsp" />
        <div className="livebadge">
          <span className="dot" />
          <span className="mono">{t('livebadge')}</span>
        </div>
        <button className="btn" onClick={onRefresh} disabled={isRefreshing}>
          {isRefreshing ? t('results.scanning') : t('results.rescan')}
        </button>
        <button className="btn" onClick={onEdit}>
          {t('editConditions')}
        </button>
        <AutoRefreshToggle enabled={autoRefreshEnabled} onChange={onToggleAutoRefresh} />
      </div>

      {/* Active-conditions palette (read-only reflection of the saved strategy) */}
      <div className="rail-lbl">{t('activeStrategy')}</div>
      <div className="condpal">
        <span className="cond on" data-logic={config.logic}>
          {config.logic === 'AND' ? t('results.yourConditionsAll') : t('results.yourConditionsAny')}
        </span>
        {config.conditions.map((c) => {
          const param = conditionParam(c);
          return (
            <span key={c.type} className="cond on">
              {plabel(c.type)}
              {param ? ` · ${param}` : ''}
            </span>
          );
        })}
      </div>
      <div className="scannote">
        <svg viewBox="0 0 24 24" aria-hidden>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 8v4M12 16h.01" />
        </svg>
        {t('note')}
      </div>

      {/* Results head — count · freshness · no ranking */}
      <div className="pghead">
        <h1 style={{ fontSize: '14px' }}>{t('results.title')}</h1>
        <span className="mono" data-testid="scan-freshness" aria-live="polite">
          {resultsMeta}
        </span>
      </div>

      {/* Full matches — on current readings only */}
      {full.length === 0 ? (
        staleFull.length === 0 &&
        response.unavailable.length === 0 && (
          <p className="scannote">
            {config.logic === 'AND' ? t('results.noMarketsAll') : t('results.noMarketsAny')}
          </p>
        )
      ) : (
        <div className="resgrid">
          {full.map((m) => (
            <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
          ))}
        </div>
      )}

      {/* Full matches on a STALE reading — held back from "maintenant" */}
      {staleFull.length > 0 && (
        <section className="mt-5">
          <div className="rail-lbl">{t('results.olderTitle')}</div>
          <p className="scannote">{t('results.olderBody')}</p>
          <div className="resgrid">
            {staleFull.map((m) => (
              <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
            ))}
          </div>
        </section>
      )}

      {/* Combos without a fresh reading — honest degraded cards, never fabricated */}
      {response.unavailable.length > 0 && (
        <div className="resgrid mt-3">
          {response.unavailable.map((u) => (
            <div key={`${u.instrument}:${u.timeframe}`} className="combo na">
              <div className="t1">
                <span className="nm">{instrumentLabel(u.instrument)}</span>
                <span className="tf2">· {u.timeframe}</span>
              </div>
              <div className="natag">{t('unavailable')}</div>
            </div>
          ))}
        </div>
      )}

      {/* Partial matches — transparency */}
      {partial.length > 0 && (
        <Accordion type="single" collapsible className="mt-4">
          <AccordionItem value="partial">
            <AccordionTrigger className="text-sm">
              {t('results.partialMatches', { count: partial.length })}
            </AccordionTrigger>
            <AccordionContent>
              <div className="resgrid pt-2">
                {partial.map((m) => (
                  <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {/* Honest coverage footer */}
      {none.length > 0 && (
        <footer className="mt-4 border-t border-[color:var(--line)] pt-4 text-[11px] text-[color:var(--faint)]">
          <p>
            {t('results.nonePresent', {
              list: none.map((m) => `${instrumentLabel(m.instrument)} ${m.timeframe}`).join(' · '),
            })}
          </p>
          <p>{t('results.scannedFooter', { count: response.scanned })}</p>
        </footer>
      )}
    </div>
  );
}

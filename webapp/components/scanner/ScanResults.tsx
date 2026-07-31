'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import type { ConditionsConfig, ConditionsScanResponse, ScanCondition } from '@/lib/conditions/types';
import { paletteEntry } from '@/lib/conditions/palette';
import { useNow } from '@/lib/conditions/use-now';
import { cn } from '@/lib/utils';
import { ComboCard } from './ComboCard';
import { AutoRefreshToggle } from './AutoRefreshToggle';
import { instrumentLabel } from './labels';
import { useScannerLabels } from './use-scanner-labels';

const CLOCK_TICK_MS = 30_000;

type FilterKey = 'matches' | 'almost' | 'nonmatch' | 'noneval';

/**
 * Results view (SC-1). A neutral, descriptive grid — never a ranking, never a
 * score, never sorted by how many conditions a combo meets. Order is fixed
 * (market, then timeframe, as scanned).
 *
 * DISPLAY filters (Correspondances / Presque / Non correspondants / Non
 * évaluables) only SHOW or HIDE groups — no data is removed and each group
 * shows its own count. When nothing matches, an explicit "ce n'est pas une
 * erreur" state lists, per condition, how many combos meet it in ISOLATION
 * (counts that are independent and do not add up).
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

  const plabel = (type: ScanCondition['type']): string =>
    t.has(`palette.${type}_label`) ? t(`palette.${type}_label`) : paletteEntry(type)?.label ?? type;

  const isStale = (m: (typeof response.matches)[number]) => m.freshness === 'stale';
  const allStale = response.matches.length > 0 && response.matches.every(isStale);

  // Groups. "Non évaluables" = combos with no fresh reading (unavailable) OR a
  // combo whose every condition was non-evaluable (denominator 0).
  const full = response.matches.filter((m) => m.matched && !isStale(m));
  const staleFull = response.matches.filter((m) => m.matched && isStale(m));
  const almost = response.matches.filter((m) => !m.matched && m.met_count > 0);
  const nonmatch = response.matches.filter((m) => !m.matched && m.met_count === 0 && m.total > 0);
  const nonEvalCombos = response.matches.filter((m) => m.total === 0);

  const counts: Record<FilterKey, number> = {
    matches: full.length + staleFull.length,
    almost: almost.length,
    nonmatch: nonmatch.length,
    noneval: nonEvalCombos.length + response.unavailable.length,
  };

  const [active, setActive] = React.useState<Set<FilterKey>>(
    () => new Set<FilterKey>(['matches', 'almost', 'nonmatch', 'noneval']),
  );
  const show = (k: FilterKey) => active.has(k);
  const toggle = (k: FilterKey) =>
    setActive((prev) => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });

  const now = useNow(CLOCK_TICK_MS);
  const freshness = age(response.as_of, now);
  const resultsMeta = t('resultsMeta', {
    count: response.scanned,
    when: isRefreshing ? t('results.analysisInProgress') : freshness ?? t('results.lastAnalysisNone'),
  });

  // "Aucun combo" = no full match at all (fresh or stale). Per-condition ISOLATED
  // counts: how many combos meet each condition on its own. Independent counts —
  // they do not add up and do not describe any partial combination.
  const noCombo = full.length === 0 && staleFull.length === 0;
  const isolated = config.conditions.map((c) => ({
    type: c.type,
    count: response.matches.filter((m) => m.conditions_met.some((x) => x.type === c.type)).length,
  }));

  const FILTERS: Array<{ key: FilterKey; label: string }> = [
    { key: 'matches', label: t('filters.matches', { count: counts.matches }) },
    { key: 'almost', label: t('filters.almost', { count: counts.almost }) },
    { key: 'nonmatch', label: t('filters.nonmatch', { count: counts.nonmatch }) },
    { key: 'noneval', label: t('filters.noneval', { count: counts.noneval }) },
  ];

  return (
    <div className="pagewrap">
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
        <button className="btn" onClick={onRefresh} disabled={isRefreshing} title={allStale ? t('noNewClose') : undefined}>
          {isRefreshing ? t('results.scanning') : t('results.rescan')}
        </button>
        <button className="btn" onClick={onEdit}>{t('editConditions')}</button>
        <AutoRefreshToggle enabled={autoRefreshEnabled} onChange={onToggleAutoRefresh} />
      </div>

      {allStale && (
        <div className="sub" role="status" data-testid="scan-no-new-close">{t('noNewClose')}</div>
      )}

      {/* Active reading (read-only reflection of the saved conditions) */}
      <div className="rail-lbl">{t('activeStrategy')}</div>
      <div className="condpal">
        <span className="cond on" data-logic={config.logic}>
          {config.logic === 'AND' ? t('results.yourConditionsAll') : t('results.yourConditionsAny')}
        </span>
        {config.conditions.map((c) => (
          <span key={c.type} className="cond on">{plabel(c.type)}</span>
        ))}
      </div>
      <div className="scannote">{t('note')}</div>

      {/* Results head + no-ranking meta */}
      <div className="pghead">
        <h1 style={{ fontSize: '14px' }}>{t('results.title')}</h1>
        <span className="mono" data-testid="scan-freshness" aria-live="polite">{resultsMeta}</span>
      </div>

      {/* DISPLAY filters — show/hide only, each with its count, nothing removed */}
      <div className="flex flex-wrap gap-2" role="group" aria-label={t('filters.aria')}>
        {FILTERS.map((f) => (
          <button
            key={f.key}
            type="button"
            aria-pressed={show(f.key)}
            onClick={() => toggle(f.key)}
            className={cn(
              'rounded-full border px-3 py-1 text-xs font-medium transition-colors',
              show(f.key) ? 'border-foreground/40 bg-foreground/10 text-foreground' : 'border-border/60 text-muted-foreground',
            )}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* No-combo state — explicitly NOT an error; isolated per-condition counts */}
      {noCombo && (
        <section className="mt-3 rounded-lg border border-border/60 p-4" data-testid="scan-no-combo">
          <p className="text-sm text-foreground">{t('results.noComboTitle')}</p>
          <p className="mt-1 text-xs text-muted-foreground">{t('results.noComboBody')}</p>
          <div className="mt-3 space-y-1">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">{t('results.isolatedIntro')}</p>
            <ul className="space-y-1">
              {isolated.map((iso) => (
                <li key={iso.type} className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-foreground">{plabel(iso.type)}</span>
                  <span className="mono text-xs text-muted-foreground">
                    {t('results.isolatedCount', { count: iso.count })}
                  </span>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-muted-foreground">{t('results.isolatedNote')}</p>
          </div>
          {/* Product line: never offer to loosen a condition to get results. */}
          <p className="mt-3 rounded-md border-l-2 border-amber-500/60 bg-amber-500/5 p-3 text-xs text-muted-foreground">
            {t('results.noLoosening')}
          </p>
        </section>
      )}

      {/* Matches (fresh) */}
      {show('matches') && full.length > 0 && (
        <div className="resgrid">
          {full.map((m) => (
            <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
          ))}
        </div>
      )}

      {/* Matches on a stale reading — held back from "maintenant" */}
      {show('matches') && staleFull.length > 0 && (
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

      {/* Almost (partial) */}
      {show('almost') && almost.length > 0 && (
        <section className="mt-5">
          <div className="rail-lbl">{t('filters.almost', { count: almost.length })}</div>
          <div className="resgrid">
            {almost.map((m) => (
              <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
            ))}
          </div>
        </section>
      )}

      {/* Non-matching */}
      {show('nonmatch') && nonmatch.length > 0 && (
        <section className="mt-5">
          <div className="rail-lbl">{t('filters.nonmatch', { count: nonmatch.length })}</div>
          <div className="resgrid">
            {nonmatch.map((m) => (
              <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
            ))}
          </div>
        </section>
      )}

      {/* Non-evaluable — no fresh reading, or every condition non-evaluable */}
      {show('noneval') && counts.noneval > 0 && (
        <section className="mt-5">
          <div className="rail-lbl">{t('filters.noneval', { count: counts.noneval })}</div>
          <div className="resgrid">
            {nonEvalCombos.map((m) => (
              <ComboCard key={`${m.instrument}:${m.timeframe}`} match={m} locale={locale} now={now} />
            ))}
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
        </section>
      )}

      <footer className="mt-4 border-t border-[color:var(--line)] pt-4 text-[11px] text-[color:var(--faint)]">
        <p>{t('results.scannedFooter', { count: response.scanned })}</p>
      </footer>
    </div>
  );
}

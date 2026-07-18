'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { Pin, PinOff, Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  formatInstrument,
  formatRelativePast,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
import { usePinnedCombos } from '@/lib/market-reading/pins';
import {
  comboKey,
  sameCombo,
  SUPPORTED_INSTRUMENTS,
  SUPPORTED_TIMEFRAMES,
  type Combo,
} from '@/lib/market-reading/store';

interface InstrumentSidebarProps {
  combos: readonly Combo[];
  active: Combo | null;
  onSelect(combo: Combo): void;
  /** ISO candle-close timestamp of the currently loaded reading, if any. */
  activeCandleCloseTs?: string | null;
}

/** Accent/case-insensitive normalisation for the search filter. */
function normalize(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .trim();
}

/** A combo matches the query if it appears in its code or its human label. */
function comboMatches(combo: Combo, query: string): boolean {
  const q = normalize(query);
  if (!q) return true;
  const haystack = normalize(
    `${combo.instrument} ${formatInstrument(combo.instrument)} ${combo.timeframe} ${formatTimeframe(combo.timeframe)}`,
  );
  return haystack.includes(q);
}

/**
 * Left column — the V1 perimeter (XAUUSD/EURUSD × M15/H1/H4). Adds a search
 * filter (restricted to the fixed catalogue — never queries anything external)
 * and a pin feature: pinned combos float to a quick-access section at the top,
 * persisted locally (localStorage). The active combo is highlighted with a gold
 * accent bar and a freshness indicator derived from the loaded reading.
 *
 * Note: only the active combo's reading is fetched, so the freshness marker is
 * shown on the active item only (a full 6-combo freshness grid would mean six
 * background fetches — deferred).
 */
export function InstrumentSidebar({
  active,
  onSelect,
  activeCandleCloseTs,
}: InstrumentSidebarProps) {
  const t = useTranslations('app');
  const [query, setQuery] = React.useState('');
  const { pinned, isPinned, toggle } = usePinnedCombos();

  const matchingPinned = pinned.filter((c) => comboMatches(c, query));

  const groups = SUPPORTED_INSTRUMENTS.map((instrument) => ({
    instrument,
    timeframes: SUPPORTED_TIMEFRAMES.filter((timeframe) =>
      comboMatches({ instrument, timeframe }, query),
    ),
  })).filter((g) => g.timeframes.length > 0);

  const hasResults = matchingPinned.length > 0 || groups.length > 0;

  return (
    <nav aria-label={t('sidebar.navAria')} className="space-y-4">
      <div>
        <p className="px-1 pb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {t('sidebar.markets')}
        </p>
        <div className="relative">
          <Search
            className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
            aria-hidden
          />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t('sidebar.searchPlaceholder')}
            aria-label={t('sidebar.searchAria')}
            className="w-full rounded-md border border-border/60 bg-background py-2 pl-8 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-[#c9a961] focus:outline-none focus:ring-1 focus:ring-[#c9a961]"
          />
        </div>
      </div>

      {matchingPinned.length > 0 && (
        <div className="space-y-1.5">
          <p className="px-1 text-xs font-semibold uppercase tracking-wide text-[#c9a961]">
            {t('sidebar.pinned')}
          </p>
          <ul className="space-y-1">
            {matchingPinned.map((combo) => (
              <li key={`pin:${comboKey(combo)}`}>
                <ComboRow
                  combo={combo}
                  active={active}
                  onSelect={onSelect}
                  pinned
                  onTogglePin={toggle}
                  activeCandleCloseTs={activeCandleCloseTs}
                  showInstrument
                />
              </li>
            ))}
          </ul>
        </div>
      )}

      {groups.map(({ instrument, timeframes }) => (
        <div key={instrument} className="space-y-1.5">
          <p className="px-1 text-sm font-semibold text-foreground">
            {formatInstrument(instrument)}
          </p>
          <ul className="space-y-1">
            {timeframes.map((timeframe) => {
              const combo: Combo = { instrument, timeframe };
              return (
                <li key={comboKey(combo)}>
                  <ComboRow
                    combo={combo}
                    active={active}
                    onSelect={onSelect}
                    pinned={isPinned(combo)}
                    onTogglePin={toggle}
                    activeCandleCloseTs={activeCandleCloseTs}
                  />
                </li>
              );
            })}
          </ul>
        </div>
      ))}

      {!hasResults && (
        <p className="px-1 py-6 text-center text-sm text-muted-foreground">
          {t('sidebar.noResults', { query })}
        </p>
      )}
    </nav>
  );
}

/**
 * One selectable combo row: a select button (active highlight + freshness) and a
 * pin toggle. `showInstrument` adds the instrument name (used in the flat
 * "Épinglés" list, where rows aren't grouped under an instrument heading).
 */
function ComboRow({
  combo,
  active,
  onSelect,
  pinned,
  onTogglePin,
  activeCandleCloseTs,
  showInstrument = false,
}: {
  combo: Combo;
  active: Combo | null;
  onSelect(combo: Combo): void;
  pinned: boolean;
  onTogglePin(combo: Combo): void;
  activeCandleCloseTs?: string | null;
  showInstrument?: boolean;
}) {
  const t = useTranslations('app');
  const isActive = sameCombo(active, combo);
  const label = showInstrument
    ? `${formatInstrument(combo.instrument)} · ${formatTimeframe(combo.timeframe)}`
    : formatTimeframe(combo.timeframe);

  return (
    <div
      className={cn(
        'flex items-stretch rounded-md border-l-2 transition-colors',
        isActive
          ? 'border-l-[#c9a961] bg-[#c9a961]/10'
          : 'border-l-transparent hover:bg-muted',
      )}
    >
      <button
        type="button"
        onClick={() => onSelect(combo)}
        aria-current={isActive ? 'true' : undefined}
        className={cn(
          'flex min-w-0 flex-1 items-center justify-between gap-2 px-3 py-2 text-left text-sm',
          isActive
            ? 'font-medium text-foreground'
            : 'text-muted-foreground hover:text-foreground',
        )}
      >
        <span className="truncate">{label}</span>
        {isActive && <Freshness candleCloseTs={activeCandleCloseTs ?? null} />}
      </button>
      <button
        type="button"
        onClick={() => onTogglePin(combo)}
        aria-pressed={pinned}
        aria-label={
          pinned
            ? t('sidebar.unpinAria', {
                combo: `${formatInstrument(combo.instrument)} ${formatTimeframe(combo.timeframe)}`,
              })
            : t('sidebar.pinAria', {
                combo: `${formatInstrument(combo.instrument)} ${formatTimeframe(combo.timeframe)}`,
              })
        }
        title={pinned ? t('sidebar.unpin') : t('sidebar.pin')}
        className={cn(
          'flex shrink-0 items-center px-2 text-muted-foreground/60 transition-colors hover:text-[#c9a961]',
          pinned && 'text-[#c9a961]',
        )}
      >
        {pinned ? (
          <PinOff className="h-3.5 w-3.5" aria-hidden />
        ) : (
          <Pin className="h-3.5 w-3.5" aria-hidden />
        )}
      </button>
    </div>
  );
}

/**
 * Freshness marker — a gold dot plus the candle-close relative age. Mount-
 * guarded so the relative time (clock-dependent) doesn't cause a hydration
 * mismatch; renders just the dot until mounted / until data arrives.
 */
function Freshness({ candleCloseTs }: { candleCloseTs: string | null }) {
  const [now, setNow] = React.useState<Date | null>(null);

  React.useEffect(() => {
    setNow(new Date());
    const id = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground">
      <span className="h-1.5 w-1.5 rounded-full bg-[#c9a961]" aria-hidden />
      {now && candleCloseTs ? formatRelativePast(candleCloseTs, now) : null}
    </span>
  );
}

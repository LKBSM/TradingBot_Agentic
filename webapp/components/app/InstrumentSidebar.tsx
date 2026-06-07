'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';
import {
  formatInstrument,
  formatRelativePast,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
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

/**
 * Left column — the V1 perimeter (XAUUSD/EURUSD × M15/H1/H4), grouped by
 * instrument. The active combo is highlighted with a gold accent bar and a
 * freshness indicator derived from the loaded reading's candle close.
 *
 * Note: only the active combo's reading is fetched, so the freshness marker is
 * shown on the active item only (a full 6-combo freshness grid would mean six
 * background fetches — deferred).
 */
export function InstrumentSidebar({
  combos,
  active,
  onSelect,
  activeCandleCloseTs,
}: InstrumentSidebarProps) {
  return (
    <nav aria-label="Combinaisons disponibles" className="space-y-5">
      <p className="px-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
        Marchés
      </p>
      {SUPPORTED_INSTRUMENTS.map((instrument) => (
        <div key={instrument} className="space-y-1.5">
          <p className="px-1 text-sm font-semibold text-foreground">
            {formatInstrument(instrument)}
          </p>
          <ul className="space-y-1">
            {SUPPORTED_TIMEFRAMES.map((timeframe) => {
              const combo: Combo = { instrument, timeframe };
              const isActive = sameCombo(active, combo);
              return (
                <li key={comboKey(combo)}>
                  <button
                    type="button"
                    onClick={() => onSelect(combo)}
                    aria-current={isActive ? 'true' : undefined}
                    className={cn(
                      'flex w-full items-center justify-between rounded-md border-l-2 px-3 py-2 text-left text-sm transition-colors',
                      isActive
                        ? 'border-l-[#c9a961] bg-[#c9a961]/10 font-medium text-foreground'
                        : 'border-l-transparent text-muted-foreground hover:bg-muted hover:text-foreground',
                    )}
                  >
                    <span>{formatTimeframe(timeframe)}</span>
                    {isActive && (
                      <Freshness candleCloseTs={activeCandleCloseTs ?? null} />
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
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
      <span
        className="h-1.5 w-1.5 rounded-full bg-[#c9a961]"
        aria-hidden
      />
      {now && candleCloseTs ? formatRelativePast(candleCloseTs, now) : null}
    </span>
  );
}

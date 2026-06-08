import { TemporalBadge } from './TemporalBadge';
import {
  formatInstrument,
  formatPrice,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
import type { MarketReadingHeader as MarketReadingHeaderData } from '@/types/market-reading';

/**
 * Header of a market reading — instrument + timeframe, last close price, and
 * the candle-close timestamp. Purely factual: no verdict, no direction badge
 * (that lives in MarketPhasePanel as a descriptive trend label).
 */
export function MarketReadingHeader({
  header,
}: {
  header: MarketReadingHeaderData;
}) {
  return (
    <header className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 className="text-balance text-xl font-semibold leading-tight tracking-tight sm:text-2xl">
          {formatInstrument(header.instrument)}
        </h2>
        <span className="font-mono text-lg font-semibold tabular-nums text-foreground sm:text-xl">
          {formatPrice(header.close_price, header.instrument)}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-xs font-medium text-muted-foreground">
          {header.instrument} · {formatTimeframe(header.timeframe)}
        </span>
        <TemporalBadge candleCloseTs={header.candle_close_ts} />
      </div>
    </header>
  );
}

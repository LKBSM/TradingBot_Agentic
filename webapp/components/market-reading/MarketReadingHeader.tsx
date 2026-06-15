import { LivePrice } from './LivePrice';
import { TemporalBadge } from './TemporalBadge';
import {
  formatInstrument,
  formatPrice,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
import type { DailyChange } from '@/lib/market-reading/price';
import type { MarketReadingHeader as MarketReadingHeaderData } from '@/types/market-reading';

/**
 * Header of a market reading — instrument + timeframe, last price, and the
 * candle-close timestamp. Purely factual: no verdict, no direction badge (that
 * lives in MarketPhasePanel as a descriptive trend label).
 *
 * Price source: when a unified `live` price is supplied (the M15 freshest close,
 * identical across timeframes — see useLatestPrice), it is shown with the
 * descriptive daily % change. Otherwise we fall back to the per-timeframe
 * `header.close_price` (landing samples, or when the candle feed is down).
 */
export function MarketReadingHeader({
  header,
  live,
}: {
  header: MarketReadingHeaderData;
  /** Unified last price + daily change. Omitted on static/landing surfaces. */
  live?: DailyChange | null;
}) {
  return (
    <header className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 className="text-balance text-[15px] font-medium leading-tight tracking-tight">
          {formatInstrument(header.instrument)}
        </h2>
        {live ? (
          <LivePrice
            instrument={header.instrument}
            price={live.price}
            changePct={live.changePct}
          />
        ) : (
          <span className="font-mono text-lg font-medium tabular-nums text-foreground">
            {formatPrice(header.close_price, header.instrument)}
          </span>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-mono text-[11px] font-normal tabular-nums text-muted-foreground">
          {header.instrument} · {formatTimeframe(header.timeframe)}
        </span>
        <TemporalBadge candleCloseTs={header.candle_close_ts} />
      </div>
    </header>
  );
}

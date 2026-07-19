import { useTranslations } from 'next-intl';
import { LivePrice } from './LivePrice';
import { TemporalBadge } from './TemporalBadge';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
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
  marketClosed = false,
}: {
  header: MarketReadingHeaderData;
  /** Unified last price + daily change. Omitted on static/landing surfaces. */
  live?: DailyChange | null;
  /**
   * Descriptive session state — true when the spot market is closed. Shows a
   * neutral "Marché fermé" badge next to the price. A present fact, no forecast.
   */
  marketClosed?: boolean;
}) {
  const fmt = useReadingFormatters();
  const t = useTranslations('app');
  return (
    <header className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 className="text-balance text-[15px] font-medium leading-tight tracking-tight">
          {fmt.instrument(header.instrument)}
        </h2>
        <div className="flex items-center gap-2">
          {marketClosed && (
            <span
              className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-muted/70 px-2 py-0.5 text-[10px] font-semibold text-muted-foreground"
              title={t('chart.marketClosedTitle')}
              role="status"
            >
              <span
                className="inline-block h-1.5 w-1.5 rounded-full bg-muted-foreground/70"
                aria-hidden
              />
              {t('chart.marketClosed')}
            </span>
          )}
          {live ? (
            <LivePrice
              instrument={header.instrument}
              price={live.price}
              changePct={live.changePct}
            />
          ) : (
            <span className="font-mono text-lg font-medium tabular-nums text-foreground">
              {fmt.price(header.close_price, header.instrument)}
            </span>
          )}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-mono text-[11px] font-normal tabular-nums text-muted-foreground">
          {header.instrument} · {fmt.timeframe(header.timeframe)}
        </span>
        <TemporalBadge candleCloseTs={header.candle_close_ts} />
      </div>
    </header>
  );
}

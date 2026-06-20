import { DEFAULT_LOCALE } from '@/i18n';
import { SUPPORTED_INSTRUMENTS, SUPPORTED_TIMEFRAMES } from '@/lib/market-reading/perimeter';

/**
 * Deep-link from a scan match to the /app reading view for the same combo.
 * The "Analyser" button uses this — it navigates the user to GO LOOK at the
 * market/timeframe themselves (never "Trader").
 *
 * `localePrefix: 'as-needed'` (see middleware): the default locale is served
 * WITHOUT a prefix, so we omit it to avoid a needless 302 on every click.
 */
export function buildAppHref(
  locale: string,
  combo: { instrument: string; timeframe: string },
): string {
  const params = new URLSearchParams({
    instrument: combo.instrument,
    timeframe: combo.timeframe,
  });
  const prefix = locale === DEFAULT_LOCALE ? '' : `/${locale}`;
  return `${prefix}/app?${params.toString()}`;
}

/**
 * Resolve an (instrument, timeframe) pair from URL query into a valid Combo,
 * or null when absent/out-of-perimeter. Used by the /app page to honour the
 * deep-link without ever trusting arbitrary query values.
 */
export function resolveComboFromQuery(
  instrument: string | undefined,
  timeframe: string | undefined,
): { instrument: string; timeframe: string } | null {
  if (!instrument || !timeframe) return null;
  const okInstrument = (SUPPORTED_INSTRUMENTS as readonly string[]).includes(instrument);
  const okTimeframe = (SUPPORTED_TIMEFRAMES as readonly string[]).includes(timeframe);
  if (!okInstrument || !okTimeframe) return null;
  return { instrument, timeframe };
}

'use client';

import { useTranslations } from 'next-intl';

/**
 * Locale-aware market display name — « Gold (XAU/USD) » (en) / « Or (XAU/USD) »
 * (fr) / « Oro (XAU/USD) » (es) — read from the SAME
 * `reading.labels.instrument_<code>` keys the reading HEADER uses. Routing the
 * sidebar, chat and mobile header through this hook makes the market's name
 * identical to the header in every language (I18N-1): the FR-baseline
 * `formatInstrument` used to leak « Or » on the English locale because it read a
 * single frozen French label from the registry.
 *
 * The instrument CODE (XAUUSD, EURUSD…) is NEVER translated — it is the key
 * suffix, not a value. An unknown code (e.g. the fictional TESTMKT, which has no
 * `instrument_*` key) falls back to the raw code, exactly like the header does.
 */
export function useInstrumentLabel(): (code: string) => string {
  const t = useTranslations('reading');
  return (code: string) => {
    const key = `labels.instrument_${code}`;
    return t.has(key) ? t(key) : code;
  };
}

'use client';

import * as React from 'react';
import Link from 'next/link';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useTranslations } from 'next-intl';
import { CalendarClock, CandlestickChart, Layers, Radar, Settings } from 'lucide-react';
import { cn } from '@/lib/utils';
import { MiaLogo } from '@/components/brand/MiaLogo';
import { useLocalizedHref } from '@/lib/i18n/href';
import { resolveComboFromQuery } from '@/lib/conditions/app-link';
import { formatInstrument, formatTimeframe } from '@/lib/market-reading/formatters';
import {
  DEFAULT_INSTRUMENT,
  DEFAULT_TIMEFRAME,
  DISPLAY_TIMEFRAMES,
  SUPPORTED_INSTRUMENTS,
  type Combo,
} from '@/lib/market-reading/store';
import { SearchField, Freshbox } from './primitives';

const DEFAULT_COMBO: Combo = { instrument: DEFAULT_INSTRUMENT, timeframe: DEFAULT_TIMEFRAME };

/** Two-glyph mono badge shown in the market row (reference `.mkt .ic`). */
const MARKET_GLYPH: Record<string, string> = { XAUUSD: 'Au', EURUSD: '€' };
function marketGlyph(instrument: string): string {
  return MARKET_GLYPH[instrument] ?? instrument.slice(0, 2);
}

/** Accent/case-insensitive match for the market search box. */
function normalize(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .trim();
}

interface SpaceLink {
  key: string;
  href: string;
  label: string;
  Icon: typeof CandlestickChart;
}

/**
 * The product-shell rail (left column, all product routes). Four stacked
 * sections from the validated reference: market search, MARCHÉS (instrument
 * selector), UNITÉ DE TEMPS (timeframe pills), ESPACE (route nav), then a
 * Freshbox + educational microcopy pinned to the bottom.
 *
 * The active combo is read from — and written to — the URL (`?instrument=&
 * timeframe=`), which /app already treats as the single source of truth. So the
 * rail never holds combo state of its own: picking a market or timeframe simply
 * navigates to /app with that query, and /app reflects it. Detection/data are
 * never touched here; this is navigation + presentation only.
 */
export function ShellRail({ activeSpace }: { activeSpace: string }) {
  const t = useTranslations();
  const lh = useLocalizedHref();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [query, setQuery] = React.useState('');

  const active =
    resolveComboFromQuery(
      searchParams.get('instrument') ?? undefined,
      searchParams.get('timeframe') ?? undefined,
    ) ?? DEFAULT_COMBO;

  const onApp = activeSpace === 'app';

  // Navigate to /app with the chosen combo. Already on /app → replace (no history
  // spam, no scroll jump), mirroring AppWorkspace.handleSelect; elsewhere → push.
  const goToCombo = React.useCallback(
    (combo: Combo) => {
      const href = lh(
        `/app?instrument=${combo.instrument}&timeframe=${combo.timeframe}`,
      );
      if (onApp) router.replace(href, { scroll: false });
      else router.push(href);
    },
    [lh, onApp, router],
  );

  const markets = SUPPORTED_INSTRUMENTS.filter((instrument) => {
    const q = normalize(query);
    if (!q) return true;
    return normalize(`${instrument} ${formatInstrument(instrument)}`).includes(q);
  });

  const spaces: SpaceLink[] = [
    { key: 'app', href: lh('/app'), label: 'App', Icon: CandlestickChart },
    // SC-2e: « Scanner » ouvre le mode « Décrire » par défaut ; la palette de
    // conditions reste à un clic via la bascule en tête de page. `activeSpace`
    // reste 'scanner' pour /scanner ET /scanner/decrire (1er segment de route).
    { key: 'scanner', href: lh('/scanner/decrire'), label: t('nav.scanner'), Icon: Radar },
    { key: 'zones', href: lh('/zones'), label: t('nav.zones'), Icon: Layers },
    { key: 'actualites', href: lh('/actualites'), label: t('nav.calendar'), Icon: CalendarClock },
    { key: 'compte', href: lh('/compte'), label: t('nav.account'), Icon: Settings },
  ];

  return (
    <aside className="rail" aria-label={t('app.sidebar.navAria')}>
      {/* Connected-surface brand — the horizontal lockup, home link (BRD-2). */}
      <Link href={lh('/')} className="rail-brand" aria-label={t('nav.brandHomeAria')}>
        <MiaLogo variant="horizontal" height={22} decorative />
      </Link>
      <SearchField
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={t('app.sidebar.searchPlaceholder')}
        aria-label={t('app.sidebar.searchAria')}
      />

      {/* MARCHÉS */}
      <div>
        <div className="rail-lbl">{t('app.sidebar.markets')}</div>
        {markets.map((instrument) => {
          const isActive = onApp && active.instrument === instrument;
          return (
            <button
              key={instrument}
              type="button"
              className={cn('mkt', isActive && 'on')}
              aria-current={isActive ? 'true' : undefined}
              onClick={() => goToCombo({ instrument, timeframe: active.timeframe })}
            >
              <span className="ic mono" aria-hidden>
                {marketGlyph(instrument)}
              </span>
              <span className="nm">{formatInstrument(instrument)}</span>
            </button>
          );
        })}
      </div>

      {/* UNITÉ DE TEMPS */}
      <div>
        <div className="rail-lbl">{t('app.rail.timeframe')}</div>
        <div style={{ display: 'flex', gap: 5, padding: '0 4px' }}>
          {DISPLAY_TIMEFRAMES.map((timeframe) => {
            const isActive = onApp && active.timeframe === timeframe;
            return (
              <button
                key={timeframe}
                type="button"
                className={cn('tf', isActive && 'on')}
                aria-pressed={isActive}
                aria-label={formatTimeframe(timeframe)}
                onClick={() => goToCombo({ instrument: active.instrument, timeframe })}
              >
                {timeframe}
              </button>
            );
          })}
        </div>
      </div>

      {/* ESPACE (route nav) */}
      <div>
        <div className="rail-lbl">{t('app.rail.space')}</div>
        {spaces.map(({ key, href, label, Icon }) => {
          const isActive = activeSpace === key;
          return (
            <Link
              key={key}
              href={href}
              className={cn('nl', isActive && 'on')}
              aria-current={isActive ? 'page' : undefined}
            >
              <Icon aria-hidden />
              {label}
            </Link>
          );
        })}
      </div>

      {/* Freshbox + educational microcopy */}
      <div className="railfoot">
        <Freshbox
          line1={t('landing.hero.badgeLive')}
          line2={`${formatInstrument(active.instrument)} · ${formatTimeframe(active.timeframe)}`}
        />
        <p
          style={{
            fontSize: 'var(--fs-legal)',
            color: 'var(--faint)',
            lineHeight: 1.5,
            padding: '8px 3px 0',
          }}
        >
          {t('legal.disclaimer.chart')}
        </p>
      </div>
    </aside>
  );
}

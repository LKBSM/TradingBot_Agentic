'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import { useMtfTrends } from '@/lib/market-reading/hooks';
import { MTF_TREND_ORDER } from '@/lib/market-reading/mtf-trend';
import { countActiveZones, deriveTrendMaturity } from '@/lib/market-reading/regime-facts';
import { formatZoneShortTime } from '@/lib/zones/lifecycle';
import type {
  BOSRecent,
  CHOCHRecent,
  MarketReadingHeader,
  MarketReadingRegime,
  MarketReadingStructure,
} from '@/types/market-reading';
import { HelpContent } from './HelpContent';

const REGIME_HELP_KEYS = ['regime', 'trend', 'vol', 'mat', 'align', 'dens'];

function arrowOf(dir: 'bullish' | 'bearish' | 'neutral' | 'ranging' | null): string {
  return dir === 'bullish' ? '↑' : dir === 'bearish' ? '↓' : '→';
}

function latestBreak<T extends BOSRecent | CHOCHRecent>(
  events: T[] | undefined,
  fallback: T | null | undefined,
): T | null {
  let best: T | null = null;
  for (const e of events ?? []) {
    if (!best || new Date(e.broken_at).getTime() > new Date(best.broken_at).getTime()) best = e;
  }
  return best ?? fallback ?? null;
}

interface RegimeCardProps {
  regime: MarketReadingRegime;
  structure: MarketReadingStructure;
  header: MarketReadingHeader;
  openHelp: string | null;
  onToggleHelp: (key: string) => void;
}

/**
 * "Régime de marché" card (mission UI-2c). Every measure carries a NAMED source
 * sub-line fed by the engine (never a number without its origin) and a "?" that
 * opens a static help panel (one open at a time across the page). "Dernier
 * événement" reads the break-event HISTORY (`*_events`), so it no longer shows
 * "non disponible" when the point-in-time break isn't on the last bar. The trend
 * is labelled with the timeframe it is measured on — the fix for the apparent
 * "trend vs alignment" contradiction (it's a labelling gap, not a calc bug).
 */
export function RegimeCard({
  regime,
  structure,
  header,
  openHelp,
  onToggleHelp,
}: RegimeCardProps) {
  const t = useTranslations('app.reg2');
  const treg = useTranslations('app.desktop.reg');
  const tr = useTranslations('reading');
  const fmt = useReadingFormatters();
  const { trends } = useMtfTrends(header.instrument);

  // Maturity — bars since the most recent CHOCH (from history), origin named.
  const maturity = deriveTrendMaturity(structure, header);

  // Alignment — dominant direction over the available upper/lower TFs.
  const avail = MTF_TREND_ORDER.filter(({ key }) => trends[key] != null);
  const dirs = avail.map(({ key }) =>
    trends[key] === 'bullish' ? 'up' : trends[key] === 'bearish' ? 'down' : 'flat',
  );
  const up = dirs.filter((d) => d === 'up').length;
  const down = dirs.filter((d) => d === 'down').length;
  const flat = dirs.filter((d) => d === 'flat').length;
  let domArrow = '→';
  let aligned = flat;
  if (up >= down && up >= flat) {
    domArrow = '↑';
    aligned = up;
  } else if (down >= up && down >= flat) {
    domArrow = '↓';
    aligned = down;
  }
  const alignValue = avail.length > 0 ? `${aligned}/${avail.length} TF ${domArrow}` : null;
  const alignSub =
    avail.length > 0
      ? avail.map(({ key, label }) => `${label} ${arrowOf(trends[key])}`).join(' · ')
      : null;

  // Last structural event — most recent of the BOS / CHOCH history.
  const lc = latestBreak(structure.choch_events, structure.choch);
  const lb = latestBreak(structure.bos_events, structure.bos);
  let last: (BOSRecent | CHOCHRecent) | null = null;
  let lastKind: 'CHOCH' | 'BOS' | null = null;
  if (lc && lb) {
    const cNewer = new Date(lc.broken_at).getTime() >= new Date(lb.broken_at).getTime();
    last = cNewer ? lc : lb;
    lastKind = cNewer ? 'CHOCH' : 'BOS';
  } else if (lc) {
    last = lc;
    lastKind = 'CHOCH';
  } else if (lb) {
    last = lb;
    lastKind = 'BOS';
  }

  const density = countActiveZones(structure);

  interface Cell {
    label: string;
    value: string | null;
    sub: string | null;
    help?: string;
    mono?: boolean;
  }
  const cells: Cell[] = [
    {
      label: treg('trend'),
      value: fmt.trend(regime.trend).label,
      sub: t('trendSub', { tf: header.timeframe }),
      help: 'trend',
    },
    {
      label: treg('volatility'),
      value: fmt.volatility(regime.volatility_observed).label,
      sub: t('volSub'),
      help: 'vol',
    },
    {
      label: treg('maturity'),
      value: maturity?.bars != null ? t('matValue', { count: maturity.bars }) : null,
      sub: maturity ? t('matSub', { time: formatZoneShortTime(maturity.brokenAt) }) : null,
      help: 'mat',
      mono: true,
    },
    {
      label: treg('alignment'),
      value: alignValue,
      sub: alignSub,
      help: 'align',
      mono: true,
    },
    {
      label: treg('lastEvent'),
      value: last ? `${lastKind} ${last.direction === 'bullish' ? '↑' : '↓'}` : null,
      sub: last ? formatZoneShortTime(last.broken_at) : null,
    },
    {
      label: treg('density'),
      value: `${density.ob} OB · ${density.fvg} FVG`,
      sub: t('densSub', { tf: header.timeframe }),
      help: 'dens',
      mono: true,
    },
  ];

  const helpInThisCard = openHelp != null && REGIME_HELP_KEYS.includes(openHelp);

  return (
    <div className="card">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M3 17l6-6 4 4 8-8" />
          <path d="M21 3v6h-6" />
        </svg>
        <h3>{tr('regime.title')}</h3>
        <span className="hsp" />
        <button
          type="button"
          className={cn('hbtn', openHelp === 'regime' && 'on')}
          aria-label={t('helpAria')}
          aria-expanded={openHelp === 'regime'}
          onClick={() => onToggleHelp('regime')}
        >
          ?
        </button>
      </div>
      <div className="reggrid">
        {cells.map((c) => (
          <div className="reg" key={c.label}>
            <div className="k">
              {c.label}
              {c.help && (
                <button
                  type="button"
                  className={cn('ihelp', openHelp === c.help && 'on')}
                  aria-label={t('measureHelpAria')}
                  aria-expanded={openHelp === c.help}
                  onClick={() => onToggleHelp(c.help as string)}
                >
                  ?
                </button>
              )}
            </div>
            <div
              className={cn('v', c.mono && 'mono')}
              style={c.value == null ? { color: 'var(--faint)', fontStyle: 'italic' } : undefined}
            >
              {c.value ?? tr('regime.unavailable')}
            </div>
            {c.sub && <div className="sub2">{c.sub}</div>}
          </div>
        ))}
      </div>
      {helpInThisCard && (
        <div className="infobox on">
          <HelpContent helpKey={openHelp as string} />
        </div>
      )}
    </div>
  );
}

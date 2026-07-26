'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import type {
  LiquidityKind,
  LiquidityStatus,
  MarketReadingStructure,
} from '@/types/market-reading';
import { HelpContent } from './HelpContent';

type SideFilter = 'all' | 'BSL' | 'SSL';
type StateFilter = 'all' | 'intact' | 'swept' | 'broken';

const LINE_CLASS: Record<LiquidityStatus, string> = {
  intact: 'lline',
  swept: 'lline swept',
  broken: 'lline broken',
};
const BADGE_CLASS: Record<LiquidityStatus, string> = {
  intact: 'zs-a',
  swept: 'zs-f',
  broken: 'zs-m',
};
const SRC_KEY: Record<LiquidityKind, string> = {
  equal_highs: 'eqh',
  equal_lows: 'eql',
  range_high: 'high',
  range_low: 'low',
};

interface LiquidityCardProps {
  structure: MarketReadingStructure;
  instrument: string;
  price: number | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  openHelp: string | null;
  onToggleHelp: (key: string) => void;
}

/**
 * "Liquidité externe" card (mission UI-2c). The full external-liquidity pocket
 * list (BSL/SSL), scrollable, with a collapsible side/state filter. Per row: a
 * status line-marker, side, level (mono), origin (EQH/EQL/Sommet/Creux from the
 * engine `kind`), state, live distance-to-price and an honest fact. Clicking a
 * row highlights that LEVEL on the chart (same id-lock as the zones). No per-row
 * TF chip (no per-pocket timeframe data). Sort by distance; a price tick updates
 * distances WITHOUT reordering.
 */
export function LiquidityCard({
  structure,
  instrument,
  price,
  selectedId,
  onSelect,
  openHelp,
  onToggleHelp,
}: LiquidityCardProps) {
  const t = useTranslations('app.liq2');
  const fmt = useReadingFormatters();
  const [sortOpen, setSortOpen] = React.useState(false);
  const [side, setSide] = React.useState<SideFilter>('all');
  const [state, setState] = React.useState<StateFilter>('all');

  const pools = React.useMemo(() => structure.liquidity_pools ?? [], [structure]);

  const priceRef = React.useRef(price);
  priceRef.current = price;

  const ordered = React.useMemo(() => {
    const list = pools.filter(
      (l) =>
        (side === 'all' || l.side.toUpperCase() === side) &&
        (state === 'all' || l.status === state),
    );
    const p = priceRef.current;
    if (p != null && Number.isFinite(p)) {
      list.sort((a, b) => Math.abs(a.level - p) - Math.abs(b.level - p));
    }
    return list;
    // price excluded (priceRef) — order frozen between filter changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pools, side, state]);

  function distNode(level: number): React.ReactNode {
    if (price == null || !Number.isFinite(price)) return null;
    const d = ((level - price) / price) * 100;
    const pct = Math.abs(d).toFixed(2).replace('.', ',');
    return d >= 0 ? t('dist.above', { pct }) : t('dist.below', { pct });
  }

  const helpOn = openHelp === 'liq';

  return (
    <div className="card">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M4 8h16M4 12h16M4 16h16" />
        </svg>
        <h3>{t('title')}</h3>
        <span className="hsp" />
        <span className="badge2">{t('pochesCount', { count: ordered.length })}</span>
        <button
          type="button"
          className={cn('hbtn', helpOn && 'on')}
          aria-label={t('helpAria')}
          aria-expanded={helpOn}
          onClick={() => onToggleHelp('liq')}
        >
          ?
        </button>
        <button
          type="button"
          className={cn('hbtn', sortOpen && 'on')}
          aria-label={t('sortAria')}
          aria-expanded={sortOpen}
          onClick={() => setSortOpen((v) => !v)}
        >
          <svg viewBox="0 0 24 24" aria-hidden>
            <path d="M4 7h10M18 7h2M4 12h4M12 12h8M4 17h12M20 17h0" />
            <circle cx="16" cy="7" r="2" />
            <circle cx="10" cy="12" r="2" />
            <circle cx="18" cy="17" r="2" />
          </svg>
        </button>
      </div>

      <div className={cn('ctrlrow', sortOpen && 'on')}>
        <div className="fsec">{t('filterSide')}</div>
        <div className="fgrp">
          {(['all', 'BSL', 'SSL'] as const).map((v) => (
            <button
              key={v}
              type="button"
              className={cn('fchip', side === v && 'on')}
              aria-pressed={side === v}
              onClick={() => setSide(v)}
            >
              {t(`side.${v === 'all' ? 'all' : v === 'BSL' ? 'bsl' : 'ssl'}`)}
            </button>
          ))}
        </div>
        <div className="fsec">{t('filterState')}</div>
        <div className="fgrp">
          {(['all', 'intact', 'swept', 'broken'] as const).map((v) => (
            <button
              key={v}
              type="button"
              className={cn('fchip', state === v && 'on')}
              aria-pressed={state === v}
              onClick={() => setState(v)}
            >
              {t(`st.${v}`)}
            </button>
          ))}
        </div>
      </div>

      <div className="zlist">
        {ordered.length === 0 ? (
          <div className="zempty">
            {t('empty1')}
            <br />
            {t('empty2')}
          </div>
        ) : (
          ordered.map((l) => {
            const selected = l.id === selectedId;
            return (
              <div
                key={l.id}
                role="button"
                tabIndex={0}
                className={cn('zrow', selected && 'sel')}
                aria-pressed={selected}
                onClick={() => onSelect(l.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelect(l.id);
                  }
                }}
              >
                <div className="zr1">
                  <span className="lmini">
                    <span className={LINE_CLASS[l.status]} />
                  </span>
                  <span className="tagx neu">{l.side.toUpperCase()}</span>
                  <span className="zr">{fmt.price(l.level, instrument)}</span>
                  <span className="zsrc">{t(`src.${SRC_KEY[l.kind]}`)}</span>
                  <span className={cn('zstate', BADGE_CLASS[l.status])}>{t(`badge.${l.status}`)}</span>
                </div>
                <div className="zr2">
                  <span className="zd">{distNode(l.level)}</span>
                  <span className="zn">{t(`fact.${l.status}`)}</span>
                </div>
              </div>
            );
          })
        )}
      </div>

      {helpOn && (
        <div className="infobox on">
          <HelpContent helpKey="liq" />
        </div>
      )}
      <div className="listhint">{t('hint')}</div>
    </div>
  );
}

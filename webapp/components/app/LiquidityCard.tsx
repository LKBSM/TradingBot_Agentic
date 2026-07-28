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
import { FilterChipGroup } from './FilterChipGroup';
import { useMultiFilter } from '@/lib/market-reading/use-multi-filter';
import { useChartViewOptional } from '@/lib/chart/viewState';

const SIDE_VALUES = ['BSL', 'SSL'] as const;
const STATE_VALUES = ['intact', 'swept', 'broken'] as const;

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
  openHelp,
  onToggleHelp,
}: LiquidityCardProps) {
  const t = useTranslations('app.liq2');
  const fmt = useReadingFormatters();
  // VZ-1 — a pocket is a LEVEL, not a zone: it selects on the unified selection's
  // distinct LEVEL channel (horizontal line + frameLevel), never the zone id-lock.
  const { selection, select, clearSelection } = useChartViewOptional();
  const selectedId =
    selection?.family === 'level' && selection.kind === 'liquidity' ? selection.id : null;
  const selectPocket = React.useCallback(
    (pool: { id: string; level: number; side: 'bsl' | 'ssl' }) => {
      if (selectedId === pool.id) {
        clearSelection();
        return;
      }
      select({
        family: 'level',
        kind: 'liquidity',
        id: pool.id,
        price: pool.level,
        label: `${pool.side.toUpperCase()} · ${fmt.price(pool.level, instrument)}`,
        side: pool.side,
      });
    },
    [selectedId, select, clearSelection, fmt, instrument],
  );
  const [sortOpen, setSortOpen] = React.useState(false);
  const sideFilter = useMultiFilter(SIDE_VALUES);
  const stateFilter = useMultiFilter(STATE_VALUES);
  const noneSelected = sideFilter.noneSelected || stateFilter.noneSelected;

  const pools = React.useMemo(() => structure.liquidity_pools ?? [], [structure]);

  const priceRef = React.useRef(price);
  priceRef.current = price;

  const ordered = React.useMemo(() => {
    // (BSL ∪ SSL) ET (intacte ∪ balayée ∪ cassée). No selection in a group ⇒
    // zero rows — NEVER a silent fallback to "show all".
    if (noneSelected) return [];
    const list = pools.filter(
      (l) =>
        sideFilter.selected.has(l.side.toUpperCase() as (typeof SIDE_VALUES)[number]) &&
        stateFilter.selected.has(l.status),
    );
    const p = priceRef.current;
    if (p != null && Number.isFinite(p)) {
      list.sort((a, b) => Math.abs(a.level - p) - Math.abs(b.level - p));
    }
    return list;
    // price excluded (priceRef) — order frozen between filter changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pools, sideFilter.selected, stateFilter.selected, noneSelected]);

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
        <span className="badge2">{t('countFiltered', { n: ordered.length, m: pools.length })}</span>
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
        <FilterChipGroup
          label={t('filterSide')}
          filter={sideFilter}
          resetLabel={t('reset')}
          options={[
            { value: 'BSL', label: t('side.bsl') },
            { value: 'SSL', label: t('side.ssl') },
          ]}
        />
        <FilterChipGroup
          label={t('filterState')}
          filter={stateFilter}
          resetLabel={t('reset')}
          options={[
            { value: 'intact', label: t('st.intact') },
            { value: 'swept', label: t('st.swept') },
            { value: 'broken', label: t('st.broken') },
          ]}
        />
      </div>

      <div className="zlist">
        {noneSelected ? (
          <div className="zempty">
            {sideFilter.noneSelected ? t('noneSide') : t('noneState')}
          </div>
        ) : ordered.length === 0 ? (
          <div className="zempty">
            {t('noMatch1')}
            <br />
            {t('noMatch2')}
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
                onClick={() => selectPocket({ id: l.id, level: l.level, side: l.side })}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    selectPocket({ id: l.id, level: l.level, side: l.side });
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

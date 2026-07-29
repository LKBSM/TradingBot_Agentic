'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import {
  collectZones,
  fillFraction,
  formatZoneShortTime,
  priceRelation,
  type ZoneLifecycle,
} from '@/lib/zones/lifecycle';
import type { BOSRecent, CHOCHRecent, MarketReadingStructure } from '@/types/market-reading';
import { HelpContent } from './HelpContent';
import { FilterChipGroup } from './FilterChipGroup';
import { useMultiFilter } from '@/lib/market-reading/use-multi-filter';
import { useChartViewOptional } from '@/lib/chart/viewState';

/** ISO-8601 → epoch seconds, or null when unparseable. */
function isoToSec(iso: string): number | null {
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
}

const TYPE_VALUES = ['ob', 'fvg'] as const;
const STATE_VALUES = ['active', 'tested', 'mitig'] as const;
type SortKey = 'near' | 'recent' | 'big';

/**
 * 3-way DISPLAY state from the real engine fields (mission UI-2c): terminal
 * (mitigated OB / filled FVG / invalidated) → "mitig"; touched-but-alive
 * (tested / partially_filled) → "tested"; untouched active → "active". No
 * fabricated state, and never a "×N" count (the engine tracks none).
 */
function zoneState3(z: ZoneLifecycle): 'active' | 'tested' | 'mitig' {
  if (z.status === 'mitigated' || z.status === 'filled' || z.status === 'invalidated') return 'mitig';
  if (z.tested || z.status === 'partially_filled') return 'tested';
  return 'active';
}

const STATE_CLASS: Record<'active' | 'tested' | 'mitig', string> = {
  active: 'zs-a',
  tested: 'zs-f',
  mitig: 'zs-m',
};

function dirArrow(dir: ZoneLifecycle['direction']): string {
  return dir === 'bearish' ? ' ↓' : dir === 'bullish' ? ' ↑' : '';
}
function dirTone(dir: ZoneLifecycle['direction']): 'bear' | 'bull' | 'neu' {
  return dir === 'bearish' ? 'bear' : dir === 'bullish' ? 'bull' : 'neu';
}

/** Most recent event of a break-history list (or the point-in-time fallback). */
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

interface StructureCardProps {
  structure: MarketReadingStructure;
  instrument: string;
  /** The unit this panel operates on — labelled like the Régime panel (TF-1 E3). */
  timeframe: string;
  /** Current price — drives the distance badges (recomputed every render / tick). */
  price: number | null;
  /** Chart-highlighted zone id (single source of truth for the selected row). */
  selectedId: string | null;
  /** Toggle chart highlight for a zone id (via the id-lock). */
  onSelect: (id: string) => void;
  /** Open the Zones page on this zone's card (deep-link). */
  onOpenZone: (id: string) => void;
  /** Page-level help key open (one at a time across the dashboard), + setter. */
  openHelp: string | null;
  onToggleHelp: (key: string) => void;
}

/**
 * "Structure de marché" card (mission UI-2c). The FULL detected zone list
 * (no front truncation), scrollable at a fixed height, with a collapsible
 * type/state/sort panel, the CHOCH/BOS context rows, and per-row: type+dir,
 * price band, engine state, live distance-to-price, an honest fact, and an "En
 * savoir plus" deep-link. Clicking a row highlights the zone on the chart via
 * the existing id-lock. Confluence "N TF" badge and per-row TF chip are removed
 * (no per-zone data). Sorting only re-runs on a criterion change — a price tick
 * updates the distances WITHOUT reordering the list.
 */
export function StructureCard({
  structure,
  instrument,
  timeframe,
  price,
  selectedId,
  onSelect,
  onOpenZone,
  openHelp,
  onToggleHelp,
}: StructureCardProps) {
  const t = useTranslations('app.struct');
  const fmt = useReadingFormatters();
  // VZ-1 — BOS/CHOCH events select on the unified selection's distinct EVENT
  // channel (confirmation-candle marker + broken-level line + frameEvent), never
  // the zone id-lock. Zones keep flowing through `onSelect` (id-lock) above.
  const { selection, select, clearSelection } = useChartViewOptional();
  const selectedEventId = selection?.family === 'event' ? selection.id : null;
  const selectEvent = React.useCallback(
    (kind: 'bos' | 'choch', ev: BOSRecent | CHOCHRecent) => {
      const atSec = isoToSec(ev.broken_at);
      if (atSec == null) return;
      const id = `${kind}:${atSec}:${ev.level}`;
      if (selectedEventId === id) {
        clearSelection();
        return;
      }
      select({ family: 'event', id, kind, direction: ev.direction, level: ev.level, atSec });
    },
    [selectedEventId, select, clearSelection],
  );
  const [sortOpen, setSortOpen] = React.useState(false);
  const typeFilter = useMultiFilter(TYPE_VALUES);
  const stateFilter = useMultiFilter(STATE_VALUES);
  const noneSelected = typeFilter.noneSelected || stateFilter.noneSelected;
  const [sort, setSort] = React.useState<SortKey>('near');

  // Price is read through a ref inside the ordering memo so a tick never
  // reorders the list (mission §A) — only a criterion/data change re-sorts.
  const priceRef = React.useRef(price);
  priceRef.current = price;

  const allZones = React.useMemo(() => collectZones(structure), [structure]);

  // TF-1 E2 — the header states total AND active so it reconciles with the
  // Régime « Densité » tile (which counts active/open only): total = active +
  // consumed. Same active definition as Densité (OB active + FVG active/partial).
  const activeCount = React.useMemo(
    () =>
      structure.order_blocks.filter((z) => z.status === 'active').length +
      structure.fair_value_gaps.filter(
        (z) => z.status === 'active' || z.status === 'partially_filled',
      ).length,
    [structure],
  );

  const ordered = React.useMemo(() => {
    // (OB ∪ FVG) ET (active ∪ testée ∪ mitigée). No selection in a group ⇒ zero
    // rows — NEVER a silent fallback to "show all".
    if (noneSelected) return [];
    const zones = allZones.filter(
      (z) =>
        typeFilter.selected.has(z.kind as (typeof TYPE_VALUES)[number]) &&
        stateFilter.selected.has(zoneState3(z)),
    );
    const p = priceRef.current;
    const dist = (z: ZoneLifecycle) => {
      const r = priceRelation(z, p);
      return r && r.position !== 'inside' ? r.distance : 0;
    };
    const byRecent = (a: ZoneLifecycle, b: ZoneLifecycle) =>
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    zones.sort((a, b) =>
      sort === 'near'
        ? dist(a) - dist(b) || byRecent(a, b)
        : sort === 'big'
          ? b.levelHigh - b.levelLow - (a.levelHigh - a.levelLow) || byRecent(a, b)
          : byRecent(a, b),
    );
    return zones;
    // price excluded on purpose (see priceRef) — order is frozen between sorts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allZones, typeFilter.selected, stateFilter.selected, sort, noneSelected]);

  const choch = latestBreak(structure.choch_events, structure.choch);
  const bos = latestBreak(structure.bos_events, structure.bos);

  function distNode(z: ZoneLifecycle): React.ReactNode {
    const rel = priceRelation(z, price);
    if (!rel) return null;
    if (rel.position === 'inside') return <b>{t('dist.inside')}</b>;
    const pct = ((rel.distance / (price as number)) * 100).toFixed(2).replace('.', ',');
    return rel.position === 'above' ? t('dist.above', { pct }) : t('dist.below', { pct });
  }

  function factNote(z: ZoneLifecycle): string {
    const s = zoneState3(z);
    if (z.kind === 'fvg') {
      if (z.status === 'filled') return t('fact.filled');
      const frac = fillFraction(z);
      if (frac != null && frac > 0) return t('fact.partial', { pct: Math.round(frac * 100) });
      if (s === 'active') return t('fact.fvgOpen');
    }
    if (s === 'mitig') {
      return z.mitigatedAt
        ? t('fact.mitigatedAt', { time: formatZoneShortTime(z.mitigatedAt) })
        : t('fact.mitigated');
    }
    if (!z.tested) return t('fact.neverTested');
    return t('fact.tested');
  }

  const helpOn = openHelp === 'struct';

  return (
    <div className="card">
      <div className="card-h">
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M3 12h4l3-7 4 14 3-7h4" />
        </svg>
        <h3>{t('title')}</h3>
        {/* TF-1 E3 — the panel carries its unit of time, like the Régime panel. */}
        <span className="badge2" aria-label={t('unitLabel', { tf: timeframe })}>{timeframe}</span>
        <span className="hsp" />
        <span className="badge2">{t('countFiltered', { n: ordered.length, m: allZones.length, active: activeCount })}</span>
        <button
          type="button"
          className={cn('hbtn', helpOn && 'on')}
          aria-label={t('helpAria')}
          aria-expanded={helpOn}
          onClick={() => onToggleHelp('struct')}
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
          label={t('filterType')}
          filter={typeFilter}
          resetLabel={t('reset')}
          options={[
            { value: 'ob', label: t('type.ob') },
            { value: 'fvg', label: t('type.fvg') },
          ]}
        />
        <FilterChipGroup
          label={t('filterState')}
          filter={stateFilter}
          resetLabel={t('reset')}
          options={[
            { value: 'active', label: t('st.active') },
            { value: 'tested', label: t('st.tested') },
            { value: 'mitig', label: t('st.mitig') },
          ]}
        />
        <div className="fsec">{t('filterSort')}</div>
        <div className="fgrp">
          {(['near', 'recent', 'big'] as const).map((v) => (
            <button
              key={v}
              type="button"
              className={cn('fchip', sort === v && 'on')}
              aria-pressed={sort === v}
              onClick={() => setSort(v)}
            >
              {t(`sort.${v}`)}
            </button>
          ))}
        </div>
      </div>

      {choch && (
        <div
          role="button"
          tabIndex={0}
          className={cn('strow', selectedEventId === `choch:${isoToSec(choch.broken_at)}:${choch.level}` && 'sel')}
          aria-pressed={selectedEventId === `choch:${isoToSec(choch.broken_at)}:${choch.level}`}
          aria-label={t('evSelectAria', { kind: 'CHOCH' })}
          onClick={() => selectEvent('choch', choch)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              selectEvent('choch', choch);
            }
          }}
        >
          <span className={cn('tagx', choch.direction === 'bullish' ? 'bull' : 'bear')}>
            CHOCH{dirArrow(choch.direction)}
          </span>
          <span className="d">{t('evChoch')}</span>
          <span className="t">{formatZoneShortTime(choch.broken_at)}</span>
        </div>
      )}
      {bos && (
        <div
          role="button"
          tabIndex={0}
          className={cn('strow', selectedEventId === `bos:${isoToSec(bos.broken_at)}:${bos.level}` && 'sel')}
          aria-pressed={selectedEventId === `bos:${isoToSec(bos.broken_at)}:${bos.level}`}
          aria-label={t('evSelectAria', { kind: 'BOS' })}
          onClick={() => selectEvent('bos', bos)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              selectEvent('bos', bos);
            }
          }}
        >
          <span className={cn('tagx', bos.direction === 'bullish' ? 'bull' : 'bear')}>
            BOS{dirArrow(bos.direction)}
          </span>
          <span className="d">
            {bos.direction === 'bullish' ? t('evBosBull') : t('evBosBear')}
          </span>
          <span className="t">{formatZoneShortTime(bos.broken_at)}</span>
        </div>
      )}

      <div className="zlist">
        {noneSelected ? (
          <div className="zempty">
            {typeFilter.noneSelected ? t('noneType') : t('noneState')}
          </div>
        ) : ordered.length === 0 ? (
          <div className="zempty">
            {t('noMatch1')}
            <br />
            {t('noMatch2')}
          </div>
        ) : (
          ordered.map((z) => {
            const s = zoneState3(z);
            const selected = z.id === selectedId;
            return (
              <div
                key={z.id}
                role="button"
                tabIndex={0}
                className={cn('zrow', selected && 'sel')}
                aria-pressed={selected}
                onClick={() => onSelect(z.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelect(z.id);
                  }
                }}
              >
                <div className="zr1">
                  <span className={cn('tagx', dirTone(z.direction))}>
                    {z.kind.toUpperCase()}
                    {dirArrow(z.direction)}
                  </span>
                  <span className="zr">{fmt.band(z.levelLow, z.levelHigh, instrument)}</span>
                  <span className={cn('zstate', STATE_CLASS[s])}>{t(`badge.${s}`)}</span>
                </div>
                <div className="zr2">
                  <span className="zd">{distNode(z)}</span>
                  <span className="zn">{factNote(z)}</span>
                  <button
                    type="button"
                    className="zbtn"
                    onClick={(e) => {
                      e.stopPropagation();
                      onOpenZone(z.id);
                    }}
                  >
                    {t('more')}
                    <svg viewBox="0 0 24 24" aria-hidden>
                      <path d="M4 12h15M13 6l6 6-6 6" />
                    </svg>
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {helpOn && (
        <div className="infobox on">
          <HelpContent helpKey="struct" />
        </div>
      )}
      <div className="listhint">{t('hint')}</div>
    </div>
  );
}

'use client';

import { useCallback, useMemo, type ReactNode } from 'react';
import { useTranslations } from 'next-intl';
import { Shapes } from 'lucide-react';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { cn } from '@/lib/utils';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import type { GlossaryKey } from '@/lib/glossary';
import { useChartViewOptional } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import type {
  BOSRecent,
  CHOCHRecent,
  FairValueGap,
  LiquidityPool,
  LiquidityStatus,
  MarketReadingStructure,
  OBImportance,
  OrderBlock,
} from '@/types/market-reading';
import { ZoneList } from './ZoneList';

/** ISO-8601 → epoch seconds, or null when unparseable. */
function isoToSec(iso: string): number | null {
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
}

// Importance / status weights driving the collapsed ordering (lower = surfaced
// first). Display-only ranking — detection is untouched.
const OB_IMPORTANCE_RANK: Record<OBImportance, number> = {
  high: 0,
  medium: 1,
  low: 2,
};
// FVGs carry no importance field; their lifecycle status stands in for it so OB
// and FVG get the same « importance puis proximité » treatment.
const FVG_STATUS_RANK: Record<FairValueGap['status'], number> = {
  active: 0,
  partially_filled: 1,
  filled: 2,
};

// Liquidity ordering — surface in-play pockets first: intact, then swept, then
// broken (terminal). Display-only ranking; detection is untouched.
const LIQUIDITY_STATUS_RANK: Record<LiquidityStatus, number> = {
  intact: 0,
  swept: 1,
  broken: 2,
};

/**
 * Section "Structure" — renders the Smart Money Concept block factually:
 * BOS / CHOCH, Order Blocks, Fair Value Gaps, retest in progress. Every line is
 * descriptive (a market fact), never prescriptive.
 */
export function StructureSection({
  structure,
  instrument,
  closePrice,
}: {
  structure: MarketReadingStructure;
  instrument: string;
  /** Current close price — drives the proximity tie-break in the zone lists. */
  closePrice?: number;
}) {
  const t = useTranslations('reading.structure');
  const fmt = useReadingFormatters();
  const { bos, choch, order_blocks, fair_value_gaps, retest_in_progress } =
    structure;
  const structure_liquidity_pools = structure.liquidity_pools;
  const liquidity_pools = useMemo(
    () => structure_liquidity_pools ?? [],
    [structure_liquidity_pools],
  );

  // Click-to-chart wiring (display/navigation only). We reuse the EXISTING chart
  // view channel the M.I.A Agent drives: clicking a zone asks the chart to
  // `focus_zone` (centre) + `highlight_zone` (emphasise) it by its REAL engine
  // id. Optional provider — outside the /app workspace `applyActions` is a no-op,
  // so the list stays readable with no chart wired in. Detection is never touched.
  const { view: chartView, applyActions, selection, select, clearSelection } =
    useChartViewOptional();
  const fmtPrice = fmt.price;

  // The id verrou: the ONLY zones a focus/highlight may reference are the ones
  // the engine emitted in THIS structure — identical to AppWorkspace's set.
  // A click can never invent or move a zone.
  const validZoneIds = useMemo(() => {
    const ids = new Set<string>();
    for (const ob of order_blocks) ids.add(ob.id);
    for (const fvg of fair_value_gaps) ids.add(fvg.id);
    return ids;
  }, [order_blocks, fair_value_gaps]);

  // The selected entry mirrors the chart's highlighted zone (single source of
  // truth) — the highlight persists as the selection until another is clicked.
  const selectedZoneId = chartView.highlightZoneId;

  const selectZone = useCallback(
    (zoneId: string) => {
      // Toggle: clicking the ALREADY-selected zone deselects it (drop the blue
      // highlight + un-zoom), so a second click undoes the first.
      if (zoneId === selectedZoneId) {
        applyActions(
          coerceViewActions([{ action: 'clear_highlight', params: {} }], validZoneIds),
        );
        return;
      }
      // Re-validate through the same Couche-4 coercion (defence in depth): a
      // focus/highlight on an unknown id is dropped rather than mis-applied.
      const actions = coerceViewActions(
        [
          { action: 'focus_zone', params: { zone_id: zoneId } },
          { action: 'highlight_zone', params: { zone_id: zoneId } },
        ],
        validZoneIds,
      );
      applyActions(actions);
    },
    [applyActions, validZoneIds, selectedZoneId],
  );

  // VZ-1 — events (BOS/CHOCH) and liquidity pockets select on the unified
  // selection's DISTINCT channels (never the zone id-lock): an event carries its
  // confirmation time + broken level; a pocket is a horizontal LEVEL.
  const selectedEventId = selection?.family === 'event' ? selection.id : null;
  const selectEvent = useCallback(
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
  const selectedLevelId =
    selection?.family === 'level' && selection.kind === 'liquidity' ? selection.id : null;
  const selectPocket = useCallback(
    (pool: LiquidityPool) => {
      if (selectedLevelId === pool.id) {
        clearSelection();
        return;
      }
      select({
        family: 'level',
        kind: 'liquidity',
        id: pool.id,
        price: pool.level,
        label: `${pool.side.toUpperCase()} · ${fmtPrice(pool.level, instrument)}`,
        side: pool.side,
      });
    },
    [selectedLevelId, select, clearSelection, fmtPrice, instrument],
  );

  // Surfacing coherence (founder eval 2026-06-08): the engine emits `bos` only
  // on a FRESH break at the last close (by design — see market_reading_mappers
  // F6), while a retest is armed BARS AFTER that break, with `bos` already null.
  // Without this, the BOS row said "aucune cassure récente" while the retest row
  // said "retest de cassure (BOS)" — a logical contradiction. When the live
  // retest references a prior break, we state that instead of denying it. This
  // is a copy/surfacing fix only — no detection threshold is touched.
  const bosUnderRetest =
    !bos && retest_in_progress?.type === 'bos_retest';
  const chochUnderRetest =
    !choch && retest_in_progress?.type === 'choch_retest';

  const hasAnything =
    bos ||
    choch ||
    order_blocks.length > 0 ||
    fair_value_gaps.length > 0 ||
    liquidity_pools.length > 0 ||
    retest_in_progress;

  // In-play pockets first (intact → swept → broken), then by price level. The
  // engine already caps the set; this is display ordering only.
  const sortedLiquidity = useMemo(
    () =>
      [...liquidity_pools].sort(
        (a, b) =>
          LIQUIDITY_STATUS_RANK[a.status] - LIQUIDITY_STATUS_RANK[b.status] ||
          b.level - a.level,
      ),
    [liquidity_pools],
  );

  return (
    <AccordionItem value="structure">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <Shapes className="h-4 w-4 text-muted-foreground" aria-hidden />
          <span>{t('title')}</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        {/* Cadrage éditorial (niveau 1.5) — structures décrites au présent. */}
        <p className="mb-3 text-xs text-muted-foreground">{t('intro')}</p>
        {!hasAnything ? (
          <p className="text-sm text-muted-foreground">{t('empty')}</p>
        ) : (
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Row
              label={t('bosLabel')}
              termKey="bos"
              value={
                bos
                  ? `${fmt.price(bos.level, instrument)} · ${fmt.direction(bos.direction)} · ${fmt.validationFem(bos.validation_status)}`
                  : bosUnderRetest
                    ? t('bosRetest', { price: fmt.price(retest_in_progress!.level, instrument) })
                    : t('bosNone')
              }
              onSelect={bos ? () => selectEvent('bos', bos) : undefined}
              selected={
                bos != null &&
                selectedEventId === `bos:${isoToSec(bos.broken_at)}:${bos.level}`
              }
              selectAria={t('bosLabel')}
            />
            <Row
              label={t('chochLabel')}
              termKey="choch"
              value={
                choch
                  ? `${fmt.price(choch.level, instrument)} · ${fmt.direction(choch.direction)} · ${fmt.validationFem(choch.validation_status)}`
                  : chochUnderRetest
                    ? t('chochRetest', { price: fmt.price(retest_in_progress!.level, instrument) })
                    : t('chochNone')
              }
              onSelect={choch ? () => selectEvent('choch', choch) : undefined}
              selected={
                choch != null &&
                selectedEventId === `choch:${isoToSec(choch.broken_at)}:${choch.level}`
              }
              selectAria={t('chochLabel')}
            />
            <ZoneRow label={t('obLabel')} termKey="order_block">
              {order_blocks.length > 0 ? (
                <ZoneList<OrderBlock>
                  zones={order_blocks}
                  price={closePrice}
                  noun="zone"
                  importanceRank={(ob) => OB_IMPORTANCE_RANK[ob.importance]}
                  band={(ob) => [ob.level_low, ob.level_high]}
                  isActive={(ob) => ob.status === 'active'}
                  dedupKey={(ob) =>
                    `${ob.level_low}|${ob.level_high}|${ob.importance}|${ob.status}`
                  }
                  renderLabel={(ob) =>
                    // PR #51 label purge: OB shows band · status only — the
                    // "importance {X}" conviction score was removed from the
                    // display (the price band suffices). Detection/sort untouched.
                    t('obItem', {
                      band: fmt.band(ob.level_low, ob.level_high, instrument),
                      status: fmt.obStatus(ob.status),
                    })
                  }
                  idOf={(ob) => ob.id}
                  onSelect={selectZone}
                  selectedZoneId={selectedZoneId}
                />
              ) : (
                <span className="text-sm font-medium text-foreground">{t('obEmpty')}</span>
              )}
            </ZoneRow>
            <ZoneRow label={t('fvgLabel')} termKey="fvg">
              {fair_value_gaps.length > 0 ? (
                <ZoneList<FairValueGap>
                  zones={fair_value_gaps}
                  price={closePrice}
                  noun="zone"
                  importanceRank={(fvg) => FVG_STATUS_RANK[fvg.status]}
                  band={(fvg) => [fvg.level_low, fvg.level_high]}
                  isActive={(fvg) => fvg.status === 'active'}
                  dedupKey={(fvg) =>
                    `${fvg.level_low}|${fvg.level_high}|${fvg.status}`
                  }
                  renderLabel={(fvg) =>
                    `${fmt.band(fvg.level_low, fvg.level_high, instrument)} · ${fmt.fvgStatus(fvg.status)}`
                  }
                  idOf={(fvg) => fvg.id}
                  onSelect={selectZone}
                  selectedZoneId={selectedZoneId}
                />
              ) : (
                <span className="text-sm font-medium text-foreground">{t('fvgEmpty')}</span>
              )}
            </ZoneRow>
            <ZoneRow label={t('liquidityLabel')} termKey="liquidity">
              {sortedLiquidity.length > 0 ? (
                <LiquidityList
                  pools={sortedLiquidity}
                  instrument={instrument}
                  onSelect={selectPocket}
                  selectedId={selectedLevelId}
                />
              ) : (
                <span className="text-sm font-medium text-foreground">{t('liquidityEmpty')}</span>
              )}
            </ZoneRow>
            <Row
              label={t('retestLabel')}
              termKey="retest"
              value={
                retest_in_progress
                  ? `${fmt.price(retest_in_progress.level, instrument)} · ${fmt.retestType(retest_in_progress.type)}`
                  : t('retestNone')
              }
              className="sm:col-span-2"
            />
          </dl>
        )}
        <p className="mt-4 text-xs italic text-muted-foreground">{t('footer')}</p>
      </AccordionContent>
    </AccordionItem>
  );
}

function Row({
  label,
  value,
  termKey,
  className,
  onSelect,
  selected,
  selectAria,
}: {
  label: string;
  value: string;
  /** When set, the label becomes a vulgarisation tooltip (ⓘ + /methodology link). */
  termKey?: GlossaryKey;
  className?: string;
  /** VZ-1 — when set, the value becomes a focusable button that frames the chart. */
  onSelect?: () => void;
  selected?: boolean;
  selectAria?: string;
}) {
  return (
    <div className={cn(className)}>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">
        {termKey ? <InfoTooltip termKey={termKey}>{label}</InfoTooltip> : label}
      </dt>
      <dd className="mt-1 text-sm font-medium text-foreground">
        {onSelect ? (
          <button
            type="button"
            aria-pressed={selected}
            aria-label={selectAria}
            onClick={onSelect}
            className={cn(
              'w-full rounded px-1.5 py-1 text-left transition-colors -mx-1.5',
              'hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              selected && 'bg-primary/10 ring-1 ring-primary/40',
            )}
          >
            {value}
          </button>
        ) : (
          value
        )}
      </dd>
    </div>
  );
}

/**
 * Read-only list of external liquidity pockets. Mirrors the chart's per-side
 * palette (BSL blue-teal / SSL rose-violet) so the panel and the price lines
 * read as one. Strictly descriptive: WHERE liquidity rests + its intact / swept
 * / broken state — no target, draw or bias. Not clickable (these are levels, not
 * focusable boxes), unlike Order Block / FVG zones.
 */
const LIQUIDITY_DOT: Record<LiquidityPool['side'], string> = {
  bsl: 'rgb(79, 163, 199)',
  ssl: 'rgb(199, 127, 163)',
};

function LiquidityList({
  pools,
  instrument,
  onSelect,
  selectedId,
}: {
  pools: LiquidityPool[];
  instrument: string;
  /** VZ-1 — select this pocket as a LEVEL on the chart (frame + line). */
  onSelect?: (pool: LiquidityPool) => void;
  selectedId?: string | null;
}) {
  const fmt = useReadingFormatters();
  return (
    <ul className="flex flex-col gap-1.5">
      {pools.map((p) => {
        const status = fmt.liquidityStatus(p.status);
        const selected = selectedId === p.id;
        return (
          <li
            key={p.id}
            role={onSelect ? 'button' : undefined}
            tabIndex={onSelect ? 0 : undefined}
            aria-pressed={onSelect ? selected : undefined}
            onClick={onSelect ? () => onSelect(p) : undefined}
            onKeyDown={
              onSelect
                ? (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      onSelect(p);
                    }
                  }
                : undefined
            }
            /* flex-wrap + min-w-0: on a 390px phone the widest BSL/SSL line
               (price · side · kind · status) wraps instead of overflowing the
               panel horizontally. */
            className={cn(
              'flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-sm font-medium text-foreground',
              onSelect &&
                'cursor-pointer rounded px-1.5 py-1 -mx-1.5 transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
              selected && 'bg-primary/10 ring-1 ring-primary/40',
            )}
          >
            <span
              className="inline-block h-2 w-2 shrink-0 rounded-full"
              style={{
                backgroundColor: LIQUIDITY_DOT[p.side],
                opacity: p.status === 'broken' ? 0.45 : 1,
              }}
              aria-hidden
            />
            <span className="tabular-nums">{fmt.price(p.level, instrument)}</span>
            <span className="text-muted-foreground">·</span>
            <span className="font-semibold tracking-wide">{p.side.toUpperCase()}</span>
            <span className="text-muted-foreground">{fmt.liquidityKind(p.kind)}</span>
            <span className="text-muted-foreground">·</span>
            <span
              className={cn(
                status.tone === 'warn' ? 'text-sentinel-warn' : 'text-muted-foreground',
                p.status === 'broken' && 'line-through opacity-70',
              )}
            >
              {status.label}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/** Like `Row`, but the value is arbitrary content (the collapsible zone list). */
function ZoneRow({
  label,
  termKey,
  children,
}: {
  label: string;
  termKey?: GlossaryKey;
  children: ReactNode;
}) {
  return (
    <div className="sm:col-span-2">
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">
        {termKey ? <InfoTooltip termKey={termKey}>{label}</InfoTooltip> : label}
      </dt>
      <dd className="mt-1.5">{children}</dd>
    </div>
  );
}

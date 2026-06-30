'use client';

import { useCallback, useMemo, type ReactNode } from 'react';
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
import {
  formatBand,
  formatDirection,
  formatFvgStatus,
  formatLiquidityKind,
  formatLiquiditySideShort,
  formatLiquidityStatus,
  formatObImportance,
  formatObStatus,
  formatPrice,
  formatRetestType,
  formatValidationStatus,
} from '@/lib/market-reading/formatters';
import type {
  FairValueGap,
  LiquidityPool,
  LiquidityStatus,
  MarketReadingStructure,
  OBImportance,
  OrderBlock,
} from '@/types/market-reading';
import { ZoneList } from './ZoneList';

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
  const { bos, choch, order_blocks, fair_value_gaps, retest_in_progress } =
    structure;
  const liquidity_pools = structure.liquidity_pools ?? [];

  // Click-to-chart wiring (display/navigation only). We reuse the EXISTING chart
  // view channel the M.I.A Agent drives: clicking a zone asks the chart to
  // `focus_zone` (centre) + `highlight_zone` (emphasise) it by its REAL engine
  // id. Optional provider — outside the /app workspace `applyActions` is a no-op,
  // so the list stays readable with no chart wired in. Detection is never touched.
  const { view: chartView, applyActions } = useChartViewOptional();

  // The id verrou: the ONLY zones a focus/highlight may reference are the ones
  // the engine emitted in THIS structure — identical to AppWorkspace's set.
  // A click can never invent or move a zone.
  const validZoneIds = useMemo(() => {
    const ids = new Set<string>();
    for (const ob of order_blocks) ids.add(ob.id);
    for (const fvg of fair_value_gaps) ids.add(fvg.id);
    return ids;
  }, [order_blocks, fair_value_gaps]);

  const selectZone = useCallback(
    (zoneId: string) => {
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
    [applyActions, validZoneIds],
  );

  // The selected entry mirrors the chart's highlighted zone (single source of
  // truth) — the highlight persists as the selection until another is clicked.
  const selectedZoneId = chartView.highlightZoneId;

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
          <span aria-hidden>📐</span>
          <span>Structure de marché</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        {/* Cadrage éditorial (niveau 1.5) — les structures sont décrites au
            présent : une cassure reste affichée tant que le moteur la considère
            active (en attente / retest), et disparaît dès sa reprise ou son
            invalidation. Les zones OB/FVG sont indiquées à leur formation. */}
        <p className="mb-3 text-xs text-muted-foreground">
          Structures décrites au présent : affichées tant qu’elles restent
          vérifiables, retirées dès leur reprise ou invalidation. Les zones Order
          Block et Fair Value Gap sont indiquées à leur formation.
        </p>
        {!hasAnything ? (
          <p className="text-sm text-muted-foreground">
            Aucun élément structurel notable sur la dernière bougie.
          </p>
        ) : (
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Row
              label="Cassure de structure (BOS)"
              termKey="bos"
              value={
                bos
                  ? `${formatPrice(bos.level, instrument)} · ${formatDirection(bos.direction)} · ${formatValidationStatus(bos.validation_status)}`
                  : bosUnderRetest
                    ? `cassure antérieure en cours de retest (${formatPrice(retest_in_progress!.level, instrument)})`
                    : 'aucune cassure récente'
              }
            />
            <Row
              label="Changement de caractère (CHOCH)"
              termKey="choch"
              value={
                choch
                  ? `${formatPrice(choch.level, instrument)} · ${formatDirection(choch.direction)} · ${formatValidationStatus(choch.validation_status)}`
                  : chochUnderRetest
                    ? `changement antérieur en cours de retest (${formatPrice(retest_in_progress!.level, instrument)})`
                    : 'aucun changement récent'
              }
            />
            <ZoneRow label="Order Blocks" termKey="order_block">
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
                    `${formatBand(ob.level_low, ob.level_high, instrument)} · importance ${formatObImportance(ob.importance)} · ${formatObStatus(ob.status)}`
                  }
                  idOf={(ob) => ob.id}
                  onSelect={selectZone}
                  selectedZoneId={selectedZoneId}
                />
              ) : (
                <span className="text-sm font-medium text-foreground">
                  aucun bloc significatif
                </span>
              )}
            </ZoneRow>
            <ZoneRow label="Fair Value Gaps" termKey="fvg">
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
                    `${formatBand(fvg.level_low, fvg.level_high, instrument)} · ${formatFvgStatus(fvg.status)}`
                  }
                  idOf={(fvg) => fvg.id}
                  onSelect={selectZone}
                  selectedZoneId={selectedZoneId}
                />
              ) : (
                <span className="text-sm font-medium text-foreground">
                  aucune zone détectée
                </span>
              )}
            </ZoneRow>
            <ZoneRow label="Liquidité externe (BSL / SSL)" termKey="liquidity">
              {sortedLiquidity.length > 0 ? (
                <LiquidityList pools={sortedLiquidity} instrument={instrument} />
              ) : (
                <span className="text-sm font-medium text-foreground">
                  aucune poche de liquidité détectée
                </span>
              )}
            </ZoneRow>
            <Row
              label="Retest en cours"
              termKey="retest"
              value={
                retest_in_progress
                  ? `${formatPrice(retest_in_progress.level, instrument)} · ${formatRetestType(retest_in_progress.type)}`
                  : 'aucun retest en cours'
              }
              className="sm:col-span-2"
            />
          </dl>
        )}
        <p className="mt-4 text-xs italic text-muted-foreground">
          Lecture descriptive — une invalidation structurelle est un fait de
          marché, pas un stop-loss imposé.
        </p>
      </AccordionContent>
    </AccordionItem>
  );
}

function Row({
  label,
  value,
  termKey,
  className,
}: {
  label: string;
  value: string;
  /** When set, the label becomes a vulgarisation tooltip (ⓘ + /methodology link). */
  termKey?: GlossaryKey;
  className?: string;
}) {
  return (
    <div className={cn(className)}>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">
        {termKey ? <InfoTooltip termKey={termKey}>{label}</InfoTooltip> : label}
      </dt>
      <dd className="mt-1 text-sm font-medium text-foreground">{value}</dd>
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
}: {
  pools: LiquidityPool[];
  instrument: string;
}) {
  return (
    <ul className="flex flex-col gap-1.5">
      {pools.map((p) => {
        const status = formatLiquidityStatus(p.status);
        return (
          <li
            key={p.id}
            className="flex items-center gap-2 text-sm font-medium text-foreground"
          >
            <span
              className="inline-block h-2 w-2 shrink-0 rounded-full"
              style={{
                backgroundColor: LIQUIDITY_DOT[p.side],
                opacity: p.status === 'broken' ? 0.45 : 1,
              }}
              aria-hidden
            />
            <span className="tabular-nums">{formatPrice(p.level, instrument)}</span>
            <span className="text-muted-foreground">·</span>
            <span className="font-semibold tracking-wide">
              {formatLiquiditySideShort(p.side)}
            </span>
            <span className="text-muted-foreground">
              {formatLiquidityKind(p.kind)}
            </span>
            <span className="text-muted-foreground">·</span>
            <span
              className={cn(
                status.tone === 'warn' ? 'text-amber-600 dark:text-amber-500' : 'text-muted-foreground',
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

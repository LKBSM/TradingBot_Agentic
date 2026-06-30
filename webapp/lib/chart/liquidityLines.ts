/**
 * Pure builder for the chart's external-liquidity PRICE LINES.
 *
 * Turns the descriptive `structure.liquidity_pools` the backend already emits
 * (EQH/EQL + range extremes, each with an intact/swept/broken state) into
 * horizontal price-line descriptors. NOTHING here detects, recomputes, or
 * projects — it only reads engine-emitted pockets and decides how to draw them.
 * Split out of `ReadingChart` so the dedup/sort logic is unit-testable without a
 * canvas / lightweight-charts instance.
 *
 * Posture (niveau 1.5 strict): a liquidity line states WHERE resting liquidity
 * sits and WHETHER it is intact / swept / broken — a past, observable fact. It
 * carries no target, draw, bias or probability.
 */
import {
  formatLiquidityKind,
  formatLiquiditySide,
  formatLiquiditySideShort,
  formatLiquidityStatus,
} from '@/lib/market-reading/formatters';
import type {
  LiquidityPool,
  LiquiditySide,
  LiquidityStatus,
  MarketReadingStructure,
} from '@/types/market-reading';

/** Short kind code for the compact axis title (BSL EQH / SSL RL …). */
const KIND_SHORT: Record<LiquidityPool['kind'], string> = {
  equal_highs: 'EQH',
  equal_lows: 'EQL',
  range_high: 'RH',
  range_low: 'RL',
};

/** A render-ready liquidity line — plain primitives, no charting dependency. */
export interface LiquidityLine {
  id: string;
  /** Resting-liquidity price level. */
  price: number;
  side: LiquiditySide;
  status: LiquidityStatus;
  /** Compact axis label, e.g. "BSL · EQH". */
  title: string;
  /** Full descriptive label for tooltips / a11y, e.g. "liquidité acheteuse … · sommets égaux · balayée". */
  description: string;
}

export interface BuildLiquidityLinesOptions {
  /**
   * Include `broken` pockets (terminal — the level was closed through). Off by
   * default for the chart so consumed levels don't clutter the canvas; the
   * Structure panel still lists them. Display choice only; detection untouched.
   */
  includeBroken?: boolean;
}

/** Draw priority so intact lines layer on top of swept, swept on top of broken. */
const STATUS_RANK: Record<LiquidityStatus, number> = {
  broken: 0,
  swept: 1,
  intact: 2,
};

/**
 * Build the liquidity price lines from a structure.
 *
 * Rules (descriptive only):
 *   · one line per pocket at its `level`;
 *   · pockets with a non-finite level are skipped;
 *   · `broken` pockets are excluded unless `includeBroken` is set;
 *   · output is sorted by status priority (broken < swept < intact) so the more
 *     relevant lines are created last (drawn on top).
 */
export function buildLiquidityLines(
  structure: MarketReadingStructure,
  options: BuildLiquidityLinesOptions = {},
): LiquidityLine[] {
  const { includeBroken = false } = options;
  const pools = structure.liquidity_pools ?? [];
  const lines: LiquidityLine[] = [];

  for (const p of pools) {
    if (!Number.isFinite(p.level)) continue;
    if (p.status === 'broken' && !includeBroken) continue;
    const sideShort = formatLiquiditySideShort(p.side);
    lines.push({
      id: p.id,
      price: p.level,
      side: p.side,
      status: p.status,
      title: `${sideShort} · ${KIND_SHORT[p.kind]}`,
      description: `${formatLiquiditySide(p.side)} · ${formatLiquidityKind(p.kind)} · ${formatLiquidityStatus(p.status).label}`,
    });
  }

  lines.sort((a, b) => STATUS_RANK[a.status] - STATUS_RANK[b.status]);
  return lines;
}

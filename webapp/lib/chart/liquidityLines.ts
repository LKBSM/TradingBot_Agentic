/**
 * Pure builder for the chart's external-liquidity LINE SEGMENTS.
 *
 * Turns the descriptive `structure.liquidity_pools` the backend already emits
 * (EQH/EQL + range extremes, each with an intact/swept/broken state) into
 * horizontal segment descriptors, TIME-BOUNDED like the OB/FVG boxes: a segment
 * starts at the pocket's formation (`created_at`) and — once price has touched
 * the level — FREEZES its right end at the first contact (`swept_at` /
 * `broken_at`, both engine-emitted). NOTHING here detects, recomputes, or
 * projects — it only reads engine-emitted pockets and decides how to draw them.
 * Split out of `ReadingChart` so the logic is unit-testable without a canvas /
 * lightweight-charts instance.
 *
 * Posture (niveau 1.5 strict): a liquidity line states WHERE resting liquidity
 * sits and WHETHER it is intact / swept (prise) / broken (cassée) — a past,
 * observable fact. It carries no target, draw, bias or probability. A swept
 * pocket is NEVER removed: touched is not broken — it stays, frozen, in its
 * own style.
 */
import {
  formatLiquidityKind,
  formatLiquiditySide,
  formatLiquiditySideShort,
  formatLiquidityStatus,
} from '@/lib/market-reading/formatters';
import { isoToSec } from '@/lib/chart/zoneLayout';
import type {
  LiquidityKind,
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

/** A render-ready liquidity segment — plain primitives, no charting dependency. */
export interface LiquidityLine {
  id: string;
  /** Resting-liquidity price level. */
  price: number;
  side: LiquiditySide;
  status: LiquidityStatus;
  /** Compact axis label, e.g. "BSL · EQH". */
  title: string;
  /** Full descriptive label for tooltips / a11y, e.g. "liquidité acheteuse … · sommets égaux · prise". */
  description: string;
  /** Short on-chart label, e.g. "Liquidité achat · intacte". */
  chartLabel: string;
  /**
   * Formation time (x-start anchor), UNIX seconds. NaN when `created_at` is
   * unparseable — the renderer then anchors at the left edge of the loaded
   * window rather than dropping a real pocket.
   */
  createdSec: number;
  /**
   * FIRST CONTACT (x-end freeze anchor), UNIX seconds, or null while the pocket
   * is intact — an intact segment extends to the current bar, like an active OB.
   * Read from the engine's `swept_at` / `broken_at` (earliest non-null): a
   * broken pocket that was swept first froze where price FIRST touched it.
   * Defensive: a swept/broken pocket missing both timestamps yields null (the
   * segment extends to the current bar in its state's style — never invented).
   */
  contactSec: number | null;
}

/**
 * Localized label producers for the segment's human-readable strings, INJECTED
 * by the React caller (this module is a pure, hook-free lib). Each returns the
 * already-localized label for one enum value. Optional: when omitted, the
 * built-in French formatters are used, so existing callers keep their output.
 */
export interface LiquidityLineLabels {
  /** Full side label, e.g. "liquidité acheteuse …". */
  side(side: LiquiditySide): string;
  /** Short side code for the compact axis title, e.g. "BSL". */
  sideShort(side: LiquiditySide): string;
  /** Short on-chart side tag, e.g. "Liquidité achat". */
  sideChart(side: LiquiditySide): string;
  /** Pocket kind label, e.g. "sommets égaux". */
  kind(kind: LiquidityKind): string;
  /** Status label, e.g. "prise" / "intacte" / "cassée". */
  status(status: LiquidityStatus): string;
}

/** Built-in French labels — the pure formatters, wrapped as a {@link LiquidityLineLabels}. */
const FR_LIQUIDITY_LABELS: LiquidityLineLabels = {
  side: (s) => formatLiquiditySide(s),
  sideShort: (s) => formatLiquiditySideShort(s),
  sideChart: (s) => SIDE_CHART_LABEL[s],
  kind: (k) => formatLiquidityKind(k),
  status: (st) => formatLiquidityStatus(st).label,
};

export interface BuildLiquidityLinesOptions {
  /**
   * Display filter: show only `intact` pockets (hides swept + broken segments).
   * Reversible, display-only — the pools themselves are untouched and the
   * Structure panel still lists every state. Default: everything visible.
   */
  intactOnly?: boolean;
  /**
   * Localized label producers (side / kind / status). Injected by the React
   * caller for i18n; defaults to the built-in French formatters so existing
   * callers stay backward-compatible.
   */
  labels?: LiquidityLineLabels;
}

/** Draw priority so intact lines layer on top of swept, swept on top of broken. */
const STATUS_RANK: Record<LiquidityStatus, number> = {
  broken: 0,
  swept: 1,
  intact: 2,
};

/** Side label for the on-chart tag — deliberately short ("achat"/"vente"). */
const SIDE_CHART_LABEL: Record<LiquiditySide, string> = {
  bsl: 'Liquidité achat',
  ssl: 'Liquidité vente',
};

/**
 * Earliest engine-emitted contact timestamp of a pool, UNIX seconds, or null
 * while intact / when no timestamp is parseable. Read-only over `swept_at` /
 * `broken_at` — never derived from price data.
 */
export function poolContactSec(pool: LiquidityPool): number | null {
  if (pool.status === 'intact') return null;
  const swept = isoToSec(pool.swept_at ?? null);
  const broken = isoToSec(pool.broken_at ?? null);
  const candidates = [swept, broken].filter((s) => Number.isFinite(s));
  return candidates.length ? Math.min(...candidates) : null;
}

/**
 * Build the liquidity segments from a structure.
 *
 * Rules (descriptive only):
 *   · one segment per pocket at its `level`, EVERY state included (a swept or
 *     broken pocket stays visible, frozen at its contact, in its own style);
 *   · pockets with a non-finite level are skipped;
 *   · `intactOnly` hides swept + broken segments (display filter, reversible);
 *   · output is sorted by status priority (broken < swept < intact) so the more
 *     relevant lines are created last (drawn on top).
 */
export function buildLiquidityLines(
  structure: MarketReadingStructure,
  options: BuildLiquidityLinesOptions = {},
): LiquidityLine[] {
  const { intactOnly = false, labels = FR_LIQUIDITY_LABELS } = options;
  const pools = structure.liquidity_pools ?? [];
  const lines: LiquidityLine[] = [];

  for (const p of pools) {
    if (!Number.isFinite(p.level)) continue;
    if (intactOnly && p.status !== 'intact') continue;
    const sideShort = labels.sideShort(p.side);
    const status = labels.status(p.status);
    lines.push({
      id: p.id,
      price: p.level,
      side: p.side,
      status: p.status,
      title: `${sideShort} · ${KIND_SHORT[p.kind]}`,
      description: `${labels.side(p.side)} · ${labels.kind(p.kind)} · ${status}`,
      chartLabel: `${labels.sideChart(p.side)} · ${status}`,
      createdSec: isoToSec(p.created_at),
      contactSec: poolContactSec(p),
    });
  }

  lines.sort((a, b) => STATUS_RANK[a.status] - STATUS_RANK[b.status]);
  return lines;
}

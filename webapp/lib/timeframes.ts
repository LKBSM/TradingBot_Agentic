// TF-1 — the frontend timeframe registry. Everything about a timeframe derives
// from TIMEFRAME_SPECS (generated from config/timeframes.json, the SAME source
// the backend reads). No component enumerates timeframes: they call these
// helpers. Adding a seventh unit = edit the JSON + regenerate, nothing here.

import { TIMEFRAME_SPECS, type TimeframeSpec } from './timeframes.generated';

export type { TimeframeSpec };
export { TIMEFRAME_SPECS };

const BY_ID: Record<string, TimeframeSpec> = Object.fromEntries(
  TIMEFRAME_SPECS.map((s) => [s.id, s]),
);

export function tfSpec(tf: string): TimeframeSpec | null {
  return BY_ID[(tf ?? '').toUpperCase()] ?? null;
}

export function tfSeconds(tf: string): number | null {
  return tfSpec(tf)?.seconds ?? null;
}

export function tfMinutes(tf: string): number | null {
  return tfSpec(tf)?.minutes ?? null;
}

export function tfLabelLong(tf: string): string {
  return tfSpec(tf)?.labelLong ?? tf;
}

export function tfDateFormat(tf: string): string | null {
  return tfSpec(tf)?.dateFormat ?? null;
}

export const ALL_TIMEFRAME_IDS: readonly string[] = TIMEFRAME_SPECS.map((s) => s.id);
export const PERIMETER_TIMEFRAMES: readonly string[] = TIMEFRAME_SPECS.filter(
  (s) => s.perimeter,
).map((s) => s.id);
export const REFERENCE_TIMEFRAMES: readonly string[] = TIMEFRAME_SPECS.filter(
  (s) => s.reference,
).map((s) => s.id);

/** id → seconds. Replaces the scattered TF_SECONDS / INTERVAL_SECONDS maps. */
export const TF_SECONDS: Record<string, number> = Object.fromEntries(
  TIMEFRAME_SPECS.map((s) => [s.id, s.seconds]),
);
/** id → minutes. Replaces the scattered TIMEFRAME_MINUTES copies. */
export const TF_MINUTES: Record<string, number> = Object.fromEntries(
  TIMEFRAME_SPECS.map((s) => [s.id, s.minutes]),
);
/** id → FR baseline label. Replaces formatters' TIMEFRAME_LABEL. */
export const TF_LABEL_LONG: Record<string, string> = Object.fromEntries(
  TIMEFRAME_SPECS.map((s) => [s.id, s.labelLong]),
);

/** Perimeter units strictly SLOWER than tf (ascending); empty at the top. */
export function upperTimeframes(tf: string): string[] {
  const base = tfSpec(tf);
  if (!base) return [];
  return TIMEFRAME_SPECS.filter((s) => s.perimeter && s.minutes > base.minutes).map((s) => s.id);
}

/** The immediately faster perimeter unit, or null at the bottom. */
export function lowerTimeframe(tf: string): string | null {
  const base = tfSpec(tf);
  if (!base) return null;
  const faster = TIMEFRAME_SPECS.filter((s) => s.perimeter && s.minutes < base.minutes);
  return faster.length ? faster[faster.length - 1]!.id : null;
}

/**
 * MTF alignment set (TF-1 decision C): perimeter units ABOVE the viewed one; at
 * the top of the perimeter (D1) fall back to the reference unit above (W1);
 * above that, none — the tile then reports "no higher unit" honestly.
 */
export function alignmentTimeframes(tf: string): string[] {
  const above = upperTimeframes(tf);
  if (above.length) return above;
  const base = tfSpec(tf);
  if (!base) return [];
  return TIMEFRAME_SPECS.filter((s) => s.reference && s.minutes > base.minutes).map((s) => s.id);
}

export function isSessionRelevant(tf: string): boolean {
  return tfSpec(tf)?.sessionRelevant ?? true;
}

export function isPrevLevelsRelevant(tf: string): boolean {
  return tfSpec(tf)?.prevLevelsRelevant ?? true;
}

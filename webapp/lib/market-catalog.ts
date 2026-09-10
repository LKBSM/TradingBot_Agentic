// MKT-1 TEST UX — the display-only market catalogue, and the flag that gates it.
//
// WHY THIS FILE IS SEPARATE FROM lib/markets.ts
// ---------------------------------------------
// lib/markets.ts is the REAL registry: what the engine follows, what the backend
// serves, what a deep-link may validly target. This file is a catalogue of market
// NAMES used to test how the interface holds at 100 entries, before the data for
// most of them exists. The two must never be confused, so they never merge:
//
//   • nothing here feeds ALL_MARKET_IDS or SUPPORTED_INSTRUMENTS;
//   • the catalogue carries no priceDecimals and no timeframes, so there is
//     structurally nothing to format a price or request a candle with;
//   • every consumer asks isCatalogOnly() before doing anything with an id.
//
// A PLAIN module (no 'use client'): the flag is read by server components too.

import { CATALOG_ENTRIES, CATALOG_GROUPS, type CatalogEntry, type CatalogGroup } from './market-catalog.generated';
import { ALL_MARKET_IDS } from './markets';

export type { CatalogEntry, CatalogGroup };
export { CATALOG_GROUPS };

/**
 * The UX-test catalogue gate. OFF by default EVERYWHERE, production included:
 * only an explicit `NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST=1` on the founder's
 * own environment turns it on.
 *
 * `NEXT_PUBLIC_` is not a style choice — Next.js exposes no other variable to
 * the browser, and the market list is a client component.
 *
 * MEASURED, not assumed: Next only inlines a NEXT_PUBLIC_ variable that EXISTS
 * at build time. Unset, the expression stays dynamic, so there is no dead-code
 * elimination and the ~8 kB catalogue (a list of public market names, nothing
 * sensitive) still ships in the client bundle — inert. What the flag guarantees
 * is behaviour, not bundle size: nothing renders, nothing is fetched.
 *
 * Truthiness follows the existing repo convention (cf. middleware.ts): '1' or 'true'.
 */
export const CATALOG_UX_TEST_ENABLED =
  process.env.NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST === '1' ||
  process.env.NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST === 'true';

const REAL_IDS = new Set<string>(ALL_MARKET_IDS);

/**
 * Catalogue entries the engine does NOT follow — the real registry always wins.
 * Deriving this by subtraction (rather than curating a second list) means a
 * market promoted into config/markets.json leaves the test catalogue on its own,
 * with no second edit to forget.
 */
export const CATALOG_ONLY_ENTRIES: readonly CatalogEntry[] = CATALOG_UX_TEST_ENABLED
  ? CATALOG_ENTRIES.filter((e) => !REAL_IDS.has(e.id))
  : [];
// Guarded by the flag even though isCatalogOnly() already checks it: with the
// flag off every derived structure below is empty, so no consumer ever walks the
// 100-entry table. (It does NOT remove the table from the bundle — verified on a
// real build: an unset NEXT_PUBLIC_ variable is not inlined, so no dead-code
// elimination happens. The table ships, inert.)

const CATALOG_ONLY_BY_ID: Record<string, CatalogEntry> = Object.fromEntries(
  CATALOG_ONLY_ENTRIES.map((e) => [e.id, e]),
);

/**
 * True when `id` is shown only because the UX-test catalogue is on — i.e. there
 * is NO real data behind it and every data path must refuse to fetch.
 *
 * Returns false when the flag is off, so a stale deep-link to a catalogue market
 * in a normal build falls through to the ordinary "unsupported combination"
 * path rather than to a state that would not exist there.
 */
export function isCatalogOnly(id: string | null | undefined): boolean {
  if (!CATALOG_UX_TEST_ENABLED) return false;
  return Boolean(id) && (id as string).toUpperCase() in CATALOG_ONLY_BY_ID;
}

/** Catalogue metadata for a display-only market (null for a real or unknown one). */
export function catalogEntry(id: string | null | undefined): CatalogEntry | null {
  return CATALOG_ONLY_BY_ID[(id ?? '').toUpperCase()] ?? null;
}

/** Display label for a catalogue-only market; falls back to the raw id. */
export function catalogLabel(id: string): string {
  return catalogEntry(id)?.label ?? id;
}

/** Short mono badge for a catalogue-only market. */
export function catalogGlyph(id: string): string {
  return catalogEntry(id)?.glyph ?? (id ?? '').slice(0, 2);
}

/**
 * How many markets are listed for display only. Rendered in the "mode test UX"
 * banner so the founder always sees the real ratio — computed, never written
 * down, so it stays true if the catalogue changes.
 */
export const CATALOG_ONLY_COUNT = CATALOG_ONLY_ENTRIES.length;

/** Catalogue-only entries of one group, in catalogue order. */
export function catalogEntriesByGroup(group: CatalogGroup): readonly CatalogEntry[] {
  return CATALOG_ONLY_ENTRIES.filter((e) => e.group === group);
}

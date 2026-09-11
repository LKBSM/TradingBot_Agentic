import { afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ALL_MARKET_IDS } from '@/lib/markets';
import { SUPPORTED_INSTRUMENTS } from '@/lib/market-reading/perimeter';

/**
 * MKT-1 test UX — the flag and the catalogue/registry separation.
 *
 * The flag is read at MODULE LOAD (it is inlined at build time in a real build),
 * so each case stubs the env then re-imports the module through vi.resetModules.
 */

const REPO = resolve(process.cwd(), '..');
const MODULE = '@/lib/market-catalog';

async function loadWith(value: string | undefined) {
  vi.resetModules();
  if (value === undefined) {
    vi.stubEnv('NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST', '');
    delete process.env.NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST;
  } else {
    vi.stubEnv('NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST', value);
  }
  return import(MODULE);
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe('MKT-1 flag — OFF by default', () => {
  it('a standard build with NO env var set has the catalogue disabled', async () => {
    const m = await loadWith(undefined);
    expect(m.CATALOG_UX_TEST_ENABLED).toBe(false);
  });

  it('isCatalogOnly() answers false for every catalogue market while the flag is off', async () => {
    const m = await loadWith(undefined);
    // The ids come from the GENERATED module, which does not depend on the flag —
    // otherwise this would trivially pass on an empty list.
    const { CATALOG_ENTRIES } = await import('@/lib/market-catalog.generated');
    expect(CATALOG_ENTRIES.length).toBeGreaterThan(50);
    for (const entry of CATALOG_ENTRIES) {
      expect(m.isCatalogOnly(entry.id), `${entry.id} must be inert while the flag is off`).toBe(false);
    }
  });

  it('the derived catalogue is EMPTY while the flag is off, so a build can drop it', async () => {
    const m = await loadWith(undefined);
    // Not merely gated at call time: every derived structure is empty, so no
    // consumer walks the catalogue. (The table still SHIPS in the bundle — an
    // unset NEXT_PUBLIC_ var is not inlined, so there is no dead-code
    // elimination. The guarantee here is behavioural, not bundle size.)
    expect(m.CATALOG_ONLY_ENTRIES).toEqual([]);
    expect(m.CATALOG_ONLY_COUNT).toBe(0);
  });

  it('only an explicit 1 / true opens the gate', async () => {
    for (const on of ['1', 'true']) {
      const m = await loadWith(on);
      expect(m.CATALOG_UX_TEST_ENABLED, `value ${on}`).toBe(true);
    }
    // Anything else stays closed — no accidental truthiness ("0", "yes", "on"…).
    for (const off of ['0', 'false', 'yes', 'on', 'TRUE', ' 1', '']) {
      const m = await loadWith(off);
      expect(m.CATALOG_UX_TEST_ENABLED, `value ${JSON.stringify(off)}`).toBe(false);
    }
  });
});

describe('MKT-1 — the test catalogue never touches the real perimeter', () => {
  it('no catalogue-only market is a supported instrument, flag on or off', async () => {
    const m = await loadWith('1');
    expect(m.CATALOG_UX_TEST_ENABLED).toBe(true);
    for (const entry of m.CATALOG_ONLY_ENTRIES) {
      expect(SUPPORTED_INSTRUMENTS, `${entry.id} leaked into the perimeter`).not.toContain(entry.id);
      expect(ALL_MARKET_IDS, `${entry.id} leaked into the registry`).not.toContain(entry.id);
    }
  });

  it('the real registry still holds exactly the markets the engine follows', () => {
    // The catalogue must not have grown the product perimeter as a side effect.
    expect([...ALL_MARKET_IDS].sort()).toEqual(['EURUSD', 'XAUUSD']);
  });

  it('a market present in BOTH files counts as real, never as catalogue-only', async () => {
    const m = await loadWith('1');
    const raw = JSON.parse(
      readFileSync(resolve(REPO, 'config/market_catalog_ux_test.json'), 'utf-8'),
    );
    const catalogueIds: string[] = raw.markets.map((e: { id: string }) => e.id);
    // The catalogue does list gold and EUR/USD — they ARE popular markets.
    expect(catalogueIds).toContain('XAUUSD');
    expect(catalogueIds).toContain('EURUSD');
    // …and they are deduplicated out of the display-only set, so the column
    // never shows the same market twice, once real and once empty.
    expect(m.isCatalogOnly('XAUUSD')).toBe(false);
    expect(m.isCatalogOnly('EURUSD')).toBe(false);
    expect(m.CATALOG_ONLY_COUNT).toBe(catalogueIds.length - 2);
  });

  it('catalogue entries carry NO price precision and NO timeframe', () => {
    // Structural honesty: with neither field there is nothing to format a price
    // with, and no unit to request a candle for.
    const raw = JSON.parse(
      readFileSync(resolve(REPO, 'config/market_catalog_ux_test.json'), 'utf-8'),
    );
    for (const entry of raw.markets) {
      expect(entry).not.toHaveProperty('priceDecimals');
      expect(entry).not.toHaveProperty('timeframes');
    }
  });
});

describe('MKT-1 — catalogue integrity', () => {
  it('market-catalog.generated.ts is in sync with the JSON', async () => {
    const m = await loadWith('1');
    const raw = JSON.parse(
      readFileSync(resolve(REPO, 'config/market_catalog_ux_test.json'), 'utf-8'),
    );
    const { CATALOG_ENTRIES } = await import('@/lib/market-catalog.generated');
    const fromJson = raw.markets.map((e: Record<string, unknown>) => ({
      id: e.id,
      label: e.label,
      symbol: e.symbol,
      group: e.group,
      glyph: e.glyph,
    }));
    const fromGen = CATALOG_ENTRIES.map((e) => ({
      id: e.id,
      label: e.label,
      symbol: e.symbol,
      group: e.group,
      glyph: e.glyph,
    }));
    expect(fromGen, 'run `node scripts/gen_market_catalog.mjs`').toEqual(fromJson);
    expect(m.CATALOG_ONLY_COUNT).toBeGreaterThan(0);
  });

  it('lists 100 markets, unique ids, every group non-empty', async () => {
    const { CATALOG_ENTRIES } = await import('@/lib/market-catalog.generated');
    const m = await loadWith('1');
    expect(CATALOG_ENTRIES).toHaveLength(100);
    expect(new Set(CATALOG_ENTRIES.map((e) => e.id)).size).toBe(100);
    for (const group of m.CATALOG_GROUPS) {
      expect(m.catalogEntriesByGroup(group).length, `group ${group} is empty`).toBeGreaterThan(0);
    }
  });
});

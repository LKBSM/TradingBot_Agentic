import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  MARKET_SPECS,
  ALL_MARKET_IDS,
  MARKET_PRICE_DECIMALS,
} from '@/lib/markets';

const REPO = resolve(process.cwd(), '..');

describe('MKT-1 — the frontend registry is the single source', () => {
  it('markets.generated.ts is in sync with config/markets.json', () => {
    const raw = JSON.parse(readFileSync(resolve(REPO, 'config/markets.json'), 'utf-8'));
    const fromJson = raw.markets.map((m: Record<string, unknown>) => ({
      id: m.id,
      label: m.label,
      symbol: m.symbol,
      type: m.type,
      priceDecimals: m.priceDecimals,
      glyph: m.glyph,
      timeframes: m.timeframes,
    }));
    const fromGen = MARKET_SPECS.map((s) => ({
      id: s.id,
      label: s.label,
      symbol: s.symbol,
      type: s.type,
      priceDecimals: s.priceDecimals,
      glyph: s.glyph,
      timeframes: [...s.timeframes],
    }));
    expect(fromGen, 'run `node scripts/gen_markets.mjs` after editing the JSON').toEqual(fromJson);
  });

  it('the two V1 markets carry their conventional precision (symptom guard)', () => {
    expect(ALL_MARKET_IDS).toContain('XAUUSD');
    expect(ALL_MARKET_IDS).toContain('EURUSD');
    expect(MARKET_PRICE_DECIMALS['XAUUSD']).toBe(2);
    expect(MARKET_PRICE_DECIMALS['EURUSD']).toBe(5);
  });

  it('every market timeframe id exists in the timeframe registry (TF-1)', () => {
    const tfRaw = JSON.parse(readFileSync(resolve(REPO, 'config/timeframes.json'), 'utf-8'));
    const known = new Set(tfRaw.timeframes.map((e: { id: string }) => e.id));
    for (const m of MARKET_SPECS) {
      for (const tf of m.timeframes) {
        expect(known, `${m.id} references unknown timeframe ${tf}`).toContain(tf);
      }
    }
  });
});

// Scan webapp source for a re-introduced hardcoded market list/map — the thing
// this mission exists to abolish. Every enumeration must derive from @/lib/markets.
const FORBIDDEN: [RegExp, string][] = [
  [/\[\s*['"]XAUUSD['"]\s*,\s*['"]EURUSD['"]/, 'inline market array'],
  [/new Set\(\s*\[\s*['"]XAUUSD['"]/, 'inline market Set'],
  [/XAUUSD:\s*['"][^'"]/, 'inline market label map'],
  [/XAUUSD:\s*\d/, 'inline market decimals map'],
  [/\bMARKET_GLYPH\b\s*[:=]\s*\{/, 'inline market glyph map'],
];
const ALLOW = new Set(['markets.ts', 'markets.generated.ts']);

/**
 * What the guard scans: PRODUCT source only. A test may legitimately pin a
 * fixed list of symbols as a fixture — `lib/market-reading/session.test.ts`
 * checks that `isTwentyFourSevenMarket` (a regex over the symbol, NOT a
 * registry lookup) answers "no" for gold, FX and indices. That is not the
 * enumeration MKT-1 abolishes.
 *
 * The guard already expressed that intent, but by DIRECTORY only
 * (`__tests__`). Four test files sit next to their module in
 * `lib/market-reading/`, so the guard shipped red on 2026-08-21 against a test
 * written on 2026-07-19. Excluding test files by NAME as well as by directory
 * is the same rule, applied consistently.
 */
const IS_TEST_FILE = /\.(test|spec)\.tsx?$/;

function walk(dir: string, hits: string[], seen: { files: number }): void {
  for (const name of readdirSync(dir)) {
    const p = resolve(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (name === 'node_modules' || name === '__tests__' || name === '.next') continue;
      walk(p, hits, seen);
    } else if (/\.(ts|tsx)$/.test(name) && !ALLOW.has(name) && !IS_TEST_FILE.test(name)) {
      seen.files += 1;
      const text = readFileSync(p, 'utf-8');
      for (const [re, what] of FORBIDDEN) {
        if (re.test(text)) hits.push(`${p.replace(REPO, '')} → ${what}`);
      }
    }
  }
}

describe('MKT-1 GUARD — no hardcoded market list in the webapp', () => {
  it('every enumeration derives from lib/markets', () => {
    const hits: string[] = [];
    const seen = { files: 0 };
    walk(resolve(process.cwd(), 'lib'), hits, seen);
    walk(resolve(process.cwd(), 'components'), hits, seen);
    // A broken exclusion could empty the perimeter and make the guard pass on
    // nothing at all. Assert it really read the product source.
    expect(seen.files, 'the guard must scan a non-empty perimeter').toBeGreaterThan(100);
    expect(hits, `derive from @/lib/markets instead:\n${hits.join('\n')}`).toEqual([]);
  });
});

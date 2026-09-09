/**
 * MIA-4S — the frozen scenario has ONE source, and this proves it.
 *
 * `config/demo_illustration.json` is what the simulated agent reads;
 * `lp1/data.ts` is what the page draws. Two copies of the same numbers is a
 * drift waiting to happen — the day they disagree, M.I.A describes a zone the
 * visitor cannot see, which is exactly the kind of quiet lie this product
 * refuses. So they are compared here, field by field.
 *
 * If this test fails: fix whichever file is wrong, never "align" it by hand in
 * both places without deciding which one is the truth.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DEMO_LEVELS, DEMO_REGIME, DEMO_ZONES } from '../data';

interface Band {
  id: string;
  low: number;
  high: number;
  fill_pct?: number;
}

const scenario = JSON.parse(
  readFileSync(join(process.cwd(), '..', 'config', 'demo_illustration.json'), 'utf-8'),
) as {
  current_price: number;
  structure: {
    choch: { level: number };
    bos: { level: number };
    order_blocks: Band[];
    fair_value_gaps: Band[];
    liquidity_pools: Array<{ id: string; level: number }>;
  };
  regime: Record<string, number>;
};

const band = (id: string): Band => {
  const all = [...scenario.structure.order_blocks, ...scenario.structure.fair_value_gaps];
  const found = all.find((b) => b.id === id);
  if (!found) throw new Error(`the scenario has no zone "${id}"`);
  return found;
};

const pool = (id: string) => {
  const found = scenario.structure.liquidity_pools.find((p) => p.id === id);
  if (!found) throw new Error(`the scenario has no pocket "${id}"`);
  return found;
};

describe('the drawn scenario and the scenario M.I.A reads are the same one', () => {
  it('the price levels match', () => {
    expect(DEMO_LEVELS.currentPrice).toBe(scenario.current_price);
    expect(DEMO_LEVELS.chochLevel).toBe(scenario.structure.choch.level);
    expect(DEMO_LEVELS.bosLevel).toBe(scenario.structure.bos.level);
    expect(DEMO_LEVELS.liqIntact).toBe(pool('demo-bsl-1').level);
    expect(DEMO_LEVELS.liqSwept).toBe(pool('demo-ssl-1').level);
  });

  it('the structure tab draws the zones the agent describes', () => {
    expect([DEMO_LEVELS.obLow, DEMO_LEVELS.obHigh]).toEqual([
      band('demo-ob-1').low,
      band('demo-ob-1').high,
    ]);
    expect([DEMO_LEVELS.fvgLow, DEMO_LEVELS.fvgHigh]).toEqual([
      band('demo-fvg-1').low,
      band('demo-fvg-1').high,
    ]);
  });

  it('the three zones of the zones tab are the three zones of the scenario', () => {
    const expected: Array<[string, string]> = [
      ['untested', 'demo-ob-1'],
      ['tested', 'demo-fvg-1'],
      ['filled', 'demo-ob-2'],
    ];
    for (const [key, id] of expected) {
      const zone = DEMO_ZONES.find((z) => z.key === key);
      expect(zone, `no demo zone "${key}"`).toBeTruthy();
      expect([zone!.low, zone!.high]).toEqual([band(id).low, band(id).high]);
    }
    // the partially filled one agrees on HOW filled it is
    expect(DEMO_ZONES.find((z) => z.key === 'tested')!.fill).toBe(band('demo-fvg-1').fill_pct);
  });

  it('the régime numbers match', () => {
    expect(Number(DEMO_REGIME.recentAtr.replace(',', '.'))).toBe(scenario.regime.recent_atr);
    expect(Number(DEMO_REGIME.baselineAtr.replace(',', '.'))).toBe(scenario.regime.baseline_atr);
    expect(Number(DEMO_REGIME.ratio.replace(',', '.'))).toBe(scenario.regime.ratio);
    expect(Number(DEMO_REGIME.lowThreshold.replace(',', '.'))).toBe(scenario.regime.low_threshold);
    expect(Number(DEMO_REGIME.highThreshold.replace(',', '.'))).toBe(scenario.regime.high_threshold);
  });
});

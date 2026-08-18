import { describe, expect, it } from 'vitest';
import { zoneGaugeLayout, GAUGE_MARGIN_RATIO } from '../gauge';
import type { ZoneLifecycle } from '../lifecycle';

// The layout only reads levelLow / levelHigh — a thin cast keeps the fixtures honest.
function zone(low: number, high: number): ZoneLifecycle {
  return { levelLow: low, levelHigh: high } as ZoneLifecycle;
}

describe('zoneGaugeLayout (VZ-3)', () => {
  it('always places the band in the middle 50% of the track, whatever the zone height', () => {
    const cases: Array<[number, number]> = [
      [100, 110],
      [4407.84, 4413.53],
      [2388, 2392],
      [0.5, 0.5001],
    ];
    for (const [lo, hi] of cases) {
      const g = zoneGaugeLayout(zone(lo, hi), lo - 1); // price below, out or in
      expect(g.bandLowPct).toBeCloseTo(25, 6);
      expect(g.bandHighPct).toBeCloseTo(75, 6);
    }
    expect(GAUGE_MARGIN_RATIO).toBe(0.5);
  });

  it('price ABOVE the zone but within the window → marker right, bracket from the high edge', () => {
    const z = zone(100, 110); // height 10, window [95, 115]
    const g = zoneGaugeLayout(z, 112); // inside window (< 115)
    expect(g.state).toBe('above');
    expect(g.outOfWindow).toBe(false);
    expect(g.refEdge).toBe('high');
    expect(g.pricePct).toBeCloseTo(((112 - 95) / 20) * 100, 6); // 85
    expect(g.bracket).not.toBeNull();
    expect(g.bracket!.fromPct).toBeCloseTo(75, 6);
    expect(g.bracket!.toPct).toBeCloseTo(85, 6);
  });

  it('price BELOW the zone but within the window → marker left, bracket to the low edge', () => {
    const z = zone(100, 110); // window [95, 115]
    const g = zoneGaugeLayout(z, 98);
    expect(g.state).toBe('below');
    expect(g.outOfWindow).toBe(false);
    expect(g.refEdge).toBe('low');
    expect(g.pricePct).toBeCloseTo(((98 - 95) / 20) * 100, 6); // 15
    expect(g.bracket!.fromPct).toBeCloseTo(15, 6);
    expect(g.bracket!.toPct).toBeCloseTo(25, 6);
  });

  it('price INSIDE the band → marker in the band, NO bracket', () => {
    const z = zone(100, 110);
    const g = zoneGaugeLayout(z, 105);
    expect(g.state).toBe('inside');
    expect(g.bracket).toBeNull();
    expect(g.outOfWindow).toBe(false);
    expect(g.pricePct).toBeGreaterThan(25);
    expect(g.pricePct).toBeLessThan(75);
  });

  it('price ABOVE the window → outAbove, pinned to 100%, no bracket', () => {
    const z = zone(100, 110); // window [95, 115]
    const g = zoneGaugeLayout(z, 200);
    expect(g.state).toBe('outAbove');
    expect(g.outOfWindow).toBe(true);
    expect(g.pricePct).toBe(100);
    expect(g.bracket).toBeNull();
    expect(g.refEdge).toBe('high');
  });

  it('price BELOW the window → outBelow, pinned to 0%, no bracket', () => {
    const z = zone(100, 110);
    const g = zoneGaugeLayout(z, 10);
    expect(g.state).toBe('outBelow');
    expect(g.outOfWindow).toBe(true);
    expect(g.pricePct).toBe(0);
    expect(g.bracket).toBeNull();
    expect(g.refEdge).toBe('low');
  });

  it('degenerate zero-height zone does not divide by zero', () => {
    const g = zoneGaugeLayout(zone(100, 100), 100);
    expect(Number.isFinite(g.bandLowPct)).toBe(true);
    expect(Number.isFinite(g.pricePct)).toBe(true);
  });
});

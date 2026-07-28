import { describe, expect, it } from 'vitest';
import {
  animateCamera,
  easeInOutCubic,
  frameEvent,
  frameLevel,
  frameZone,
  ZONE_OCCUPANCY_MAX,
  ZONE_OCCUPANCY_MIN,
  FRAME_MARGIN_FRAC,
  type CameraFrame,
} from '../focusController';

/**
 * VZ-1 — the framing math encodes the mission's cadrage rules verbatim. These
 * tests assert the invariants that make the gesture legible, so a future tweak
 * that breaks 25–60% occupancy / the visible couple / the readability fallback
 * fails loudly.
 */

describe('frameZone (mission VZ-1c — wide frame, zone 5–15%, ≥60 bars)', () => {
  const barSec = 900;
  // Formation exactly 100 bars back so the base frame shows the whole life.
  const base = { startSec: 1_000_000, lastSec: 1_090_000, barSec, bandLow: 2380, bandHigh: 2390 };

  it('sizes the window so the band occupies 5–15% of the visible height', () => {
    const f = frameZone(base);
    const height = f.priceMax! - f.priceMin!;
    const occupancy = (2390 - 2380) / height;
    expect(occupancy).toBeGreaterThanOrEqual(ZONE_OCCUPANCY_MIN);
    expect(occupancy).toBeLessThanOrEqual(ZONE_OCCUPANCY_MAX);
    // Target ~10% (relaxed from 20%): the zone is spotted, not dominant.
    expect(occupancy).toBeCloseTo(0.1, 2);
  });

  it('shows at least 60 candles — even for a freshly-formed zone', () => {
    // The old bug: formation→current collapsed to ~6 bars for a recent zone.
    const recent = { ...base, startSec: base.lastSec - 6 * barSec };
    const f = frameZone(recent);
    const bars = (f.to - f.from) / barSec;
    expect(bars).toBeGreaterThanOrEqual(60);
    // …and never absurdly wide (candles stay distinct).
    expect(bars).toBeLessThanOrEqual(130);
  });

  it('keeps the current bar visible (frame extends to/just past it)', () => {
    const f = frameZone(base);
    expect(f.to).toBeGreaterThanOrEqual(base.lastSec);
  });

  it('keeps ≥40% vertical margin above and below the band (no price)', () => {
    const f = frameZone(base);
    const height = f.priceMax! - f.priceMin!;
    expect((f.priceMax! - 2390) / height).toBeGreaterThanOrEqual(0.4 - 1e-9);
    expect((2380 - f.priceMin!) / height).toBeGreaterThanOrEqual(0.4 - 1e-9);
  });

  it('folds the current price into the view when the gap allows it', () => {
    // Price just outside the base window → it must sit inside the framed window.
    const f = frameZone({ ...base, price: 2460 });
    expect(f.priceMax!).toBeGreaterThanOrEqual(2460);
    const occ = (2390 - 2380) / (f.priceMax! - f.priceMin!);
    expect(occ).toBeGreaterThanOrEqual(ZONE_OCCUPANCY_MIN - 1e-9);
  });

  it('does NOT dezoom past the 5% floor for a far price (legibility first)', () => {
    const f = frameZone({ ...base, price: 3000 });
    const occ = (2390 - 2380) / (f.priceMax! - f.priceMin!);
    expect(occ).toBeGreaterThanOrEqual(ZONE_OCCUPANCY_MIN - 1e-9);
    expect(f.priceMax!).toBeLessThan(3000); // the far price is not forced in
  });

  it('includes the formation bar when it falls within the frame', () => {
    const f = frameZone(base);
    expect(f.from).toBeLessThan(1_000_000); // left of the formation bar
    expect(f.to).toBeGreaterThan(1_090_000); // right of the current bar
  });
});

describe('frameEvent (mission §B — 20 before / 10 after, level+candle together)', () => {
  const barSec = 900;
  it('centres on the confirmation bar with ≥20 bars before', () => {
    const atSec = 1_050_000;
    const f = frameEvent({ atSec, lastSec: 1_100_000, barSec, level: 2400 });
    expect(f.from).toBeLessThanOrEqual(atSec - 20 * barSec);
  });

  it('keeps the broken level AND the confirmation candle visible together', () => {
    const f = frameEvent({
      atSec: 1_050_000,
      lastSec: 1_100_000,
      barSec,
      level: 2400,
      candleLow: 2395,
      candleHigh: 2404,
    });
    // The full couple (level 2400 + candle 2395..2404) sits inside the window.
    expect(f.priceMin!).toBeLessThanOrEqual(2395);
    expect(f.priceMax!).toBeGreaterThanOrEqual(2404);
  });

  it('never runs the right edge past the current bar', () => {
    const atSec = 1_099_000;
    const lastSec = 1_100_000;
    const f = frameEvent({ atSec, lastSec, barSec, level: 2400 });
    expect(f.to).toBeLessThanOrEqual(lastSec + barSec * 2);
  });
});

describe('frameLevel (mission §B — level+price together, readability fallback)', () => {
  const barSec = 900;
  const lastSec = 1_100_000;
  it('shows the level and the current price together when they are close', () => {
    const f = frameLevel({
      level: 2410,
      price: 2400,
      lastSec,
      barSec,
      context: { low: 2398, high: 2402 },
    });
    expect(f.edge).toBeNull();
    expect(f.priceMin!).toBeLessThanOrEqual(2400);
    expect(f.priceMax!).toBeGreaterThanOrEqual(2410);
  });

  it('falls back to a legible window + edge indicator when the level is far', () => {
    // Level 5000 vs price 2400 with a tight recent range → fitting both would
    // crush the candles, so we keep a readable window and flag the edge.
    const f = frameLevel({
      level: 5000,
      price: 2400,
      lastSec,
      barSec,
      context: { low: 2398, high: 2402 },
    });
    expect(f.edge).toBe('above'); // the level sits above the framed window
    const height = f.priceMax! - f.priceMin!;
    const recentRange = 2402 - 2398;
    // Candles stay distinct: the recent range keeps a real share of the window.
    expect(recentRange / height).toBeGreaterThan(0.1);
    // The far level is NOT forced into the window (that is the whole point).
    expect(f.priceMax!).toBeLessThan(5000);
  });

  it('points the edge below when a far level sits under the price', () => {
    const f = frameLevel({
      level: 100,
      price: 2400,
      lastSec,
      barSec,
      context: { low: 2398, high: 2402 },
    });
    expect(f.edge).toBe('below');
  });
});

describe('easeInOutCubic', () => {
  it('is pinned at the ends and symmetric at the midpoint', () => {
    expect(easeInOutCubic(0)).toBe(0);
    expect(easeInOutCubic(1)).toBe(1);
    expect(easeInOutCubic(0.5)).toBeCloseTo(0.5, 6);
  });
});

describe('animateCamera', () => {
  const from: CameraFrame = { from: 0, to: 100, priceMin: 10, priceMax: 20, edge: null };
  const to: CameraFrame = { from: 50, to: 150, priceMin: 30, priceMax: 40, edge: null };

  it('reduced motion jumps straight to the target (no tween) and settles', () => {
    const ranges: Array<{ from: number; to: number }> = [];
    const prices: Array<{ min: number; max: number } | null> = [];
    let done = false;
    animateCamera({
      chart: { setVisibleRange: (r) => ranges.push(r) },
      from,
      to,
      setPriceTarget: (p) => prices.push(p),
      reducedMotion: true,
      onDone: () => (done = true),
    });
    expect(ranges).toEqual([{ from: 50, to: 150 }]);
    expect(prices).toEqual([{ min: 30, max: 40 }]);
    expect(done).toBe(true);
  });

  it('tweens both axes and ends exactly on the target', () => {
    const ranges: Array<{ from: number; to: number }> = [];
    let lastPrice: { min: number; max: number } | null = null;
    // now() sequence: start=0, then ticks at 200 (mid) and 400 (end).
    const times = [0, 200, 400];
    let i = 0;
    animateCamera({
      chart: { setVisibleRange: (r) => ranges.push(r) },
      from,
      to,
      setPriceTarget: (p) => (lastPrice = p),
      durationMs: 400,
      now: () => times[Math.min(i++, times.length - 1)]!,
      raf: (cb) => cb(), // run synchronously; the loop stops itself at raw>=1
    });
    // Final frame lands on the target on both axes.
    expect(ranges[ranges.length - 1]).toEqual({ from: 50, to: 150 });
    expect(lastPrice).toEqual({ min: 30, max: 40 });
    // A mid frame existed strictly between the endpoints (real interpolation).
    const mid = ranges[0]!;
    expect(mid.from).toBeGreaterThan(0);
    expect(mid.from).toBeLessThan(50);
  });

  it('cancel stops further frames', () => {
    const ranges: Array<{ from: number; to: number }> = [];
    const pending: Array<() => void> = [];
    const cancel = animateCamera({
      chart: { setVisibleRange: (r) => ranges.push(r) },
      from,
      to,
      setPriceTarget: () => {},
      durationMs: 400,
      now: () => 0, // never reaches the end on its own
      raf: (cb) => {
        pending.push(cb);
      },
    });
    const countAfterStart = ranges.length;
    cancel();
    pending.forEach((fn) => fn()); // queued frames must no-op after cancel
    expect(ranges.length).toBe(countAfterStart);
  });
});

describe('FRAME_MARGIN_FRAC', () => {
  it('is the 15% margin the mission fixed', () => {
    expect(FRAME_MARGIN_FRAC).toBe(0.15);
  });
});

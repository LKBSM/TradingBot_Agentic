import { describe, expect, it } from 'vitest';
import {
  ACTIVE_ZONE_CAP,
  TESTED_ZONE_CAP,
  buildLiveOverlay,
  buildZoneModels,
  curateZones,
  isObInTestLive,
  isoToSec,
  openFvgBand,
  provisionalOpenFvgBand,
  zoneEndSec,
  type ZoneModel,
} from '../zoneLayout';
import type { Direction, FairValueGap, MarketReadingStructure } from '@/types/market-reading';

function ob(
  id: string,
  status: 'active' | 'mitigated' | 'invalidated',
  high: number,
  low: number,
  created: string,
  mitigated?: string,
) {
  return {
    id,
    direction: 'bullish' as const,
    level_high: high,
    level_low: low,
    importance: 'medium' as const,
    status,
    created_at: created,
    tested: status !== 'active',
    mitigated_at: mitigated ?? null,
    user_flagged: false,
  };
}

function fvg(
  id: string,
  status: 'active' | 'partially_filled' | 'filled',
  high: number,
  low: number,
  created: string,
  mitigated?: string,
  extra: { direction?: Direction; fill_level?: number | null } = {},
): FairValueGap {
  return {
    id,
    direction: extra.direction ?? ('bullish' as const),
    level_high: high,
    level_low: low,
    status,
    created_at: created,
    tested: status !== 'active',
    mitigated_at: mitigated ?? null,
    fill_level: extra.fill_level ?? null,
    user_flagged: false,
  };
}

function structure(
  order_blocks: ReturnType<typeof ob>[],
  fair_value_gaps: ReturnType<typeof fvg>[] = [],
): MarketReadingStructure {
  return {
    bos: null,
    choch: null,
    order_blocks,
    fair_value_gaps,
    retest_in_progress: null,
  };
}

describe('isoToSec', () => {
  it('parses ISO to UNIX seconds', () => {
    expect(isoToSec('2026-05-26T00:00:00+00:00')).toBe(
      Math.floor(Date.UTC(2026, 4, 26) / 1000),
    );
  });
  it('returns NaN for empty / invalid', () => {
    expect(Number.isNaN(isoToSec(null))).toBe(true);
    expect(Number.isNaN(isoToSec('not-a-date'))).toBe(true);
  });
});

describe('buildZoneModels', () => {
  it('maps OB/FVG and reads mitigated_at (read-only)', () => {
    const s = structure(
      [ob('a', 'mitigated', 102, 100, '2026-05-26T05:00:00+00:00', '2026-05-26T08:00:00+00:00')],
      [fvg('f', 'active', 99, 98, '2026-05-26T06:00:00+00:00')],
    );
    const models = buildZoneModels(s);
    expect(models).toHaveLength(2);
    const a = models.find((m) => m.id === 'a')!;
    expect(a.tested).toBe(true);
    expect(a.mitigatedSec).toBe(isoToSec('2026-05-26T08:00:00+00:00'));
    const f = models.find((m) => m.id === 'f')!;
    expect(f.tested).toBe(false);
    expect(f.mitigatedSec).toBeNull();
  });

  it('NEVER surfaces a consumed zone (invalidated OB / filled FVG dropped)', () => {
    const s = structure(
      [ob('inv', 'invalidated', 102, 100, '2026-05-26T05:00:00+00:00')],
      [fvg('filled', 'filled', 99, 98, '2026-05-26T06:00:00+00:00')],
    );
    expect(buildZoneModels(s)).toEqual([]);
  });

  it('drops a zone whose formation time is unparseable (cannot anchor)', () => {
    const s = structure([ob('bad', 'active', 102, 100, 'nope')]);
    expect(buildZoneModels(s)).toEqual([]);
  });
});

describe('openFvgBand', () => {
  it('keeps the full band for an active gap (no fill_level)', () => {
    expect(openFvgBand(fvg('f', 'active', 101, 100, '2026-05-26T00:00:00Z'))).toEqual({
      high: 101,
      low: 100,
    });
  });

  it('shrinks a bullish partial fill to the still-open portion (top down)', () => {
    // Fills from above → open part below the penetration → high = fill_level.
    const f = fvg('f', 'partially_filled', 101, 100, '2026-05-26T00:00:00Z', undefined, {
      direction: 'bullish',
      fill_level: 100.5,
    });
    expect(openFvgBand(f)).toEqual({ high: 100.5, low: 100 });
  });

  it('shrinks a bearish partial fill to the still-open portion (bottom up)', () => {
    // Fills from below → open part above the penetration → low = fill_level.
    const f = fvg('f', 'partially_filled', 101, 100, '2026-05-26T00:00:00Z', undefined, {
      direction: 'bearish',
      fill_level: 100.5,
    });
    expect(openFvgBand(f)).toEqual({ high: 101, low: 100.5 });
  });

  it('never shrinks when fill_level sits outside the band or direction is unknown', () => {
    const outOfBand = fvg('f', 'partially_filled', 101, 100, '2026-05-26T00:00:00Z', undefined, {
      direction: 'bullish',
      fill_level: 99, // below the band → ignored
    });
    expect(openFvgBand(outOfBand)).toEqual({ high: 101, low: 100 });
    // direction null (doc-example shape): can't tell which side filled → keep full band.
    const noDir: FairValueGap = {
      id: 'g',
      direction: null,
      level_high: 101,
      level_low: 100,
      status: 'partially_filled',
      created_at: '2026-05-26T00:00:00Z',
      tested: true,
      mitigated_at: null,
      fill_level: 100.5,
      user_flagged: false,
    };
    expect(openFvgBand(noDir)).toEqual({ high: 101, low: 100 });
  });

  it('feeds the shrunk band through buildZoneModels', () => {
    const s = structure(
      [],
      [
        fvg('f', 'partially_filled', 101, 100, '2026-05-26T06:00:00+00:00', '2026-05-26T08:00:00+00:00', {
          direction: 'bullish',
          fill_level: 100.5,
        }),
      ],
    );
    const m = buildZoneModels(s).find((z) => z.id === 'f')!;
    expect(m.high).toBe(100.5);
    expect(m.low).toBe(100);
  });
});

describe('curateZones', () => {
  const mk = (
    id: string,
    tested: boolean,
    high: number,
    low: number,
    createdSec: number,
  ): ZoneModel => ({
    id,
    kind: 'ob',
    high,
    low,
    createdSec,
    mitigatedSec: tested ? createdSec + 3600 : null,
    tested,
    label: 'Order Block',
  });

  it('splits active vs tested and caps each independently', () => {
    const zones: ZoneModel[] = [];
    for (let i = 0; i < 8; i += 1) zones.push(mk(`act${i}`, false, 100 + i, 99 + i, 1000 + i));
    for (let i = 0; i < 6; i += 1) zones.push(mk(`tst${i}`, true, 100 + i, 99 + i, 1000 + i));
    const { active, tested } = curateZones(zones, 100);
    expect(active).toHaveLength(ACTIVE_ZONE_CAP);
    expect(tested).toHaveLength(TESTED_ZONE_CAP);
    expect(active.every((z) => !z.tested)).toBe(true);
    expect(tested.every((z) => z.tested)).toBe(true);
  });

  it('keeps the zones nearest to the current price (proximity weighs in)', () => {
    // Five active zones at increasing distance from price=100; cap=4 → the
    // farthest (mid ~140) must be dropped.
    const zones = [
      mk('p100', false, 101, 99, 5000), // mid 100
      mk('p110', false, 111, 109, 5000), // mid 110
      mk('p120', false, 121, 119, 5000), // mid 120
      mk('p130', false, 131, 129, 5000), // mid 130
      mk('p140', false, 141, 139, 5000), // mid 140 (farthest)
    ];
    const { active } = curateZones(zones, 100);
    expect(active).toHaveLength(4);
    expect(active.map((z) => z.id)).not.toContain('p140');
  });

  it('prefers more recent zones when proximity ties', () => {
    // All same mid-price → recency decides; oldest dropped at cap.
    const zones = [
      mk('old', false, 101, 99, 1000),
      mk('mid', false, 101, 99, 2000),
      mk('new', false, 101, 99, 3000),
    ];
    const { active } = curateZones(zones, 100, { active: 2 });
    expect(active).toHaveLength(2);
    expect(active.map((z) => z.id)).toContain('new');
    expect(active.map((z) => z.id)).not.toContain('old');
  });

  it('is deterministic (stable id tie-break)', () => {
    const zones = [mk('b', false, 101, 99, 1000), mk('a', false, 101, 99, 1000)];
    const r1 = curateZones(zones, 100).active.map((z) => z.id);
    const r2 = curateZones(zones, 100).active.map((z) => z.id);
    expect(r1).toEqual(r2);
  });
});

describe('zoneEndSec (right-edge time anchor)', () => {
  const lastBarSec = 2000;
  const formingSec = 2900; // live forming bar, further right than the last closed bar

  it('extends an ACTIVE zone to the live forming bar (current candle)', () => {
    expect(
      zoneEndSec({ tested: false, mitigatedSec: null }, { lastBarSec, formingSec }),
    ).toBe(formingSec);
  });

  it('extends an ACTIVE zone to the last closed bar when no tick is streaming', () => {
    // formingSec null → current bar IS the last closed bar (default view preserved).
    expect(
      zoneEndSec({ tested: false, mitigatedSec: null }, { lastBarSec, formingSec: null }),
    ).toBe(lastBarSec);
  });

  it('keeps a MITIGATED zone BOUNDED at its mitigation point (never over-extended)', () => {
    const mitigatedSec = 1500; // resolved before the current bar
    expect(
      zoneEndSec({ tested: true, mitigatedSec }, { lastBarSec, formingSec }),
    ).toBe(mitigatedSec);
  });

  it('falls back to the current bar for a tested zone with no usable mitigation point', () => {
    expect(
      zoneEndSec({ tested: true, mitigatedSec: null }, { lastBarSec, formingSec }),
    ).toBe(formingSec);
  });
});

// ─── Live (provisional, intra-candle) interaction ─────────────────────────────

describe('provisionalOpenFvgBand', () => {
  it('shrinks a bullish gap further as the live price eats it top-down', () => {
    // Active gap [100,101]; price at 100.4 (inside) → open band shrinks to [100,100.4].
    const f = fvg('f', 'active', 101, 100, '2026-05-26T00:00:00Z', undefined, {
      direction: 'bullish',
    });
    expect(provisionalOpenFvgBand(f, 100.4)).toEqual({ high: 100.4, low: 100 });
  });

  it('shrinks a bearish gap further as the live price eats it bottom-up', () => {
    const f = fvg('f', 'active', 101, 100, '2026-05-26T00:00:00Z', undefined, {
      direction: 'bearish',
    });
    expect(provisionalOpenFvgBand(f, 100.6)).toEqual({ high: 101, low: 100.6 });
  });

  it('starts from the CONFIRMED open band (deeper than a prior partial fill)', () => {
    // Bullish gap already partially filled to 100.5 → confirmed open band [100,100.5].
    // Live price 100.2 is deeper → provisional open band [100,100.2].
    const f = fvg('f', 'partially_filled', 101, 100, '2026-05-26T00:00:00Z', undefined, {
      direction: 'bullish',
      fill_level: 100.5,
    });
    expect(provisionalOpenFvgBand(f, 100.2)).toEqual({ high: 100.2, low: 100 });
  });

  it('returns null on a retreat (price above the confirmed front) — reverts cleanly', () => {
    const f = fvg('f', 'partially_filled', 101, 100, '2026-05-26T00:00:00Z', undefined, {
      direction: 'bullish',
      fill_level: 100.5,
    });
    // Price 100.8 is ABOVE the confirmed front (100.5) → no further shrink.
    expect(provisionalOpenFvgBand(f, 100.8)).toBeNull();
  });

  it('returns null for a filled gap, unknown direction, price outside, or non-finite', () => {
    const filled = fvg('f', 'filled', 101, 100, '2026-05-26T00:00:00Z', undefined, {
      direction: 'bullish',
    });
    expect(provisionalOpenFvgBand(filled, 100.4)).toBeNull();
    // Unknown direction — built as a literal (the fvg() helper defaults null → bullish).
    const noDir: FairValueGap = {
      id: 'g',
      direction: null,
      level_high: 101,
      level_low: 100,
      status: 'active',
      created_at: '2026-05-26T00:00:00Z',
      tested: false,
      mitigated_at: null,
      fill_level: null,
      user_flagged: false,
    };
    expect(provisionalOpenFvgBand(noDir, 100.4)).toBeNull();
    const active = fvg('h', 'active', 101, 100, '2026-05-26T00:00:00Z', undefined, {
      direction: 'bullish',
    });
    expect(provisionalOpenFvgBand(active, 105)).toBeNull(); // outside
    expect(provisionalOpenFvgBand(active, Number.NaN)).toBeNull();
  });
});

describe('isObInTestLive', () => {
  it('flags an ACTIVE OB the price is currently inside', () => {
    expect(isObInTestLive(ob('a', 'active', 102, 100, '2026-05-26T00:00:00Z'), 101)).toBe(true);
  });

  it('does NOT flag a mitigated/invalidated OB (already candle-confirmed)', () => {
    expect(isObInTestLive(ob('a', 'mitigated', 102, 100, '2026-05-26T00:00:00Z'), 101)).toBe(false);
    expect(isObInTestLive(ob('a', 'invalidated', 102, 100, '2026-05-26T00:00:00Z'), 101)).toBe(false);
  });

  it('does NOT flag when the price is outside the band or non-finite', () => {
    const a = ob('a', 'active', 102, 100, '2026-05-26T00:00:00Z');
    expect(isObInTestLive(a, 99)).toBe(false);
    expect(isObInTestLive(a, Number.NaN)).toBe(false);
  });
});

describe('buildLiveOverlay', () => {
  it('collects provisional FVG fronts and in-test OBs from the live price', () => {
    const s = structure(
      [
        ob('ob_in', 'active', 102, 100, '2026-05-26T00:00:00Z'), // price 100.4 inside
        ob('ob_out', 'active', 99, 98, '2026-05-26T00:00:00Z'), // price outside
      ],
      [
        fvg('fvg_in', 'active', 101, 100, '2026-05-26T00:00:00Z', undefined, {
          direction: 'bullish',
        }),
      ],
    );
    const overlay = buildLiveOverlay(s, 100.4);
    expect(overlay.fvgFronts).toEqual([{ id: 'fvg_in', high: 100.4, low: 100 }]);
    expect([...overlay.obInTest]).toEqual(['ob_in']);
  });

  it('returns an empty overlay for a null / non-finite price (no live state)', () => {
    const s = structure([ob('a', 'active', 102, 100, '2026-05-26T00:00:00Z')]);
    expect(buildLiveOverlay(s, null)).toEqual({ fvgFronts: [], obInTest: new Set() });
    expect(buildLiveOverlay(s, Number.NaN)).toEqual({ fvgFronts: [], obInTest: new Set() });
  });
});

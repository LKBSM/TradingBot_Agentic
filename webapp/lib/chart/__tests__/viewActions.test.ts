import { describe, expect, it } from 'vitest';
import {
  applyChartViewAction,
  coerceViewAction,
  coerceViewActions,
  DEFAULT_CHART_VIEW,
  type ChartViewState,
  type ViewAction,
} from '../viewActions';
import { applyZoneVisibility, filterZoneModels, type ZoneModel } from '../zoneLayout';

const ZONES = new Set(['ob_1', 'fvg_2']);

// ─── coerceViewAction — the inviolable line (display-only whitelist) ───────────

describe('coerceViewAction', () => {
  it('accepts a layer toggle', () => {
    expect(
      coerceViewAction(
        { action: 'set_layer_visibility', params: { layer: 'fvg', visible: false } },
        ZONES,
      ),
    ).toEqual({ action: 'set_layer_visibility', params: { layer: 'fvg', visible: false } });
  });

  it('accepts a MULTI-layer toggle and de-dupes (fvg + ob in one action)', () => {
    expect(
      coerceViewAction(
        { action: 'set_layer_visibility', params: { layers: ['fvg', 'ob'], visible: false } },
        ZONES,
      ),
    ).toEqual({
      action: 'set_layer_visibility',
      params: { layers: ['fvg', 'ob'], visible: false },
    });
    // duplicate collapsed, order preserved
    expect(
      coerceViewAction(
        { action: 'set_layer_visibility', params: { layers: ['ob', 'ob', 'fvg'], visible: true } },
        ZONES,
      ),
    ).toEqual({
      action: 'set_layer_visibility',
      params: { layers: ['ob', 'fvg'], visible: true },
    });
  });

  it('rejects a MULTI-layer toggle with a bad/empty/all/mixed list', () => {
    // "all" is not addressable inside an explicit subset.
    expect(
      coerceViewAction(
        { action: 'set_layer_visibility', params: { layers: ['fvg', 'all'], visible: false } },
        ZONES,
      ),
    ).toBeNull();
    // empty list, non-array, junk element
    expect(
      coerceViewAction({ action: 'set_layer_visibility', params: { layers: [], visible: false } }, ZONES),
    ).toBeNull();
    expect(
      coerceViewAction({ action: 'set_layer_visibility', params: { layers: 'fvg', visible: false } }, ZONES),
    ).toBeNull();
    // mixing `layer` and `layers` is ambiguous → dropped
    expect(
      coerceViewAction(
        { action: 'set_layer_visibility', params: { layer: 'fvg', layers: ['ob'], visible: false } },
        ZONES,
      ),
    ).toBeNull();
    // a non-bool visible still fails
    expect(
      coerceViewAction(
        { action: 'set_layer_visibility', params: { layers: ['fvg', 'ob'], visible: 'yes' } },
        ZONES,
      ),
    ).toBeNull();
  });

  it('rejects an off-list (create/move/resize) action', () => {
    for (const action of ['create_zone', 'place_ob', 'move_zone', 'resize_fvg']) {
      expect(coerceViewAction({ action, params: {} }, ZONES)).toBeNull();
    }
  });

  it('rejects any geometry-shaped param (no band edit possible)', () => {
    for (const key of ['price', 'level', 'level_high', 'level_low', 'high', 'low']) {
      expect(
        coerceViewAction(
          { action: 'focus_zone', params: { zone_id: 'ob_1', [key]: 2000 } },
          ZONES,
        ),
      ).toBeNull();
    }
  });

  it('rejects focus/highlight on an invented zone id', () => {
    expect(
      coerceViewAction({ action: 'focus_zone', params: { zone_id: 'ob_2000' } }, ZONES),
    ).toBeNull();
    expect(
      coerceViewAction({ action: 'highlight_zone', params: { zone_id: 'nope' } }, ZONES),
    ).toBeNull();
  });

  it('accepts hide/isolate on known zone ids and de-dupes', () => {
    expect(
      coerceViewAction({ action: 'hide_zones', params: { zone_ids: ['ob_1', 'fvg_2'] } }, ZONES),
    ).toEqual({ action: 'hide_zones', params: { zone_ids: ['ob_1', 'fvg_2'] } });
    expect(
      coerceViewAction({ action: 'isolate_zones', params: { zone_ids: ['ob_1', 'ob_1'] } }, ZONES),
    ).toEqual({ action: 'isolate_zones', params: { zone_ids: ['ob_1'] } });
  });

  it('rejects hide/isolate when ANY id is invented (nothing hidden)', () => {
    // « masque l'OB à 4160 » with no matching real zone → whole action dropped.
    expect(
      coerceViewAction({ action: 'hide_zones', params: { zone_ids: ['ob_1', 'ob_4160'] } }, ZONES),
    ).toBeNull();
    expect(
      coerceViewAction({ action: 'isolate_zones', params: { zone_ids: ['ghost'] } }, ZONES),
    ).toBeNull();
  });

  it('rejects hide/isolate with an empty or non-array list', () => {
    expect(coerceViewAction({ action: 'hide_zones', params: { zone_ids: [] } }, ZONES)).toBeNull();
    expect(
      coerceViewAction({ action: 'hide_zones', params: { zone_ids: 'ob_1' } }, ZONES),
    ).toBeNull();
  });

  it('show_zones: no ids restores all; an explicit list is validated', () => {
    expect(coerceViewAction({ action: 'show_zones', params: {} }, ZONES)).toEqual({
      action: 'show_zones',
      params: {},
    });
    expect(
      coerceViewAction({ action: 'show_zones', params: { zone_ids: ['ob_1'] } }, ZONES),
    ).toEqual({ action: 'show_zones', params: { zone_ids: ['ob_1'] } });
    expect(
      coerceViewAction({ action: 'show_zones', params: { zone_ids: ['ghost'] } }, ZONES),
    ).toBeNull();
  });

  it('accepts focus/highlight on a known zone id', () => {
    expect(
      coerceViewAction({ action: 'focus_zone', params: { zone_id: 'ob_1' } }, ZONES),
    ).toEqual({ action: 'focus_zone', params: { zone_id: 'ob_1' } });
  });

  it('clamps filter thresholds and drops an empty filter', () => {
    expect(
      coerceViewAction(
        { action: 'filter_zones', params: { proximity_pct: 999, min_size_pct: -5 } },
        ZONES,
      ),
    ).toEqual({ action: 'filter_zones', params: { proximity_pct: 10, min_size_pct: 0 } });
    expect(coerceViewAction({ action: 'filter_zones', params: {} }, ZONES)).toBeNull();
  });

  it('validates instrument/timeframe enums', () => {
    expect(
      coerceViewAction(
        { action: 'set_instrument_timeframe', params: { instrument: 'EURUSD', timeframe: 'H4' } },
        ZONES,
      ),
    ).toEqual({
      action: 'set_instrument_timeframe',
      params: { instrument: 'EURUSD', timeframe: 'H4' },
    });
    expect(
      coerceViewAction(
        { action: 'set_instrument_timeframe', params: { instrument: 'BTCUSD', timeframe: 'H4' } },
        ZONES,
      ),
    ).toBeNull();
  });

  it('accepts no-param framing actions', () => {
    for (const action of ['focus_price', 'fit_chart', 'reset_view'] as const) {
      expect(coerceViewAction({ action }, ZONES)).toEqual({ action, params: {} });
    }
  });

  it('rejects junk', () => {
    expect(coerceViewAction('masque les FVG', ZONES)).toBeNull();
    expect(coerceViewAction(null, ZONES)).toBeNull();
    expect(coerceViewAction({ action: 'fit_chart', params: [] }, ZONES)).toEqual({
      action: 'fit_chart',
      params: {},
    });
  });
});

describe('coerceViewActions', () => {
  it('keeps valid actions and drops invalid ones', () => {
    const raw = [
      { action: 'set_layer_visibility', params: { layer: 'ob', visible: false } },
      { action: 'create_zone', params: { price: 2000 } }, // dropped
      { action: 'focus_zone', params: { zone_id: 'ob_1' } },
    ];
    const out = coerceViewActions(raw, ZONES);
    expect(out.map((a) => a.action)).toEqual(['set_layer_visibility', 'focus_zone']);
  });

  it('returns [] for non-array input', () => {
    expect(coerceViewActions(null, ZONES)).toEqual([]);
  });
});

// ─── applyChartViewAction — reducer ───────────────────────────────────────────

describe('applyChartViewAction', () => {
  it('toggles a single layer', () => {
    const next = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'set_layer_visibility',
      params: { layer: 'fvg', visible: false },
    });
    expect(next.layers).toEqual({ fvg: false, ob: true, breaks: true, liquidity: true });
  });

  it('toggles MULTIPLE layers in one action (fvg + ob), leaving the rest intact', () => {
    const next = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'set_layer_visibility',
      params: { layers: ['fvg', 'ob'], visible: false },
    });
    expect(next.layers).toEqual({ fvg: false, ob: false, breaks: true, liquidity: true });
  });

  it('layer "all" toggles every overlay', () => {
    const next = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'set_layer_visibility',
      params: { layer: 'all', visible: false },
    });
    expect(next.layers).toEqual({ fvg: false, ob: false, breaks: false, liquidity: false });
  });

  it('merges filter changes', () => {
    const next = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'filter_zones',
      params: { active_only: true, min_size_pct: 0.2 },
    });
    expect(next.filter.activeOnly).toBe(true);
    expect(next.filter.minSizePct).toBe(0.2);
    expect(next.filter.proximityOnly).toBe(false); // untouched
  });

  it('focus actions bump the nonce each time', () => {
    const a = applyChartViewAction(DEFAULT_CHART_VIEW, { action: 'fit_chart', params: {} });
    const b = applyChartViewAction(a, { action: 'focus_price', params: {} });
    expect(a.focus?.kind).toBe('fit');
    expect(b.focus?.kind).toBe('price');
    expect((b.focus?.nonce ?? 0) > (a.focus?.nonce ?? 0)).toBe(true);
  });

  it('focus_zone carries the zone id', () => {
    const next = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'focus_zone',
      params: { zone_id: 'ob_1' },
    });
    expect(next.focus).toMatchObject({ kind: 'zone', zoneId: 'ob_1' });
  });

  it('highlight sets the highlighted id', () => {
    const next = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'highlight_zone',
      params: { zone_id: 'fvg_2' },
    });
    expect(next.highlightZoneId).toBe('fvg_2');
  });

  it('hide_zones unions ids into the hidden set (reversible)', () => {
    const a = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'hide_zones',
      params: { zone_ids: ['ob_1'] },
    });
    expect(a.hiddenZoneIds).toEqual(['ob_1']);
    const b = applyChartViewAction(a, { action: 'hide_zones', params: { zone_ids: ['fvg_2', 'ob_1'] } });
    expect(new Set(b.hiddenZoneIds)).toEqual(new Set(['ob_1', 'fvg_2']));
  });

  it('isolate_zones sets the isolation allow-list', () => {
    const next = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'isolate_zones',
      params: { zone_ids: ['ob_1'] },
    });
    expect(next.isolatedZoneIds).toEqual(['ob_1']);
  });

  it('show_zones with no ids restores all masks', () => {
    const masked = applyChartViewAction(
      applyChartViewAction(DEFAULT_CHART_VIEW, { action: 'hide_zones', params: { zone_ids: ['ob_1'] } }),
      { action: 'isolate_zones', params: { zone_ids: ['fvg_2'] } },
    );
    const restored = applyChartViewAction(masked, { action: 'show_zones', params: {} });
    expect(restored.hiddenZoneIds).toEqual([]);
    expect(restored.isolatedZoneIds).toBeNull();
  });

  it('show_zones with ids un-hides them (and re-adds under active isolation)', () => {
    let s = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'hide_zones',
      params: { zone_ids: ['ob_1', 'fvg_2'] },
    });
    s = applyChartViewAction(s, { action: 'isolate_zones', params: { zone_ids: ['ob_3'] } });
    s = applyChartViewAction(s, { action: 'show_zones', params: { zone_ids: ['ob_1'] } });
    expect(s.hiddenZoneIds).toEqual(['fvg_2']); // ob_1 un-hidden
    expect(new Set(s.isolatedZoneIds!)).toEqual(new Set(['ob_3', 'ob_1'])); // re-added to isolation
  });

  it('reset_view restores defaults', () => {
    const dirty: ChartViewState = {
      layers: { fvg: false, ob: false, breaks: false, liquidity: false },
      filter: { activeOnly: true, proximityOnly: true, proximityPct: 2, minSizePct: 1 },
      focus: { kind: 'zone', zoneId: 'ob_1', nonce: 9 },
      highlightZoneId: 'ob_1',
      hiddenZoneIds: ['ob_1'],
      isolatedZoneIds: ['fvg_2'],
    };
    expect(applyChartViewAction(dirty, { action: 'reset_view', params: {} })).toEqual(
      DEFAULT_CHART_VIEW,
    );
  });

  it('set_instrument_timeframe is not a render change (state unchanged)', () => {
    const action: ViewAction = {
      action: 'set_instrument_timeframe',
      params: { instrument: 'EURUSD', timeframe: 'H1' },
    };
    expect(applyChartViewAction(DEFAULT_CHART_VIEW, action)).toBe(DEFAULT_CHART_VIEW);
  });
});

// ─── filterZoneModels — display filter over DETECTED zones (hide, never edit) ──

function zone(id: string, high: number, low: number, tested = false): ZoneModel {
  return {
    id,
    kind: id.startsWith('ob') ? 'ob' : 'fvg',
    high,
    low,
    createdSec: 1000,
    mitigatedSec: tested ? 2000 : null,
    tested,
    label: id,
  };
}

describe('filterZoneModels', () => {
  const price = 2000;
  const zones = [
    zone('ob_near', 2002, 1998, false), // mid 2000, 0.2% tall, at price
    zone('ob_far', 2102, 2098, false), // mid 2100, far
    zone('fvg_tested', 2001, 1999, true), // tested, near
    zone('fvg_tiny', 2000.1, 1999.9, false), // 0.01% tall, near
  ];

  it('default filter keeps everything', () => {
    const out = filterZoneModels(zones, price, DEFAULT_CHART_VIEW.filter);
    expect(out).toHaveLength(4);
  });

  it('activeOnly drops tested zones', () => {
    const out = filterZoneModels(zones, price, {
      ...DEFAULT_CHART_VIEW.filter,
      activeOnly: true,
    });
    expect(out.map((z) => z.id)).not.toContain('fvg_tested');
  });

  it('minSizePct drops zones smaller than the threshold', () => {
    const out = filterZoneModels(zones, price, {
      ...DEFAULT_CHART_VIEW.filter,
      minSizePct: 0.1, // 0.1% of 2000 = 2.0 tall
    });
    expect(out.map((z) => z.id)).not.toContain('fvg_tiny');
    expect(out.map((z) => z.id)).toContain('ob_near');
  });

  it('proximityOnly keeps only zones within the window', () => {
    const out = filterZoneModels(zones, price, {
      ...DEFAULT_CHART_VIEW.filter,
      proximityOnly: true,
      proximityPct: 0.5, // ±0.5% of 2000 = ±10
    });
    expect(out.map((z) => z.id)).not.toContain('ob_far');
    expect(out.map((z) => z.id)).toContain('ob_near');
  });

  it('does not mutate the input array', () => {
    const copy = [...zones];
    filterZoneModels(zones, price, { ...DEFAULT_CHART_VIEW.filter, activeOnly: true });
    expect(zones).toEqual(copy);
  });
});

// ─── applyZoneVisibility — per-id masking (hide / isolate), reversible ─────────

describe('applyZoneVisibility', () => {
  const zones = [
    zone('ob_1', 2002, 1998),
    zone('ob_2', 2102, 2098),
    zone('fvg_3', 2001, 1999),
  ];

  it('keeps everything with no masks (default state)', () => {
    expect(applyZoneVisibility(zones, [], null).map((z) => z.id)).toEqual([
      'ob_1',
      'ob_2',
      'fvg_3',
    ]);
  });

  it('hide removes a real zone by id and is reversible', () => {
    const hidden = applyZoneVisibility(zones, ['ob_1'], null);
    expect(hidden.map((z) => z.id)).toEqual(['ob_2', 'fvg_3']); // ob_1 gone
    // Reversibility: dropping the mask shows it again (same input zones).
    const restored = applyZoneVisibility(zones, [], null);
    expect(restored.map((z) => z.id)).toContain('ob_1');
  });

  it('isolate shows ONLY the listed zones', () => {
    const iso = applyZoneVisibility(zones, [], ['ob_2']);
    expect(iso.map((z) => z.id)).toEqual(['ob_2']);
  });

  it('hide composes with isolate (intersection minus hidden)', () => {
    const out = applyZoneVisibility(zones, ['ob_2'], ['ob_1', 'ob_2']);
    expect(out.map((z) => z.id)).toEqual(['ob_1']); // ob_2 isolated-in but then hidden
  });

  it('does not mutate the input array', () => {
    const copy = [...zones];
    applyZoneVisibility(zones, ['ob_1'], ['ob_2']);
    expect(zones).toEqual(copy);
  });
});

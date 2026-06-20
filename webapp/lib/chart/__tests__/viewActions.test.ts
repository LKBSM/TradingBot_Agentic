import { describe, expect, it } from 'vitest';
import {
  applyChartViewAction,
  coerceViewAction,
  coerceViewActions,
  DEFAULT_CHART_VIEW,
  type ChartViewState,
  type ViewAction,
} from '../viewActions';
import { filterZoneModels, type ZoneModel } from '../zoneLayout';

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
    expect(next.layers).toEqual({ fvg: false, ob: true, breaks: true });
  });

  it('layer "all" toggles every overlay', () => {
    const next = applyChartViewAction(DEFAULT_CHART_VIEW, {
      action: 'set_layer_visibility',
      params: { layer: 'all', visible: false },
    });
    expect(next.layers).toEqual({ fvg: false, ob: false, breaks: false });
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

  it('reset_view restores defaults', () => {
    const dirty: ChartViewState = {
      layers: { fvg: false, ob: false, breaks: false },
      filter: { activeOnly: true, proximityOnly: true, proximityPct: 2, minSizePct: 1 },
      focus: { kind: 'zone', zoneId: 'ob_1', nonce: 9 },
      highlightZoneId: 'ob_1',
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

/**
 * Chart VIEW actions — the display-only contract the chatbot emits and the front
 * applies to the RENDER only. Mirror of the backend Couche 4 whitelist
 * (`src/intelligence/chatbot/view_action_filter.py`).
 *
 * The inviolable line (same as the backend):
 *   · The action vocabulary is a CLOSED set. No create/place/move/resize verb.
 *   · No action carries a price/level/geometry field.
 *   · focus_zone / highlight_zone reference an EXISTING detected zone id.
 *
 * `coerceViewActions` re-validates whatever the backend returned against the
 * SAME rules (defence in depth) AND against the zone ids currently on screen —
 * so a stale id (e.g. the combo changed between request and apply) is dropped
 * rather than mis-applied. This module NEVER touches detection; it only decides
 * how to change the chart's display state.
 */

export type ChartLayer = 'fvg' | 'ob' | 'breaks' | 'all';

export const ALLOWED_LAYERS: readonly ChartLayer[] = ['fvg', 'ob', 'breaks', 'all'];

export const SUPPORTED_INSTRUMENTS = ['XAUUSD', 'EURUSD'] as const;
export const SUPPORTED_TIMEFRAMES = ['M15', 'H1', 'H4'] as const;

/** Geometry-shaped keys that must NEVER appear on a view action. */
const GEOMETRY_KEYS = new Set([
  'price',
  'prices',
  'level',
  'level_high',
  'level_low',
  'high',
  'low',
  'top',
  'bottom',
  'band',
  'open',
  'close',
]);

export type ViewAction =
  | { action: 'set_layer_visibility'; params: { layer: ChartLayer; visible: boolean } }
  | {
      action: 'filter_zones';
      params: {
        active_only?: boolean;
        proximity_only?: boolean;
        proximity_pct?: number;
        min_size_pct?: number;
      };
    }
  | { action: 'focus_zone'; params: { zone_id: string } }
  | { action: 'highlight_zone'; params: { zone_id: string } }
  | { action: 'focus_price'; params: Record<string, never> }
  | { action: 'fit_chart'; params: Record<string, never> }
  | { action: 'reset_view'; params: Record<string, never> }
  | {
      action: 'set_instrument_timeframe';
      params: { instrument: string; timeframe: string };
    };

const PROXIMITY_PCT_RANGE: readonly [number, number] = [0.05, 10.0];
const MIN_SIZE_PCT_RANGE: readonly [number, number] = [0.0, 10.0];

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function hasGeometryKey(params: Record<string, unknown>): boolean {
  return Object.keys(params).some((k) => GEOMETRY_KEYS.has(k));
}

/**
 * Re-validate a raw action (defence in depth) and return a normalised ViewAction,
 * or null if it is not admissible. `validZoneIds` is the set of zone ids the
 * chart can currently resolve — a focus/highlight on an unknown id is dropped.
 */
export function coerceViewAction(
  raw: unknown,
  validZoneIds: ReadonlySet<string>,
): ViewAction | null {
  if (!isPlainObject(raw)) return null;
  const action = raw.action;
  const params = isPlainObject(raw.params) ? raw.params : {};
  if (hasGeometryKey(params)) return null;

  switch (action) {
    case 'set_layer_visibility': {
      const layer = params.layer;
      const visible = params.visible;
      if (typeof layer !== 'string' || !ALLOWED_LAYERS.includes(layer as ChartLayer)) {
        return null;
      }
      if (typeof visible !== 'boolean') return null;
      return { action, params: { layer: layer as ChartLayer, visible } };
    }
    case 'filter_zones': {
      const out: {
        active_only?: boolean;
        proximity_only?: boolean;
        proximity_pct?: number;
        min_size_pct?: number;
      } = {};
      if (typeof params.active_only === 'boolean') out.active_only = params.active_only;
      if (typeof params.proximity_only === 'boolean') {
        out.proximity_only = params.proximity_only;
      }
      if (typeof params.proximity_pct === 'number' && Number.isFinite(params.proximity_pct)) {
        out.proximity_pct = clamp(params.proximity_pct, ...PROXIMITY_PCT_RANGE);
      }
      if (typeof params.min_size_pct === 'number' && Number.isFinite(params.min_size_pct)) {
        out.min_size_pct = clamp(params.min_size_pct, ...MIN_SIZE_PCT_RANGE);
      }
      if (Object.keys(out).length === 0) return null;
      return { action, params: out };
    }
    case 'focus_zone':
    case 'highlight_zone': {
      const zoneId = params.zone_id;
      if (typeof zoneId !== 'string' || !zoneId.trim()) return null;
      if (!validZoneIds.has(zoneId)) return null;
      return { action, params: { zone_id: zoneId } };
    }
    case 'focus_price':
    case 'fit_chart':
    case 'reset_view':
      return { action, params: {} };
    case 'set_instrument_timeframe': {
      const instrument = params.instrument;
      const timeframe = params.timeframe;
      if (
        typeof instrument !== 'string' ||
        !SUPPORTED_INSTRUMENTS.includes(instrument as (typeof SUPPORTED_INSTRUMENTS)[number])
      ) {
        return null;
      }
      if (
        typeof timeframe !== 'string' ||
        !SUPPORTED_TIMEFRAMES.includes(timeframe as (typeof SUPPORTED_TIMEFRAMES)[number])
      ) {
        return null;
      }
      return { action, params: { instrument, timeframe } };
    }
    default:
      return null;
  }
}

/** Coerce a list of raw actions, dropping any that fail validation. */
export function coerceViewActions(
  raw: unknown,
  validZoneIds: ReadonlySet<string>,
): ViewAction[] {
  if (!Array.isArray(raw)) return [];
  const out: ViewAction[] = [];
  for (const item of raw) {
    const coerced = coerceViewAction(item, validZoneIds);
    if (coerced) out.push(coerced);
  }
  return out;
}

// ─── View state + reducer ─────────────────────────────────────────────────────

export interface ChartLayers {
  fvg: boolean;
  ob: boolean;
  breaks: boolean;
}

export interface ChartFilter {
  activeOnly: boolean;
  proximityOnly: boolean;
  /** Proximity window as a % of the current price (mid within ±pct). */
  proximityPct: number;
  /** Minimum zone band height as a % of price, or null (no size filter). */
  minSizePct: number | null;
}

/** A one-shot focus command; `nonce` re-triggers the chart even if unchanged. */
export interface FocusCommand {
  kind: 'zone' | 'price' | 'fit';
  zoneId?: string;
  nonce: number;
}

export interface ChartViewState {
  layers: ChartLayers;
  filter: ChartFilter;
  focus: FocusCommand | null;
  highlightZoneId: string | null;
}

export const DEFAULT_CHART_VIEW: ChartViewState = {
  layers: { fvg: true, ob: true, breaks: true },
  filter: { activeOnly: false, proximityOnly: false, proximityPct: 0.5, minSizePct: null },
  focus: null,
  highlightZoneId: null,
};

/**
 * Apply ONE validated render action to the chart view state. `set_instrument_
 * timeframe` is NOT a render action (it changes the active combo) and is handled
 * by the dispatcher — it returns the state unchanged here.
 */
export function applyChartViewAction(
  state: ChartViewState,
  action: ViewAction,
): ChartViewState {
  switch (action.action) {
    case 'set_layer_visibility': {
      const { layer, visible } = action.params;
      if (layer === 'all') {
        return { ...state, layers: { fvg: visible, ob: visible, breaks: visible } };
      }
      return { ...state, layers: { ...state.layers, [layer]: visible } };
    }
    case 'filter_zones': {
      const p = action.params;
      return {
        ...state,
        filter: {
          activeOnly: p.active_only ?? state.filter.activeOnly,
          proximityOnly: p.proximity_only ?? state.filter.proximityOnly,
          proximityPct: p.proximity_pct ?? state.filter.proximityPct,
          minSizePct:
            p.min_size_pct !== undefined ? p.min_size_pct : state.filter.minSizePct,
        },
      };
    }
    case 'focus_zone':
      return { ...state, focus: { kind: 'zone', zoneId: action.params.zone_id, nonce: nextNonce(state) } };
    case 'focus_price':
      return { ...state, focus: { kind: 'price', nonce: nextNonce(state) } };
    case 'fit_chart':
      return { ...state, focus: { kind: 'fit', nonce: nextNonce(state) } };
    case 'highlight_zone':
      return { ...state, highlightZoneId: action.params.zone_id };
    case 'reset_view':
      return { ...DEFAULT_CHART_VIEW };
    case 'set_instrument_timeframe':
      return state; // handled by the dispatcher (combo change), not a render action.
    default:
      return state;
  }
}

function nextNonce(state: ChartViewState): number {
  return (state.focus?.nonce ?? 0) + 1;
}

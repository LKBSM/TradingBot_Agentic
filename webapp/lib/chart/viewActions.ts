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

export type ChartLayer = 'fvg' | 'ob' | 'breaks' | 'liquidity' | 'all';

export const ALLOWED_LAYERS: readonly ChartLayer[] = [
  'fvg',
  'ob',
  'breaks',
  'liquidity',
  'all',
];

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
  | { action: 'hide_zones'; params: { zone_ids: string[] } }
  | { action: 'isolate_zones'; params: { zone_ids: string[] } }
  | { action: 'show_zones'; params: { zone_ids?: string[] } }
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
 * Validate a list of zone ids against the on-screen detected set (same id lock as
 * focus/highlight, generalised). Returns a de-duplicated (order-preserving) list,
 * or null if the input is not an array or any id is missing/blank/unknown — an
 * invented id rejects the whole action so masking it hides nothing.
 */
function coerceZoneIdList(
  raw: unknown,
  validZoneIds: ReadonlySet<string>,
): string[] | null {
  if (!Array.isArray(raw)) return null;
  const seen = new Set<string>();
  const out: string[] = [];
  for (const id of raw) {
    if (typeof id !== 'string' || !id.trim()) return null;
    if (!validZoneIds.has(id)) return null;
    if (!seen.has(id)) {
      seen.add(id);
      out.push(id);
    }
  }
  return out;
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
    case 'hide_zones':
    case 'isolate_zones': {
      const ids = coerceZoneIdList(params.zone_ids, validZoneIds);
      // A single invented id (or an empty list) drops the whole action — masking
      // a non-existent zone hides nothing, exactly like focus on an invented id.
      if (ids === null || ids.length === 0) return null;
      return { action, params: { zone_ids: ids } };
    }
    case 'show_zones': {
      // No ids → restore every masked zone. An explicit list is validated the
      // same way (a stale id can't slip through); an empty list also restores.
      if (params.zone_ids === undefined || params.zone_ids === null) {
        return { action, params: {} };
      }
      const ids = coerceZoneIdList(params.zone_ids, validZoneIds);
      if (ids === null) return null;
      return { action, params: { zone_ids: ids } };
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
  /** External liquidity pockets (BSL/SSL) as horizontal price-level lines. */
  liquidity: boolean;
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
  /**
   * Detected zones explicitly REMOVED from the display by id (`hide_zones`).
   * Reversible via `show_zones` / `reset_view`. The zones still exist in the
   * engine — this is display state only, never a detection mutation.
   */
  hiddenZoneIds: string[];
  /**
   * When non-null, ONLY these detected zones are displayed (`isolate_zones`);
   * every other zone is hidden. `null` means "no isolation" (show all). Cleared
   * by `show_zones` (no ids) / `reset_view`.
   */
  isolatedZoneIds: string[] | null;
}

export const DEFAULT_CHART_VIEW: ChartViewState = {
  layers: { fvg: true, ob: true, breaks: true, liquidity: true },
  filter: { activeOnly: false, proximityOnly: false, proximityPct: 0.5, minSizePct: null },
  focus: null,
  highlightZoneId: null,
  hiddenZoneIds: [],
  isolatedZoneIds: null,
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
        return {
          ...state,
          layers: { fvg: visible, ob: visible, breaks: visible, liquidity: visible },
        };
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
    case 'hide_zones': {
      // Union the targeted ids into the hidden set (display-only, reversible).
      const hidden = new Set(state.hiddenZoneIds);
      for (const id of action.params.zone_ids) hidden.add(id);
      return { ...state, hiddenZoneIds: [...hidden] };
    }
    case 'isolate_zones':
      // Show ONLY these zones; replaces any prior isolation.
      return { ...state, isolatedZoneIds: [...action.params.zone_ids] };
    case 'show_zones': {
      const ids = action.params.zone_ids;
      if (!ids || ids.length === 0) {
        // Restore all: drop every mask and any isolation.
        return { ...state, hiddenZoneIds: [], isolatedZoneIds: null };
      }
      // Un-hide the named ids; if an isolation is active, add them back into it.
      const restore = new Set(ids);
      const hiddenZoneIds = state.hiddenZoneIds.filter((id) => !restore.has(id));
      const isolatedZoneIds =
        state.isolatedZoneIds === null
          ? null
          : [...new Set([...state.isolatedZoneIds, ...ids])];
      return { ...state, hiddenZoneIds, isolatedZoneIds };
    }
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

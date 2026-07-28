'use client';

import * as React from 'react';
import {
  applyChartViewAction,
  DEFAULT_CHART_VIEW,
  selectionKey,
  type ChartSelection,
  type ChartViewState,
  type ReferenceLevel,
  type ViewAction,
} from './viewActions';

/**
 * Holds the chatbot-controllable CHART VIEW state (layer visibility, zone
 * filters, focus/zoom command) AND the unified single SELECTION (VZ-1) for the
 * /app + /zones workspaces. This is a DISPLAY layer only — it never holds or
 * touches detection data; the chart reads it to decide what to render and how to
 * frame it.
 *
 * VZ-1 — one gesture, product-wide. `selection` is the SINGLE source of truth for
 * the currently-selected element across all three families (zone / event /
 * level) and every panel. `highlightZoneId` (the blue zone emphasis) and
 * `referenceLevel` (the traced calendar line) are now DERIVED from it, so exactly
 * one element is ever selected: clicking a level deselects the active zone, and
 * vice-versa. Legacy consumers that read `view.highlightZoneId` / `referenceLevel`
 * keep working unchanged.
 *
 * `set_instrument_timeframe` is a combo change, not a render change, so it is NOT
 * stored here: `applyActions` forwards it to the caller via `onComboChange`.
 */
interface ChartViewContextValue {
  view: ChartViewState;
  /**
   * Apply already-validated render actions to the view state. Combo-change
   * actions (`set_instrument_timeframe`) are forwarded to `onComboChange`
   * instead of mutating the render state. Selection actions (`focus_zone` /
   * `highlight_zone` / `clear_highlight`) are routed to the unified selection so
   * the chatbot and the panels share ONE selection.
   */
  applyActions(
    actions: ViewAction[],
    onComboChange?: (combo: { instrument: string; timeframe: string }) => void,
  ): void;
  /** Restore the default view (all layers visible, no filter, no selection). */
  reset(): void;
  /** The single selected element (zone / event / level), or null. */
  selection: ChartSelection | null;
  /**
   * Select an element (SET semantics — replaces any prior selection in any
   * family/panel, so only one is ever selected). Re-click/Escape toggling is the
   * caller's responsibility (compare against `selection` and call
   * `clearSelection`) — it keeps the intent explicit at the call site.
   */
  select(sel: ChartSelection): void;
  /** Deselect whatever is selected (drop the emphasis; the chart restores its view). */
  clearSelection(): void;
  /**
   * The calendar-derived reference marker currently traced on the chart, or null.
   * DERIVED from `selection` (a level of kind 'reference'). This is a DISTINCT
   * channel from the chatbot-controllable `view`: it never enters
   * `coerceViewAction` (which bans price/level fields), so tracing a reference
   * level cannot weaken the zone id-lock.
   */
  referenceLevel: ReferenceLevel | null;
  /**
   * Trace a reference level (replaces any current selection), or null to clear
   * it. Compatibility wrapper over the unified selection so the Régime UI keeps
   * its call site; it routes through `select` / `clearSelection`.
   */
  setReferenceLevel(level: ReferenceLevel | null): void;
  /**
   * Reset the render state when the active combo CHANGES (NAV-05). Hidden /
   * isolated / selected elements are keyed to a specific combo; without this they
   * linger across a combo switch. No-op while the combo key is unchanged, so the
   * intended /app↔/zones sharing for the SAME combo is preserved. The first call
   * only records the key (never wipes the arrival view).
   */
  resetForCombo(comboKey: string): void;
}

const ChartViewContext = React.createContext<ChartViewContextValue | null>(null);

export function ChartViewProvider({ children }: { children: React.ReactNode }) {
  const [reducerView, setReducerView] = React.useState<ChartViewState>(DEFAULT_CHART_VIEW);
  // The unified selection is the SINGLE source of truth (VZ-1). It lives OUTSIDE
  // the chatbot view-action reducer on purpose — event/level geometry must never
  // be reachable from the coerceViewAction pipeline (the zone id-lock).
  const [selection, setSelection] = React.useState<ChartSelection | null>(null);

  const select = React.useCallback((sel: ChartSelection) => {
    setSelection(sel);
  }, []);
  const clearSelection = React.useCallback(() => setSelection(null), []);

  // VZ-1 — Escape deselects the active element anywhere the provider is mounted
  // (the keyboard mirror of re-clicking). A no-op when nothing is selected, so it
  // never swallows Escape from other surfaces (chat input, dialogs).
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelection((cur) => (cur ? null : cur));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const applyActions = React.useCallback(
    (
      actions: ViewAction[],
      onComboChange?: (combo: { instrument: string; timeframe: string }) => void,
    ) => {
      if (actions.length === 0) return;
      for (const action of actions) {
        switch (action.action) {
          case 'set_instrument_timeframe':
            onComboChange?.({
              instrument: action.params.instrument,
              timeframe: action.params.timeframe,
            });
            break;
          // Selection actions → the unified selection (shared with the panels).
          // `focus_zone` and `highlight_zone` arrive together; both SET the same
          // zone selection (idempotent), so the pair selects exactly once.
          case 'focus_zone':
          case 'highlight_zone':
            setSelection({ family: 'zone', id: action.params.zone_id });
            break;
          case 'clear_highlight':
            setSelection(null);
            break;
          case 'reset_view':
            setSelection(null);
            setReducerView(applyChartViewAction(DEFAULT_CHART_VIEW, action));
            break;
          // Everything else (layers / filter / hide / isolate / show / focus_price
          // / fit_chart) is a pure render action handled by the reducer.
          default:
            setReducerView((prev) => applyChartViewAction(prev, action));
        }
      }
    },
    [],
  );

  const reset = React.useCallback(() => {
    setReducerView(DEFAULT_CHART_VIEW);
    setSelection(null);
  }, []);

  const setReferenceLevel = React.useCallback((level: ReferenceLevel | null) => {
    if (!level) {
      // Clear only if a reference level is what's currently selected.
      setSelection((cur) =>
        cur && cur.family === 'level' && cur.kind === 'reference' ? null : cur,
      );
      return;
    }
    setSelection({
      family: 'level',
      kind: 'reference',
      id: `ref:${level.label}`,
      price: level.price,
      label: level.label,
    });
  }, []);

  const lastComboKeyRef = React.useRef<string | null>(null);
  const resetForCombo = React.useCallback((comboKey: string) => {
    if (lastComboKeyRef.current === comboKey) return;
    const isFirst = lastComboKeyRef.current === null;
    lastComboKeyRef.current = comboKey;
    // First observation just records the key; only a real CHANGE wipes the view
    // so masks/isolation/selection from the previous combo don't leak forward.
    if (!isFirst) {
      setReducerView(DEFAULT_CHART_VIEW);
      setSelection(null);
    }
  }, []);

  // DERIVE the legacy channels from the single selection so every existing
  // consumer (chart, panels, tests) keeps reading `view.highlightZoneId` /
  // `referenceLevel` unchanged, while there is only ONE selection underneath.
  const view = React.useMemo<ChartViewState>(
    () => ({
      ...reducerView,
      highlightZoneId: selection?.family === 'zone' ? selection.id : null,
    }),
    [reducerView, selection],
  );

  const referenceLevel = React.useMemo<ReferenceLevel | null>(
    () =>
      selection && selection.family === 'level' && selection.kind === 'reference'
        ? { price: selection.price, label: selection.label }
        : null,
    [selection],
  );

  const value = React.useMemo<ChartViewContextValue>(
    () => ({
      view,
      applyActions,
      reset,
      resetForCombo,
      selection,
      select,
      clearSelection,
      referenceLevel,
      setReferenceLevel,
    }),
    [
      view,
      applyActions,
      reset,
      resetForCombo,
      selection,
      select,
      clearSelection,
      referenceLevel,
      setReferenceLevel,
    ],
  );

  return (
    <ChartViewContext.Provider value={value}>{children}</ChartViewContext.Provider>
  );
}

export function useChartView(): ChartViewContextValue {
  const ctx = React.useContext(ChartViewContext);
  if (!ctx) {
    throw new Error('useChartView must be used inside a <ChartViewProvider />.');
  }
  return ctx;
}

const NOOP_VIEW: ChartViewContextValue = {
  view: { ...DEFAULT_CHART_VIEW },
  applyActions: () => {},
  reset: () => {},
  resetForCombo: () => {},
  selection: null,
  select: () => {},
  clearSelection: () => {},
  referenceLevel: null,
  setReferenceLevel: () => {},
};

/**
 * Like `useChartView` but tolerant of a missing provider — returns the DEFAULT
 * view (all layers visible, no filter, no selection) so a component used outside
 * the /app workspace (or in an isolated unit test) renders exactly the
 * pre-existing behaviour. Use this for the render-time consumers (chart column);
 * use the strict `useChartView` only where a provider is guaranteed.
 */
export function useChartViewOptional(): ChartViewContextValue {
  return React.useContext(ChartViewContext) ?? NOOP_VIEW;
}

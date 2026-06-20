'use client';

import * as React from 'react';
import {
  applyChartViewAction,
  DEFAULT_CHART_VIEW,
  type ChartViewState,
  type ViewAction,
} from './viewActions';

/**
 * Holds the chatbot-controllable CHART VIEW state (layer visibility, zone
 * filters, focus/zoom command, highlighted zone) for the /app workspace. This is
 * a DISPLAY layer only — it never holds or touches detection data; the chart
 * reads it to decide what to render and how to frame it.
 *
 * `set_instrument_timeframe` is a combo change, not a render change, so it is NOT
 * stored here: `applyActions` forwards it to the caller via `onComboChange`.
 */
interface ChartViewContextValue {
  view: ChartViewState;
  /**
   * Apply already-validated render actions to the view state. Combo-change
   * actions (`set_instrument_timeframe`) are forwarded to `onComboChange`
   * instead of mutating the render state.
   */
  applyActions(
    actions: ViewAction[],
    onComboChange?: (combo: { instrument: string; timeframe: string }) => void,
  ): void;
  /** Restore the default view (all layers visible, no filter, no highlight). */
  reset(): void;
}

const ChartViewContext = React.createContext<ChartViewContextValue | null>(null);

export function ChartViewProvider({ children }: { children: React.ReactNode }) {
  const [view, setView] = React.useState<ChartViewState>(DEFAULT_CHART_VIEW);

  const applyActions = React.useCallback(
    (
      actions: ViewAction[],
      onComboChange?: (combo: { instrument: string; timeframe: string }) => void,
    ) => {
      if (actions.length === 0) return;
      setView((prev) => {
        let next = prev;
        for (const action of actions) {
          if (action.action === 'set_instrument_timeframe') {
            onComboChange?.({
              instrument: action.params.instrument,
              timeframe: action.params.timeframe,
            });
            continue; // not a render action
          }
          next = applyChartViewAction(next, action);
        }
        return next;
      });
    },
    [],
  );

  const reset = React.useCallback(() => setView(DEFAULT_CHART_VIEW), []);

  const value = React.useMemo<ChartViewContextValue>(
    () => ({ view, applyActions, reset }),
    [view, applyActions, reset],
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
  view: DEFAULT_CHART_VIEW,
  applyActions: () => {},
  reset: () => {},
};

/**
 * Like `useChartView` but tolerant of a missing provider — returns the DEFAULT
 * view (all layers visible, no filter, no focus) so a component used outside the
 * /app workspace (or in an isolated unit test) renders exactly the pre-existing
 * behaviour. Use this for the render-time consumers (chart column); use the
 * strict `useChartView` only where a provider is guaranteed (the dispatcher).
 */
export function useChartViewOptional(): ChartViewContextValue {
  return React.useContext(ChartViewContext) ?? NOOP_VIEW;
}

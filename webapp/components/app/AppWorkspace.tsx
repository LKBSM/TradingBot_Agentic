'use client';

import * as React from 'react';
import { useChat } from '@/components/chat/ChatProvider';
import { AppChatSidebar } from './AppChatSidebar';
import { InstrumentSidebar } from './InstrumentSidebar';
import { MobileWorkspace } from './MobileWorkspace';
import { ReadingColumn } from './ReadingColumn';
import { useIsMobile } from '@/lib/use-media-query';
import { useMarketReading, type ReadingSource } from '@/lib/market-reading/hooks';
import {
  ActiveComboProvider,
  useActiveCombo,
  type Combo,
} from '@/lib/market-reading/store';
import { useChartView } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import { READING_DATA_SOURCE } from '@/lib/mockReadings';
import type { MarketReading } from '@/types/market-reading';

const POLL_MS = 60_000;

export interface AppWorkspaceProps {
  /** Combo selected on mount (null = none). The /app route defaults to XAU M15. */
  initialCombo?: Combo | null;
  /**
   * Zone id to focus on the chart once the reading loads — set by the "Analyser"
   * deep-link (`?focus=`). Validated against the on-screen zone-id lock before
   * dispatch, so an unknown/stale id is a graceful no-op. Display-only.
   */
  initialFocusZoneId?: string | null;
  /** Reading source — defaults to the module flag; overridable for tests. */
  dataSource?: ReadingSource;
}

/**
 * Shared shape handed to both the desktop and mobile layouts so the data /
 * selection logic lives in exactly one place (WorkspaceInner).
 */
export interface WorkspaceViewProps {
  combos: readonly Combo[];
  active: Combo | null;
  onSelect(combo: Combo): void;
  reading: MarketReading | null;
  isLoading: boolean;
  isRefreshing: boolean;
  error: Error | null;
  onRetry(): void;
  /** Candle/reading source, forwarded to ReadingColumn's chart feed. */
  dataSource: ReadingSource;
}

/**
 * /app workspace. Desktop (≥768px) shows three columns; mobile (<768px) shows
 * a tabbed layout (Marchés · Lecture · Chat). The active combo lives in
 * ActiveComboProvider; the reading is fetched + polled (60s) via
 * useMarketReading; the chat context follows the active combo via openForCombo.
 */
export function AppWorkspace({
  initialCombo = null,
  initialFocusZoneId = null,
  dataSource = READING_DATA_SOURCE,
}: AppWorkspaceProps = {}) {
  // ChartViewProvider is now mounted in the locale layout (shared with /zones),
  // so this surface just consumes it via useChartView().
  return (
    <ActiveComboProvider initial={initialCombo}>
      <WorkspaceInner dataSource={dataSource} initialFocusZoneId={initialFocusZoneId} />
    </ActiveComboProvider>
  );
}

function WorkspaceInner({
  dataSource,
  initialFocusZoneId = null,
}: {
  dataSource: ReadingSource;
  initialFocusZoneId?: string | null;
}) {
  const { active, select, combos } = useActiveCombo();
  const { openForCombo, viewActionSignal } = useChat();
  const { applyActions } = useChartView();
  const isMobile = useIsMobile();

  const { data, isLoading, isRefreshing, error, refresh } = useMarketReading(
    active?.instrument ?? null,
    active?.timeframe ?? null,
    { pollMs: POLL_MS, source: dataSource },
  );

  // Keep the chat context aligned with the selected combo.
  React.useEffect(() => {
    if (active) openForCombo(active);
  }, [active, openForCombo]);

  // Zone ids the chart can currently resolve — the ONLY zones a focus/highlight
  // view action may reference (defence in depth on top of the backend Couche 4).
  const validZoneIds = React.useMemo(() => {
    const ids = new Set<string>();
    const s = data?.structure;
    if (s) {
      for (const ob of s.order_blocks ?? []) ids.add(ob.id);
      for (const fvg of s.fair_value_gaps ?? []) ids.add(fvg.id);
    }
    return ids;
  }, [data]);

  // Apply the chatbot's display-only view actions to the chart RENDER. We
  // re-validate the raw actions against the on-screen zones, then dispatch:
  // `set_instrument_timeframe` changes the active combo (via select), every
  // other action changes the chart view state only. Detection is never touched.
  const lastViewNonceRef = React.useRef(0);
  React.useEffect(() => {
    if (!viewActionSignal) return;
    if (viewActionSignal.nonce === lastViewNonceRef.current) return;
    lastViewNonceRef.current = viewActionSignal.nonce;
    const actions = coerceViewActions(viewActionSignal.actions, validZoneIds);
    applyActions(actions, select);
  }, [viewActionSignal, validZoneIds, applyActions, select]);

  // Honour an "Analyser" deep-link (?focus=<zone_id>) from the /zones page: once
  // the reading is loaded and the id resolves to an on-screen zone, focus +
  // highlight it ONCE. Re-validated through the same id-lock — an unknown/stale
  // id is dropped, never mis-applied. Display-only; detection is never touched.
  const focusDispatchedRef = React.useRef(false);
  React.useEffect(() => {
    if (focusDispatchedRef.current) return;
    if (!initialFocusZoneId || !validZoneIds.has(initialFocusZoneId)) return;
    focusDispatchedRef.current = true;
    const actions = coerceViewActions(
      [
        { action: 'focus_zone', params: { zone_id: initialFocusZoneId } },
        { action: 'highlight_zone', params: { zone_id: initialFocusZoneId } },
      ],
      validZoneIds,
    );
    applyActions(actions, select);
  }, [initialFocusZoneId, validZoneIds, applyActions, select]);

  // ID lock, honest half: when the deep-linked zone id does NOT resolve in the
  // loaded reading (zone consumed/expired between /zones and here), say so
  // discreetly — the app opens normally on the combo, and the stale zone is
  // NEVER redrawn from memory. Checked ONCE at the first loaded reading (the
  // dispatch effect above runs first, so a resolved focus never notices).
  const [staleFocusNotice, setStaleFocusNotice] = React.useState(false);
  const focusCheckedRef = React.useRef(false);
  React.useEffect(() => {
    if (focusCheckedRef.current) return;
    if (!initialFocusZoneId || !data) return;
    focusCheckedRef.current = true;
    if (!focusDispatchedRef.current && !validZoneIds.has(initialFocusZoneId)) {
      setStaleFocusNotice(true);
    }
  }, [initialFocusZoneId, data, validZoneIds]);

  const view: WorkspaceViewProps = {
    combos,
    active,
    onSelect: select,
    reading: data,
    isLoading,
    isRefreshing,
    error,
    onRetry: refresh,
    dataSource,
  };

  return (
    <>
      {staleFocusNotice && (
        <div className="container-wide pt-4" role="status" aria-live="polite">
          <div className="flex items-center justify-between gap-3 rounded-md border border-border/70 bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
            <span>
              Cette zone n’est plus détectée — la lecture actuelle du marché est
              affichée.
            </span>
            <button
              type="button"
              aria-label="Fermer ce message"
              onClick={() => setStaleFocusNotice(false)}
              className="shrink-0 rounded px-1 font-medium transition-colors hover:text-foreground"
            >
              ✕
            </button>
          </div>
        </div>
      )}
      {isMobile ? <MobileWorkspace {...view} /> : <DesktopWorkspace {...view} />}
    </>
  );
}

function DesktopWorkspace({
  combos,
  active,
  onSelect,
  reading,
  isLoading,
  isRefreshing,
  error,
  onRetry,
  dataSource,
}: WorkspaceViewProps) {
  return (
    <div className="container-wide py-6">
      <div className="grid grid-cols-1 gap-6 md:grid-cols-[240px_minmax(0,1fr)_360px] md:items-start">
        <InstrumentSidebar
          combos={combos}
          active={active}
          onSelect={onSelect}
          activeCandleCloseTs={reading?.header.candle_close_ts ?? null}
        />

        <ReadingColumn
          active={active}
          reading={reading}
          isLoading={isLoading}
          isRefreshing={isRefreshing}
          error={error}
          onRetry={onRetry}
          dataSource={dataSource}
        />

        <div className="md:sticky md:top-6 md:h-[calc(100vh-7rem)]">
          <AppChatSidebar active={active} onSelectCombo={onSelect} />
        </div>
      </div>
    </div>
  );
}

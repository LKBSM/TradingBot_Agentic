'use client';

import * as React from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useTranslations } from 'next-intl';
import { useChat } from '@/components/chat/ChatProvider';
import { AppChatSidebar } from './AppChatSidebar';
import { InstrumentSidebar } from './InstrumentSidebar';
import { MobileWorkspace } from './MobileWorkspace';
import { ReadingColumn } from './ReadingColumn';
import { useIsMobile } from '@/lib/use-media-query';
import { useMarketReading, type ReadingSource } from '@/lib/market-reading/hooks';
import {
  ActiveComboProvider,
  comboKey,
  sameCombo,
  useActiveCombo,
  type Combo,
} from '@/lib/market-reading/store';
import { resolveComboFromQuery } from '@/lib/conditions/app-link';
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
  const t = useTranslations('app');
  const { active, select, combos } = useActiveCombo();
  const { openForCombo, viewActionSignal } = useChat();
  const { applyActions, resetForCombo } = useChartView();
  const isMobile = useIsMobile();
  // useIsMobile reports `false` until its effect runs, so the first paint would
  // show the DESKTOP layout for a frame before flipping to mobile — a flash +
  // remount of the columns (NAV-10). Gate the layout choice on `mounted` so we
  // render a neutral placeholder until the real breakpoint is known; both
  // `mounted` and the media query resolve in the same post-mount flush, so the
  // next paint is the correct layout directly.
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // The URL is the source of truth for the active combo (NAV-01/02/04). Reading
  // it via useSearchParams makes it reactive to client navigation, deep-links
  // arriving while already on /app, AND the browser back/forward buttons — none
  // of which re-seed the provider's initial state.
  const urlCombo = React.useMemo(
    () =>
      resolveComboFromQuery(
        searchParams.get('instrument') ?? undefined,
        searchParams.get('timeframe') ?? undefined,
      ),
    [searchParams],
  );

  // URL → state: reflect the query combo into the active selection whenever it
  // changes. Guarded by comboKey equality so this never fights the state→URL
  // write below (no replace/effect loop).
  React.useEffect(() => {
    if (urlCombo && !sameCombo(urlCombo, active)) {
      select(urlCombo);
    }
  }, [urlCombo, active, select]);

  // State → URL: a user pick writes the combo into the query (shallow replace,
  // no scroll) so it survives navigation and is shareable/bookmarkable. A manual
  // pick also clears any lingering ?focus= deep-link (that zone belonged to the
  // previous selection). Snappy: select() updates immediately; the URL→state
  // effect then sees an equal combo and no-ops.
  const handleSelect = React.useCallback(
    (combo: Combo) => {
      select(combo);
      const params = new URLSearchParams(searchParams.toString());
      params.set('instrument', combo.instrument);
      params.set('timeframe', combo.timeframe);
      params.delete('focus');
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [select, searchParams, pathname, router],
  );

  const { data, isLoading, isRefreshing, error, refresh } = useMarketReading(
    active?.instrument ?? null,
    active?.timeframe ?? null,
    { pollMs: POLL_MS, source: dataSource },
  );

  // Keep the chat context aligned with the selected combo, and clear any chart
  // masks/isolation/highlight carried over from a previous combo (NAV-05).
  React.useEffect(() => {
    if (active) {
      openForCombo(active);
      resetForCombo(comboKey(active));
    }
  }, [active, openForCombo, resetForCombo]);

  // Structure ids the chart can currently resolve — the ONLY structures a
  // focus/highlight/mask view action may reference (defence in depth on top of
  // the backend Couche 4). Liquidity pockets are id-maskable exactly like
  // OB/FVG zones, so their engine-emitted ids belong to the same lock.
  const validZoneIds = React.useMemo(() => {
    const ids = new Set<string>();
    const s = data?.structure;
    if (s) {
      for (const ob of s.order_blocks ?? []) ids.add(ob.id);
      for (const fvg of s.fair_value_gaps ?? []) ids.add(fvg.id);
      for (const pool of s.liquidity_pools ?? []) ids.add(pool.id);
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
    // Route the combo-change action through handleSelect (not raw select) so a
    // chatbot `set_instrument_timeframe` also updates the URL — otherwise the
    // URL→state sync effect would immediately revert it to the stale query.
    applyActions(actions, handleSelect);
  }, [viewActionSignal, validZoneIds, applyActions, handleSelect]);

  // Honour an "Analyser" deep-link (?focus=<zone_id>) from the /zones page. The
  // focus id is read from the (reactive) URL so a SECOND deep-link — or a
  // back/forward that lands on a different ?focus= — re-dispatches instead of
  // being swallowed by a once-per-mount latch (NAV-03). We track the last id we
  // dispatched, not a boolean. Re-validated through the id-lock; an unknown/stale
  // id is dropped, never mis-applied. Display-only; detection is never touched.
  const focusZoneId = searchParams.get('focus') ?? initialFocusZoneId;
  const lastFocusDispatchedRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (!focusZoneId || focusZoneId === lastFocusDispatchedRef.current) return;
    if (!validZoneIds.has(focusZoneId)) return; // wait for the reading / or stale
    lastFocusDispatchedRef.current = focusZoneId;
    const actions = coerceViewActions(
      [
        { action: 'focus_zone', params: { zone_id: focusZoneId } },
        { action: 'highlight_zone', params: { zone_id: focusZoneId } },
      ],
      validZoneIds,
    );
    applyActions(actions, select);
  }, [focusZoneId, validZoneIds, applyActions, select]);

  // ID lock, honest half: when the deep-linked zone id does NOT resolve in the
  // loaded reading (zone consumed/expired between /zones and here), say so
  // discreetly — the app opens normally on the combo, and the stale zone is
  // NEVER redrawn from memory. Re-evaluated per focus id (not once-per-mount) so
  // it clears/re-checks as the URL focus changes (NAV-03/UI-17).
  const [staleFocusNotice, setStaleFocusNotice] = React.useState(false);
  const lastFocusCheckedRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (!focusZoneId) {
      lastFocusCheckedRef.current = null;
      setStaleFocusNotice(false);
      return;
    }
    if (!data || focusZoneId === lastFocusCheckedRef.current) return;
    lastFocusCheckedRef.current = focusZoneId;
    const dispatched = focusZoneId === lastFocusDispatchedRef.current;
    setStaleFocusNotice(!dispatched && !validZoneIds.has(focusZoneId));
  }, [focusZoneId, data, validZoneIds]);

  const view: WorkspaceViewProps = {
    combos,
    active,
    onSelect: handleSelect,
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
            <span>{t('staleFocus.message')}</span>
            <button
              type="button"
              aria-label={t('staleFocus.dismiss')}
              onClick={() => setStaleFocusNotice(false)}
              className="shrink-0 rounded px-1 font-medium transition-colors hover:text-foreground"
            >
              ✕
            </button>
          </div>
        </div>
      )}
      {!mounted ? (
        <div className="container-wide py-6" aria-busy="true" aria-live="polite">
          <div className="flex min-h-[60vh] items-center justify-center">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-primary" />
          </div>
        </div>
      ) : isMobile ? (
        <MobileWorkspace {...view} />
      ) : (
        <DesktopWorkspace {...view} />
      )}
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

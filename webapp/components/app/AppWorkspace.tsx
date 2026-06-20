'use client';

import * as React from 'react';
import { useChat } from '@/components/chat/ChatProvider';
import { AppChatSidebar } from './AppChatSidebar';
import { InstrumentSidebar } from './InstrumentSidebar';
import { MobileWorkspace } from './MobileWorkspace';
import { ReadingColumn } from './ReadingColumn';
import { useIsMobile } from '@/lib/use-media-query';
import { useMarketReading } from '@/lib/market-reading/hooks';
import {
  ActiveComboProvider,
  useActiveCombo,
  type Combo,
} from '@/lib/market-reading/store';
import type { MarketReading } from '@/types/market-reading';

const POLL_MS = 60_000;

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
}

/**
 * /app workspace. Desktop (≥768px) shows three columns; mobile (<768px) shows
 * a tabbed layout (Marchés · Lecture · Chat). The active combo lives in
 * ActiveComboProvider; the reading is fetched + polled (60s) via
 * useMarketReading; the chat context follows the active combo via openForCombo.
 */
export function AppWorkspace({
  initialCombo = null,
}: {
  /** Pre-selected combo (e.g. from a Scanner deep-link). Defaults to none. */
  initialCombo?: Combo | null;
} = {}) {
  return (
    <ActiveComboProvider initial={initialCombo}>
      <WorkspaceInner />
    </ActiveComboProvider>
  );
}

function WorkspaceInner() {
  const { active, select, combos } = useActiveCombo();
  const { openForCombo } = useChat();
  const isMobile = useIsMobile();

  const { data, isLoading, isRefreshing, error, refresh } = useMarketReading(
    active?.instrument ?? null,
    active?.timeframe ?? null,
    { pollMs: POLL_MS },
  );

  // Keep the chat context aligned with the selected combo.
  React.useEffect(() => {
    if (active) openForCombo(active);
  }, [active, openForCombo]);

  const view: WorkspaceViewProps = {
    combos,
    active,
    onSelect: select,
    reading: data,
    isLoading,
    isRefreshing,
    error,
    onRetry: refresh,
  };

  if (isMobile) {
    return <MobileWorkspace {...view} />;
  }

  return <DesktopWorkspace {...view} />;
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
        />

        <div className="md:sticky md:top-6 md:h-[calc(100vh-7rem)]">
          <AppChatSidebar active={active} />
        </div>
      </div>
    </div>
  );
}

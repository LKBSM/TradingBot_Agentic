'use client';

import * as React from 'react';
import { useChat } from '@/components/chat/ChatProvider';
import { AppChatSidebar } from './AppChatSidebar';
import { InstrumentSidebar } from './InstrumentSidebar';
import { ReadingColumn } from './ReadingColumn';
import { useMarketReading } from '@/lib/market-reading/hooks';
import { ActiveComboProvider, useActiveCombo } from '@/lib/market-reading/store';

const POLL_MS = 60_000;

/**
 * /app workspace — three columns on desktop:
 *   left   · instruments (6 combos, active + freshness)
 *   centre · the selected combo's detailed reading (+ loading/error/empty)
 *   right  · the permanent Sentinel chat sidebar
 *
 * The active combo lives in ActiveComboProvider; the reading is fetched +
 * polled (60s) via useMarketReading; the chat context is bound to the active
 * combo via openForCombo (no modal). Mobile tabs are layered on in 5.B.4.
 */
export function AppWorkspace() {
  return (
    <ActiveComboProvider>
      <WorkspaceInner />
    </ActiveComboProvider>
  );
}

function WorkspaceInner() {
  const { active, select, combos } = useActiveCombo();
  const { openForCombo } = useChat();

  const { data, isLoading, isRefreshing, error, refresh } = useMarketReading(
    active?.instrument ?? null,
    active?.timeframe ?? null,
    { pollMs: POLL_MS },
  );

  // Keep the chat context aligned with the selected combo.
  React.useEffect(() => {
    if (active) openForCombo(active);
  }, [active, openForCombo]);

  return (
    <div className="container-wide py-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[240px_minmax(0,1fr)_360px] lg:items-start">
        <InstrumentSidebar
          combos={combos}
          active={active}
          onSelect={select}
          activeCandleCloseTs={data?.header.candle_close_ts ?? null}
        />

        <ReadingColumn
          active={active}
          reading={data}
          isLoading={isLoading}
          isRefreshing={isRefreshing}
          error={error}
          onRetry={refresh}
        />

        <div className="lg:sticky lg:top-6 lg:h-[calc(100vh-7rem)]">
          <AppChatSidebar active={active} />
        </div>
      </div>
    </div>
  );
}

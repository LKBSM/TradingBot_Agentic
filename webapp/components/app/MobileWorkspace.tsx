'use client';

import { LineChart, ListTree, MessageCircle } from 'lucide-react';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AppChatSidebar } from './AppChatSidebar';
import { InstrumentSidebar } from './InstrumentSidebar';
import { ReadingColumn } from './ReadingColumn';
import type { WorkspaceViewProps } from './AppWorkspace';
import {
  formatInstrument,
  formatTimeframe,
} from '@/lib/market-reading/formatters';
import type { Combo } from '@/lib/market-reading/store';

type MobileTab = 'markets' | 'reading' | 'chat';

/**
 * Stacked layout (<1280px — phone + tablet) — a single full-height column with
 * a sticky header (active combo) and sticky bottom tab bar (Marchés · Lecture ·
 * Chat), in the spirit of a native mobile app. Selecting a combo jumps to the
 * Lecture tab. Heights use `svh` (small viewport) so the tab bar stays put when
 * mobile browser chrome expands/collapses; the bottom bar reserves the
 * home-indicator safe area.
 */
export function MobileWorkspace({
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
  const t = useTranslations('app');
  const [tab, setTab] = React.useState<MobileTab>('markets');

  function handleSelect(combo: Combo) {
    onSelect(combo);
    setTab('reading');
  }

  const headerLabel = active
    ? `${formatInstrument(active.instrument)} · ${formatTimeframe(active.timeframe)}`
    : t('mobile.workspaceTitle');

  return (
    <Tabs
      value={tab}
      onValueChange={(v) => setTab(v as MobileTab)}
      /* Fixed height (not min-h) so the content region can flex and the Chat
         tab's input pins to the bottom instead of floating mid-scroll. The
         viewport `interactive-widget: resizes-content` shrinks svh when the
         keyboard opens, keeping the input visible. */
      className="flex h-[calc(100svh-3.5rem)] flex-col"
    >
      <header className="shrink-0 border-b border-border/60 bg-background/95 px-4 py-3">
        <p className="truncate text-sm font-semibold text-foreground">
          {headerLabel}
        </p>
      </header>

      {/* Each tab fills this region (absolute inset-0) and owns its own scroll:
          Marchés/Lecture scroll their content; Chat hands the height to the
          sidebar, whose messages scroll while the input stays docked. */}
      <div className="relative flex-1 min-h-0">
        <TabsContent
          value="markets"
          className="absolute inset-0 mt-0 overflow-y-auto px-4 py-4"
        >
          <InstrumentSidebar
            combos={combos}
            active={active}
            onSelect={handleSelect}
            activeCandleCloseTs={reading?.header.candle_close_ts ?? null}
          />
        </TabsContent>

        <TabsContent
          value="reading"
          className="absolute inset-0 mt-0 overflow-y-auto px-4 py-4"
        >
          <ReadingColumn
            active={active}
            reading={reading}
            isLoading={isLoading}
            isRefreshing={isRefreshing}
            error={error}
            onRetry={onRetry}
            dataSource={dataSource}
          />
        </TabsContent>

        <TabsContent value="chat" className="absolute inset-0 mt-0 p-2">
          {/* onSelect (not handleSelect): picking a recent discussion swaps
              the combo while staying on the Chat tab. AppChatSidebar is h-full
              → its messages scroll and the input docks at the bottom. */}
          <AppChatSidebar active={active} onSelectCombo={onSelect} />
        </TabsContent>
      </div>

      <TabsList className="sticky bottom-0 z-10 grid h-auto w-full grid-cols-3 rounded-none border-t border-border/60 bg-background p-0 pb-[env(safe-area-inset-bottom)]">
        <MobileTabTrigger value="markets" label={t('mobile.tabMarkets')}>
          <ListTree className="h-4 w-4" aria-hidden />
        </MobileTabTrigger>
        <MobileTabTrigger value="reading" label={t('mobile.tabReading')}>
          <LineChart className="h-4 w-4" aria-hidden />
        </MobileTabTrigger>
        <MobileTabTrigger value="chat" label={t('mobile.tabChat')}>
          <MessageCircle className="h-4 w-4" aria-hidden />
        </MobileTabTrigger>
      </TabsList>
    </Tabs>
  );
}

function MobileTabTrigger({
  value,
  label,
  children,
}: {
  value: MobileTab;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <TabsTrigger
      value={value}
      className="flex h-14 flex-col items-center justify-center gap-1 rounded-none text-[11px] data-[state=active]:bg-muted"
    >
      {children}
      {label}
    </TabsTrigger>
  );
}

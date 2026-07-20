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
 * Mobile layout (<768px) — a single full-height column with a sticky header
 * (active combo) and sticky bottom tab bar (Marchés · Lecture · Chat), in the
 * spirit of a native mobile app. Selecting a combo jumps to the Lecture tab.
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
      className="flex min-h-[calc(100vh-4rem)] flex-col"
    >
      <header className="sticky top-0 z-10 border-b border-border/60 bg-background/95 px-4 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <p className="truncate text-sm font-semibold text-foreground">
          {headerLabel}
        </p>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        <TabsContent value="markets" className="mt-0">
          <InstrumentSidebar
            combos={combos}
            active={active}
            onSelect={handleSelect}
            activeCandleCloseTs={reading?.header.candle_close_ts ?? null}
          />
        </TabsContent>

        <TabsContent value="reading" className="mt-0">
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

        <TabsContent value="chat" className="mt-0">
          <div className="h-[70vh]">
            {/* onSelect (not handleSelect): picking a recent discussion swaps
                the combo while staying on the Chat tab. */}
            <AppChatSidebar active={active} onSelectCombo={onSelect} />
          </div>
        </TabsContent>
      </div>

      <TabsList className="sticky bottom-0 z-10 grid h-auto w-full grid-cols-3 rounded-none border-t border-border/60 bg-background p-0">
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

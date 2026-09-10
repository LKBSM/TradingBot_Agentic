'use client';

import {
  HelpCircle,
  Info,
  LayoutPanelTop,
  LineChart,
  PanelLeftClose,
  PanelRightClose,
  RotateCcw,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { MiaPanel } from '@/components/chat/MiaPanel';
import { type WelcomeSuggestion } from '@/components/chat/ChatWelcome';
import { useChat } from '@/components/chat/ChatProvider';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { formatInstrument, formatTimeframe } from '@/lib/market-reading/formatters';
import type { Combo } from '@/lib/market-reading/store';

/** Icons for the on-brand starter questions (text is localized in-component). */
const STARTER_META: ReadonlyArray<{ id: string; icon: React.ReactNode }> = [
  { id: 'structure', icon: <LineChart className="h-4 w-4" aria-hidden /> },
  { id: 'choch', icon: <HelpCircle className="h-4 w-4" aria-hidden /> },
  { id: 'order-blocks', icon: <LayoutPanelTop className="h-4 w-4" aria-hidden /> },
];

/**
 * VZ-4 — starters for the zone surfaces. They are CONTEXTUAL through the SAME
 * mechanism the panel already had (the `starters` prop + the shared `focus`), not
 * a second injection path: when the one conversation is oriented on a zone, the
 * empty-state chips ask about THAT zone.
 *
 * All three are factual questions the agent answers from its tools — formation,
 * containment, contact count. There is deliberately NO predictive chip: the
 * diagnostic showed « ça va rebondir ? » is not intercepted by any deterministic
 * layer (Couche 1 covers jailbreak / trade / persona / advice, Couche 3 filters
 * action-recommendation-timing-risk tokens), so shipping such a chip would invite
 * the one answer this surface must never produce.
 */
const ZONE_STARTER_META: ReadonlyArray<{ id: string; icon: React.ReactNode }> = [
  { id: 'formed', icon: <LineChart className="h-4 w-4" aria-hidden /> },
  { id: 'nested', icon: <LayoutPanelTop className="h-4 w-4" aria-hidden /> },
  { id: 'tested', icon: <HelpCircle className="h-4 w-4" aria-hidden /> },
];

/**
 * /app disposition of the SINGLE M.I.A panel (MIA-3). This is not a second panel:
 * it wraps the shared {@link MiaPanel} and only supplies the /app-specific chrome
 * — the live combo context label, the bubble↔column display-mode toggle, the
 * reset action, and the mode-preference status line. The conversation itself (one
 * product-wide thread) lives entirely in MiaPanel via the shared ChatProvider.
 */
export function AppChatSidebar({
  active,
  displayMode,
  onSetDisplayMode,
}: {
  active: Combo | null;
  /**
   * Current display disposition on /app (desktop ≥1100): `'column'` docked or
   * `'bubble'` reduced. Drives the toggle button. Omitted by the mobile
   * workspace (chat is a tab there, no toggle).
   */
  displayMode?: 'column' | 'bubble';
  /** Switch disposition; with `displayMode`, renders the header toggle (≥1100). */
  onSetDisplayMode?: (mode: 'column' | 'bubble') => void;
}) {
  const t = useTranslations('app');
  const tz = useTranslations('zones');
  const { turns, resetTurns, focus } = useChat();
  const empty = turns.length === 0;

  const STARTERS: ReadonlyArray<WelcomeSuggestion> = STARTER_META.map((s) => ({
    id: s.id,
    text: t(`chat.starter_${s.id}`),
    icon: s.icon,
  }));
  const ZONE_STARTERS: ReadonlyArray<WelcomeSuggestion> = ZONE_STARTER_META.map((s) => ({
    id: `zone-${s.id}`,
    text: tz(`detail.starters.${s.id}`),
    icon: s.icon,
  }));
  // The subject drives the chips — one panel, one conversation, contextual chips.
  const starters = focus?.kind === 'zone' ? ZONE_STARTERS : STARTERS;

  const contextLabel = active
    ? `· ${formatInstrument(active.instrument)} · ${formatTimeframe(active.timeframe)}`
    : `· ${t('chat.pickComboPrompt')}`;

  const headerActions = (
    <TooltipProvider delayDuration={200}>
      {displayMode && onSetDisplayMode && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              size="icon"
              variant="ghost"
              aria-label={
                displayMode === 'column'
                  ? t('chat.collapseToBubble')
                  : t('chat.dockToColumn')
              }
              onClick={() =>
                onSetDisplayMode(displayMode === 'column' ? 'bubble' : 'column')
              }
              // ≥1100 only (APP-1): below that the centre would be too narrow for
              // a legible chart, so no column disposition is offered.
              className="hidden text-muted-foreground min-[1100px]:inline-flex min-[1100px]:h-8 min-[1100px]:w-8"
            >
              {displayMode === 'column' ? (
                <PanelRightClose className="h-4 w-4" aria-hidden />
              ) : (
                <PanelLeftClose className="h-4 w-4" aria-hidden />
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            {displayMode === 'column'
              ? t('chat.collapseToBubble')
              : t('chat.dockToColumn')}
          </TooltipContent>
        </Tooltip>
      )}
      {!empty && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              size="icon"
              variant="ghost"
              aria-label={t('chat.reset')}
              onClick={resetTurns}
              className="h-11 w-11 text-muted-foreground xl:h-8 xl:w-8"
            >
              <RotateCcw className="h-4 w-4" aria-hidden />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">{t('chat.reset')}</TooltipContent>
        </Tooltip>
      )}
    </TooltipProvider>
  );

  // APP-1 — mode status, only where the disposition toggle lives (never mobile).
  const statusLine =
    displayMode && onSetDisplayMode ? (
      <p
        data-testid="mia-mode-status"
        className="mt-1.5 flex items-center gap-1 text-[10px] leading-tight text-muted-foreground/80"
      >
        <Info className="h-3 w-3 shrink-0" aria-hidden />
        <span className="hidden min-[1100px]:inline">{t('chat.modeNotSynced')}</span>
        <span className="min-[1100px]:hidden">{t('chat.columnNeedsWidth')}</span>
      </p>
    ) : undefined;

  return (
    <MiaPanel
      ariaLabel={t('chat.asideAria')}
      title="M.I.A Agent"
      contextLabel={contextLabel}
      headerActions={headerActions}
      statusLine={statusLine}
      welcomeTitle={active ? t('chat.welcomeTitleActive') : t('chat.welcomeTitleIdle')}
      welcomeSubtitle={
        active ? t('chat.welcomeSubtitleActive') : t('chat.welcomeSubtitleIdle')
      }
      starters={active ? starters : []}
      offlineNote={t('chat.offlineNote')}
      complianceLine={t('chat.complianceLine')}
    />
  );
}

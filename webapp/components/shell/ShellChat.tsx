'use client';

import * as React from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { MessageCircle } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useLocalizedHref } from '@/lib/i18n/href';
import { resolveComboFromQuery } from '@/lib/conditions/app-link';
import { AppChatSidebar } from '@/components/app/AppChatSidebar';
import { useChatColumn } from './ChatColumnContext';
import { cn } from '@/lib/utils';
import type { Combo } from '@/lib/market-reading/store';

const DEFAULT_COMBO: Combo = { instrument: 'XAUUSD', timeframe: 'M15' };

/**
 * The chat surface of the product shell (/app only). It hosts the existing
 * AppChatSidebar unchanged — the shared ChatProvider (locale layout) keeps this
 * in lockstep with the /app workspace.
 *
 * Responsive (mission UI-2b), driven by shell.css media queries — one instance,
 * no width JS:
 *   · ≥1280px      → TWO dispositions the user toggles (persisted): column mode
 *                    docks `.chatcol` as a grid cell; bubble mode reduces it to
 *                    the floating `.chat-fab`, which opens the SAME off-canvas
 *                    drawer so the centre reclaims full width without losing the
 *                    conversation (the sidebar is never unmounted).
 *   · 768–1279px   → off-canvas drawer; a floating button (`.chat-fab`) slides it
 *                    in over a backdrop, so the reference centre column keeps its
 *                    full width. Closed via the backdrop or Escape.
 *   · <768px       → hidden here (MobileWorkspace owns the Chat tab).
 * `open` is the drawer state (bubble mode + tablet band); `chatOpen` from the
 * shared context is the ≥1280 disposition (column vs bubble), persisted.
 */
export function ShellChat() {
  const router = useRouter();
  const lh = useLocalizedHref();
  const searchParams = useSearchParams();
  const t = useTranslations('app');
  const [open, setOpen] = React.useState(false);
  const { open: chatOpen, show: showColumn, hide: hideColumn } = useChatColumn();

  // Single display-mode control shared with the sidebar header. Switching to
  // column mode also closes the drawer (it becomes the docked column); switching
  // to bubble mode leaves the drawer closed so only the bubble shows.
  const setDisplayMode = React.useCallback(
    (mode: 'column' | 'bubble') => {
      if (mode === 'column') {
        showColumn();
        setOpen(false);
      } else {
        hideColumn();
      }
    },
    [showColumn, hideColumn],
  );

  const active =
    resolveComboFromQuery(
      searchParams.get('instrument') ?? undefined,
      searchParams.get('timeframe') ?? undefined,
    ) ?? DEFAULT_COMBO;

  const onSelectCombo = React.useCallback(
    (combo: Combo) => {
      router.replace(
        lh(`/app?instrument=${combo.instrument}&timeframe=${combo.timeframe}`),
        { scroll: false },
      );
    },
    [lh, router],
  );

  // Escape closes the drawer (no-op when docked, since it's never "open" there).
  React.useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  return (
    <>
      <button
        type="button"
        className="chat-fab"
        aria-label={t('chat.openPanel')}
        aria-expanded={open}
        onClick={() => setOpen(true)}
      >
        <MessageCircle aria-hidden />
      </button>
      <div
        className={cn('chat-backdrop', open && 'show')}
        onClick={() => setOpen(false)}
        aria-hidden
      />
      <aside className={cn('chatcol', open && 'open')}>
        {/* The header toggle switches disposition on desktop (≥1280): "reduce to
            bubble" when docked, "dock to column" when in the bubble's drawer. The
            tablet band (768–1279) has no column mode, so AppChatSidebar CSS-hides
            the button there; the drawer closes via the backdrop/Escape. */}
        <AppChatSidebar
          active={active}
          onSelectCombo={onSelectCombo}
          displayMode={chatOpen ? 'column' : 'bubble'}
          onSetDisplayMode={setDisplayMode}
        />
      </aside>
    </>
  );
}

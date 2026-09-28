'use client';

import * as React from 'react';

/**
 * M.I.A display disposition on the chat spaces (desktop only): the single,
 * coherent state the user toggles between the two layouts from the page itself —
 *   · `open === true`  → COLUMN mode: M.I.A docked as the right column.
 *   · `open === false` → BUBBLE mode: M.I.A reduced to the floating bubble
 *                        (`.chat-fab`), the centre reclaims the full width.
 *
 * The DEFAULT is per space, because the two spaces spend their width very
 * differently:
 *   · /app — COLUMN. The chart is the subject and the column reads beside it.
 *   · /zones — BUBBLE. The docked column narrows the cards column to 670px,
 *     which (a) wraps the « single controls row » onto two lines, costing 46px
 *     of vertical budget, and (b) squeezes every card's text. /zones is a LIST:
 *     its job is to show zones, so M.I.A starts out of the way and one click
 *     brings it back.
 *
 * The choice is persisted per space in localStorage so it survives visits (like
 * the saved readings) AND so a choice made on /app never moves /zones. /app
 * keeps its historical key, so nobody's existing preference is reset. Below the
 * desktop threshold the shell keeps its own responsive behaviour (tablet
 * off-canvas drawer, phone tab) — this state only drives the desktop
 * disposition. `show()` = go to column, `hide()` = go to bubble.
 */
const DEFAULT_OPEN_BY_SPACE: Readonly<Record<string, boolean>> = {
  app: true,
  zones: false,
  actualites: true,
};

/** Column unless the space is known to be better served by the bubble. */
function defaultOpenFor(space: string): boolean {
  return DEFAULT_OPEN_BY_SPACE[space] ?? true;
}

/** /app keeps the original key so existing preferences survive this change. */
function storageKeyFor(space: string): string {
  return `mia.${space || 'app'}.chat-column-open`;
}

interface ChatColumnValue {
  /** True in column mode (docked), false in bubble mode. Desktop ≥1280 only. */
  open: boolean;
  /** True once the persisted preference has been read on the client. */
  hydrated: boolean;
  toggle(): void;
  show(): void;
  hide(): void;
}

const ChatColumnCtx = React.createContext<ChatColumnValue | null>(null);

export function ChatColumnProvider({
  space,
  children,
}: {
  /** Active product space (`app`, `zones`, …) — it picks the default and the key. */
  space: string;
  children: React.ReactNode;
}) {
  // The server paint uses the space's default, so the common case never flashes.
  // The persisted preference is read post-mount; a user who chose the other
  // disposition sees it settle on the next paint.
  const [open, setOpen] = React.useState(() => defaultOpenFor(space));
  const [hydrated, setHydrated] = React.useState(false);

  // Re-read on every space change: the shell stays mounted across client-side
  // navigation, so /app → /zones must pick up the other space's key and default
  // rather than carry the previous page's disposition over.
  React.useEffect(() => {
    let next = defaultOpenFor(space);
    try {
      const stored = window.localStorage.getItem(storageKeyFor(space));
      // Both directions are explicit: with a per-space default, '1' is a real
      // choice on /zones (« keep the column here ») and not just the default.
      if (stored === '0') next = false;
      else if (stored === '1') next = true;
    } catch {
      /* localStorage unavailable (private mode / SSR) — keep the default. */
    }
    setOpen(next);
    setHydrated(true);
  }, [space]);

  const persist = React.useCallback(
    (next: boolean) => {
      setOpen(next);
      try {
        window.localStorage.setItem(storageKeyFor(space), next ? '1' : '0');
      } catch {
        /* ignore — the choice still applies for the session. */
      }
    },
    [space],
  );

  const value = React.useMemo<ChatColumnValue>(
    () => ({
      open,
      hydrated,
      toggle: () => persist(!open),
      show: () => persist(true),
      hide: () => persist(false),
    }),
    [open, hydrated, persist],
  );

  return <ChatColumnCtx.Provider value={value}>{children}</ChatColumnCtx.Provider>;
}

/**
 * Read the docked-column visibility. Returns a safe no-op default when used
 * outside the provider (e.g. AppChatSidebar mounted by the mobile workspace),
 * so consumers never need to guard for null.
 */
export function useChatColumn(): ChatColumnValue {
  const ctx = React.useContext(ChatColumnCtx);
  if (ctx) return ctx;
  return {
    open: true,
    hydrated: true,
    toggle: () => {},
    show: () => {},
    hide: () => {},
  };
}

'use client';

import * as React from 'react';

/**
 * M.I.A display disposition on /app (desktop ≥1280 only): the single, coherent
 * state the user toggles between the two layouts from the page itself —
 *   · `open === true`  → COLUMN mode: M.I.A docked as the right column.
 *   · `open === false` → BUBBLE mode: M.I.A reduced to the floating bubble
 *                        (`.chat-fab`), the centre reclaims the full width.
 *
 * Column is the DEFAULT (product decision). The choice is persisted in
 * localStorage so it survives visits (like the saved readings). Below 1280px the
 * shell keeps its own responsive behaviour (tablet off-canvas drawer, phone
 * tab) — this state only drives the desktop disposition, so the drawer/tab are
 * never affected by it. `show()` = go to column, `hide()` = go to bubble.
 */
const STORAGE_KEY = 'mia.app.chat-column-open';

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

export function ChatColumnProvider({ children }: { children: React.ReactNode }) {
  // Default to shown so the first server paint (and the common case) renders the
  // column. The persisted preference is read post-mount; if the user had hidden
  // it, the column collapses on the next paint — a one-frame flash only for users
  // who chose to hide, and always in the safe direction (M.I.A visible first).
  const [open, setOpen] = React.useState(true);
  const [hydrated, setHydrated] = React.useState(false);

  React.useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored === '0') setOpen(false);
    } catch {
      /* localStorage unavailable (private mode / SSR) — keep the default. */
    }
    setHydrated(true);
  }, []);

  const persist = React.useCallback((next: boolean) => {
    setOpen(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next ? '1' : '0');
    } catch {
      /* ignore — the choice still applies for the session. */
    }
  }, []);

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

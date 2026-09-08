'use client';

import * as React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useLocale, useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { useLocalizedHref } from '@/lib/i18n/href';
import { MiaLogo } from '@/components/brand/MiaLogo';
import { SkipLink } from '@/components/a11y/SkipLink';
import { ShellRail } from './ShellRail';
import { ShellChat } from './ShellChat';
import { MobileSpaceNav } from './MobileSpaceNav';
import { ChatColumnProvider, useChatColumn } from './ChatColumnContext';
import './shell.css';
import './pages.css';

/**
 * Product shell (mission UI-1) — the terminal-style frame shared by every product
 * route. Three columns on /app (rail · center · docked chat), two columns
 * elsewhere (rail · center). The page CONTENT renders untouched in the center;
 * the shell only supplies the coquille, the rail navigation and the chat column.
 *
 * Structure (grid) and colours come entirely from the literal design tokens via
 * shell.css, so all four themes are covered. No detection/data lives here.
 */
/**
 * Product spaces that dock the shared M.I.A chat column (MIA-3). /app and /zones
 * both get the exact same shell chat column; other spaces stay two-column.
 */
const CHAT_SPACES = new Set(['app', 'zones', 'actualites']);

export function ProductShell({ children }: { children: React.ReactNode }) {
  // The chat-column visibility is shared across the shell frame (grid + chat
  // hide button) AND the page content (DesktopReading's reopen affordance), so
  // the provider wraps both. A component can't consume the context it provides,
  // hence the inner ShellFrame consumer.
  return (
    <ChatColumnProvider>
      <ShellFrame>{children}</ShellFrame>
    </ChatColumnProvider>
  );
}

function ShellFrame({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const locale = useLocale();
  const lh = useLocalizedHref();
  const t = useTranslations();
  const { open: chatOpen } = useChatColumn();

  // Which product route are we on? Strip a leading `/<locale>` (non-default
  // locales are prefixed) then take the first segment. Product routes are flat
  // (app / scanner / zones / compte), so the first segment is the space id.
  const activeSpace = React.useMemo(() => {
    let p = pathname;
    if (p === `/${locale}`) p = '/';
    else if (p.startsWith(`/${locale}/`)) p = p.slice(locale.length + 1);
    return p.split('/').filter(Boolean)[0] ?? '';
  }, [pathname, locale]);

  const isApp = activeSpace === 'app';
  // MIA-3 — M.I.A is one panel docked by the SHELL on every chat space, so /zones
  // gets the exact same column as /app (same component, full height). /app keeps
  // its MobileWorkspace tab on phones; the other chat spaces are "standalone"
  // (no tab system) and get the floating drawer on phones instead.
  const hasChat = CHAT_SPACES.has(activeSpace);
  const isStandaloneChat = hasChat && !isApp;

  return (
    <div
      className={cn(
        'app-shell',
        !hasChat && 'no-chat',
        // Phone chat spaces without a tab system (e.g. /zones) — a marker class so
        // the CSS keeps the floating chat drawer reachable below 768px.
        isStandaloneChat && 'chat-standalone',
        // Bubble mode: drop the docked column and let the centre reclaim its
        // width; M.I.A becomes the floating drawer (never `no-chat`, whose
        // `:not(.no-chat)` media rules drive the tablet drawer we keep).
        hasChat && !chatOpen && 'chat-collapsed',
      )}
    >
      <SkipLink />
      {/* Mobile-only brand bar (<768px, where the rail is hidden): the mark as a
          home link, so every product route keeps the brand on phones (BRD-2). */}
      <Link href={lh('/')} className="shell-mbrand" aria-label={t('nav.brandHomeAria')}>
        <MiaLogo variant="compact" decorative height={20} />
      </Link>
      <ShellRail activeSpace={activeSpace} />
      <div id="main" className="center">
        {children}
        {/* CLN-1 §5 — the single educational/legal disclaimer per page. On desktop
            the rail footer (ShellRail) carries it; the rail is hidden < 768px, so
            this mobile-only copy keeps EXACTLY ONE disclaimer visible at both
            viewports (never zero, never two). Every inline per-surface duplicate
            (the /app header line, the scanner combo note, the /zones M.I.A note)
            was removed so nothing stacks on top of it. */}
        <p className="shell-mdisclaimer" role="note">
          {t('legal.disclaimer.chart')}
        </p>
      </div>
      {hasChat && <ShellChat />}
      {/* Mobile space nav (<768px): every non-/app surface loses the rail there,
          so it gets a bottom tab bar (chat spaces like /zones keep it too — the
          floating chat drawer sits above it). /app keeps MobileWorkspace's tabs. */}
      {!isApp && <MobileSpaceNav activeSpace={activeSpace} />}
    </div>
  );
}

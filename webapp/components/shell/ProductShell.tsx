'use client';

import * as React from 'react';
import { usePathname } from 'next/navigation';
import { useLocale } from 'next-intl';
import { cn } from '@/lib/utils';
import { SkipLink } from '@/components/a11y/SkipLink';
import { ShellRail } from './ShellRail';
import { ShellChat } from './ShellChat';
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
export function ProductShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const locale = useLocale();

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

  return (
    <div className={cn('app-shell', !isApp && 'no-chat')}>
      <SkipLink />
      <ShellRail activeSpace={activeSpace} />
      <div id="main" className="center">
        {children}
      </div>
      {isApp && <ShellChat />}
    </div>
  );
}

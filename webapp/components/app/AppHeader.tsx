'use client';

import Link from 'next/link';
import { CandlestickChart, HelpCircle, Layers, Radar } from 'lucide-react';
import { LocaleToggle } from '@/components/LocaleToggle';
import { ThemeToggle } from '@/components/theme-toggle';
import { Badge } from '@/components/ui/badge';
import { AccountMenu } from './AccountMenu';

/**
 * Product header for the /app workspace. Unlike the landing's marketing nav
 * (Démo · Honnêteté · Tarifs · FAQ), this carries NO marketing anchors — only a
 * brand mark on the left and a utility cluster on the right (plan badge, help,
 * language, theme, account menu). Market navigation lives in the left column of
 * the workspace, not here. The marketing links remain reachable from the
 * account menu ("Le site"), so they live only on the landing.
 */
export function AppHeader() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/85 backdrop-blur">
      <div className="container-wide flex h-14 items-center justify-between gap-4">
        <Link
          href="/app"
          className="flex items-center gap-2 text-sm font-semibold tracking-tight"
          aria-label="MIA Markets — espace de lecture"
        >
          <span
            aria-hidden
            className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-amber-400 to-amber-600 text-xs font-bold text-white shadow-sm"
          >
            M
          </span>
          <span>MIA Markets</span>
        </Link>

        <div className="flex items-center gap-2">
          <Link
            href="/app"
            aria-label="App — espace de lecture"
            className="inline-flex h-9 items-center gap-1.5 rounded-md px-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <CandlestickChart className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">App</span>
          </Link>
          <Link
            href="/zones"
            aria-label="Zones — cycle de vie des zones détectées"
            className="inline-flex h-9 items-center gap-1.5 rounded-md px-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <Layers className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">Zones</span>
          </Link>
          <Link
            href="/scanner"
            aria-label="Scanner de conditions"
            className="inline-flex h-9 items-center gap-1.5 rounded-md px-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <Radar className="h-4 w-4" aria-hidden />
            <span className="hidden sm:inline">Scanner</span>
          </Link>
          <Badge variant="secondary" className="hidden sm:inline-flex">
            Accès libre
          </Badge>
          <Link
            href="/methodology"
            aria-label="Aide — méthodologie et glossaire"
            className="inline-flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <HelpCircle className="h-4 w-4" aria-hidden />
          </Link>
          <LocaleToggle />
          <ThemeToggle />
          <AccountMenu />
        </div>
      </div>
    </header>
  );
}

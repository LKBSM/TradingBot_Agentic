'use client';

import Link from 'next/link';
import { ChevronDown, ExternalLink, LogOut, User } from 'lucide-react';
import * as React from 'react';
import { LocaleToggle } from '@/components/LocaleToggle';
import { cn } from '@/lib/utils';

/**
 * Account menu for the /app product header — a lightweight, dependency-free
 * dropdown (no @radix dropdown package needed). Trigger = avatar + chevron;
 * panel holds the account items.
 *
 * Posture: the app currently runs in personal-testing mode (no real auth /
 * session — see TESTING_MODE), so these entries are the product shell. The
 * marketing surfaces (Honnêteté, FAQ, Tarifs) live ONLY on the landing; here we
 * link back to them rather than duplicating the marketing nav in the header.
 */
export function AccountMenu() {
  const [open, setOpen] = React.useState(false);
  const rootRef = React.useRef<HTMLDivElement>(null);

  // Close on outside click + Escape.
  React.useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Menu du compte"
        className="flex items-center gap-1 rounded-full border border-border/70 py-0.5 pl-0.5 pr-1.5 transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span
          aria-hidden
          className="flex h-7 w-7 items-center justify-center rounded-full bg-secondary text-secondary-foreground"
        >
          <User className="h-4 w-4" />
        </span>
        <ChevronDown
          className={cn('h-3.5 w-3.5 text-muted-foreground transition-transform', open && 'rotate-180')}
          aria-hidden
        />
      </button>

      {open && (
        <div
          role="menu"
          aria-label="Compte"
          className="absolute right-0 z-50 mt-2 w-56 overflow-hidden rounded-lg border border-border/70 bg-popover p-1 text-popover-foreground shadow-md"
        >
          <Link
            href="/#tarifs"
            role="menuitem"
            className="flex items-center justify-between rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
            onClick={() => setOpen(false)}
          >
            Abonnement
          </Link>

          <div className="flex items-center justify-between gap-2 px-3 py-2 text-sm">
            <span className="text-muted-foreground">Langue</span>
            <LocaleToggle />
          </div>

          <div className="my-1 h-px bg-border/60" role="separator" />

          <p className="px-3 pb-1 pt-1 text-[11px] uppercase tracking-wide text-muted-foreground/70">
            Le site
          </p>
          <Link
            href="/#honnetete"
            role="menuitem"
            className="flex items-center justify-between rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
            onClick={() => setOpen(false)}
          >
            Honnêteté
            <ExternalLink className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
          </Link>
          <Link
            href="/#faq"
            role="menuitem"
            className="flex items-center justify-between rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
            onClick={() => setOpen(false)}
          >
            FAQ
            <ExternalLink className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
          </Link>

          <div className="my-1 h-px bg-border/60" role="separator" />

          <Link
            href="/"
            role="menuitem"
            className="flex items-center justify-between rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:bg-accent focus-visible:outline-none"
            onClick={() => setOpen(false)}
          >
            Se déconnecter
            <LogOut className="h-3.5 w-3.5" aria-hidden />
          </Link>
        </div>
      )}
    </div>
  );
}

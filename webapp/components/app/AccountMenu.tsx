'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useTranslations } from 'next-intl';
import { ChevronDown, CreditCard, ExternalLink, LogIn, LogOut, User, UserPlus } from 'lucide-react';
import * as React from 'react';
import { LocaleToggle } from '@/components/LocaleToggle';
import { useAuth } from '@/lib/auth/store';
import { cn } from '@/lib/utils';

/**
 * Account menu for the /app product header — a lightweight, dependency-free
 * dropdown. Trigger = avatar + chevron; panel adapts to the session:
 *   - logged out → Connexion / Inscription
 *   - logged in  → username, Mon compte, Se déconnecter (real logout)
 *
 * The marketing surfaces (Honnêteté, FAQ, Tarifs) live ONLY on the landing; we
 * link back to them here rather than duplicating the marketing nav.
 */
export function AccountMenu() {
  const t = useTranslations('app');
  const [open, setOpen] = React.useState(false);
  const rootRef = React.useRef<HTMLDivElement>(null);
  const { account, isAuthenticated, logout } = useAuth();
  const router = useRouter();

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

  async function onLogout() {
    setOpen(false);
    await logout();
    router.push('/');
  }

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t('account.menuAria')}
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
          aria-label={t('account.panelAria')}
          className="absolute right-0 z-50 mt-2 w-56 overflow-hidden rounded-lg border border-border/70 bg-popover p-1 text-popover-foreground shadow-md"
        >
          {isAuthenticated ? (
            <>
              <p className="truncate px-3 pb-1 pt-2 text-sm font-medium text-foreground">
                {account?.username}
              </p>
              <Link
                href="/compte"
                role="menuitem"
                className="flex items-center justify-between rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
                onClick={() => setOpen(false)}
              >
                {t('account.myAccount')}
                <User className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
              </Link>
              <Link
                href="/abonnement"
                role="menuitem"
                className="flex items-center justify-between rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
                onClick={() => setOpen(false)}
              >
                {t('account.subscription')}
                <CreditCard className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
              </Link>
            </>
          ) : (
            <>
              <Link
                href="/connexion"
                role="menuitem"
                className="flex items-center justify-between rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
                onClick={() => setOpen(false)}
              >
                {t('account.login')}
                <LogIn className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
              </Link>
              <Link
                href="/inscription"
                role="menuitem"
                className="flex items-center justify-between rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
                onClick={() => setOpen(false)}
              >
                {t('account.signup')}
                <UserPlus className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
              </Link>
            </>
          )}

          <div className="flex items-center justify-between gap-2 px-3 py-2 text-sm">
            <span className="text-muted-foreground">{t('account.language')}</span>
            <LocaleToggle />
          </div>

          <div className="my-1 h-px bg-border/60" role="separator" />

          <p className="px-3 pb-1 pt-1 text-[11px] uppercase tracking-wide text-muted-foreground/70">
            {t('account.site')}
          </p>
          <Link
            href="/#honnetete"
            role="menuitem"
            className="flex items-center justify-between rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
            onClick={() => setOpen(false)}
          >
            {t('account.honesty')}
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

          {isAuthenticated && (
            <>
              <div className="my-1 h-px bg-border/60" role="separator" />
              <button
                type="button"
                role="menuitem"
                onClick={onLogout}
                className="flex w-full items-center justify-between rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:bg-accent focus-visible:outline-none"
              >
                {t('account.logout')}
                <LogOut className="h-3.5 w-3.5" aria-hidden />
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

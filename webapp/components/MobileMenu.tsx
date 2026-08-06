'use client';

import Link from 'next/link';
import { Menu, User } from 'lucide-react';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import { LocaleToggle } from '@/components/LocaleToggle';
import { useAuth } from '@/lib/auth/store';
import { BRAND_NAME } from '@/lib/brand';

/**
 * Phone navigation drawer (`sm:hidden`). Below 640px the inline nav clusters
 * (marketing anchors + App/Zones/Scanner + account/language) overflow the top
 * bar, so they collapse into this burger-triggered sheet instead of being
 * dropped or clipped. Each row is a ≥44px touch target; tapping a link closes
 * the sheet (SheetClose). Two variants:
 *   - `marketing` : landing anchors + product links + account
 *   - `app`       : product links + help + account (no marketing anchors)
 */

const PRODUCT_LINKS = [
  { href: '/app', key: 'appLink' as const },
  { href: '/zones', key: 'zones' as const },
  { href: '/scanner', key: 'scanner' as const },
];

// LP-2: marketing anchors match the desktop nav (M.I.A · Démo · Outils ·
// Tarifs · FAQ). Labels are localized via the `nav` namespace.
const MARKETING_ANCHORS = [
  { href: '/#mia', key: 'mia' as const },
  { href: '/#demo', key: 'demo' as const },
  { href: '/#outils', key: 'tools' as const },
  { href: '/#tarifs', key: 'pricing' as const },
  { href: '/#faq', key: 'faq' as const },
];

const ROW =
  'flex min-h-[44px] items-center rounded-md px-3 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring';

export function MobileMenu({ variant }: { variant: 'marketing' | 'app' }) {
  const { isAuthenticated } = useAuth();
  const t = useTranslations('nav');
  const [open, setOpen] = React.useState(false);
  // Product links are gated to signed-in users on the marketing surface (they
  // have no access otherwise); the /app burger always shows them.
  const showProductLinks = variant === 'app' || isAuthenticated;

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <button
          type="button"
          aria-label="Ouvrir le menu"
          className="inline-flex h-11 w-11 items-center justify-center rounded-md text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:hidden"
        >
          <Menu className="h-5 w-5" aria-hidden />
        </button>
      </SheetTrigger>
      <SheetContent
        side="right"
        className="w-72 gap-0 pb-[env(safe-area-inset-bottom)]"
      >
        <SheetHeader className="text-left">
          <SheetTitle>{BRAND_NAME}</SheetTitle>
        </SheetHeader>

        <nav aria-label="Navigation" className="mt-6 flex flex-col gap-1">
          {variant === 'marketing' && (
            <>
              {MARKETING_ANCHORS.map((a) => (
                <SheetClose asChild key={a.href}>
                  <Link href={a.href} className={ROW}>
                    {t(a.key)}
                  </Link>
                </SheetClose>
              ))}
              <hr className="my-2 border-border/60" />
            </>
          )}

          {showProductLinks &&
            PRODUCT_LINKS.map((l) => (
              <SheetClose asChild key={l.href}>
                <Link href={l.href} className={ROW}>
                  {t(l.key)}
                </Link>
              </SheetClose>
            ))}

          {variant === 'app' && (
            <SheetClose asChild>
              <Link href="/methodology" className={ROW}>
                {t('methodology')}
              </Link>
            </SheetClose>
          )}

          <hr className="my-2 border-border/60" />

          {variant === 'marketing' && !isAuthenticated && (
            <SheetClose asChild>
              <Link
                href="/inscription"
                className="flex min-h-[44px] items-center justify-center rounded-md bg-primary px-3 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {t('tryFree')}
              </Link>
            </SheetClose>
          )}

          <SheetClose asChild>
            <Link
              href={isAuthenticated ? '/compte' : '/connexion'}
              className={ROW}
            >
              <User className="mr-2 h-4 w-4" aria-hidden />
              {isAuthenticated ? t('account') : t('login')}
            </Link>
          </SheetClose>

          <div className="px-1 pt-2">
            <LocaleToggle />
          </div>
        </nav>
      </SheetContent>
    </Sheet>
  );
}

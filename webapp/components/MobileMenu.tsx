'use client';

import Link from 'next/link';
import { Menu, User } from 'lucide-react';
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
  { href: '/app', label: 'App' },
  { href: '/zones', label: 'Zones' },
  { href: '/scanner', label: 'Scanner' },
] as const;

const MARKETING_ANCHORS = [
  { href: '/#demo', label: 'Démo' },
  { href: '/#honnetete', label: 'Honnêteté' },
  { href: '/#tarifs', label: 'Tarifs' },
  { href: '/#faq', label: 'FAQ' },
] as const;

const ROW =
  'flex min-h-[44px] items-center rounded-md px-3 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring';

export function MobileMenu({ variant }: { variant: 'marketing' | 'app' }) {
  const { isAuthenticated } = useAuth();
  const [open, setOpen] = React.useState(false);

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
                    {a.label}
                  </Link>
                </SheetClose>
              ))}
              <hr className="my-2 border-border/60" />
            </>
          )}

          {PRODUCT_LINKS.map((l) => (
            <SheetClose asChild key={l.href}>
              <Link href={l.href} className={ROW}>
                {l.label}
              </Link>
            </SheetClose>
          ))}

          {variant === 'app' && (
            <SheetClose asChild>
              <Link href="/methodology" className={ROW}>
                Méthodologie & glossaire
              </Link>
            </SheetClose>
          )}

          <hr className="my-2 border-border/60" />

          <SheetClose asChild>
            <Link
              href={isAuthenticated ? '/compte' : '/connexion'}
              className={ROW}
            >
              <User className="mr-2 h-4 w-4" aria-hidden />
              {isAuthenticated ? 'Compte' : 'Connexion'}
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

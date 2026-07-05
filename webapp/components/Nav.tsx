'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { User } from 'lucide-react';
import { AppHeader } from '@/components/app/AppHeader';
import { LocaleToggle } from '@/components/LocaleToggle';
import { ThemeToggle } from '@/components/theme-toggle';
import { useAuth } from '@/lib/auth/store';
import { SUPPORTED_LOCALES } from '@/i18n';

const ANCHORS = [
  { href: '#demo', label: 'Démo' },
  { href: '#honnetete', label: 'Honnêteté' },
  { href: '#tarifs', label: 'Tarifs' },
  { href: '#faq', label: 'FAQ' },
] as const;

/**
 * Is the current route a product surface (the /app workspace or the /scanner
 * page)? Both wear the product header (brand → /app + Scanner button) so the
 * user can move between the reading workspace and the scanner freely.
 * `localePrefix: 'as-needed'` means FR (default) has no prefix (`/app`), but we
 * strip a leading locale segment defensively in case a prefixed locale ships.
 */
function isAppRoute(pathname: string): boolean {
  const segs = pathname.split('/').filter(Boolean);
  const first = segs[0] as (typeof SUPPORTED_LOCALES)[number] | undefined;
  const rest = first && SUPPORTED_LOCALES.includes(first) ? segs.slice(1) : segs;
  return rest[0] === 'app' || rest[0] === 'scanner' || rest[0] === 'zones';
}

/**
 * Top navigation. On the marketing landing it shows the section anchors (Démo ·
 * Honnêteté · Tarifs · FAQ). On the /app workspace it swaps to the product
 * header (brand + utility cluster only) — the marketing nav lives ONLY on the
 * landing.
 */
/**
 * Session-aware account control for the marketing nav. Logged out → "Connexion";
 * logged in → "Compte" (links to /compte). Stays quiet during the initial /me
 * probe to avoid a flash of the wrong state.
 */
function NavAccountLink() {
  const { isAuthenticated, loading } = useAuth();
  if (loading) {
    return <span className="h-9 w-20" aria-hidden />;
  }
  return (
    <Link
      href={isAuthenticated ? '/compte' : '/connexion'}
      className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <User className="h-4 w-4" aria-hidden />
      <span className="hidden sm:inline">{isAuthenticated ? 'Compte' : 'Connexion'}</span>
    </Link>
  );
}

export function Nav() {
  const pathname = usePathname() ?? '/';
  if (isAppRoute(pathname)) {
    return <AppHeader />;
  }

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/85 backdrop-blur">
      <div className="container-prose flex h-14 items-center justify-between gap-4">
        <Link
          href="/"
          className="flex items-center gap-2 text-sm font-semibold tracking-tight"
          aria-label="MIA Markets — retour à l'accueil"
        >
          <span
            aria-hidden
            className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-amber-400 to-amber-600 text-xs font-bold text-white shadow-sm"
          >
            M
          </span>
          <span>MIA Markets</span>
        </Link>

        <nav aria-label="Sections du site" className="hidden sm:block">
          <ul className="flex items-center gap-1 text-sm">
            {ANCHORS.map((a) => (
              <li key={a.href}>
                <Link
                  href={a.href}
                  className="rounded-md px-3 py-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {a.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        {/* gap-1 + px-2 sous sm : sur 390px le cluster débordait et le
            LocaleToggle recouvrait le ThemeToggle (toggle thème incliquable
            sur mobile — bug attrapé par l'e2e mobile-iphone-12). */}
        <div className="flex items-center gap-1 sm:gap-2">
          <Link
            href="/zones"
            className="rounded-md px-2 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:px-3"
          >
            Zones
          </Link>
          <Link
            href="/scanner"
            className="rounded-md px-2 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:px-3"
          >
            Scanner
          </Link>
          <NavAccountLink />
          {/* V1 = FR-only (middleware 302 en/de/es → fr) : le sélecteur de
              langue reste desktop-only tant que l'i18n n'est pas activée. */}
          <div className="hidden sm:block">
            <LocaleToggle />
          </div>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

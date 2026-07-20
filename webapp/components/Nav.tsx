'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useTranslations } from 'next-intl';
import { User } from 'lucide-react';
import { AppHeader } from '@/components/app/AppHeader';
import { LocaleToggle } from '@/components/LocaleToggle';
import { MobileMenu } from '@/components/MobileMenu';
import { ThemeToggle } from '@/components/theme-toggle';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { BRAND_NAME, BRAND_BASELINE } from '@/lib/brand';
import { SUPPORTED_LOCALES } from '@/i18n';

// Anchors keep their hrefs in code; the visible label is pulled from the
// `nav` message namespace by key so every locale renders in its own language.
const ANCHORS = [
  { href: '#demo', key: 'demo' },
  { href: '#honnetete', key: 'honesty' },
  { href: '#tarifs', key: 'pricing' },
  { href: '#faq', key: 'faq' },
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
  const t = useTranslations('nav');
  const lh = useLocalizedHref();
  if (loading) {
    return <span className="h-9 w-20" aria-hidden />;
  }
  return (
    <Link
      href={lh(isAuthenticated ? '/compte' : '/connexion')}
      className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <User className="h-4 w-4" aria-hidden />
      <span className="hidden sm:inline">{isAuthenticated ? t('account') : t('login')}</span>
    </Link>
  );
}

export function Nav() {
  const pathname = usePathname() ?? '/';
  const t = useTranslations('nav');
  const lh = useLocalizedHref();
  if (isAppRoute(pathname)) {
    return <AppHeader />;
  }

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/85 backdrop-blur">
      <div className="container-prose flex h-14 items-center justify-between gap-4">
        <Link
          href={lh('/')}
          className="flex items-center gap-2 text-sm font-semibold tracking-tight"
          aria-label={t('brandHomeAria')}
        >
          <span
            aria-hidden
            className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-amber-400 to-amber-600 text-xs font-bold text-white shadow-sm"
          >
            M
          </span>
          <span className="flex flex-col leading-none">
            <span>{BRAND_NAME}</span>
            <span className="mt-0.5 hidden text-[10px] font-normal tracking-tight text-muted-foreground lg:block">
              {BRAND_BASELINE}
            </span>
          </span>
        </Link>

        <nav aria-label={t('sectionsAria')} className="hidden sm:block">
          <ul className="flex items-center gap-1 text-sm">
            {ANCHORS.map((a) => (
              <li key={a.href}>
                <Link
                  href={a.href}
                  className="rounded-md px-3 py-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {t(a.key)}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        {/* Sous sm (390px) le cluster débordait : App/Zones/Scanner + compte +
            langue passent dans le tiroir burger (MobileMenu). Restent toujours
            visibles : le thème et le burger. */}
        <div className="flex items-center gap-1 sm:gap-2">
          <div className="hidden items-center gap-1 sm:flex sm:gap-2">
            <Link
              href={lh('/app')}
              className="rounded-md px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              App
            </Link>
            <Link
              href={lh('/zones')}
              className="rounded-md px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {t('zones')}
            </Link>
            <Link
              href={lh('/scanner')}
              className="rounded-md px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              {t('scanner')}
            </Link>
            <NavAccountLink />
            {/* Sélecteur de langue desktop-only (le cluster mobile est déjà dense
                sur 390px ; les liens localisés ci-dessus couvrent toutes locales). */}
            <LocaleToggle />
          </div>
          <ThemeToggle />
          <MobileMenu variant="marketing" />
        </div>
      </div>
    </header>
  );
}

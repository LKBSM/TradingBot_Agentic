'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useTranslations } from 'next-intl';
import { User } from 'lucide-react';
import { AppHeader } from '@/components/app/AppHeader';
import { BrandMark } from '@/components/BrandMark';
import { LocaleToggle } from '@/components/LocaleToggle';
import { MobileMenu } from '@/components/MobileMenu';
import { ThemeMenu } from '@/components/theme/ThemeMenu';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { BRAND_NAME, BRAND_BASELINE } from '@/lib/brand';
import { SUPPORTED_LOCALES } from '@/i18n';

// Anchors keep their hrefs in code; the visible label is pulled from the
// `nav` message namespace by key so every locale renders in its own language.
// LP-2: the visitor sees ONLY marketing anchors here — the product links
// (App/Zones/Scanner) are gated behind authentication further down.
const ANCHORS = [
  { href: '#mia', key: 'mia' },
  { href: '#demo', key: 'demo' },
  { href: '#outils', key: 'tools' },
  { href: '#tarifs', key: 'pricing' },
  { href: '#faq', key: 'faq' },
] as const;

/**
 * Is the current route a product surface (the /app workspace, /scanner or
 * /zones)? They wear the product header, not the marketing nav.
 */
function isAppRoute(pathname: string): boolean {
  const segs = pathname.split('/').filter(Boolean);
  const first = segs[0] as (typeof SUPPORTED_LOCALES)[number] | undefined;
  const rest = first && SUPPORTED_LOCALES.includes(first) ? segs.slice(1) : segs;
  return rest[0] === 'app' || rest[0] === 'scanner' || rest[0] === 'zones';
}

const PRODUCT_LINK =
  'rounded-md px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring';

/**
 * Right-hand cluster of the marketing nav. LP-2 rule: App/Zones/Scanner appear
 * ONLY when authenticated — a visitor has no access, so surfacing them is pure
 * frustration. A visitor instead sees a secondary "Se connecter" and a primary
 * "Essayer gratuitement". During the initial /me probe we treat the session as
 * logged-out (show the try/login CTAs) to avoid flashing product links.
 */
function NavCluster() {
  const { isAuthenticated, loading } = useAuth();
  const t = useTranslations('nav');
  const lh = useLocalizedHref();

  if (isAuthenticated && !loading) {
    return (
      <>
        <Link href={lh('/app')} className={PRODUCT_LINK}>App</Link>
        <Link href={lh('/zones')} className={PRODUCT_LINK}>{t('zones')}</Link>
        <Link href={lh('/scanner')} className={PRODUCT_LINK}>{t('scanner')}</Link>
        <Link
          href={lh('/compte')}
          className="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <User className="h-4 w-4" aria-hidden />
          <span className="hidden sm:inline">{t('account')}</span>
        </Link>
        <LocaleToggle />
      </>
    );
  }

  return (
    <>
      <Link
        href={lh('/connexion')}
        className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {t('login')}
      </Link>
      <Link
        href={lh('/inscription')}
        className="rounded-md bg-primary px-3 py-1.5 text-sm font-semibold text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {t('tryFree')}
      </Link>
      <LocaleToggle />
    </>
  );
}

/**
 * Top navigation. On the marketing landing it shows the section anchors and the
 * session-aware cluster. On the /app workspace it swaps to the product header.
 */
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
          title={BRAND_BASELINE}
        >
          <BrandMark size={28} />
          <span>{BRAND_NAME}</span>
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

        {/* Sous sm (390px) le cluster passe dans le tiroir burger (MobileMenu).
            Restent toujours visibles : le thème et le burger. */}
        <div className="flex items-center gap-1 sm:gap-2">
          <div className="hidden items-center gap-1 sm:flex sm:gap-2">
            <NavCluster />
          </div>
          <ThemeMenu />
          <MobileMenu variant="marketing" />
        </div>
      </div>
    </header>
  );
}

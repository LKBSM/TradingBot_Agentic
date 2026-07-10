import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ShieldCheck } from 'lucide-react';
import { BRAND_BASELINE } from '@/lib/brand';

/**
 * Footer partagé (rendu une seule fois dans `[locale]/layout`), internationalisé
 * (i18n Étape 1 + réconcilié avec le nettoyage claims de main).
 *
 * Règle éditoriale (nettoyage claims 2026-07-04) : le footer n'affirme QUE ce
 * qui est vrai et vérifiable aujourd'hui, et ne liste que des liens dont la
 * cible existe. Les libellés viennent du namespace `footer` ; seuls les hrefs
 * restent en dur. `/#…` : le footer est global, les sections vivent sur la
 * landing — le lien doit fonctionner depuis /app, /scanner, /zones aussi.
 */
export const LEGAL_LINKS = [
  { href: '/conditions', key: 'terms' },
  { href: '/confidentialite', key: 'privacy' },
  { href: 'mailto:contact@mia.markets', key: 'contact' },
] as const;

export const PRODUCT_LINKS = [
  { href: '/#multi-marche', key: 'multiAsset' },
  { href: '/#conversations', key: 'chatbot' },
  { href: '/#avant-apres', key: 'beforeAfter' },
  { href: '/#honnetete', key: 'transparency' },
  { href: '/methodology', key: 'methodology' },
  { href: '/methodology#attributions', key: 'attributions' },
  { href: '/#tarifs', key: 'pricing' },
  { href: '/#faq', key: 'faq' },
] as const;

export function Footer() {
  const t = useTranslations('footer');
  // String so the year renders literally (no locale number-grouping / Eastern digits).
  const year = String(new Date().getFullYear());

  return (
    <footer role="contentinfo" className="border-t border-border/60 bg-muted/30">
      <div className="container-wide space-y-8 py-12">
        <div className="grid gap-8 lg:grid-cols-[1.5fr_1fr_1fr]">
          {/* Brand + Early Access badge */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <p className="text-sm font-semibold tracking-tight">MIA Markets</p>
              <span className="inline-flex items-center gap-1 rounded-full border border-sentinel-bull/40 bg-sentinel-bull/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-sentinel-bull">
                <ShieldCheck className="h-2.5 w-2.5" aria-hidden />
                {t('earlyAccessBadge')}
              </span>
            </div>
            {/* Brand baseline kept in English in every locale (the "MIA" acronym
                expansion) — see lib/brand.ts. */}
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground/80">
              {BRAND_BASELINE} for Markets
            </p>
            <p className="max-w-md text-xs text-muted-foreground">{t('description')}</p>
            <p className="text-[11px] text-muted-foreground">
              <span className="font-mono">mia.markets</span> ·{' '}
              <a
                href="mailto:contact@mia.markets"
                className="underline-offset-2 hover:text-foreground hover:underline"
              >
                contact@mia.markets
              </a>
            </p>
          </div>

          {/* Produit */}
          <nav aria-label={t('productHeading')} className="text-xs">
            <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              {t('productHeading')}
            </p>
            <ul className="space-y-2">
              {PRODUCT_LINKS.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    className="text-muted-foreground transition-colors hover:text-foreground focus-visible:underline focus-visible:outline-none"
                  >
                    {t(`product.${link.key}`)}
                  </a>
                </li>
              ))}
            </ul>
          </nav>

          {/* Légal */}
          <nav aria-label={t('legalHeading')} className="text-xs">
            <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              {t('legalHeading')}
            </p>
            <ul className="space-y-2">
              {LEGAL_LINKS.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-muted-foreground transition-colors hover:text-foreground focus-visible:underline focus-visible:outline-none"
                  >
                    {t(`legal.${link.key}`)}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </div>

        <div
          className="rounded-md border border-dashed border-border/70 bg-card/50 px-3 py-2 text-[11px] italic leading-relaxed text-muted-foreground"
          data-legal-pending="footer-disclaimer"
        >
          {t('disclaimer')}
        </div>

        <p className="text-[11px] text-muted-foreground">{t('copyright', { year })}</p>
      </div>
    </footer>
  );
}

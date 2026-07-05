import Link from 'next/link';
import { ShieldCheck } from 'lucide-react';

/**
 * Footer partagé (rendu une seule fois dans `[locale]/layout`).
 *
 * Règle éditoriale (nettoyage claims 2026-07-04) : le footer n'affirme QUE ce
 * qui est vrai et vérifiable aujourd'hui. Il ne liste que des liens dont la
 * cible existe réellement dans le repo. Les pages légales manquantes
 * (mentions légales, cookies) seront ajoutées ici quand elles existeront —
 * ne PAS réintroduire de lien mort ni de claim de conformité non sourcé.
 */
export const LEGAL_LINKS = [
  { href: '/conditions', label: 'Conditions d’utilisation' },
  { href: '/confidentialite', label: 'Confidentialité' },
  { href: 'mailto:contact@mia.markets', label: 'Contact' },
] as const;

// Ancres préfixées `/#` : le footer est global, les sections vivent sur la
// landing — le lien doit fonctionner depuis /app, /scanner, /zones aussi.
export const PRODUCT_LINKS = [
  { href: '/#multi-marche', label: 'Multi-actifs' },
  { href: '/#conversations', label: 'Chatbot M.I.A Agent' },
  { href: '/#avant-apres', label: 'Avant / Après' },
  { href: '/#honnetete', label: 'Transparence' },
  { href: '/methodology', label: 'Méthodologie' },
  { href: '/methodology#attributions', label: 'Attributions' },
  { href: '/#tarifs', label: 'Tarifs' },
  { href: '/#faq', label: 'FAQ' },
] as const;

export function Footer() {
  return (
    <footer
      role="contentinfo"
      className="border-t border-border/60 bg-muted/30"
    >
      <div className="container-wide space-y-8 py-12">
        <div className="grid gap-8 lg:grid-cols-[1.5fr_1fr_1fr]">
          {/* Brand + Early Access badge */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <p className="text-sm font-semibold tracking-tight">
                MIA Markets
              </p>
              <span className="inline-flex items-center gap-1 rounded-full border border-sentinel-bull/40 bg-sentinel-bull/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-sentinel-bull">
                <ShieldCheck className="h-2.5 w-2.5" aria-hidden />
                Accès anticipé
              </span>
            </div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground/80">
              Multi-asset Intelligence Assistant for Markets
            </p>
            <p className="max-w-md text-xs text-muted-foreground">
              Indicateur de marché conversationnel. Posture éducative,
              compréhension augmentée du marché — pas une performance
              financière promise.
            </p>
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
          <nav aria-label="Produit" className="text-xs">
            <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Produit
            </p>
            <ul className="space-y-2">
              {PRODUCT_LINKS.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    className="text-muted-foreground transition-colors hover:text-foreground focus-visible:underline focus-visible:outline-none"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>

          {/* Légal */}
          <nav aria-label="Mentions légales" className="text-xs">
            <p className="mb-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Légal
            </p>
            <ul className="space-y-2">
              {LEGAL_LINKS.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-muted-foreground transition-colors hover:text-foreground focus-visible:underline focus-visible:outline-none"
                  >
                    {link.label}
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
          Démonstration en accès anticipé · Lecture algorithmique éducative ·
          Ne constitue ni un signal de trading, ni un conseil en
          investissement personnalisé · Performances passées non
          indicatives des performances futures · Trading à effet de
          levier comporte un risque élevé de perte.
        </div>

        <p className="text-[11px] text-muted-foreground">
          © {new Date().getFullYear()} MIA Markets — tous droits
          réservés.
        </p>
      </div>
    </footer>
  );
}

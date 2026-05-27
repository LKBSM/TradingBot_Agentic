import Link from 'next/link';

// LEGAL-PENDING: footer legal links + compliance bandeau wording — to be
// finalised with the legal terminal output (CGU/CGV, Privacy, Mentions
// légales URL paths, mediation, RGPD DSAR endpoint). For now the hrefs
// are placeholders that scroll to a non-existent fragment; the legal pass
// will route them to real pages.
const LEGAL_LINKS = [
  { href: '#legal-mentions', label: 'Mentions légales' },
  { href: '#legal-cgu', label: 'CGU' },
  { href: '#legal-privacy', label: 'Confidentialité' },
  { href: '#legal-mediation', label: 'Médiateur' },
  { href: '#contact', label: 'Contact' },
] as const;

export function Footer() {
  return (
    <footer
      role="contentinfo"
      className="border-t border-border/60 bg-muted/30"
    >
      <div className="container-prose space-y-6 py-10">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-1">
            <p className="text-sm font-semibold tracking-tight">
              MIA Markets
            </p>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground/80">
              Multi-asset Intelligence Assistant for Markets
            </p>
            <p className="max-w-md text-xs text-muted-foreground">
              Indicateur de marché conversationnel. Posture éducative,
              conformité UE 2024/2811 par construction.
            </p>
          </div>
          <nav aria-label="Mentions légales" className="text-xs">
            <ul className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-1">
              {LEGAL_LINKS.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-muted-foreground transition-colors hover:text-foreground focus-visible:underline focus-visible:outline-none"
                    data-legal-pending="footer-link"
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
          Démonstration paper-trading · Lecture algorithmique éducative · Ne
          constitue ni un signal de trading, ni un conseil en investissement
          personnalisé · Disponibilité restreinte hors UE
          (US/QC/UK/OFAC bloqués).
        </div>

        <p className="text-[11px] text-muted-foreground">
          © {new Date().getFullYear()} MIA Markets — tous droits réservés. ·{' '}
          <span className="font-mono">mia.markets</span>
        </p>
      </div>
    </footer>
  );
}

import type { Metadata } from 'next';
import Link from 'next/link';
import { ArrowLeft, ShieldCheck } from 'lucide-react';

export const metadata: Metadata = {
  title: 'Politique de confidentialité',
  description:
    'Politique de confidentialité de MIA Markets — données traitées, base légale, droits RGPD / Loi 25.',
};

/**
 * /confidentialite — placeholder STRUCTURÉ. The full, lawyer-reviewed privacy
 * policy is mission ④ (terminal légal). This page already exposes the spine of
 * the document (controller, data, basis, rights, contact) so the consent
 * checkbox at registration links to a real, honest page — but the canonical
 * legally-binding text is explicitly pending.
 */
const SECTIONS = [
  {
    title: '1. Responsable du traitement',
    body: 'Loukmane Bessam (exploitant unique) — loukmanebessam@gmail.com.',
  },
  {
    title: '2. Données traitées',
    body: 'Nom d’utilisateur, adresse e-mail, mot de passe (haché, jamais en clair), consentements (version + horodatage), données de session. Aucune donnée de carte bancaire n’est collectée par MIA Markets.',
  },
  {
    title: '3. Bases légales',
    body: 'Exécution du contrat (compte et service), intérêt légitime (sécurité), consentement (acceptation des Conditions et de la présente Politique à l’inscription).',
  },
  {
    title: '4. Vos droits (RGPD / Loi 25)',
    body: 'Accès, rectification, effacement, portabilité, limitation et opposition. Toute demande à loukmanebessam@gmail.com — réponse sous 30 jours.',
  },
  {
    title: '5. Conservation',
    body: 'Les données de compte sont conservées pour la durée du compte. Les sessions expirent automatiquement. Les détails de conservation par catégorie seront précisés dans la version définitive.',
  },
  {
    title: '6. Contact',
    body: 'Pour toute question relative à vos données : loukmanebessam@gmail.com.',
  },
] as const;

export default function ConfidentialitePage() {
  return (
    <div className="container-prose py-12 sm:py-16">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-1.5 text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
      >
        <ArrowLeft className="h-3.5 w-3.5" aria-hidden />
        Retour à l’accueil
      </Link>

      <header className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          Politique de confidentialité
        </h1>
        <div
          className="flex items-start gap-2 rounded-md border border-dashed border-border/70 bg-card/50 p-3 text-sm text-muted-foreground"
          data-legal-pending="privacy-placeholder"
        >
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>
            Version préliminaire structurée. Le texte définitif (relu
            juridiquement) sera publié avec le terminal légal. La structure
            ci-dessous décrit fidèlement les traitements en vigueur.
          </span>
        </div>
      </header>

      <div className="mt-8 space-y-6">
        {SECTIONS.map((s) => (
          <section key={s.title} className="space-y-1.5">
            <h2 className="text-lg font-semibold text-foreground">{s.title}</h2>
            <p className="leading-relaxed text-muted-foreground">{s.body}</p>
          </section>
        ))}
      </div>
    </div>
  );
}

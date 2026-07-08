import {
  Activity,
  CalendarClock,
  Gauge,
  GitCompare,
  Layers,
  MessagesSquare,
  Radar,
  ShieldCheck,
  Workflow,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';

/**
 * Section « Comment ça marche + points forts ».
 *
 * Objectif : définir clairement le produit AVANT les démos, puis expliquer
 * ce que chacune des démos de la page prouve concrètement. Reste strictement
 * descriptif — aucune promesse de gain, aucun signal, aucune référence
 * réglementaire non vérifiée (garde-fou `tests/claims-cleanup.test.ts`).
 */

interface Step {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  body: string;
}

const STEPS: ReadonlyArray<Step> = [
  {
    icon: Activity,
    title: '1 · Structure du marché',
    body: "Le moteur repère les empreintes réelles des gros flux : cassures de structure (BOS), changements de caractère (CHOCH), Order Blocks, Fair Value Gaps et retests. Ni indicateurs empilés, ni interprétation subjective.",
  },
  {
    icon: Gauge,
    title: '2 · Régime & volatilité',
    body: "Il qualifie la tendance, la maturité du mouvement et l'amplitude moyenne des bougies — volatilité normale ou élevée. Ce qui est incertain est affiché comme incertain, jamais masqué.",
  },
  {
    icon: CalendarClock,
    title: '3 · Événements macro',
    body: "Il surveille le calendrier (FOMC, BCE, NFP…) et se met en pause à l'approche des publications à fort impact, le temps que la volatilité retombe.",
  },
  {
    icon: MessagesSquare,
    title: '4 · Lecture narrée + chatbot',
    body: "Tout est traduit en une lecture claire, puis M.I.A Agent (Claude) répond à vos questions avec ce contexte injecté. Il explique, il contextualise — et il refuse tout ordre d'achat ou de vente.",
  },
];

interface Strength {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  body: string;
  href: string;
  proof: string;
}

const STRENGTHS: ReadonlyArray<Strength> = [
  {
    icon: Layers,
    title: 'Multi-actifs, multi-horizons',
    body: 'La même rigueur méthodologique sur l’or et l’euro, de M15 à H4 — le BTC arrive après validation du moteur.',
    href: '#multi-marche',
    proof: 'Démo · Trois lectures',
  },
  {
    icon: Radar,
    title: 'Un chatbot qui a le contexte',
    body: 'M.I.A Agent connaît la lecture en cours : structure, régime, événements. Il vulgarise, et refuse les questions qu’il ne doit pas répondre.',
    href: '#conversations',
    proof: 'Démo · Conversations',
  },
  {
    icon: GitCompare,
    title: 'Une seule lecture assumée',
    body: 'Fini les trois indicateurs qui se contredisent : un cadre unique, un verdict descriptif, l’incertitude toujours visible.',
    href: '#avant-apres',
    proof: 'Démo · Avant / Après',
  },
  {
    icon: ShieldCheck,
    title: 'Honnête par construction',
    body: 'Zéro promesse de gain, zéro signal de trade — une posture éducative qu’on assume publiquement.',
    href: '#honnetete',
    proof: 'Section · Transparence',
  },
];

export function HowItWorksSection() {
  return (
    <section
      id="fonctionnement"
      aria-labelledby="how-title"
      className="container-wide space-y-12 py-16 sm:py-20"
    >
      <header className="max-w-2xl space-y-3">
        <Badge
          variant="secondary"
          className="text-[11px] uppercase tracking-wider"
        >
          <Workflow className="mr-1 h-3 w-3" aria-hidden />
          Comment ça marche
        </Badge>
        <h2
          id="how-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          De la bougie brute à une lecture que vous comprenez.
        </h2>
        <p className="text-pretty text-muted-foreground">
          MIA lit le marché comme un analyste structurel, puis vous l’explique
          en langage clair. Voici les quatre étapes — et, plus bas, ce que
          chaque démo de cette page vous montre en conditions réelles.
        </p>
      </header>

      {/* Les 4 étapes */}
      <ol className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step) => {
          const Icon = step.icon;
          return (
            <li
              key={step.title}
              className="rounded-2xl border border-border/60 bg-card/50 p-5"
            >
              <Icon className="h-5 w-5 text-primary" aria-hidden />
              <h3 className="mt-3 text-sm font-semibold tracking-tight">
                {step.title}
              </h3>
              <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                {step.body}
              </p>
            </li>
          );
        })}
      </ol>

      {/* Points forts + ancrage vers les démos */}
      <div className="space-y-5">
        <h3 className="text-lg font-semibold tracking-tight">
          Ce que les démos prouvent
        </h3>
        <div className="grid gap-5 sm:grid-cols-2">
          {STRENGTHS.map((s) => {
            const Icon = s.icon;
            return (
              <a
                key={s.title}
                href={s.href}
                className="group flex gap-4 rounded-2xl border border-border/60 bg-card/50 p-5 transition-colors hover:border-primary/40 hover:bg-card"
              >
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <Icon className="h-5 w-5" aria-hidden />
                </span>
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h4 className="text-sm font-semibold tracking-tight">
                      {s.title}
                    </h4>
                    <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground/80">
                      {s.proof}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {s.body}
                  </p>
                  <span className="inline-block text-xs font-medium text-primary underline-offset-4 group-hover:underline">
                    Voir la démo →
                  </span>
                </div>
              </a>
            );
          })}
        </div>
      </div>
    </section>
  );
}

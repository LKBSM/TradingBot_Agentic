import { ShieldCheck, AlertTriangle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

/**
 * Section 5 — « Honnêteté » (Engagement public).
 *
 * Pleine largeur, fond sombre subtil. C'est le moat de MIA : on assume
 * publiquement notre posture — un outil de compréhension, pas une promesse
 * de gain.
 *
 * Post-pivot 2026-05-27 (Chantier 5.C) : retrait des chiffres de backtest
 * (Profit Factor, Deflated Sharpe Ratio, PBO) et de la colonne « edge
 * mesurable » — ces métriques relevaient du positionnement pré-pivot
 * « système de trading » et n'ont pas leur place sur un indicateur descriptif
 * (edge_claim=false, niveau 1.5 strict).
 *
 * Conservé : la citation « Engagement public » (lock 2) + les colonnes
 * « Ce que nous ne ferons jamais » et « Ce que nous faisons aujourd'hui ».
 *
 * Cette section EXISTE parce qu'elle constitue la défense MIFID II / UE
 * 2024/2811 et le différenciateur "fini de mentir" face aux fournisseurs
 * de signaux qui annoncent 90 % de win-rate.
 */
export function HonestConfidenceSection() {
  return (
    <section
      id="honnetete"
      aria-labelledby="honest-confidence-title"
      className="bg-foreground/[0.02] py-16 sm:py-24 dark:bg-foreground/[0.03]"
    >
      <div className="container-wide">
        <header className="mb-10 max-w-3xl">
          <Badge
            variant="outline"
            className="mb-3 border-sentinel-warn/40 text-[11px] uppercase tracking-wider text-sentinel-warn"
          >
            <ShieldCheck className="mr-1 h-3 w-3" aria-hidden />
            Honnêteté conformelle
          </Badge>
          <h2
            id="honest-confidence-title"
            className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl lg:text-4xl"
          >
            Un indicateur honnête commence par dire ce qu&apos;il ne fait pas.
          </h2>
          <p className="mt-4 max-w-2xl text-pretty text-muted-foreground">
            MIA décrit le marché — la structure, le régime, les événements.
            Elle ne promet aucun gain et ne se présente pas comme un système
            de trading. Nous l&apos;assumons publiquement, parce que c&apos;est
            la seule posture défendable pour un outil de compréhension.
          </p>
        </header>

        {/* Citation imposée (lock 2) */}
        <figure className="max-w-3xl">
          <blockquote className="border-l-2 border-sentinel-warn pl-5 text-balance text-base italic text-foreground sm:text-lg">
            Aucun indicateur de marché ne devrait promettre des gains. Nous
            n&apos;en faisons pas. Ce que nous offrons, c&apos;est une
            compréhension augmentée du marché — pas une performance
            financière.
          </blockquote>
          <figcaption className="mt-3 text-xs text-muted-foreground">
            — Engagement public MIA Markets, 27 mai 2026
          </figcaption>
        </figure>

        {/* Que faisons-nous alors */}
        <div className="mt-12 grid gap-6 lg:grid-cols-2">
          <ValueColumn
            icon={<AlertTriangle className="h-5 w-5" aria-hidden />}
            title="Ce que nous ne ferons jamais"
            items={[
              'Publier des « 90 % win-rate ».',
              'Promettre un rendement chiffré.',
              'Vendre une lecture comme un signal de trade.',
              'Nous présenter comme un système de trading rentable.',
            ]}
            tone="bad"
          />
          <ValueColumn
            icon={<ShieldCheck className="h-5 w-5" aria-hidden />}
            title={`Ce que nous faisons aujourd${'’'}hui`}
            items={[
              'Une lecture structurée du marché en temps réel.',
              "Une incertitude affichée, jamais masquée.",
              'Un chatbot qui refuse les ordres de trade.',
              'Une méthodologie publique et reproductible.',
            ]}
            tone="good"
          />
        </div>
      </div>
    </section>
  );
}

interface ValueColumnProps {
  icon: React.ReactNode;
  title: string;
  items: ReadonlyArray<string>;
  tone: 'bad' | 'good';
}

function ValueColumn({ icon, title, items, tone }: ValueColumnProps) {
  const iconClasses = {
    bad: 'text-sentinel-bear',
    good: 'text-sentinel-bull',
  } as const;
  const bulletClasses = {
    bad: 'bg-sentinel-bear',
    good: 'bg-sentinel-bull',
  } as const;

  return (
    <div>
      <h3 className="flex items-center gap-2 text-sm font-semibold tracking-tight">
        <span className={iconClasses[tone]}>{icon}</span>
        <span>{title}</span>
      </h3>
      <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
        {items.map((item) => (
          <li key={item} className="flex items-start gap-2">
            <span
              className={cn(
                'mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full',
                bulletClasses[tone],
              )}
              aria-hidden
            />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

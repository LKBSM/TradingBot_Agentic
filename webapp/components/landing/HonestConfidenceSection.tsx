import { ShieldCheck, AlertTriangle, FileText } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

/**
 * Section 5 — « Honnêteté Conformelle ».
 *
 * Pleine largeur, fond sombre subtil. C'est le moat de MIA : on assume
 * publiquement nos chiffres défavorables.
 *
 *   - Profit Factor backtest 7 ans = 0,786 (sous PF=1)
 *   - Sous-performance vs buy-and-hold = −318 pp
 *   - A1 verdict 2026-05-01 : DSR=0,000 · PBO=0,500 · edge non détecté
 *   - edge_claim = false dans le contrat InsightSignal v2.1.0
 *
 * + citation imposée par lock 2 :
 *   « Aucun indicateur de marché ne devrait promettre des gains. Nous n'en
 *    faisons pas. Ce que nous offrons, c'est une compréhension augmentée
 *    du marché — pas une performance financière. »
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
            Voici ce que notre backtest dit. Et il ne dit pas ce que vous
            espérez entendre.
          </h2>
          <p className="mt-4 max-w-2xl text-pretty text-muted-foreground">
            Sur 7 ans d&apos;historique XAU/USD, le scoring actuel de MIA{' '}
            <strong className="text-foreground">ne bat pas</strong> une
            stratégie passive. Nous le publions — pas parce que c&apos;est
            confortable, mais parce qu&apos;un indicateur honnête commence
            par dire ce qu&apos;il ne sait pas faire.
          </p>
        </header>

        {/* Quatre chiffres durs */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Profit Factor backtest"
            value="0,786"
            unit="sur 7 ans"
            tone="bad"
            footnote="Reference : XAU/USD M15 · 2019-01 → 2025-12 · coûts inclus."
          />
          <StatCard
            label="Sous-performance vs buy & hold"
            value="−318"
            unit="points · 7 ans"
            tone="bad"
            footnote="Stratégie longue/short v. simple détention or."
          />
          <StatCard
            label="Deflated Sharpe Ratio"
            value="0,000"
            unit="hypothèse multiple-testing"
            tone="bad"
            footnote="Verdict A1 stack 2026-05-01 · LightGBM 2 niveaux."
          />
          <StatCard
            label="Probability of Backtest Overfitting"
            value="0,50"
            unit="cible &lt; 0,33"
            tone="warn"
            footnote="CPCV 28 paths · 19 features."
          />
        </div>

        {/* Citation imposée (lock 2) */}
        <figure className="mt-12 max-w-3xl">
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
        <div className="mt-12 grid gap-6 lg:grid-cols-3">
          <ValueColumn
            icon={<AlertTriangle className="h-5 w-5" aria-hidden />}
            title="Ce que nous ne ferons jamais"
            items={[
              'Publier des « 90 % win-rate ».',
              'Promettre un rendement chiffré.',
              'Vendre une lecture comme un signal de trade.',
              'Cacher un backtest défavorable.',
            ]}
            tone="bad"
          />
          <ValueColumn
            icon={<ShieldCheck className="h-5 w-5" aria-hidden />}
            title={`Ce que nous faisons aujourd${'’'}hui`}
            items={[
              'Une lecture structurée du marché en temps réel.',
              "Une jauge d'incertitude conformelle, affichée.",
              'Un chatbot qui refuse les ordres de trade.',
              'Tous nos chiffres publics — bons ou mauvais.',
            ]}
            tone="good"
          />
          <ValueColumn
            icon={<FileText className="h-5 w-5" aria-hidden />}
            title="Ce sur quoi nous travaillons"
            items={[
              'Améliorer la valeur prédictive du scoring.',
              "Atteindre un edge mesurable (DSR > 1, PF lo > 1,05).",
              'Tester en paper-trading avant toute promesse.',
              'Publier les progrès — comme les revers.',
            ]}
            tone="neutral"
          />
        </div>

        {/* Notes techniques bas de section */}
        <p className="mt-10 max-w-3xl text-xs leading-relaxed text-muted-foreground">
          <strong className="font-medium text-foreground">
            Note technique.
          </strong>{' '}
          Le contrat <code className="rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[10px]">
            InsightSignal v2.1.0
          </code>{' '}
          expose explicitement le champ{' '}
          <code className="rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[10px]">
            edge_claim = false
          </code>
          . Tant que ce champ n&apos;est pas passé à <code className="rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[10px]">true</code>,
          MIA s&apos;interdit de présenter ses lectures comme un signal de
          trade rentable. Méthodologie : Deflated Sharpe Ratio (Bailey &amp;
          López de Prado, 2014) · CPCV (López de Prado, 2018) · PBO (Bailey
          et al., 2017). Audit interne reproductible (rapport{' '}
          <code className="rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[10px]">
            a1_verdict_2026.md
          </code>).
        </p>
      </div>
    </section>
  );
}

interface StatCardProps {
  label: string;
  value: string;
  unit: string;
  footnote: string;
  tone: 'bad' | 'warn' | 'good';
}

function StatCard({ label, value, unit, footnote, tone }: StatCardProps) {
  const toneClasses = {
    bad: 'border-sentinel-bear/30 bg-sentinel-bear/[0.04]',
    warn: 'border-sentinel-warn/30 bg-sentinel-warn/[0.04]',
    good: 'border-sentinel-bull/30 bg-sentinel-bull/[0.04]',
  } as const;
  const valueClasses = {
    bad: 'text-sentinel-bear',
    warn: 'text-sentinel-warn',
    good: 'text-sentinel-bull',
  } as const;

  return (
    <article
      className={cn(
        'rounded-2xl border p-5 sm:p-6',
        toneClasses[tone],
      )}
    >
      <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p
        className={cn(
          'mt-2 flex items-baseline gap-1.5 tabular-nums',
          valueClasses[tone],
        )}
      >
        <span className="text-3xl font-semibold sm:text-4xl">{value}</span>
        <span className="text-xs font-normal text-muted-foreground">
          {unit}
        </span>
      </p>
      <p className="mt-3 text-[11px] leading-snug text-muted-foreground">
        {footnote}
      </p>
    </article>
  );
}

interface ValueColumnProps {
  icon: React.ReactNode;
  title: string;
  items: ReadonlyArray<string>;
  tone: 'bad' | 'good' | 'neutral';
}

function ValueColumn({ icon, title, items, tone }: ValueColumnProps) {
  const iconClasses = {
    bad: 'text-sentinel-bear',
    good: 'text-sentinel-bull',
    neutral: 'text-muted-foreground',
  } as const;
  const bulletClasses = {
    bad: 'bg-sentinel-bear',
    good: 'bg-sentinel-bull',
    neutral: 'bg-muted-foreground',
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

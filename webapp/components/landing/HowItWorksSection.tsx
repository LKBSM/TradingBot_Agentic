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
import { useTranslations } from 'next-intl';
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
  key: string;
}

const STEPS: ReadonlyArray<Step> = [
  { icon: Activity, key: 'marketStructure' },
  { icon: Gauge, key: 'regimeVolatility' },
  { icon: CalendarClock, key: 'macroEvents' },
  { icon: MessagesSquare, key: 'narratedChatbot' },
];

interface Strength {
  icon: React.ComponentType<{ className?: string }>;
  key: string;
  href: string;
}

const STRENGTHS: ReadonlyArray<Strength> = [
  { icon: Layers, key: 'multiAsset', href: '#multi-marche' },
  { icon: Radar, key: 'contextChatbot', href: '#conversations' },
  { icon: GitCompare, key: 'singleReading', href: '#avant-apres' },
  { icon: ShieldCheck, key: 'honestByDesign', href: '#honnetete' },
];

export function HowItWorksSection() {
  const t = useTranslations('landing.howItWorks');

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
          {t('kicker')}
        </Badge>
        <h2
          id="how-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          {t('title')}
        </h2>
        <p className="text-pretty text-muted-foreground">
          {t('subtitle')}
        </p>
      </header>

      {/* Les 4 étapes */}
      <ol className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step) => {
          const Icon = step.icon;
          return (
            <li
              key={step.key}
              className="rounded-2xl border border-border/60 bg-card/50 p-5"
            >
              <Icon className="h-5 w-5 text-primary" aria-hidden />
              <h3 className="mt-3 text-sm font-semibold tracking-tight">
                {t(`steps.${step.key}.title`)}
              </h3>
              <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                {t(`steps.${step.key}.body`)}
              </p>
            </li>
          );
        })}
      </ol>

      {/* Points forts + ancrage vers les démos */}
      <div className="space-y-5">
        <h3 className="text-lg font-semibold tracking-tight">
          {t('strengthsHeading')}
        </h3>
        <div className="grid gap-5 sm:grid-cols-2">
          {STRENGTHS.map((s) => {
            const Icon = s.icon;
            return (
              <a
                key={s.key}
                href={s.href}
                className="group flex gap-4 rounded-2xl border border-border/60 bg-card/50 p-5 transition-colors hover:border-primary/40 hover:bg-card"
              >
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <Icon className="h-5 w-5" aria-hidden />
                </span>
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h4 className="text-sm font-semibold tracking-tight">
                      {t(`strengths.${s.key}.title`)}
                    </h4>
                    <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground/80">
                      {t(`strengths.${s.key}.proof`)}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {t(`strengths.${s.key}.body`)}
                  </p>
                  <span className="inline-block text-xs font-medium text-primary underline-offset-4 group-hover:underline">
                    {t('seeDemo')}
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

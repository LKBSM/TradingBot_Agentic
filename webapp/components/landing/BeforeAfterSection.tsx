import { ArrowRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { LandingReadingChart } from './LandingReadingChart';

/**
 * Section 4 — « L'avant / L'après ».
 *
 * À gauche, le chaos habituel (trois indicateurs génériques empilés qui se
 * contredisent) ; à droite, la VRAIE lecture MIA : le graphique produit
 * (`ReadingChart`, chandeliers + zones SMC lues dans la structure), la même
 * surface que la vue /app — pas un mock. Aucun texte « regardez comme c'est
 * mieux » : la composition fait le job.
 */
export function BeforeAfterSection() {
  return (
    <section
      id="avant-apres"
      aria-labelledby="before-after-title"
      className="container-wide py-16 sm:py-20"
    >
      <header className="mb-8 max-w-2xl">
        <Badge
          variant="secondary"
          className="mb-3 text-[11px] uppercase tracking-wider"
        >
          <ArrowRight className="mr-1 h-3 w-3" aria-hidden />
          Avant · Après
        </Badge>
        <h2
          id="before-after-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          La même heure de marché, deux façons de la lire.
        </h2>
        <p className="mt-3 text-pretty text-muted-foreground">
          La plupart des traders empilent trois indicateurs qui se contredisent.
          MIA lit la structure du marché et l&apos;affiche telle quelle — voici
          le graphique réel du produit, à droite.
        </p>
      </header>

      <div className="grid items-stretch gap-5 sm:gap-6 lg:grid-cols-2">
        <BeforeCard />
        <AfterCard />
      </div>
    </section>
  );
}

function BeforeCard() {
  return (
    <article
      className="relative flex flex-col gap-4 rounded-2xl border border-border/60 bg-muted/30 p-5 shadow-sm sm:p-6"
      aria-labelledby="before-card-title"
    >
      <header className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Approche classique
          </p>
          <h3
            id="before-card-title"
            className="mt-0.5 text-base font-semibold tracking-tight"
          >
            Trois indicateurs empilés
          </h3>
        </div>
        <Badge variant="secondary" className="text-[10px]">
          3 signaux contradictoires
        </Badge>
      </header>

      <BeforeChart />

      <ul className="space-y-1.5 text-xs text-muted-foreground">
        <li className="flex items-start gap-2">
          <span
            className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-bull"
            aria-hidden
          />
          <span>
            <strong className="font-medium text-foreground">RSI 28</strong>{' '}
            — survente, signal d&apos;achat.
          </span>
        </li>
        <li className="flex items-start gap-2">
          <span
            className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-bear"
            aria-hidden
          />
          <span>
            <strong className="font-medium text-foreground">MACD</strong>{' '}
            — croisement baissier, signal de vente.
          </span>
        </li>
        <li className="flex items-start gap-2">
          <span
            className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-neutral"
            aria-hidden
          />
          <span>
            <strong className="font-medium text-foreground">Bollinger</strong>{' '}
            — prix sur la bande inférieure, signal ambigu.
          </span>
        </li>
      </ul>

      <p className="mt-auto rounded-md bg-background/60 px-3 py-2 text-xs italic text-muted-foreground">
        « Bon, je fais quoi ? »
      </p>
    </article>
  );
}

function AfterCard() {
  return (
    <article
      className="relative flex flex-col gap-4 rounded-2xl border border-primary/30 bg-card p-5 shadow-md sm:p-6"
      aria-labelledby="after-card-title"
    >
      <header className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Lecture MIA · Or (XAU/USD) · 15 min
          </p>
          <h3
            id="after-card-title"
            className="mt-0.5 text-base font-semibold tracking-tight"
          >
            Le vrai graphique du produit, rien de caché.
          </h3>
        </div>
        <Badge variant="default" className="text-[10px]">
          1 lecture assumée
        </Badge>
      </header>

      {/* Le graphique produit réel : chandeliers + zones SMC (OB / FVG),
          niveaux de cassure (BOS / CHOCH) et poches de liquidité. */}
      <LandingReadingChart />

      <ul className="space-y-1.5 text-xs text-muted-foreground">
        <li className="flex items-start gap-2">
          <span
            className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-bull"
            aria-hidden
          />
          <span>
            <strong className="font-medium text-foreground">Biais haussier</strong>{' '}
            — cassure de structure confirmée et retest en cours, décrits sans
            score directif.
          </span>
        </li>
        <li className="flex items-start gap-2">
          <span
            className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-neutral"
            aria-hidden
          />
          <span>
            <strong className="font-medium text-foreground">Zones dessinées</strong>{' '}
            — Order Block et Fair Value Gap haussiers sous le prix, poches de
            liquidité repérées.
          </span>
        </li>
        <li className="flex items-start gap-2">
          <span
            className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-warn"
            aria-hidden
          />
          <span>
            <strong className="font-medium text-foreground">Volatilité annoncée</strong>{' '}
            — un FOMC approche : MIA signale l&apos;élargissement probable, sans
            décider à ta place.
          </span>
        </li>
      </ul>

      <p className="mt-auto rounded-md bg-muted/40 px-3 py-2 text-xs italic text-muted-foreground">
        « Voilà ce qu&apos;on voit. À toi de décider si tu agis. »
      </p>
    </article>
  );
}

/**
 * Chart "avant" : trois courbes désynchronisées (RSI, MACD, prix dans BB)
 * pour évoquer la cacophonie sans simuler un vrai chart. SVG pur, viewbox
 * 0 0 400 140, stroke uniquement.
 */
function BeforeChart() {
  return (
    <svg
      viewBox="0 0 400 140"
      className="h-32 w-full rounded-lg bg-background/40"
      role="img"
      aria-labelledby="before-chart-title before-chart-desc"
    >
      <title id="before-chart-title">Trois indicateurs contradictoires</title>
      <desc id="before-chart-desc">
        Un graphique illustratif où RSI, MACD et bandes de Bollinger envoient
        des signaux opposés sur la même fenêtre temporelle.
      </desc>

      {/* Bandes Bollinger — gris clair */}
      <path
        d="M 10 35 Q 100 28, 200 42 T 390 38"
        fill="none"
        stroke="hsl(var(--muted-foreground) / 0.25)"
        strokeWidth="1"
        strokeDasharray="3 3"
      />
      <path
        d="M 10 95 Q 100 88, 200 102 T 390 98"
        fill="none"
        stroke="hsl(var(--muted-foreground) / 0.25)"
        strokeWidth="1"
        strokeDasharray="3 3"
      />

      {/* Prix — neutre */}
      <path
        d="M 10 70 L 50 60 L 90 85 L 130 55 L 170 90 L 210 65 L 250 95 L 290 60 L 330 80 L 370 70"
        fill="none"
        stroke="hsl(var(--foreground) / 0.7)"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* RSI overlay — bullish (vert) */}
      <path
        d="M 10 110 L 80 105 L 150 100 L 220 95 L 290 92 L 370 90"
        fill="none"
        stroke="hsl(var(--sentinel-bull))"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <text
        x="375"
        y="88"
        className="fill-[hsl(var(--sentinel-bull))] text-[8px] font-medium"
        textAnchor="end"
      >
        RSI ↑
      </text>

      {/* MACD overlay — bearish (rouge) */}
      <path
        d="M 10 20 L 80 25 L 150 30 L 220 32 L 290 38 L 370 42"
        fill="none"
        stroke="hsl(var(--sentinel-bear))"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <text
        x="375"
        y="50"
        className="fill-[hsl(var(--sentinel-bear))] text-[8px] font-medium"
        textAnchor="end"
      >
        MACD ↓
      </text>

      {/* Annotation BB */}
      <text
        x="375"
        y="32"
        className="fill-muted-foreground text-[8px]"
        textAnchor="end"
      >
        BB band
      </text>
    </svg>
  );
}

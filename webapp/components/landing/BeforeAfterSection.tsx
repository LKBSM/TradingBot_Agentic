import { ArrowRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

/**
 * Section 4 — « L'avant / L'après ».
 *
 * Comparaison visuelle SVG : à gauche, le chaos habituel (RSI + MACD + BB
 * empilés, signaux contradictoires) ; à droite, la lecture MIA (un verdict,
 * une jauge, un blackout, c'est tout). Aucun texte « regardez comme c'est
 * mieux » — la composition fait le job.
 *
 * SVG pur (pas de chart lib) pour perf Lighthouse + a11y title/desc.
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
          MIA propose une seule lecture — et l&apos;assume.
        </p>
      </header>

      <div className="grid gap-5 sm:gap-6 lg:grid-cols-2">
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
            RSI + MACD + Bollinger empilés
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

      <p className="mt-1 rounded-md bg-background/60 px-3 py-2 text-xs italic text-muted-foreground">
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
            Lecture MIA
          </p>
          <h3
            id="after-card-title"
            className="mt-0.5 text-base font-semibold tracking-tight"
          >
            Un verdict, une conviction, un cadre.
          </h3>
        </div>
        <Badge variant="default" className="text-[10px]">
          1 lecture assumée
        </Badge>
      </header>

      <AfterChart />

      <ul className="space-y-1.5 text-xs text-muted-foreground">
        <li className="flex items-start gap-2">
          <span
            className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-bull"
            aria-hidden
          />
          <span>
            <strong className="font-medium text-foreground">Biais haussier</strong>{' '}
            — conviction modérée (62/100), 8 composantes alignées sur 11.
          </span>
        </li>
        <li className="flex items-start gap-2">
          <span
            className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-warn"
            aria-hidden
          />
          <span>
            <strong className="font-medium text-foreground">Blackout</strong>{' '}
            — FOMC dans 3 h, prudence demandée.
          </span>
        </li>
        <li className="flex items-start gap-2">
          <span
            className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-neutral"
            aria-hidden
          />
          <span>
            <strong className="font-medium text-foreground">Incertitude conformelle</strong>{' '}
            ±0.8 % à 1 h — affichée, pas masquée.
          </span>
        </li>
      </ul>

      <p className="mt-1 rounded-md bg-muted/40 px-3 py-2 text-xs italic text-muted-foreground">
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

/**
 * Chart "après" : un seul prix propre, une zone de conviction haussière
 * teintée vert pâle, une zone d'incertitude conformelle gris-foncé. Pas de
 * faux indicateur — la lecture parle d'elle-même.
 */
function AfterChart() {
  return (
    <svg
      viewBox="0 0 400 140"
      className="h-32 w-full rounded-lg bg-background/40"
      role="img"
      aria-labelledby="after-chart-title after-chart-desc"
    >
      <title id="after-chart-title">Lecture MIA — biais haussier conditionnel</title>
      <desc id="after-chart-desc">
        Le même graphique avec une seule lecture : zone haussière conditionnelle
        marquée, incertitude conformelle représentée par une bande grise autour
        du prix.
      </desc>

      {/* Cône d'incertitude conformelle */}
      <path
        d="M 200 70 L 370 30 L 370 110 L 200 70 Z"
        fill="hsl(var(--muted-foreground) / 0.12)"
      />

      {/* Trajectoire projetée centrale */}
      <path
        d="M 200 70 L 370 60"
        fill="none"
        stroke="hsl(var(--sentinel-bull))"
        strokeWidth="1.5"
        strokeDasharray="4 3"
        strokeLinecap="round"
      />

      {/* Prix passé — clean */}
      <path
        d="M 10 95 L 50 88 L 90 92 L 130 80 L 170 82 L 200 70"
        fill="none"
        stroke="hsl(var(--foreground))"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Point d'entrée du verdict */}
      <circle
        cx="200"
        cy="70"
        r="3.5"
        fill="hsl(var(--sentinel-bull))"
        stroke="hsl(var(--background))"
        strokeWidth="1.5"
      />

      {/* Annotations */}
      <text
        x="200"
        y="58"
        className="fill-[hsl(var(--sentinel-bull))] text-[9px] font-semibold"
        textAnchor="middle"
      >
        Biais ↑ · 62
      </text>
      <text
        x="370"
        y="120"
        className="fill-muted-foreground text-[8px]"
        textAnchor="end"
      >
        Incertitude ±0.8 % · 1 h
      </text>

      {/* Marker blackout FOMC */}
      <line
        x1="320"
        y1="10"
        x2="320"
        y2="130"
        stroke="hsl(var(--sentinel-warn))"
        strokeWidth="1"
        strokeDasharray="2 2"
        opacity="0.7"
      />
      <text
        x="324"
        y="22"
        className="fill-[hsl(var(--sentinel-warn))] text-[8px] font-medium"
      >
        FOMC
      </text>
    </svg>
  );
}

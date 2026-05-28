import { Clock } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { ComingSoonCard } from './ComingSoonCard';
import { InsightGalleryClient } from './InsightGalleryClient';
import { SAMPLE_SIGNALS } from '@/lib/mocks';

/**
 * Section 2 — « MIA lit plusieurs marchés ».
 *
 * Aucun texte explicatif "multi-actifs" — la cartographie le démontre :
 * XAU M15 + EUR H1 actifs + 1 placeholder grisé "Bientôt" (BTC ou US500
 * — D4 instruments lock : XAU + EUR seuls en GA, les autres post-S16).
 *
 * Le visiteur scrolle, voit 3 cartes, comprend sans qu'on ait à le dire.
 */
export function MultiMarketSection() {
  const xau = SAMPLE_SIGNALS[0]; // XAU M15 bullish
  const eur = SAMPLE_SIGNALS[1]; // EUR H1 bearish

  if (!xau || !eur) return null;

  return (
    <section
      id="multi-marche"
      aria-labelledby="multi-marche-title"
      className="container-wide py-16 sm:py-20"
    >
      <header className="mb-8 max-w-2xl">
        <Badge
          variant="secondary"
          className="mb-3 text-[11px] uppercase tracking-wider"
        >
          <Clock className="mr-1 h-3 w-3" aria-hidden />
          Trois lectures · à l&apos;instant
        </Badge>
        <h2
          id="multi-marche-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          Sur chaque marché, le même cadre de lecture.
        </h2>
        <p className="mt-3 text-pretty text-muted-foreground">
          Or, devises, indices — la même rigueur méthodologique, la même
          honnêteté sur l&apos;incertitude.
        </p>
      </header>

      <InsightGalleryClient
        signals={[xau, eur]}
        renderAfter={<ComingSoonCard label="BTC/USD" subtitle="Bientôt" />}
        gridClassName="grid-cols-1 gap-5 sm:gap-6 lg:grid-cols-3"
      />
    </section>
  );
}

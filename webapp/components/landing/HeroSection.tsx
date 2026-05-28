import Link from 'next/link';
import { ArrowDown, ShieldCheck } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { SAMPLE_SIGNALS } from '@/lib/mocks';

/**
 * Landing hero — positioning statement + honest track record + dual CTA.
 * Above-the-fold on mobile 375px. The track-record line is the DG-120
 * pépite ("track-record honnête en hero") — it cites the same XAU figures
 * that the rest of the system can substantiate.
 */
export function HeroSection() {
  // We sample stats from the bullish XAU mock as a stand-in for the public
  // paper-trading aggregate. F5 backend integration will replace this with
  // a real /api/v1/forward-test/snapshot read.
  const sample = SAMPLE_SIGNALS[0];
  const hist = sample?.historical_stats;

  return (
    <section
      aria-labelledby="hero-title"
      className="container-prose pb-12 pt-12 sm:pt-20"
    >
      <Badge variant="secondary" className="mb-5 text-[11px] uppercase tracking-wider">
        <ShieldCheck className="mr-1 h-3 w-3" aria-hidden />
        Indicateur · pas un service de signaux
      </Badge>

      <h1
        id="hero-title"
        className="text-balance text-4xl font-semibold leading-tight tracking-tight sm:text-5xl"
      >
        Comprenez le marché — sans qu&apos;on vous dise quoi faire.
      </h1>

      <p className="mt-6 max-w-2xl text-pretty text-lg text-muted-foreground">
        MIA Markets lit l&apos;or, l&apos;euro et les indices avec les méthodes des
        salles de marché institutionnelles — et un chatbot (Sentinel) qui
        répond à toutes vos questions, en français, en temps réel.
      </p>

      <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:items-center">
        <Button asChild size="lg" className="w-full sm:w-auto">
          <Link href="#demo">
            Voir une lecture en direct
            <ArrowDown aria-hidden />
          </Link>
        </Button>
        <Button asChild size="lg" variant="outline" className="w-full sm:w-auto">
          {/* LEGAL-PENDING: CTA wording + tunnel d'inscription — branchement
              Stripe + auth viendra au sprint d'intégration. */}
          <Link href="#tarifs">Découvrir les formules</Link>
        </Button>
      </div>

      {hist && (
        <div className="mt-10 rounded-xl border border-border bg-card/60 p-5 sm:p-6">
          <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Track record honnête · XAU M15
          </p>
          <p className="mt-2 text-balance text-base sm:text-lg">
            <span className="font-semibold tabular-nums">
              {hist.similar_setups_n} setups
            </span>{' '}
            sur 7 ans de walk-forward, profit factor{' '}
            <span className="font-semibold tabular-nums">
              {hist.profit_factor.toFixed(2).replace('.', ',')}
            </span>{' '}
            <span className="text-muted-foreground tabular-nums">
              (IC 95 %&nbsp;
              {hist.profit_factor_ci95[0].toFixed(2).replace('.', ',')} –
              &nbsp;
              {hist.profit_factor_ci95[1].toFixed(2).replace('.', ',')})
            </span>
            .
          </p>
          {/* LEGAL-PENDING: edge_claim disclosure wording — to be finalised
              with the legal terminal. Stays factual, no performance promise. */}
          <p className="mt-2 text-xs italic text-muted-foreground">
            Performance passée non garante de la performance future. Aucun
            edge n&apos;est revendiqué tant que les critères de validation
            empirique ne sont pas franchis.
          </p>
        </div>
      )}
    </section>
  );
}

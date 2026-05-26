import { MarketReadingCard } from '@/components/insight/MarketReadingCard';
import { SAMPLE_SIGNALS } from '@/lib/mocks';

/**
 * Sprint F2 demo gallery — renders the three mocked signals through the
 * MarketReadingCard hero layer. The proper landing (hero + how-it-works +
 * pricing + footer) is built in F5.
 */
export default function LandingPage() {
  return (
    <div className="container-prose space-y-12 py-10 sm:py-16">
      <header className="space-y-3">
        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Sprint F3 · démo hero + sections collapsibles
        </p>
        <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
          Smart Sentinel — Lecture de marché
        </h1>
        <p className="max-w-2xl text-pretty text-muted-foreground">
          Trois lectures algorithmiques mockées. Hero permanent + cinq sections
          dépliables (Structure / Régime / Volatilité / Événements / Historique).
          Le chatbot pilier arrive au sprint F4.
        </p>
      </header>

      <section className="space-y-6">
        {SAMPLE_SIGNALS.map((signal, idx) => (
          <MarketReadingCard
            key={signal.id}
            signal={signal}
            // Démo F3 : la première carte ouvre l'historique par défaut pour
            // exposer le hero différenciateur (PF + IC95%). Les deux autres
            // restent toutes collapsées (état "neuf" attendu en prod).
            defaultOpenSections={idx === 0 ? ['history'] : undefined}
          />
        ))}
      </section>
    </div>
  );
}

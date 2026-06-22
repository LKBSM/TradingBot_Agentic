import type { Metadata } from 'next';
import { ScannerWorkspace } from '@/components/scanner/ScannerWorkspace';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';

export const metadata: Metadata = {
  title: 'Scanner de conditions',
  description:
    'Définis tes conditions structurelles et vois sur quels marchés et timeframes elles sont présentes en ce moment. Outil de lecture descriptif — à toi le jugement.',
};

/**
 * Scanner page — the user composes present-tense structural conditions and the
 * tool shows where they are met right now across XAU/USD and EUR/USD on M15/H1/H4.
 * Descriptive and read-only: it reuses readings the detection engine already
 * produced and never recommends action.
 */
export default async function ScannerPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  // The multi-market scanner is a paid feature: require full access (subscriber
  // or owner). Visitors → login; free accounts → a clean upsell paywall. Open
  // while the gate is OFF (testing phase).
  return (
    <SubscriptionGate
      requireFullAccess
      paywallTitle="Le scanner est réservé aux abonnés"
      paywallDescription="Le scanner multi-marchés balaie XAU/USD et EUR/USD sur M15, H1 et H4. Passe à l’abonnement pour l’utiliser."
    >
      <div className="container-wide py-8">
        <ScannerWorkspace locale={locale} />
      </div>
    </SubscriptionGate>
  );
}

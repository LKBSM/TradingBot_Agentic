import type { Metadata } from 'next';
import { ScannerWorkspace } from '@/components/scanner/ScannerWorkspace';

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
  return (
    <div className="container-prose py-8">
      <ScannerWorkspace locale={locale} />
    </div>
  );
}

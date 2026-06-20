import type { Metadata } from 'next';
import { AppWorkspace } from '@/components/app/AppWorkspace';
import { resolveComboFromQuery } from '@/lib/conditions/app-link';

export const metadata: Metadata = {
  title: 'Espace de lecture',
  description:
    'Lecture de marché en direct — XAU/USD et EUR/USD sur M15, H1, H4. Structure, régime, événements et synthèse, expliqués par Sentinel.',
};

/**
 * Application view — the working surface (distinct from the marketing landing
 * at `/`). Three columns on desktop: instruments · reading · chat. The heavy
 * lifting lives in the client AppWorkspace; this server page only sets the
 * route + metadata.
 */
export default async function AppPage({
  searchParams,
}: {
  searchParams: Promise<{ instrument?: string; timeframe?: string }>;
}) {
  const sp = await searchParams;
  // Honour an optional Scanner deep-link (?instrument=&timeframe=); falls back
  // to the default (no pre-selection) for any absent / out-of-perimeter value.
  const initialCombo = resolveComboFromQuery(sp.instrument, sp.timeframe);
  return <AppWorkspace initialCombo={initialCombo} />;
}

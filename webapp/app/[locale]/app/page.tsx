import type { Metadata } from 'next';
import { AppWorkspace } from '@/components/app/AppWorkspace';
import { resolveComboFromQuery } from '@/lib/conditions/app-link';

/**
 * Default combo on load (XAU/USD M15). Defined inline rather than imported from
 * the client-only mockReadings module so this server component doesn't pull a
 * 'use client' module into the server bundle.
 */
const DEFAULT_COMBO = { instrument: 'XAUUSD', timeframe: 'M15' } as const;

export const metadata: Metadata = {
  title: 'Espace de lecture',
  description:
    'Lecture de marché en direct — XAU/USD et EUR/USD sur M15, H1, H4. Structure, régime, événements et lecture narrée, expliqués par M.I.A Agent.',
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
  // Honour an optional Scanner deep-link (?instrument=&timeframe=); fall back to
  // the default XAU/USD M15 so a direct visit still shows a populated surface.
  const initialCombo = resolveComboFromQuery(sp.instrument, sp.timeframe) ?? DEFAULT_COMBO;
  return <AppWorkspace initialCombo={initialCombo} />;
}

import type { Metadata } from 'next';
import { AppWorkspace } from '@/components/app/AppWorkspace';

/**
 * Default combo on load (XAU/USD M15). Defined inline rather than imported from
 * the client-only mockReadings module so this server component doesn't pull a
 * 'use client' module into the server bundle.
 */
const DEFAULT_COMBO = { instrument: 'XAUUSD', timeframe: 'M15' } as const;

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
export default function AppPage() {
  // Default to a XAU/USD M15 reading on load so the workspace shows a fully
  // populated "produit fini" surface immediately (mock data — see mockReadings).
  return <AppWorkspace initialCombo={DEFAULT_COMBO} />;
}

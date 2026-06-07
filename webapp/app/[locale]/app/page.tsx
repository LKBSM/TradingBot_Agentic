import type { Metadata } from 'next';
import { AppWorkspace } from '@/components/app/AppWorkspace';

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
  return <AppWorkspace />;
}

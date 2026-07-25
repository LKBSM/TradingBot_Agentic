import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { AppWorkspace } from '@/components/app/AppWorkspace';
import { SubscriptionGate } from '@/components/access/SubscriptionGate';
import { resolveComboFromQuery } from '@/lib/conditions/app-link';

/**
 * Default combo on load (XAU/USD M15). Defined inline rather than imported from
 * the client-only mockReadings module so this server component doesn't pull a
 * 'use client' module into the server bundle.
 */
const DEFAULT_COMBO = { instrument: 'XAUUSD', timeframe: 'M15' } as const;

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('app.meta.title'),
    description: t('app.meta.description'),
  };
}

/**
 * Application view — the working surface (distinct from the marketing landing
 * at `/`). Three columns on desktop: instruments · reading · chat. The heavy
 * lifting lives in the client AppWorkspace; this server page only sets the
 * route + metadata.
 */
export default async function AppPage({
  searchParams,
}: {
  searchParams: Promise<{ instrument?: string; timeframe?: string; focus?: string }>;
}) {
  const sp = await searchParams;
  // Honour an optional Scanner deep-link (?instrument=&timeframe=); fall back to
  // the default XAU/USD M15 so a direct visit still shows a populated surface.
  const initialCombo = resolveComboFromQuery(sp.instrument, sp.timeframe) ?? DEFAULT_COMBO;
  // Optional zone focus (?focus=) from the Zones page "Analyser" action. Passed
  // through as-is; the workspace validates it against the on-screen zone-id lock
  // before focusing, so an unknown/stale id is a graceful no-op.
  const initialFocusZoneId = sp.focus ?? null;
  // Gate the working surface: when enforced, a visitor is redirected to login; a
  // free account is let in (partial perimeter — XAU/USD M15) and locked combos
  // degrade to a clean upsell per request. Open while the gate is OFF (testing).
  return (
    <SubscriptionGate>
      <AppWorkspace initialCombo={initialCombo} initialFocusZoneId={initialFocusZoneId} />
    </SubscriptionGate>
  );
}

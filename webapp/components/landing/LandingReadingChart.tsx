'use client';

import * as React from 'react';
import dynamic from 'next/dynamic';
import { FIXTURE_XAU_M15 } from '@/lib/market-reading/fixtures';
import { getMockCandles } from '@/lib/mockReadings';

/**
 * Aperçu produit RÉEL pour la landing : le MÊME composant `ReadingChart` que
 * la vue /app (chandeliers TradingView Lightweight Charts + overlays SMC lus
 * directement dans la structure — BOS/CHOCH, Order Blocks, Fair Value Gaps,
 * poches de liquidité). Pas un mock SVG : c'est littéralement le graphique du
 * produit, alimenté par la fixture XAU/USD M15 et des bougies déterministes
 * (`getMockCandles`, générées localement — aucune donnée de marché externe).
 *
 * `ReadingChart` est chargé en dynamique (ssr:false) pour que lightweight-charts
 * reste hors du bundle initial de la landing et ne se charge qu'au montage de
 * cet aperçu.
 */
const ReadingChart = dynamic(
  () =>
    import('@/components/app/ReadingChart').then((m) => ({
      default: m.ReadingChart,
    })),
  {
    ssr: false,
    loading: () => (
      <div
        className="h-[260px] w-full animate-pulse rounded-lg bg-muted/40 sm:h-[300px]"
        aria-hidden
      />
    ),
  },
);

export function LandingReadingChart({ className }: { className?: string }) {
  // Deterministic mock candles framing the fixture's SMC levels (never real
  // quotes). Memoised so a re-render never regenerates the walk.
  const candles = React.useMemo(
    () => getMockCandles('XAUUSD', 'M15') ?? [],
    [],
  );

  return (
    <div className={className}>
      <ReadingChart
        candles={candles}
        structure={FIXTURE_XAU_M15.structure}
        instrument="XAUUSD"
        timeframe="M15"
        className="rounded-lg border border-border/50 bg-background/40"
        /* Fixed height on the marketing hero (the /app default is fluid). Matches
           the dynamic-import loading placeholder above so there's no layout jump. */
        heightClassName="h-[260px] sm:h-[300px]"
      />
    </div>
  );
}

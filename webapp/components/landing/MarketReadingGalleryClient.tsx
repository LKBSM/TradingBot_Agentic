'use client';

import * as React from 'react';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import { useChat } from '@/components/chat/ChatProvider';
import { cn } from '@/lib/utils';
import type { LandingSample } from '@/lib/market-reading/landing-samples';

interface MarketReadingGalleryClientProps {
  samples: ReadonlyArray<LandingSample>;
  /** Optional node rendered after the last card (typically a "Coming soon" tile). */
  renderAfter?: React.ReactNode;
  gridClassName?: string;
}

/**
 * Grille de MarketReadingCard natives (Chantier 5.C — remplace l'ancien
 * InsightGalleryClient basé sur InsightSignalV2). Accepte un slot après-cards
 * (ComingSoonCard) et câble le CTA "Demander à Sentinel" sur useChat().
 */
export function MarketReadingGalleryClient({
  samples,
  renderAfter,
  gridClassName,
}: MarketReadingGalleryClientProps) {
  const { openFor } = useChat();
  return (
    <div className={cn('grid gap-6', gridClassName)}>
      {samples.map(({ id, reading }) => (
        <MarketReadingCard
          key={id}
          reading={reading}
          onAskChatbot={() =>
            openFor({
              id,
              instrument: reading.header.instrument,
              timeframe: reading.header.timeframe,
            })
          }
        />
      ))}
      {renderAfter}
    </div>
  );
}

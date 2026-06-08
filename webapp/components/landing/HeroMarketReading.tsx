'use client';

import * as React from 'react';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import { useChat } from '@/components/chat/ChatProvider';
import type { LandingSample } from '@/lib/market-reading/landing-samples';

/**
 * Wrapper de MarketReadingCard pour le hero — applique l'animation de
 * composition progressive (fade-in-up délayée) à toute la card, et câble
 * le CTA "Demander à Sentinel" sur l'ouverture du panneau de chat avec la
 * lecture du hero injectée en contexte.
 *
 * La card garde ses sections collapsées par défaut : pas d'historique ouvert
 * en hero (lock 2 — aucun chiffre de performance au-dessus du pli). Consomme
 * désormais une MarketReading native (Chantier 5.C, plus d'InsightSignalV2).
 */
export function HeroMarketReading({ sample }: { sample: LandingSample }) {
  const { openFor } = useChat();
  const { reading } = sample;
  return (
    <div className="hero-stagger" style={{ animationDelay: '200ms' }}>
      <MarketReadingCard
        reading={reading}
        onAskChatbot={() =>
          openFor({
            id: sample.id,
            instrument: reading.header.instrument,
            timeframe: reading.header.timeframe,
          })
        }
      />
    </div>
  );
}

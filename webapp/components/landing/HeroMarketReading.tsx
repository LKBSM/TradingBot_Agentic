'use client';

import * as React from 'react';
import { MarketReadingCard } from '@/components/insight/MarketReadingCard';
import { useChat } from '@/components/chat/ChatProvider';
import type { InsightSignalV2 } from '@/types/insight';

/**
 * Wrapper de MarketReadingCard pour le hero — applique l'animation de
 * composition progressive (fade-in-up délayée) à toute la card, et câble
 * le CTA "Demander à Sentinel" sur l'ouverture du panneau de chat avec le
 * signal du hero injecté en contexte.
 *
 * La card elle-même garde toutes ses sections collapsées par défaut :
 * pas d'historique ouvert en hero (lock 2 — aucun chiffre de performance
 * au-dessus du pli, ils vivent en Section 5).
 */
export function HeroMarketReading({ signal }: { signal: InsightSignalV2 }) {
  const { openFor } = useChat();
  return (
    <div className="hero-stagger" style={{ animationDelay: '200ms' }}>
      <MarketReadingCard
        signal={signal}
        onAskChatbot={() => openFor(signal)}
      />
    </div>
  );
}

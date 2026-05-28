'use client';

import * as React from 'react';
import { MarketReadingCard } from '@/components/insight/MarketReadingCard';
import { useChat } from '@/components/chat/ChatProvider';
import { cn } from '@/lib/utils';
import type { InsightSignalV2 } from '@/types/insight';

interface InsightGalleryClientProps {
  signals: ReadonlyArray<InsightSignalV2>;
  /** Optional node rendered after the last card (typically a "Coming soon" tile). */
  renderAfter?: React.ReactNode;
  gridClassName?: string;
}

/**
 * Variante de InsightGallery qui accepte un slot après-cards (pour
 * afficher un ComingSoonCard placeholder à droite). Toujours câblée à
 * useChat() pour le CTA "Demander à Sentinel".
 */
export function InsightGalleryClient({
  signals,
  renderAfter,
  gridClassName,
}: InsightGalleryClientProps) {
  const { openFor } = useChat();
  return (
    <div className={cn('grid gap-6', gridClassName)}>
      {signals.map((signal) => (
        <MarketReadingCard
          key={signal.id}
          signal={signal}
          onAskChatbot={() => openFor(signal)}
        />
      ))}
      {renderAfter}
    </div>
  );
}

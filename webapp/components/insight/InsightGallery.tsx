'use client';

import { MarketReadingCard } from './MarketReadingCard';
import { useChat } from '@/components/chat/ChatProvider';
import type { InsightSignalV2 } from '@/types/insight';

interface InsightGalleryProps {
  signals: readonly InsightSignalV2[];
  /** Section keys to expand by default per signal index (F3 demo aid). */
  defaultOpenByIndex?: ReadonlyMap<
    number,
    ReadonlyArray<'structure' | 'regime' | 'volatility' | 'events' | 'history'>
  >;
}

/**
 * Client wrapper that bridges the static signal list to the ChatProvider —
 * each card's "ask Sentinel" CTA opens the panel pre-loaded with that
 * signal's context.
 */
export function InsightGallery({
  signals,
  defaultOpenByIndex,
}: InsightGalleryProps) {
  const { openFor } = useChat();

  return (
    <section className="space-y-6">
      {signals.map((signal, idx) => (
        <MarketReadingCard
          key={signal.id}
          signal={signal}
          onAskChatbot={() => openFor(signal)}
          defaultOpenSections={defaultOpenByIndex?.get(idx)}
        />
      ))}
    </section>
  );
}

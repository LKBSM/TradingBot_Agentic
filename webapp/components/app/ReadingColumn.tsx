'use client';

import { Loader2 } from 'lucide-react';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import { EmptyReadingState, ReadingErrorState } from './ReadingPlaceholders';
import { ReadingSkeleton } from './ReadingSkeleton';
import type { Combo } from '@/lib/market-reading/store';
import type { MarketReading } from '@/types/market-reading';

interface ReadingColumnProps {
  active: Combo | null;
  reading: MarketReading | null;
  isLoading: boolean;
  isRefreshing: boolean;
  error: Error | null;
  onRetry: () => void;
}

/**
 * Centre column — renders the detailed reading of the active combo, plus the
 * loading / refreshing / error / empty states. The "Demander à Sentinel" CTA
 * focuses the always-present chat sidebar instead of opening a modal.
 */
export function ReadingColumn({
  active,
  reading,
  isLoading,
  isRefreshing,
  error,
  onRetry,
}: ReadingColumnProps) {
  function focusChat() {
    const input = document.querySelector<HTMLTextAreaElement>(
      'textarea[aria-label="Question libre pour Sentinel"]',
    );
    input?.focus();
    input?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  let body: React.ReactNode;
  if (!active) {
    body = <EmptyReadingState />;
  } else if (error) {
    body = <ReadingErrorState error={error} onRetry={onRetry} />;
  } else if (isLoading && !reading) {
    body = <ReadingSkeleton />;
  } else if (reading) {
    body = (
      <MarketReadingCard
        reading={reading}
        onAskChatbot={focusChat}
        className="w-full border-border/60 shadow-sm"
      />
    );
  } else {
    body = <ReadingSkeleton />;
  }

  return (
    <section aria-label="Lecture de marché" className="min-w-0 space-y-3">
      {isRefreshing && reading && (
        <div
          className="flex items-center gap-2 text-xs text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
          Actualisation…
        </div>
      )}
      {body}
    </section>
  );
}

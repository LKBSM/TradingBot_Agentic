'use client';

import { Compass, LineChart, RefreshCw, ServerCrash } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  MarketReadingNotAvailableError,
  MarketReadingValidationError,
} from '@/lib/market-reading/api-client';

/**
 * Shown in the centre column when no combo is selected yet. The wording uses
 * the established "lecture" vocabulary (cf. landing copy) — descriptive, no
 * directive token.
 */
export function EmptyReadingState() {
  const t = useTranslations('app');
  return (
    <Card className="w-full border-dashed border-border/60 bg-transparent shadow-none">
      <CardContent className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
        <Compass className="h-8 w-8 text-muted-foreground/60" aria-hidden />
        <p className="max-w-xs text-sm text-muted-foreground">
          {t('placeholders.emptyReading')}
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Friendly error state — distinguishes "service not available on this
 * environment" (503) from a transient/internal failure, and offers a retry.
 * Never surfaces server internals. Presented as a clear "Données indisponibles"
 * card (jamais un écran vide).
 */
export function ReadingErrorState({
  error,
  onRetry,
}: {
  error: Error;
  onRetry: () => void;
}) {
  const t = useTranslations('app');
  const isUnavailable = error instanceof MarketReadingNotAvailableError;
  const isValidation = error instanceof MarketReadingValidationError;

  const message = isUnavailable
    ? t('placeholders.errorUnavailable')
    : isValidation
      ? t('placeholders.errorValidation')
      : t('placeholders.errorGeneric');

  return (
    <Card className="w-full border-border/60 shadow-sm">
      <CardContent className="flex flex-col items-center justify-center gap-4 px-6 py-14 text-center">
        <ServerCrash className="h-8 w-8 text-sentinel-warn" aria-hidden />
        <div className="space-y-1">
          <p className="text-sm font-semibold text-foreground">
            {t('placeholders.dataUnavailableTitle')}
          </p>
          <p className="max-w-sm text-sm text-muted-foreground">{message}</p>
        </div>
        {!isValidation && (
          <Button type="button" variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="h-4 w-4" aria-hidden />
            {t('placeholders.retry')}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Chart-specific "data unavailable" placeholder — shown inside the reading panel
 * when the candle feed for the active combo isn't connected. The textual reading
 * around it stays usable (graceful degradation, never a blank box).
 */
export function ChartUnavailable({
  onRetry,
}: {
  onRetry?: () => void;
}) {
  const t = useTranslations('app');
  return (
    <div
      role="status"
      className="flex h-full min-h-[260px] w-full flex-col items-center justify-center gap-3 rounded-md border border-dashed border-border/60 bg-muted/30 px-6 py-10 text-center"
    >
      <LineChart className="h-7 w-7 text-muted-foreground/60" aria-hidden />
      <div className="space-y-1">
        <p className="text-sm font-semibold text-foreground">
          {t('placeholders.chartUnavailableTitle')}
        </p>
        <p className="max-w-xs text-xs text-muted-foreground">
          {t('placeholders.chartUnavailableBody')}
        </p>
      </div>
      {onRetry && (
        <Button type="button" variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw className="h-4 w-4" aria-hidden />
          {t('placeholders.retry')}
        </Button>
      )}
    </div>
  );
}

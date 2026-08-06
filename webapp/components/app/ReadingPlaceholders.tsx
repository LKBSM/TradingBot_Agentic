'use client';

import * as React from 'react';
import { Compass, LineChart, Loader2, RefreshCw, ServerCrash } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  MarketReadingError,
  MarketReadingNoDataError,
  MarketReadingNotAvailableError,
  MarketReadingValidationError,
  type CandlesErrorReason,
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
  const isValidation = error instanceof MarketReadingValidationError;

  // Distinct honest copy per failure mode (PERF-1): the user must be able to tell
  // "trop lent", "serveur injoignable", "aucune donnée" and "combo non supporté"
  // apart — never one vague "données indisponibles" for all of them.
  const messageKey = error instanceof MarketReadingValidationError
    ? 'placeholders.errorValidation'
    : error instanceof MarketReadingNotAvailableError
      ? 'placeholders.errorUnavailable'
      : error instanceof MarketReadingNoDataError
        ? 'placeholders.errorNoData'
        : error instanceof MarketReadingError && error.reason === 'timeout'
          ? 'placeholders.errorTimeout'
          : error instanceof MarketReadingError && error.reason === 'network'
            ? 'placeholders.errorUnreachable'
            : 'placeholders.errorGeneric';
  const message = t(messageKey);

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
 * Progressive-loading honesty (PERF-1): after a threshold of continuous loading,
 * tell the user the fetch is still in flight so a slow load never reads as a
 * frozen screen — "ça charge" must be distinguishable from "c'est cassé". Renders
 * nothing until the threshold; the caller shows the skeleton in the meantime.
 */
export function SlowLoadHint({ afterMs = 6000 }: { afterMs?: number }) {
  const t = useTranslations('app');
  const [show, setShow] = React.useState(false);
  React.useEffect(() => {
    const id = setTimeout(() => setShow(true), afterMs);
    return () => clearTimeout(id);
  }, [afterMs]);
  if (!show) return null;
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center justify-center gap-2 pt-3 text-xs text-muted-foreground"
    >
      <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
      {t('placeholders.slowLoading')}
    </div>
  );
}

/**
 * Chart-specific "data unavailable" placeholder — shown inside the reading panel
 * when the candle feed for the active combo isn't connected. The textual reading
 * around it stays usable (graceful degradation, never a blank box).
 */
export function ChartUnavailable({
  onRetry,
  reason,
}: {
  onRetry?: () => void;
  /**
   * PERF-2 — why the candle feed is unavailable, so the placeholder says WHICH
   * (délai dépassé / serveur injoignable / aucune donnée) instead of one vague
   * line. Reuses the reading-error copy (already translated in every locale);
   * a missing/unknown reason falls back to the generic chart body.
   */
  reason?: CandlesErrorReason;
}) {
  const t = useTranslations('app');
  const bodyKey =
    reason === 'timeout'
      ? 'placeholders.errorTimeout'
      : reason === 'network'
        ? 'placeholders.errorUnreachable'
        : reason === 'nodata'
          ? 'placeholders.errorNoData'
          : 'placeholders.chartUnavailableBody';
  return (
    <div
      role="status"
      // Match the chart / loader height (UI-13) so swapping to this placeholder
      // doesn't shift the content below it (CLS).
      className="flex h-[280px] w-full flex-col items-center justify-center gap-3 rounded-md border border-dashed border-border/60 bg-muted/30 px-6 py-10 text-center sm:h-[340px]"
    >
      <LineChart className="h-7 w-7 text-muted-foreground/60" aria-hidden />
      <div className="space-y-1">
        <p className="text-sm font-semibold text-foreground">
          {t('placeholders.chartUnavailableTitle')}
        </p>
        <p className="max-w-xs text-xs text-muted-foreground">{t(bodyKey)}</p>
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

import { Compass, RefreshCw, ServerCrash } from 'lucide-react';
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
  return (
    <Card className="w-full border-dashed border-border/60 bg-transparent shadow-none">
      <CardContent className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
        <Compass className="h-8 w-8 text-muted-foreground/60" aria-hidden />
        <p className="max-w-xs text-sm text-muted-foreground">
          Sélectionnez une combinaison à gauche pour afficher sa lecture de
          marché.
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Friendly error state — distinguishes "service not available on this
 * environment" (503) from a transient/internal failure, and offers a retry.
 * Never surfaces server internals.
 */
export function ReadingErrorState({
  error,
  onRetry,
}: {
  error: Error;
  onRetry: () => void;
}) {
  const isUnavailable = error instanceof MarketReadingNotAvailableError;
  const isValidation = error instanceof MarketReadingValidationError;

  const message = isUnavailable
    ? "Le service de lecture n'est pas disponible sur cet environnement pour le moment."
    : isValidation
      ? 'Cette combinaison instrument / timeframe n’est pas prise en charge.'
      : 'La lecture n’a pas pu être récupérée. Réessayez dans un instant.';

  return (
    <Card className="w-full border-border/60 shadow-sm">
      <CardContent className="flex flex-col items-center justify-center gap-4 px-6 py-14 text-center">
        <ServerCrash className="h-8 w-8 text-sentinel-warn" aria-hidden />
        <p className="max-w-sm text-sm text-foreground">{message}</p>
        {!isValidation && (
          <Button type="button" variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="h-4 w-4" aria-hidden />
            Réessayer
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

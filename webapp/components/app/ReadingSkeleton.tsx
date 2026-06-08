import { Card, CardContent } from '@/components/ui/card';

/**
 * Loading placeholder for the centre column during the initial fetch of a
 * combo's reading. Mirrors the MarketReadingCard layout (hero + phase badges +
 * a couple of section rows) so the swap to real content doesn't jump.
 */
export function ReadingSkeleton() {
  return (
    <Card
      className="w-full border-border/60 shadow-sm"
      aria-hidden
      data-testid="reading-skeleton"
    >
      <CardContent className="space-y-6 p-5 sm:p-7">
        <div className="flex items-center justify-between">
          <div className="h-6 w-40 animate-pulse rounded bg-muted" />
          <div className="h-6 w-20 animate-pulse rounded bg-muted" />
        </div>
        <div className="h-4 w-56 animate-pulse rounded bg-muted" />
        <div className="flex gap-2">
          <div className="h-6 w-28 animate-pulse rounded-md bg-muted" />
          <div className="h-6 w-24 animate-pulse rounded-md bg-muted" />
          <div className="h-6 w-24 animate-pulse rounded-md bg-muted" />
        </div>
        <div className="space-y-3 border-t border-border/60 pt-4">
          <div className="h-10 w-full animate-pulse rounded bg-muted" />
          <div className="h-10 w-full animate-pulse rounded bg-muted" />
          <div className="h-10 w-full animate-pulse rounded bg-muted" />
        </div>
      </CardContent>
    </Card>
  );
}

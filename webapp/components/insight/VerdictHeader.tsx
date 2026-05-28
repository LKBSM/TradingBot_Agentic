import { Badge } from '@/components/ui/badge';
import {
  directionBadgeVariant,
  formatInstrument,
  formatTimeframe,
  formatVerdict,
} from '@/lib/insight-formatters';
import type { InsightSignalV2 } from '@/types/insight';

/**
 * Single-line market verdict + instrument/timeframe sub-line. Always visible
 * in the hero — must be readable by a non-trader in under 5 seconds.
 */
export function VerdictHeader({ signal }: { signal: InsightSignalV2 }) {
  const variant = directionBadgeVariant(signal.direction);
  const verdictLabel =
    variant === 'bull'
      ? 'Haussier'
      : variant === 'bear'
        ? 'Baissier'
        : 'Neutre';

  return (
    <header className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={variant} className="text-xs">
          {verdictLabel}
        </Badge>
        <span className="text-xs font-medium text-muted-foreground">
          {formatInstrument(signal)} · {formatTimeframe(signal)}
        </span>
      </div>
      <h2 className="text-balance text-xl font-semibold leading-tight tracking-tight sm:text-2xl">
        {formatVerdict(signal)}
      </h2>
    </header>
  );
}

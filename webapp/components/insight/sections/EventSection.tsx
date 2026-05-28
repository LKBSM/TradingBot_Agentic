import { AlertTriangle } from 'lucide-react';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import {
  formatNextEventCountdown,
  formatSession,
} from '@/lib/insight-formatters';
import type { InsightSignalV2 } from '@/types/insight';

export function EventSection({ signal }: { signal: InsightSignalV2 }) {
  const e = signal.event_readout;
  const countdown = formatNextEventCountdown(e.next_event_in_minutes);
  const minutesUntil = e.next_event_in_minutes ?? Number.POSITIVE_INFINITY;
  const imminent = minutesUntil <= 240; // 4h threshold (cf. DG-122 pépite)
  const sentimentTone =
    e.sentiment_score >= 0.2
      ? 'text-sentinel-bull'
      : e.sentiment_score <= -0.2
        ? 'text-sentinel-bear'
        : 'text-foreground';

  return (
    <AccordionItem value="events">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>📅</span>
          <span>Contexte événementiel</span>
          {imminent && (
            <Badge variant="warn" className="text-[10px]">
              <AlertTriangle className="mr-1 h-3 w-3" aria-hidden />
              Imminent
            </Badge>
          )}
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Row
            label="Prochain événement à fort impact"
            value={
              e.next_event_label && countdown
                ? `${e.next_event_label} · ${countdown}`
                : 'aucun événement majeur identifié'
            }
            valueClassName={imminent ? 'text-sentinel-warn' : undefined}
            className="sm:col-span-2"
          />
          <Row
            label="Fenêtre de blackout news"
            value={
              e.news_blackout_active
                ? 'active — analyse mise en pause'
                : 'inactive'
            }
            valueClassName={
              e.news_blackout_active ? 'text-sentinel-warn' : undefined
            }
          />
          <Row label="Session de marché" value={formatSession(e.session)} />
          <Row
            label="Sentiment news (24 h)"
            value={`${formatSentiment(e.sentiment_score)} · confiance ${Math.round(e.sentiment_confidence * 100)} %`}
            valueClassName={sentimentTone}
            className="sm:col-span-2"
          />
        </dl>
      </AccordionContent>
    </AccordionItem>
  );
}

function formatSentiment(score: number): string {
  if (score >= 0.4) return 'très positif';
  if (score >= 0.15) return 'positif';
  if (score >= -0.15) return 'neutre';
  if (score >= -0.4) return 'négatif';
  return 'très négatif';
}

function Row({
  label,
  value,
  className,
  valueClassName,
}: {
  label: string;
  value: string;
  className?: string;
  valueClassName?: string;
}) {
  return (
    <div className={className}>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className={cn('mt-1 text-sm font-medium', valueClassName ?? 'text-foreground')}>
        {value}
      </dd>
    </div>
  );
}

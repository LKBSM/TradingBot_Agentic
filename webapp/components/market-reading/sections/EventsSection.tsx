import { AlertTriangle, Calendar } from 'lucide-react';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import {
  formatCurrency,
  formatImpact,
  formatMinutesAgo,
  formatSurprise,
  formatTimeToEvent,
  formatTriggerType,
  type Tone,
} from '@/lib/market-reading/formatters';
import type { MarketReadingEvents } from '@/types/market-reading';

const TONE_TO_VARIANT: Record<Tone, 'bull' | 'bear' | 'neutral' | 'warn'> = {
  bull: 'bull',
  bear: 'bear',
  neutral: 'neutral',
  warn: 'warn',
};

// "Imminent" threshold for an upcoming event (minutes).
const IMMINENT_MIN = 60;

/**
 * Section "Événements" — upcoming news, news just published, and recent
 * technical triggers. Descriptive context only; the `potential_effect_description`
 * comes verbatim from the backend (niveau 1.5 enforced server-side).
 */
export function EventsSection({ events }: { events: MarketReadingEvents }) {
  const { news_upcoming, news_just_published, technical_triggers_recent } =
    events;

  const hasImminent = news_upcoming.some(
    (n) => n.time_to_event_min <= IMMINENT_MIN,
  );
  const isEmpty =
    news_upcoming.length === 0 &&
    news_just_published.length === 0 &&
    technical_triggers_recent.length === 0;

  return (
    <AccordionItem value="events">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <Calendar className="h-4 w-4 text-muted-foreground" aria-hidden />
          <span>Contexte événementiel</span>
          {hasImminent && (
            <Badge variant="warn" className="text-[10px]">
              <AlertTriangle className="mr-1 h-3 w-3" aria-hidden />
              Imminent
            </Badge>
          )}
        </span>
      </AccordionTrigger>
      <AccordionContent>
        {isEmpty ? (
          <p className="text-sm text-muted-foreground">
            Aucun événement notable à proximité de cette lecture.
          </p>
        ) : (
          <div className="space-y-5">
            {news_upcoming.length > 0 && (
              <Group title="À venir">
                {news_upcoming.map((n, i) => {
                  const impact = formatImpact(n.impact);
                  const imminent = n.time_to_event_min <= IMMINENT_MIN;
                  return (
                    <EventRow
                      key={`up-${i}`}
                      title={`${n.event} · ${formatCurrency(n.currency)}`}
                      timing={formatTimeToEvent(n.time_to_event_min)}
                      timingClassName={imminent ? 'text-sentinel-warn' : undefined}
                      impact={impact}
                      description={n.potential_effect_description}
                    />
                  );
                })}
              </Group>
            )}

            {news_just_published.length > 0 && (
              <Group title="Publié récemment">
                {news_just_published.map((n, i) => {
                  const impact = formatImpact(n.impact);
                  const surprise = n.surprise_direction
                    ? formatSurprise(n.surprise_direction)
                    : null;
                  return (
                    <EventRow
                      key={`pub-${i}`}
                      title={`${n.event} · ${formatCurrency(n.currency)}`}
                      timing={surprise}
                      impact={impact}
                      description={n.potential_effect_description}
                    />
                  );
                })}
              </Group>
            )}

            {technical_triggers_recent.length > 0 && (
              <Group title="Déclencheurs techniques récents">
                <ul className="space-y-1.5">
                  {technical_triggers_recent.map((t, i) => (
                    <li
                      key={`trig-${i}`}
                      className="flex flex-wrap items-baseline justify-between gap-x-3 text-sm"
                    >
                      <span className="font-medium text-foreground">
                        {formatTriggerType(t.type)}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {formatMinutesAgo(t.minutes_ago)}
                      </span>
                    </li>
                  ))}
                </ul>
              </Group>
            )}
          </div>
        )}
      </AccordionContent>
    </AccordionItem>
  );
}

function Group({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
        {title}
      </p>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function EventRow({
  title,
  timing,
  timingClassName,
  impact,
  description,
}: {
  title: string;
  timing: string | null;
  timingClassName?: string;
  impact: { label: string; tone: Tone };
  description: string;
}) {
  return (
    <div className="rounded-md border border-border/60 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium text-foreground">{title}</span>
        <div className="flex items-center gap-2">
          {timing && (
            <span
              className={cn('text-xs text-muted-foreground', timingClassName)}
            >
              {timing}
            </span>
          )}
          <Badge variant={TONE_TO_VARIANT[impact.tone]} className="text-[10px]">
            {impact.label}
          </Badge>
        </div>
      </div>
      <p className="mt-1.5 text-xs text-muted-foreground">{description}</p>
    </div>
  );
}

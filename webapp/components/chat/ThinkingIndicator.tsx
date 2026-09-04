import { useTranslations } from 'next-intl';
import { AgentAvatar } from './AgentAvatar';
import { useChat } from './ChatProvider';
import { formatTimeframe } from '@/lib/market-reading/formatters';
import { useInstrumentLabel } from '@/lib/market-reading/useInstrumentLabel';

/**
 * "M.I.A Agent is working" state — three pulsing dots inside an assistant-style
 * bubble, with the brand avatar. MIA-1: when the backend reports a REAL step in
 * progress (a market read it actually started), the dots are captioned with an
 * honest, visible label ("Lecture de XAU/USD M15…") built from the tool's own
 * arguments — a fixed template, never model prose. Otherwise the caption stays
 * screen-reader-only so the dots carry the generic "thinking" meaning.
 */
export function ThinkingIndicator() {
  const t = useTranslations('chat');
  const instrumentLabel = useInstrumentLabel();
  const { activity } = useChat();

  let caption: string | null = null;
  if (activity?.kind === 'tool') {
    const market = activity.instrument ? instrumentLabel(activity.instrument) : '';
    const tf = activity.timeframe ? formatTimeframe(activity.timeframe) : '';
    if (activity.tool === 'get_ob_diagnostic') {
      caption = t('activityDiagnostic');
    } else if (market && tf) {
      caption = t('activityReadingMarket', { market, timeframe: tf });
    } else {
      caption = t('activityReading');
    }
  }

  return (
    <div className="flex w-full gap-2.5" role="status" aria-live="polite" data-testid="chat-thinking">
      <AgentAvatar size="sm" className="mt-0.5" />
      <div className="flex min-w-0 flex-col gap-1">
        <div className="flex w-fit items-center gap-1.5 rounded-2xl rounded-tl-sm border border-border bg-muted/60 px-4 py-3.5">
          <span className="chat-think-dot h-1.5 w-1.5 rounded-full bg-muted-foreground" />
          <span
            className="chat-think-dot h-1.5 w-1.5 rounded-full bg-muted-foreground"
            style={{ animationDelay: '0.2s' }}
          />
          <span
            className="chat-think-dot h-1.5 w-1.5 rounded-full bg-muted-foreground"
            style={{ animationDelay: '0.4s' }}
          />
        </div>
        {caption ? (
          <span className="truncate pl-1 text-[11px] text-muted-foreground" data-testid="chat-activity">
            {caption}
          </span>
        ) : (
          <span className="sr-only">{t('thinking')}</span>
        )}
      </div>
    </div>
  );
}

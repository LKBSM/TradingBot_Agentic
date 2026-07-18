import { useTranslations } from 'next-intl';
import { AgentAvatar } from './AgentAvatar';

/**
 * Elegant "M.I.A Agent is thinking" state — three pulsing dots inside an
 * assistant-style bubble, with the brand avatar. Replaces the raw spinner.
 * The visible text is screen-reader only so the dots carry the meaning
 * visually without a noisy label.
 */
export function ThinkingIndicator() {
  const t = useTranslations('chat');
  return (
    <div className="flex w-full gap-2.5" role="status" aria-live="polite">
      <AgentAvatar size="sm" className="mt-0.5" />
      <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm border border-border bg-muted/60 px-4 py-3.5">
        <span className="chat-think-dot h-1.5 w-1.5 rounded-full bg-muted-foreground" />
        <span
          className="chat-think-dot h-1.5 w-1.5 rounded-full bg-muted-foreground"
          style={{ animationDelay: '0.2s' }}
        />
        <span
          className="chat-think-dot h-1.5 w-1.5 rounded-full bg-muted-foreground"
          style={{ animationDelay: '0.4s' }}
        />
        <span className="sr-only">{t('thinking')}</span>
      </div>
    </div>
  );
}

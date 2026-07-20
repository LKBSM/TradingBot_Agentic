'use client';

import { Loader2, SendHorizonal } from 'lucide-react';
import { useTranslations } from 'next-intl';
import * as React from 'react';
import { cn } from '@/lib/utils';
import { useChat } from './ChatProvider';

interface ChatInputProps {
  className?: string;
}

const MAX_CHARS = 2000;

/**
 * Free-text input for the chatbot. Submits to the backend via
 * `useChat().askFreeForm()` (POST /api/chatbot/message, synchronous JSON).
 * Disabled while a request is in flight to prevent overlapping calls.
 * Auto-grows the textarea up to a reasonable max height. Enter sends,
 * Shift+Enter inserts a newline.
 */
export function ChatInput({ className }: ChatInputProps) {
  const t = useTranslations('chat');
  const { askFreeForm, isLoading, activeSignal, apiAvailable } = useChat();
  const [value, setValue] = React.useState('');
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Auto-resize on every value change.
  React.useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = '0px';
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [value]);

  const offline = apiAvailable === false;
  const canSubmit =
    !isLoading && value.trim().length > 0 && activeSignal !== null;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    const question = value.trim();
    setValue('');
    try {
      await askFreeForm(question);
    } catch (err) {
      // askFreeForm already pushes an error turn — nothing else to do here.
      console.error('chat submit failed', err);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSubmit(e as unknown as React.FormEvent);
    }
  }

  const placeholder = offline
    ? t('inputPlaceholderOffline')
    : t('inputPlaceholder');

  return (
    <div className={className}>
      <form
        onSubmit={handleSubmit}
        className={cn(
          'flex items-end gap-2 rounded-2xl border border-border bg-background/80 p-2 pl-3.5 transition-shadow',
          'focus-within:border-[hsl(35_92%_55%/0.5)] focus-within:shadow-[0_0_0_3px_hsl(35_92%_55%/0.10)]',
        )}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value.slice(0, MAX_CHARS))}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          maxLength={MAX_CHARS}
          disabled={offline}
          aria-label={t('inputAria')}
          /* text-base (16px) on touch prevents iOS from zooming the page on
             focus; shrink to text-sm only on xl desktop. */
          className="flex-1 resize-none bg-transparent py-1.5 text-base leading-relaxed text-foreground placeholder:text-muted-foreground/70 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60 xl:text-sm"
        />
        <button
          type="submit"
          disabled={!canSubmit}
          aria-label={isLoading ? t('sendLoadingAria') : t('sendAria')}
          className={cn(
            // 44px tap target on touch; 36px only on xl desktop (mouse).
            'flex h-11 w-11 shrink-0 items-center justify-center rounded-xl transition-colors xl:h-9 xl:w-9',
            canSubmit
              ? 'bg-[hsl(var(--sentinel-warn))] text-[hsl(222_47%_11%)] hover:brightness-110'
              : 'cursor-not-allowed bg-muted text-muted-foreground',
          )}
        >
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <SendHorizonal className="h-4 w-4" aria-hidden />
          )}
        </button>
      </form>
      {!offline && (
        <p className="mt-2 text-center text-[11px] italic text-muted-foreground/75">
          {t('inputHint')}
        </p>
      )}
    </div>
  );
}

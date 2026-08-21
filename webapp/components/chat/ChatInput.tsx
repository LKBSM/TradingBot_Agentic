'use client';

import { Loader2, SendHorizonal } from 'lucide-react';
import { useLocale, useTranslations } from 'next-intl';
import * as React from 'react';
import { cn } from '@/lib/utils';
import { MicButton } from '@/components/dictation/MicButton';
import { useVoiceInput } from '@/lib/scanner-chat/use-voice-input';
import { useDictationCopy } from '@/lib/scanner-chat/use-dictation-copy';
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
  const locale = useLocale();
  const { askFreeForm, isLoading, activeSignal, apiAvailable } = useChat();
  const [value, setValue] = React.useState('');
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Voice dictation (browser Web Speech API) — same shared hook as the scanner,
  // /zones and /actualites chats. It only appends to the field below; what is
  // sent is exactly the visible text.
  const voice = useVoiceInput({
    locale,
    value,
    onValueChange: (next) => setValue(next.slice(0, MAX_CHARS)),
    maxLength: MAX_CHARS,
  });
  const dictationCopy = useDictationCopy();

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
          data-testid="chat-input"
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
        {voice.supported && !offline && (
          <MicButton
            listening={voice.listening}
            denied={voice.denied}
            onToggle={voice.toggle}
            startLabel={dictationCopy.startLabel}
            stopLabel={dictationCopy.stopLabel}
            className="h-11 w-11 rounded-xl xl:h-9 xl:w-9"
          />
        )}
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
      {/* Dictation feedback — the live transcript and errors are never hidden,
          and the keyboard stays fully usable underneath. */}
      {voice.supported && !offline && voice.listening && (
        <p data-testid="dictation-listening" className="mt-2 text-[11px] text-[hsl(var(--sentinel-warn))]">
          {dictationCopy.listeningLabel}
          {voice.interim ? <span className="text-muted-foreground"> — “{voice.interim}”</span> : null}
        </p>
      )}
      {voice.supported && !offline && voice.error && (
        <p data-testid="dictation-error" role="alert" className="mt-2 text-[11px] text-amber-600 dark:text-amber-500">
          {dictationCopy.errorText(voice.error)}
        </p>
      )}
      {!offline && (
        <p className="mt-2 text-center text-[11px] italic text-muted-foreground/75">
          {t('inputHint')}
        </p>
      )}
      {/* Browser-transcription notice — kept honest on every surface. */}
      {voice.supported && !offline && (
        <p data-testid="transcription-note" className="mt-1 text-center text-[11px] leading-relaxed text-muted-foreground/70">
          {dictationCopy.privacy}
        </p>
      )}
    </div>
  );
}

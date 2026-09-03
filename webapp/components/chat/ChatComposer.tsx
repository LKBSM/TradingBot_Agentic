'use client';

import { Loader2, SendHorizonal } from 'lucide-react';
import { useLocale } from 'next-intl';
import * as React from 'react';
import { cn } from '@/lib/utils';
import { MicButton } from '@/components/dictation/MicButton';
import { useVoiceInput } from '@/lib/scanner-chat/use-voice-input';
import { useDictationCopy } from '@/lib/scanner-chat/use-dictation-copy';

const DEFAULT_MAX_CHARS = 2000;

export interface ChatComposerProps {
  /** Called with the trimmed text when the user submits (Enter or send button). */
  onSubmit: (text: string) => void;
  placeholder: string;
  /** aria-label for the textarea. */
  ariaLabel: string;
  /** Keyboard hint shown as a tooltip (title) — no always-on footer line. */
  hint?: string;
  sendAria: string;
  sendLoadingAria?: string;
  /** Copy for the browser-transcription privacy note. */
  privacyNote: string;
  isLoading?: boolean;
  /** Disables the field entirely (e.g. offline). */
  disabled?: boolean;
  /**
   * Extra gate on top of « non-empty text »: e.g. /app requires an active combo.
   * Defaults to true (always submittable when there is text).
   */
  submitEnabled?: boolean;
  /** Show the transcription privacy note under the field. Defaults to true. */
  showTranscriptionNote?: boolean;
  maxChars?: number;
  className?: string;
}

/**
 * Shared, controlled chat input bar (CLN-1 §2). Extracted from the network
 * `ChatInput` so /app (Sentinel backend chat) and /zones (local, zero-credit
 * M.I.A) render the SAME composer — one format, one dictation UX, one privacy
 * note — instead of two that drift apart. The composer owns its own draft state
 * and only calls `onSubmit(text)`; each surface wires that to its own engine
 * (network `askFreeForm` vs local deterministic answer). DOM + data-testids are
 * kept identical to the previous ChatInput so existing tests keep passing.
 *
 * Auto-grows the textarea up to a max height. Enter sends, Shift+Enter inserts a
 * newline. What is sent is exactly the visible text — voice dictation only
 * appends to the field (browser Web Speech API).
 */
export function ChatComposer({
  onSubmit,
  placeholder,
  ariaLabel,
  hint,
  sendAria,
  sendLoadingAria,
  privacyNote,
  isLoading = false,
  disabled = false,
  submitEnabled = true,
  showTranscriptionNote = true,
  maxChars = DEFAULT_MAX_CHARS,
  className,
}: ChatComposerProps) {
  const locale = useLocale();
  const [value, setValue] = React.useState('');
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  const voice = useVoiceInput({
    locale,
    value,
    onValueChange: (next) => setValue(next.slice(0, maxChars)),
    maxLength: maxChars,
  });
  const dictationCopy = useDictationCopy();

  // Auto-resize on every value change.
  React.useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = '0px';
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [value]);

  const canSubmit = !isLoading && !disabled && submitEnabled && value.trim().length > 0;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    const question = value.trim();
    setValue('');
    onSubmit(question);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  }

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
          onChange={(e) => setValue(e.target.value.slice(0, maxChars))}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          maxLength={maxChars}
          disabled={disabled}
          aria-label={ariaLabel}
          title={hint}
          /* text-base (16px) on touch prevents iOS from zooming the page on
             focus; shrink to text-sm only on xl desktop. */
          className="flex-1 resize-none bg-transparent py-1.5 text-base leading-relaxed text-foreground placeholder:text-muted-foreground/70 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60 xl:text-sm"
        />
        {voice.supported && !disabled && (
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
          aria-label={isLoading && sendLoadingAria ? sendLoadingAria : sendAria}
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
      {voice.supported && !disabled && voice.listening && (
        <p data-testid="dictation-listening" className="mt-2 text-[11px] text-[hsl(var(--sentinel-warn))]">
          {dictationCopy.listeningLabel}
          {voice.interim ? <span className="text-muted-foreground"> — “{voice.interim}”</span> : null}
        </p>
      )}
      {voice.supported && !disabled && voice.error && (
        <p data-testid="dictation-error" role="alert" className="mt-2 text-[11px] text-amber-600 dark:text-amber-500">
          {dictationCopy.errorText(voice.error)}
        </p>
      )}
      {/* Browser-transcription notice — kept honest on every surface, secondary
          and on a single discreet line (CLN-1 §2). */}
      {voice.supported && !disabled && showTranscriptionNote && (
        <p data-testid="transcription-note" className="mt-1 text-center text-[11px] leading-relaxed text-muted-foreground/70">
          {privacyNote}
        </p>
      )}
    </div>
  );
}

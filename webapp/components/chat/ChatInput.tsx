'use client';

import { Loader2, SendHorizonal } from 'lucide-react';
import * as React from 'react';
import { Button } from '@/components/ui/button';
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
 * Auto-grows the textarea up to a reasonable max height.
 */
export function ChatInput({ className }: ChatInputProps) {
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

  const placeholder =
    apiAvailable === false
      ? 'Le LLM en direct n\'est pas configuré — utilise les questions suggérées ci-dessus.'
      : 'Pose une question libre à Sentinel… (Entrée pour envoyer · Maj+Entrée pour saut de ligne)';

  return (
    <form
      onSubmit={handleSubmit}
      className={cn(
        'flex items-end gap-2 rounded-xl border border-border bg-background p-2',
        className,
      )}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) =>
          setValue(e.target.value.slice(0, MAX_CHARS))
        }
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        rows={1}
        maxLength={MAX_CHARS}
        disabled={apiAvailable === false}
        aria-label="Question libre pour Sentinel"
        className="flex-1 resize-none bg-transparent px-2 py-1 text-sm leading-relaxed text-foreground placeholder:text-muted-foreground/70 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
      />
      <Button
        type="submit"
        size="icon"
        disabled={!canSubmit}
        aria-label={isLoading ? 'Réponse en cours…' : 'Envoyer la question'}
      >
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
        ) : (
          <SendHorizonal className="h-4 w-4" aria-hidden />
        )}
      </Button>
    </form>
  );
}

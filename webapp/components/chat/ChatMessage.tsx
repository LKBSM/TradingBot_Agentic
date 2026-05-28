import { Bot, User } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  text: string;
}

/**
 * Single chat bubble. User on the right (secondary), assistant on the left
 * (muted card style). Long-form assistant replies wrap and preserve newlines.
 */
export function ChatMessage({ role, text }: ChatMessageProps) {
  const isUser = role === 'user';
  return (
    <div
      className={cn(
        'flex w-full gap-2',
        isUser ? 'justify-end' : 'justify-start',
      )}
      role={role === 'assistant' ? 'status' : undefined}
    >
      {!isUser && (
        <div
          aria-hidden
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground"
        >
          <Bot className="h-4 w-4" />
        </div>
      )}
      <div
        className={cn(
          'max-w-[85%] whitespace-pre-wrap rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed',
          isUser
            ? 'rounded-tr-sm bg-primary text-primary-foreground'
            : 'rounded-tl-sm bg-muted text-foreground',
        )}
      >
        {text}
      </div>
      {isUser && (
        <div
          aria-hidden
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground"
        >
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  );
}

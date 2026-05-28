import { Bot, ShieldAlert, User } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  text: string;
  /**
   * DG-112 compliance marker — when present, a small badge below the bubble
   * tells the reader the response was either intercepted by the pre-LLM
   * refusal gate or sanitised by the post-stream forbidden-token filter.
   * The badge text is intentionally short and pedagogical — not
   * accusatory toward the user.
   */
  compliance?: {
    kind: 'refusal' | 'forbidden_token';
    category?: string;
    token?: string;
  };
}

function complianceLabel(c: NonNullable<ChatMessageProps['compliance']>): string {
  if (c.kind === 'refusal') {
    return 'Réponse pédagogique de refus (cadre compliance).';
  }
  return 'Réponse filtrée — un terme prescriptif a été remplacé par un message sûr.';
}

/**
 * Single chat bubble. User on the right (secondary), assistant on the left
 * (muted card style). Long-form assistant replies wrap and preserve newlines.
 */
export function ChatMessage({ role, text, compliance }: ChatMessageProps) {
  const isUser = role === 'user';
  return (
    <div
      className={cn(
        'flex w-full flex-col gap-1',
        isUser ? 'items-end' : 'items-start',
      )}
      role={role === 'assistant' ? 'status' : undefined}
    >
      <div
        className={cn(
          'flex w-full gap-2',
          isUser ? 'justify-end' : 'justify-start',
        )}
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
      {compliance && !isUser ? (
        <div
          className="ml-9 inline-flex items-center gap-1.5 rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-900"
          data-testid="compliance-badge"
          data-compliance-kind={compliance.kind}
        >
          <ShieldAlert className="h-3 w-3" aria-hidden />
          <span>{complianceLabel(compliance)}</span>
        </div>
      ) : null}
    </div>
  );
}

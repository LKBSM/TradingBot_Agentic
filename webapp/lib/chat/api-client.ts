import type { InsightSignalV2 } from '@/types/insight';

export interface AskOptions {
  signal: InsightSignalV2;
  question: string;
  history?: ReadonlyArray<{ role: 'user' | 'assistant'; content: string }>;
  signal_abort?: AbortSignal;
  onDelta(textChunk: string): void;
}

export type RefusalCategory =
  | 'prescriptive'
  | 'guarantee'
  | 'jailbreak'
  | 'personal_advice'
  | 'signal_request';

export interface RefusalInfo {
  /** Category from the DG-112 classifier (pre-LLM gate). */
  category: RefusalCategory;
  /** Language used for the safe-fallback text. */
  language: 'fr' | 'en';
  /** Source regex (telemetry / debug only — do NOT show to the user). */
  pattern_source?: string;
}

export interface ForbiddenTokenInfo {
  /** The offending token / phrase that slipped through the LLM. */
  token: string;
  /** Language the post-process filter detected. */
  language: 'fr' | 'en';
  /** Safe replacement text the UI should show in place of the streamed body. */
  safe_fallback: string;
}

export interface AskResult {
  /** Fully concatenated assistant reply. */
  text: string;
  /** Token usage from Anthropic, if reported. */
  usage?: unknown;
  model?: string;
  /** Set when the pre-LLM gate refused the request (DG-112). */
  refusal?: RefusalInfo;
  /** Set when the post-stream forbidden-token filter fired. */
  forbidden_token?: ForbiddenTokenInfo;
  /** Mirror of the server-side flag — true on either gate. */
  compliance_filtered?: boolean;
}

export class ChatApiUnavailableError extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
    this.name = 'ChatApiUnavailableError';
  }
}

/**
 * Stream a chatbot answer from /api/chat. Pushes incremental deltas through
 * `onDelta` and resolves with the final concatenated reply.
 *
 * Server returns a Server-Sent Events stream with events:
 *   - `delta` { text }       → token chunk
 *   - `done`  { usage, model, stop_reason } → end of stream
 *   - `error` { message }    → terminal error during generation
 *
 * 503 → ChatApiUnavailableError (caller should fall back to scripted mode).
 */
export async function askSentinel(opts: AskOptions): Promise<AskResult> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      signal: opts.signal,
      question: opts.question,
      history: opts.history,
    }),
    signal: opts.signal_abort,
  });

  if (res.status === 503) {
    const body = (await res.json().catch(() => ({}))) as {
      error?: { code?: string; message?: string };
    };
    throw new ChatApiUnavailableError(
      body.error?.code ?? 'no_api_key',
      body.error?.message ?? 'Chat API non disponible',
    );
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as {
      error?: { code?: string; message?: string };
    };
    throw new Error(body.error?.message ?? `HTTP ${res.status}`);
  }
  if (!res.body) throw new Error('Réponse sans corps depuis /api/chat');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalText = '';
  let usage: unknown = undefined;
  let model: string | undefined = undefined;
  let terminalError: string | null = null;
  let refusal: RefusalInfo | undefined;
  let forbidden: ForbiddenTokenInfo | undefined;
  let complianceFiltered = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by a blank line.
    const messages = buffer.split('\n\n');
    buffer = messages.pop() ?? '';

    for (const raw of messages) {
      const lines = raw.split('\n');
      let event = 'message';
      let data = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) event = line.slice(7);
        else if (line.startsWith('data: ')) data += line.slice(6);
      }
      if (!data) continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue;
      }
      if (event === 'delta' && isDelta(parsed)) {
        finalText += parsed.text;
        opts.onDelta(parsed.text);
      } else if (event === 'done' && isDone(parsed)) {
        usage = parsed.usage;
        model = parsed.model;
        if ((parsed as { compliance_filtered?: boolean }).compliance_filtered) {
          complianceFiltered = true;
        }
      } else if (event === 'refusal' && isRefusal(parsed)) {
        refusal = parsed;
        complianceFiltered = true;
      } else if (event === 'forbidden_token' && isForbiddenToken(parsed)) {
        forbidden = parsed;
        complianceFiltered = true;
      } else if (event === 'error' && isError(parsed)) {
        terminalError = parsed.message;
      }
    }
  }

  if (terminalError) throw new Error(terminalError);
  return {
    text: finalText,
    usage,
    model,
    refusal,
    forbidden_token: forbidden,
    compliance_filtered: complianceFiltered || undefined,
  };
}

function isDelta(x: unknown): x is { text: string } {
  return typeof x === 'object' && x !== null && typeof (x as { text?: unknown }).text === 'string';
}

function isDone(x: unknown): x is { usage?: unknown; model?: string } {
  return typeof x === 'object' && x !== null;
}

function isError(x: unknown): x is { message: string } {
  return (
    typeof x === 'object' && x !== null && typeof (x as { message?: unknown }).message === 'string'
  );
}

const REFUSAL_CATEGORIES = new Set<RefusalCategory>([
  'prescriptive',
  'guarantee',
  'jailbreak',
  'personal_advice',
  'signal_request',
]);

function isRefusal(x: unknown): x is RefusalInfo {
  if (typeof x !== 'object' || x === null) return false;
  const o = x as Partial<RefusalInfo>;
  return (
    typeof o.category === 'string' &&
    REFUSAL_CATEGORIES.has(o.category as RefusalCategory) &&
    (o.language === 'fr' || o.language === 'en')
  );
}

function isForbiddenToken(x: unknown): x is ForbiddenTokenInfo {
  if (typeof x !== 'object' || x === null) return false;
  const o = x as Partial<ForbiddenTokenInfo>;
  return (
    typeof o.token === 'string' &&
    typeof o.safe_fallback === 'string' &&
    (o.language === 'fr' || o.language === 'en')
  );
}

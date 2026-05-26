import type { InsightSignalV2 } from '@/types/insight';

export interface AskOptions {
  signal: InsightSignalV2;
  question: string;
  history?: ReadonlyArray<{ role: 'user' | 'assistant'; content: string }>;
  signal_abort?: AbortSignal;
  onDelta(textChunk: string): void;
}

export interface AskResult {
  /** Fully concatenated assistant reply. */
  text: string;
  /** Token usage from Anthropic, if reported. */
  usage?: unknown;
  model?: string;
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
      } else if (event === 'error' && isError(parsed)) {
        terminalError = parsed.message;
      }
    }
  }

  if (terminalError) throw new Error(terminalError);
  return { text: finalText, usage, model };
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

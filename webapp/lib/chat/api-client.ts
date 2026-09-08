import type { ChatSignalContext } from '@/lib/chat/types';
import { accessErrorFromResponse } from '@/lib/access/errors';

/** The api-client only needs the instrument/timeframe combo for the preamble. */
type SignalContext = Pick<ChatSignalContext, 'instrument' | 'timeframe'>;

/**
 * Conversation turn exchanged with the backend chatbot. Mirrors the Pydantic
 * `ConversationMessage` of the Chantier 4 endpoint
 * (`src/api/routes/chatbot.py`): role + non-empty content (≤ 2000 chars).
 */
export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AskOptions {
  /** Free-text user question. */
  question: string;
  /**
   * Lecture de marché actuellement ouverte dans le panneau. Optional: when set,
   * a stable, recognisable context preamble is prepended to `user_message` so
   * the backend's Haiku layer can resolve `get_market_reading(instrument,
   * timeframe)` for the right combo. See docs Chantier 5.A — Tension T1.
   */
  signal?: SignalContext | null;
  /**
   * MIA-3 — extra ORIENTATION line appended to the preamble (after the combo
   * line), e.g. a selected zone or an open publication. Pure orientation: it
   * tells M.I.A what the user is looking at, it does NOT widen what M.I.A may
   * affirm — every fact still comes from a tool result. Never fabricated: the
   * caller only ever passes an id the engine actually emitted (a clicked zone).
   */
  focus?: string | null;
  /** Prior turns (already trimmed by the caller; backend caps at 20). */
  history?: ReadonlyArray<ConversationMessage>;
  /** Abort handle wired to the request lifecycle. */
  signal_abort?: AbortSignal;
}

export interface AskResult {
  /** Assistant reply text — always present (LLM answer, refusal, or template). */
  text: string;
  /**
   * `null` on a normal answer; otherwise the reason a defence layer kicked in
   * (adversarial category, `llm_error`, `output_contaminated_*`,
   * `max_tool_turns_exceeded`). `content` already carries the user-facing
   * template in that case — the frontend only uses this to show a discreet
   * "redirected" badge.
   */
  blockedReason: string | null;
  /** Tool calls the backend executed for this turn ({name, input}). */
  toolCallsMade: ReadonlyArray<Record<string, unknown>>;
  /**
   * Display-only chart view actions the backend validated (Couche 4 whitelist).
   * RAW here — the caller re-validates them via `coerceViewActions` against the
   * zones currently on screen before applying to the chart render.
   */
  viewActions: ReadonlyArray<Record<string, unknown>>;
}

/**
 * Backend not bootstrapped (HTTP 503 — `CHATBOT_ENABLED=false`). The caller
 * should flip `apiAvailable=false` and fall back to scripted suggestions.
 */
export class ChatApiUnavailableError extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
    this.name = 'ChatApiUnavailableError';
  }
}

/**
 * Any non-503 failure (422 validation, 500 internal, network, malformed body).
 * `status` is 0 for transport/parse errors with no HTTP response.
 */
export class ChatApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ChatApiError';
  }
}

/** Backend route, proxied to FastAPI via the `/api/*` rewrite (next.config.js). */
const CHATBOT_ENDPOINT = '/api/chatbot/message';
/** MIA-1 — SSE variant: honest activity/tool status, then the validated answer. */
const CHATBOT_STREAM_ENDPOINT = '/api/chatbot/stream';

/**
 * An orchestration event streamed from the backend during a turn (MIA-1). The
 * `activity` and `tool` frames are fixed, structured status — they NEVER carry
 * model prose; only `answer` does (after Couche-3 validation). The caller uses
 * `activity`/`tool` to show an honest "working / reading market" signal.
 */
export type ChatStreamEvent =
  | { type: 'activity' }
  | { type: 'tool'; tool: string; instrument?: string; timeframe?: string }
  | { type: 'answer'; result: AskResult };

interface ChatbotMessageResponse {
  content: string;
  blocked_reason: string | null;
  tool_calls_made: Array<Record<string, unknown>>;
  view_actions?: Array<Record<string, unknown>>;
}

/**
 * Chantier 5.A — Tension T1: stable, recognisable context preamble.
 *
 * Convention frontend ↔ backend: the panel is opened *for* a specific signal,
 * but the backend endpoint is signal-agnostic ({user_message,
 * conversation_history}). We surface the active instrument/timeframe to the
 * backend by prepending this single line, so Haiku can call
 * `get_market_reading(instrument, timeframe)` on the right combo.
 *
 * Format is fixed (brackets + `Lecture en cours :` + space-separated codes)
 * precisely so it can be detected/stripped later if the architecture evolves.
 */
function withSignalContext(
  question: string,
  signal?: SignalContext | null,
  focus?: string | null,
): string {
  const lines: string[] = [];
  if (signal) lines.push(`[Lecture en cours : ${signal.instrument} ${signal.timeframe}]`);
  if (focus) lines.push(focus);
  if (lines.length === 0) return question;
  return `${lines.join('\n')}\n${question}`;
}

/**
 * Ask the Chantier 4 backend chatbot. Sends a synchronous JSON request to
 * `POST /api/chatbot/message` and resolves with the full reply — no streaming.
 *
 * Every answer flows through the 3 niveau-1.5 defence layers server-side
 * (adversarial input filter → Haiku tool use → output forbidden-tokens filter),
 * so the webapp can never bypass them.
 *
 * @throws {ChatApiUnavailableError} on HTTP 503 (chatbot not bootstrapped).
 * @throws {ChatApiError} on 422 / 500 / other HTTP errors and network/parse failures.
 */
export async function askSentinel(opts: AskOptions): Promise<AskResult> {
  const body = {
    user_message: withSignalContext(opts.question, opts.signal, opts.focus),
    conversation_history: (opts.history ?? []).map((h) => ({
      role: h.role,
      content: h.content,
    })),
  };

  let res: Response;
  try {
    res = await fetch(CHATBOT_ENDPOINT, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: opts.signal_abort,
    });
  } catch (err) {
    // Network failure / aborted / DNS — no HTTP response at all.
    const message = err instanceof Error ? err.message : 'Erreur réseau';
    throw new ChatApiError(0, `Connexion au service impossible : ${message}`);
  }

  if (res.status === 503) {
    const detail = await readErrorDetail(res);
    throw new ChatApiUnavailableError(
      'chatbot_unavailable',
      detail ?? "Le service de chat n'est pas disponible sur cet environnement.",
    );
  }

  if (res.status === 422) {
    throw new ChatApiError(
      422,
      'La question a été refusée par la validation du service (format ou longueur). Reformule plus court.',
    );
  }

  // 401/402 — the freemium gate (free tier has a daily quota). Clean upsell.
  const accessErr = await accessErrorFromResponse(res);
  if (accessErr) throw accessErr;

  if (!res.ok) {
    // 500 and anything else: never surface server internals.
    throw new ChatApiError(
      res.status,
      'Le service a rencontré une erreur interne. Réessaie dans un instant.',
    );
  }

  let parsed: ChatbotMessageResponse;
  try {
    parsed = (await res.json()) as ChatbotMessageResponse;
  } catch {
    throw new ChatApiError(res.status, 'Réponse du service illisible.');
  }

  if (typeof parsed?.content !== 'string') {
    throw new ChatApiError(res.status, 'Réponse du service malformée.');
  }

  return {
    text: parsed.content,
    blockedReason: parsed.blocked_reason ?? null,
    toolCallsMade: Array.isArray(parsed.tool_calls_made) ? parsed.tool_calls_made : [],
    viewActions: Array.isArray(parsed.view_actions) ? parsed.view_actions : [],
  };
}

/**
 * MIA-1 — Ask the backend over Server-Sent Events. Emits honest orchestration
 * events via `onEvent` (activity < 200 ms, then a tool status when a real market
 * read starts), and resolves with the SAME `AskResult` shape as {@link askSentinel}
 * once the terminal, Couche-3-validated `answer` frame arrives.
 *
 * The model's text is NEVER streamed token-by-token: the answer arrives whole,
 * already validated — so nothing shown can ever need to be retracted.
 *
 * Error posture is identical to {@link askSentinel} (503 → unavailable, 422 →
 * validation, 401/402 → access, other → generic), so callers can share fallbacks.
 *
 * @throws {ChatApiUnavailableError} on HTTP 503.
 * @throws {ChatApiError} on 422 / 500 / other HTTP errors and network/parse failures.
 */
export async function askSentinelStream(
  opts: AskOptions,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<AskResult> {
  const body = {
    user_message: withSignalContext(opts.question, opts.signal, opts.focus),
    conversation_history: (opts.history ?? []).map((h) => ({
      role: h.role,
      content: h.content,
    })),
  };

  let res: Response;
  try {
    res = await fetch(CHATBOT_STREAM_ENDPOINT, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
      body: JSON.stringify(body),
      signal: opts.signal_abort,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Erreur réseau';
    throw new ChatApiError(0, `Connexion au service impossible : ${message}`);
  }

  if (res.status === 503) {
    const detail = await readErrorDetail(res);
    throw new ChatApiUnavailableError(
      'chatbot_unavailable',
      detail ?? "Le service de chat n'est pas disponible sur cet environnement.",
    );
  }
  if (res.status === 422) {
    throw new ChatApiError(
      422,
      'La question a été refusée par la validation du service (format ou longueur). Reformule plus court.',
    );
  }
  const accessErr = await accessErrorFromResponse(res);
  if (accessErr) throw accessErr;
  if (!res.ok) {
    throw new ChatApiError(
      res.status,
      'Le service a rencontré une erreur interne. Réessaie dans un instant.',
    );
  }

  let answer: AskResult | null = null;
  const handleFrame = (raw: string) => {
    const payload = parseSseData(raw);
    if (!payload) return;
    if (payload.event === 'activity') {
      onEvent({ type: 'activity' });
    } else if (payload.event === 'tool') {
      onEvent({
        type: 'tool',
        tool: String(payload.tool ?? ''),
        instrument: typeof payload.instrument === 'string' ? payload.instrument : undefined,
        timeframe: typeof payload.timeframe === 'string' ? payload.timeframe : undefined,
      });
    } else if (payload.event === 'answer') {
      if (typeof payload.content !== 'string') {
        throw new ChatApiError(res.status, 'Réponse du service malformée.');
      }
      answer = {
        text: payload.content,
        blockedReason: (payload.blocked_reason as string | null) ?? null,
        toolCallsMade: Array.isArray(payload.tool_calls_made) ? payload.tool_calls_made : [],
        viewActions: Array.isArray(payload.view_actions) ? payload.view_actions : [],
      };
      onEvent({ type: 'answer', result: answer });
    }
  };

  const reader = res.body?.getReader?.();
  if (reader) {
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep: number;
      // Frames are separated by a blank line ("\n\n").
      while ((sep = buf.indexOf('\n\n')) >= 0) {
        handleFrame(buf.slice(0, sep));
        buf = buf.slice(sep + 2);
      }
    }
    if (buf.trim()) handleFrame(buf);
  } else {
    // Environments without a streaming body (e.g. jsdom): the whole SSE body is
    // buffered — parse it frame by frame. Same honest events, no live typing.
    const text = await res.text();
    for (const frame of text.split('\n\n')) handleFrame(frame);
  }

  if (!answer) {
    throw new ChatApiError(res.status, 'Réponse du service illisible.');
  }
  return answer;
}

/** Extract and JSON-parse the `data:` line of an SSE frame; null if none/blank. */
function parseSseData(frame: string): Record<string, unknown> | null {
  for (const line of frame.split('\n')) {
    const trimmed = line.trimEnd();
    if (trimmed.startsWith('data:')) {
      const json = trimmed.slice(5).trim();
      if (!json) return null;
      try {
        return JSON.parse(json) as Record<string, unknown>;
      } catch {
        return null;
      }
    }
  }
  return null;
}

/** Best-effort extraction of a FastAPI `{detail}` body; never throws. */
async function readErrorDetail(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return typeof body?.detail === 'string' ? body.detail : null;
  } catch {
    return null;
  }
}

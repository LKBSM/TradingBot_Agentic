import type { ChatSignalContext } from '@/lib/chat/types';

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
function withSignalContext(question: string, signal?: SignalContext | null): string {
  if (!signal) return question;
  return `[Lecture en cours : ${signal.instrument} ${signal.timeframe}]\n${question}`;
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
    user_message: withSignalContext(opts.question, opts.signal),
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

/** Best-effort extraction of a FastAPI `{detail}` body; never throws. */
async function readErrorDetail(res: Response): Promise<string | null> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return typeof body?.detail === 'string' ? body.detail : null;
  } catch {
    return null;
  }
}

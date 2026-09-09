/**
 * MIA-4S — client for the landing's simulated M.I.A agent.
 *
 * Talks to `POST /api/demo/chat/stream`, the PUBLIC showcase endpoint. It is a
 * different door from the product chat (`lib/chat/api-client.ts`): no account,
 * no session, metered quotas — so it gets its own small client rather than a
 * flag on the product one.
 *
 * As in the product, the model's prose is NEVER streamed token by token: the
 * `activity` frame says work started, and the answer arrives whole, already
 * through the output filter. Nothing shown can need to be retracted.
 *
 * Every failure mode is a first-class result, because the landing must degrade
 * to its scripted starters rather than show a broken tab:
 *   · `DemoUnavailableError` — the agent is not configured (503) or unreachable
 *   · `DemoQuotaError`       — a session / IP / daily quota was reached (429)
 */

const DEMO_STREAM_ENDPOINT = '/api/demo/chat/stream';

export interface DemoTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface DemoAnswer {
  text: string;
  /** Non-null when a defence layer replaced the answer (adversarial, filtered…). */
  blockedReason: string | null;
  /** Display-only actions, already validated AND narrowed to what the demo renders. */
  viewActions: Array<Record<string, unknown>>;
  messagesLeft: number;
}

export class DemoUnavailableError extends Error {}

export class DemoQuotaError extends Error {
  constructor(
    readonly reason: string,
    message: string,
  ) {
    super(message);
  }
}

function parseFrame(raw: string): Record<string, unknown> | null {
  const line = raw.split('\n').find((l) => l.startsWith('data:'));
  if (!line) return null;
  try {
    return JSON.parse(line.slice(5).trim()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

/**
 * Ask the simulated agent one question.
 *
 * @param onActivity called once, as soon as the turn is cleared to hit the model
 *   — the honest "it started" signal the thinking indicator is built on.
 */
export async function askDemoMia(opts: {
  question: string;
  history?: DemoTurn[];
  locale?: string;
  signal?: AbortSignal;
  onActivity?: () => void;
}): Promise<DemoAnswer> {
  let res: Response;
  try {
    res = await fetch(DEMO_STREAM_ENDPOINT, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
      body: JSON.stringify({
        user_message: opts.question,
        conversation_history: (opts.history ?? []).map((h) => ({
          role: h.role,
          content: h.content,
        })),
        locale: opts.locale,
      }),
      ...(opts.signal ? { signal: opts.signal } : null),
    });
  } catch (err) {
    // Offline, blocked, no backend in front of a static host — all the same to
    // the visitor: the tab falls back to what it can show without a network.
    throw new DemoUnavailableError(err instanceof Error ? err.message : 'network');
  }

  if (res.status === 429) {
    let reason = 'session_limit';
    let message = '';
    try {
      const body = (await res.json()) as { detail?: { reason?: string; message?: string } };
      reason = body.detail?.reason ?? reason;
      message = body.detail?.message ?? '';
    } catch {
      /* an unparseable quota body still means "quota" */
    }
    throw new DemoQuotaError(reason, message);
  }
  if (!res.ok) throw new DemoUnavailableError(`HTTP ${res.status}`);

  let answer: DemoAnswer | null = null;
  const handle = (raw: string) => {
    const payload = parseFrame(raw);
    if (!payload) return;
    if (payload.event === 'activity') {
      opts.onActivity?.();
    } else if (payload.event === 'answer' && typeof payload.content === 'string') {
      answer = {
        text: payload.content,
        blockedReason: (payload.blocked_reason as string | null) ?? null,
        viewActions: Array.isArray(payload.view_actions)
          ? (payload.view_actions as Array<Record<string, unknown>>)
          : [],
        messagesLeft:
          typeof payload.messages_left === 'number' ? payload.messages_left : 0,
      };
    }
  };

  const reader = res.body?.getReader?.();
  if (!reader) {
    // No streaming body (test doubles, some proxies): the frames still arrive,
    // just all at once — split them the same way rather than reading only the first.
    for (const frame of (await res.text()).split('\n\n')) {
      if (frame.trim()) handle(frame);
    }
  } else {
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buf.indexOf('\n\n')) >= 0) {
        handle(buf.slice(0, sep));
        buf = buf.slice(sep + 2);
      }
    }
    if (buf.trim()) handle(buf);
  }

  if (!answer) throw new DemoUnavailableError('no answer frame');
  return answer;
}

/** Layer keys the demo chart knows, in the product's own vocabulary. */
export type DemoLayerKey = 'ob' | 'fvg' | 'liq' | 'str';

const LAYER_ALIASES: Record<string, DemoLayerKey> = {
  ob: 'ob',
  fvg: 'fvg',
  liquidity: 'liq',
  liq: 'liq',
  bsl: 'liq',
  ssl: 'liq',
  breaks: 'str',
};

/**
 * Translate the validated view actions into the demo chart's four layers.
 *
 * Only what the frozen demo can actually show is applied; anything else leaves
 * the layers untouched (the server already dropped what it cannot render, this
 * is the second half of the same honesty rule).
 */
export function applyDemoViewActions(
  current: Record<DemoLayerKey, boolean>,
  actions: Array<Record<string, unknown>>,
): Record<DemoLayerKey, boolean> {
  let next = { ...current };
  const all = (value: boolean) => ({ ob: value, fvg: value, liq: value, str: value });

  for (const action of actions) {
    const name = action.action;
    const params = (action.params ?? {}) as Record<string, unknown>;

    if (name === 'reset_view' || name === 'show_zones') {
      const layers = (Array.isArray(action.demo_layers) ? action.demo_layers : [])
        .map((l) => LAYER_ALIASES[String(l)])
        .filter((l): l is DemoLayerKey => Boolean(l));
      if (name === 'show_zones' && layers.length) {
        next = { ...next };
        for (const layer of layers) next[layer] = true;
      } else {
        next = all(true); // reset_view, or show_zones with no target = restore all
      }
    } else if (name === 'set_layer_visibility') {
      const visible = params.visible !== false;
      const raw = Array.isArray(params.layers)
        ? (params.layers as unknown[])
        : [params.layer];
      if (raw.includes('all')) {
        next = all(visible);
      } else {
        for (const key of raw) {
          const layer = LAYER_ALIASES[String(key)];
          if (layer) next = { ...next, [layer]: visible };
        }
      }
    } else if (name === 'hide_zones' || name === 'isolate_zones') {
      // The server resolved the real zone ids back to the layers they belong to
      // (`demo_layers`), because the demo chart has layers, not addressable zones.
      const layers = (Array.isArray(action.demo_layers) ? action.demo_layers : [])
        .map((l) => LAYER_ALIASES[String(l)])
        .filter((l): l is DemoLayerKey => Boolean(l));
      if (layers.length) {
        next = name === 'hide_zones' ? { ...next } : all(false);
        for (const layer of layers) next[layer] = name !== 'hide_zones';
      }
    }
  }
  return next;
}

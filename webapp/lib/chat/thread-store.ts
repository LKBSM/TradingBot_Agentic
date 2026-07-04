/**
 * Chat thread persistence — CLIENT-ONLY (localStorage), Loi 25 boundary.
 *
 * The chat conversations are scoped per (instrument, timeframe) combo and kept
 * on the user's device ONLY. No server endpoint, no DB write, no cookie: the
 * company never holds this data, which keeps it outside Loi 25 obligations
 * until the privacy-policy terminal lands. Follows the established
 * localStorage pattern (cf. `lib/market-reading/pins.ts`): SSR-safe guards,
 * defensive sanitisation of whatever storage returns, try/catch on quota so
 * persistence degrades to in-memory only.
 */

import {
  SUPPORTED_INSTRUMENTS,
  SUPPORTED_TIMEFRAMES,
} from '@/lib/market-reading/perimeter';

export const STORAGE_KEY = 'mia.chatThreads.v1';

/** Hard caps so the stored payload stays small and old threads self-purge. */
export const MAX_TURNS_PER_THREAD = 40;
export const MAX_THREADS = 12;
export const MAX_SERIALIZED_CHARS = 200_000;

const VALID_SOURCES = new Set(['llm', 'scripted', 'fallback', 'error']);
const VALID_INSTRUMENTS = new Set<string>(SUPPORTED_INSTRUMENTS);
const VALID_TIMEFRAMES = new Set<string>(SUPPORTED_TIMEFRAMES);

export interface StoredTurn {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  /** Was this answer produced by the live LLM API or the scripted fallback? */
  source?: 'llm' | 'scripted' | 'fallback' | 'error';
  /** Niveau-1.5 defence redirect reason (badge display), null on a normal answer. */
  blockedReason?: string | null;
  /** Display-only mirror: this turn carried validated chart view actions. */
  viewUpdated?: boolean;
}

export interface StoredThread {
  /** Thread key — always `app:{instrument}:{timeframe}` for persisted threads. */
  id: string;
  instrument: string;
  timeframe: string;
  /** Epoch ms of the last appended turn — drives recency sort + purge order. */
  updatedAt: number;
  turns: StoredTurn[];
}

/** Only combo-scoped /app threads are persisted (landing signal chats are not). */
function isPersistableThread(t: StoredThread): boolean {
  return (
    t.id === `app:${t.instrument}:${t.timeframe}` &&
    VALID_INSTRUMENTS.has(t.instrument) &&
    VALID_TIMEFRAMES.has(t.timeframe)
  );
}

/**
 * Cap a thread's turn list. When trimming actually drops turns, also drop any
 * leading assistant turns so the kept window never starts mid-exchange.
 */
function trimTurns(turns: StoredTurn[]): StoredTurn[] {
  if (turns.length <= MAX_TURNS_PER_THREAD) return turns;
  const out = turns.slice(-MAX_TURNS_PER_THREAD);
  const firstUser = out.findIndex((t) => t.role === 'user');
  return firstUser > 0 ? out.slice(firstUser) : out;
}

function sanitizeTurn(raw: unknown, fallbackId: string): StoredTurn | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const t = raw as Record<string, unknown>;
  if (t.role !== 'user' && t.role !== 'assistant') return null;
  if (typeof t.text !== 'string') return null;
  const out: StoredTurn = {
    id: typeof t.id === 'string' && t.id.length > 0 ? t.id : fallbackId,
    role: t.role,
    text: t.text,
  };
  if (typeof t.source === 'string' && VALID_SOURCES.has(t.source)) {
    out.source = t.source as StoredTurn['source'];
  }
  if (typeof t.blockedReason === 'string') out.blockedReason = t.blockedReason;
  if (t.viewUpdated === true) out.viewUpdated = true;
  return out;
}

function sanitizeThread(raw: unknown): StoredThread | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const t = raw as Record<string, unknown>;
  if (
    typeof t.id !== 'string' ||
    typeof t.instrument !== 'string' ||
    typeof t.timeframe !== 'string' ||
    !Array.isArray(t.turns)
  ) {
    return null;
  }
  const turns: StoredTurn[] = [];
  t.turns.forEach((rawTurn, i) => {
    const turn = sanitizeTurn(rawTurn, `restored-${i}`);
    if (turn) turns.push(turn);
  });
  if (turns.length === 0) return null;
  const thread: StoredThread = {
    id: t.id,
    instrument: t.instrument,
    timeframe: t.timeframe,
    updatedAt:
      typeof t.updatedAt === 'number' && Number.isFinite(t.updatedAt)
        ? t.updatedAt
        : 0,
    turns: trimTurns(turns),
  };
  // Drop threads outside the supported perimeter (stale storage from an older
  // app version) — never trust what localStorage returns.
  return isPersistableThread(thread) ? thread : null;
}

/** Read + sanitise the persisted threads. Returns [] on SSR / corrupt storage. */
export function readThreads(): StoredThread[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const seen = new Set<string>();
    const out: StoredThread[] = [];
    for (const item of parsed) {
      const thread = sanitizeThread(item);
      if (thread && !seen.has(thread.id)) {
        seen.add(thread.id);
        out.push(thread);
      }
    }
    return out;
  } catch {
    return [];
  }
}

/**
 * Persist threads, applying every cap: non-app / empty threads are skipped,
 * turns are trimmed per thread, only the MAX_THREADS most recent threads are
 * kept, and the serialized payload is shrunk (drop oldest thread, then halve
 * the last thread's turns) until it fits the size budget.
 */
export function writeThreads(threads: StoredThread[]): void {
  if (typeof window === 'undefined') return;
  try {
    let keep = threads
      .filter((t) => isPersistableThread(t) && t.turns.length > 0)
      .map((t) => ({ ...t, turns: trimTurns(t.turns) }))
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_THREADS);
    let payload = JSON.stringify(keep);
    while (payload.length > MAX_SERIALIZED_CHARS && keep.length > 0) {
      const last = keep[keep.length - 1]!;
      if (keep.length > 1) {
        keep = keep.slice(0, -1);
      } else if (last.turns.length > 2) {
        keep = [{ ...last, turns: last.turns.slice(Math.ceil(last.turns.length / 2)) }];
      } else {
        keep = [];
      }
      payload = JSON.stringify(keep);
    }
    window.localStorage.setItem(STORAGE_KEY, payload);
  } catch {
    // Quota / privacy mode — persistence degrades to in-memory only.
  }
}

'use client';

import { useTranslations } from 'next-intl';
import * as React from 'react';
import {
  askSentinel,
  ChatApiUnavailableError,
} from '@/lib/chat/api-client';
import {
  readThreads,
  writeThreads,
  type StoredThread,
  type StoredTurn,
} from '@/lib/chat/thread-store';
import type { ChatSignalContext } from '@/lib/chat/types';

/**
 * One chat turn. Field docs live on StoredTurn (thread-store.ts) — the
 * in-memory and persisted shapes are intentionally identical so a thread can
 * round-trip through localStorage without mapping.
 */
type ChatTurn = StoredTurn;

/**
 * Display-only chart view actions returned by the last successful turn, RAW (not
 * yet validated against on-screen zones). `nonce` lets a consumer re-apply even
 * when the same action list repeats. The workspace re-validates via
 * `coerceViewActions` before touching the chart render.
 */
export interface ViewActionSignal {
  actions: ReadonlyArray<Record<string, unknown>>;
  nonce: number;
}

/** Recency-sorted summary of a combo-scoped conversation, for the recents list. */
export interface ChatThreadSummary {
  id: string;
  instrument: string;
  timeframe: string;
  /** Epoch ms of the last appended turn. */
  updatedAt: number;
  turnCount: number;
  /** Text of the last non-empty turn (may be '' for display-only answers). */
  lastText: string;
}

interface ChatContextValue {
  isOpen: boolean;
  activeSignal: ChatSignalContext | null;
  turns: ChatTurn[];
  /** Latest display-only chart actions from the chatbot (raw), or null. */
  viewActionSignal: ViewActionSignal | null;
  /** True while a backend answer is in flight (synchronous JSON, no stream). */
  isLoading: boolean;
  /**
   * Whether the backend chatbot is reachable (`true` = answering, `false` =
   * endpoint returned 503 / not bootstrapped, scripted fallback only,
   * `'unknown'` before the first call). Sticky once set to false to avoid
   * spamming the endpoint.
   */
  apiAvailable: boolean | 'unknown';
  /**
   * The N most recent combo-scoped (`app:*`) conversations, newest first.
   * Landing signal chats are excluded — they are not reachable from the /app
   * combo selector and are never persisted.
   */
  recentThreads: ReadonlyArray<ChatThreadSummary>;
  openFor(signal: ChatSignalContext): void;
  /**
   * Bind the chat context to an (instrument, timeframe) combo WITHOUT opening
   * the slide-over. Used by the permanent /app sidebar (Chantier 5.B), which
   * renders its own inline panel and must not trigger the layout Sheet.
   * Switching combos switches to that combo's own thread — the previous
   * combo's conversation is kept and restored when the user comes back.
   */
  openForCombo(combo: { instrument: string; timeframe: string }): void;
  close(): void;
  appendExchange(args: {
    questionId: string;
    text: string;
    reply: string;
    source?: ChatTurn['source'];
  }): void;
  /**
   * Submit a free-form user question. Calls POST /api/chatbot/message (3 niveau-
   * 1.5 defence layers server-side), falls back to a friendly message if the
   * backend is unavailable. Throws if no activeSignal is set.
   */
  askFreeForm(question: string): Promise<void>;
  /** Clear the ACTIVE thread only (and its persisted copy). */
  resetTurns(): void;
}

const ChatContext = React.createContext<ChatContextValue | null>(null);

/**
 * Conversation state. A single atom holding the active signal AND the per-
 * thread turn lists, so functional updaters compose within one React batch
 * (e.g. `openFor` then `appendExchange` in the same click handler lands the
 * exchange in the just-activated thread). Threads are keyed by signal id —
 * `app:{instrument}:{timeframe}` for /app combos, the real signal id for
 * landing readings. Switching threads NEVER discards the previous one; that is
 * the whole fix for "changing timeframe erases the conversation".
 */
interface ChatState {
  active: ChatSignalContext | null;
  threads: Record<string, StoredThread>;
}

const EMPTY_TURNS: ChatTurn[] = [];

function ensureThread(
  threads: Record<string, StoredThread>,
  signal: ChatSignalContext,
): Record<string, StoredThread> {
  if (threads[signal.id]) return threads;
  return {
    ...threads,
    [signal.id]: {
      id: signal.id,
      instrument: signal.instrument,
      timeframe: signal.timeframe,
      updatedAt: 0,
      turns: [],
    },
  };
}

function appendToThread(
  state: ChatState,
  threadId: string,
  meta: { instrument: string; timeframe: string },
  newTurns: ChatTurn[],
): ChatState {
  const base = state.threads[threadId] ?? {
    id: threadId,
    instrument: meta.instrument,
    timeframe: meta.timeframe,
    updatedAt: 0,
    turns: [],
  };
  return {
    ...state,
    threads: {
      ...state.threads,
      [threadId]: {
        ...base,
        updatedAt: Date.now(),
        turns: [...base.turns, ...newTurns],
      },
    },
  };
}

const MAX_RECENT_THREADS = 6;

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const t = useTranslations('chat');
  const [isOpen, setIsOpen] = React.useState(false);
  const [state, setState] = React.useState<ChatState>({
    active: null,
    threads: {},
  });
  // Mirror of the latest state for reads inside async callbacks (UI-01): the
  // history payload must reflect the CURRENT thread, not the `turns` value
  // captured when askFreeForm was created (stale on rapid/successive sends).
  const stateRef = React.useRef(state);
  React.useEffect(() => {
    stateRef.current = state;
  }, [state]);
  const [hydrated, setHydrated] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);
  const [apiAvailable, setApiAvailable] = React.useState<boolean | 'unknown'>(
    'unknown',
  );
  const [viewActionSignal, setViewActionSignal] =
    React.useState<ViewActionSignal | null>(null);
  const seqRef = React.useRef(0);
  const viewNonceRef = React.useRef(0);

  // Hydrate persisted combo threads ONCE (client-only, localStorage). Threads
  // already started this session win over their stored copy.
  React.useEffect(() => {
    const stored = readThreads();
    if (stored.length > 0) {
      setState((s) => {
        const threads = { ...s.threads };
        for (const t of stored) {
          const existing = threads[t.id];
          if (!existing || existing.turns.length === 0) threads[t.id] = t;
        }
        return { ...s, threads };
      });
      // Restored turn ids end in a numeric suffix (`user-3`, `q1-a-7`); move
      // the id counter past the highest one so new ids never collide.
      let maxSeq = seqRef.current;
      for (const t of stored) {
        for (const turn of t.turns) {
          const m = /-(\d+)$/.exec(turn.id);
          if (m) maxSeq = Math.max(maxSeq, Number(m[1]) + 1);
        }
      }
      seqRef.current = maxSeq;
    }
    setHydrated(true);
  }, []);

  // Persist on every thread change — but only after hydration has landed in
  // state, so an early write can never clobber storage with the empty initial
  // state. writeThreads applies all caps (app-only, per-thread trim, thread
  // count, size budget) and swallows quota errors. CLIENT-ONLY by design
  // (Loi 25): no server call anywhere in this provider besides askSentinel's
  // existing per-request message POST.
  React.useEffect(() => {
    if (!hydrated) return;
    writeThreads(Object.values(state.threads));
  }, [hydrated, state.threads]);

  const openFor = React.useCallback((signal: ChatSignalContext) => {
    setState((s) => ({
      active: signal,
      threads: ensureThread(s.threads, signal),
    }));
    setIsOpen(true);
  }, []);

  const openForCombo = React.useCallback(
    (combo: { instrument: string; timeframe: string }) => {
      // Minimal signal context: the chat path only reads id/instrument/
      // timeframe (api-client preamble + labels). ChatSignalContext is exactly
      // that shape, so no cast is needed anymore (Chantier 5.C). No isOpen
      // toggle → the layout Sheet stays closed on /app.
      const synthetic: ChatSignalContext = {
        id: `app:${combo.instrument}:${combo.timeframe}`,
        instrument: combo.instrument,
        timeframe: combo.timeframe,
      };
      setState((s) => ({
        active: synthetic,
        threads: ensureThread(s.threads, synthetic),
      }));
    },
    [],
  );

  const close = React.useCallback(() => setIsOpen(false), []);

  const nextId = React.useCallback((prefix: string) => {
    return `${prefix}-${seqRef.current++}`;
  }, []);

  const appendExchange = React.useCallback(
    ({
      questionId,
      text,
      reply,
      source = 'scripted',
    }: {
      questionId: string;
      text: string;
      reply: string;
      source?: ChatTurn['source'];
    }) => {
      setState((s) => {
        if (!s.active) return s;
        return appendToThread(s, s.active.id, s.active, [
          { id: `${questionId}-q-${seqRef.current++}`, role: 'user', text },
          {
            id: `${questionId}-a-${seqRef.current++}`,
            role: 'assistant',
            text: reply,
            source,
          },
        ]);
      });
    },
    [],
  );

  const resetTurns = React.useCallback(() => {
    setState((s) => {
      if (!s.active || !s.threads[s.active.id]) return s;
      const threads = { ...s.threads };
      delete threads[s.active.id];
      return { ...s, threads };
    });
  }, []);

  const activeSignal = state.active;
  const turns = activeSignal
    ? (state.threads[activeSignal.id]?.turns ?? EMPTY_TURNS)
    : EMPTY_TURNS;

  const askFreeForm = React.useCallback(
    async (question: string) => {
      if (!activeSignal) {
        throw new Error('No active signal for chat');
      }
      const trimmed = question.trim();
      if (!trimmed) return;

      // Pin the destination thread NOW: if the user switches combo while the
      // answer is in flight, the reply still lands in the thread it belongs to.
      const threadId = activeSignal.id;
      const meta = {
        instrument: activeSignal.instrument,
        timeframe: activeSignal.timeframe,
      };

      // Push the user turn immediately so the UI feels responsive.
      const userTurnId = nextId('user');
      setState((s) =>
        appendToThread(s, threadId, meta, [
          { id: userTurnId, role: 'user', text: trimmed },
        ]),
      );

      // Build history payload (last 6 turns, alternating user/assistant) from
      // the FRESH thread state (stateRef), not the closure `turns` — otherwise a
      // second message sent before the re-render would replay a stale history
      // (UI-01). We read the thread as it is BEFORE pushing the current question,
      // so the current turn is correctly excluded. Drop any empty/whitespace turn:
      // a display-only answer renders as an empty bubble and the backend's
      // ConversationMessage requires content length ≥ 1 — replaying it would 422
      // the whole request ("format ou longueur") on the next message.
      const priorTurns = stateRef.current.threads[threadId]?.turns ?? EMPTY_TURNS;
      const historyForApi = priorTurns
        .slice(-6)
        .map((t) => ({ role: t.role, content: t.text.trim() }))
        .filter((m) => m.content.length > 0);

      setIsLoading(true);

      try {
        const { text, blockedReason, viewActions } = await askSentinel({
          signal: activeSignal,
          question: trimmed,
          history: historyForApi,
        });
        setApiAvailable(true);
        setState((s) =>
          appendToThread(s, threadId, meta, [
            {
              id: nextId('asst'),
              role: 'assistant',
              text,
              source: 'llm',
              blockedReason,
              viewUpdated: (viewActions ?? []).length > 0,
            },
          ]),
        );
        // Surface display-only chart actions (raw) for the workspace to validate
        // against on-screen zones and apply to the render. Always set a fresh
        // signal (even when empty) so a consumer never replays a stale list.
        viewNonceRef.current += 1;
        setViewActionSignal({
          actions: viewActions ?? [],
          nonce: viewNonceRef.current,
        });
      } catch (err) {
        if (err instanceof ChatApiUnavailableError) {
          setApiAvailable(false);
          setState((s) =>
            appendToThread(s, threadId, meta, [
              {
                id: nextId('asst'),
                role: 'assistant',
                source: 'fallback',
                text: t('turnUnavailable'),
              },
            ]),
          );
        } else {
          const message = err instanceof Error ? err.message : t('unknownError');
          setState((s) =>
            appendToThread(s, threadId, meta, [
              {
                id: nextId('asst'),
                role: 'assistant',
                source: 'error',
                text: t('turnError', { message }),
              },
            ]),
          );
        }
      } finally {
        setIsLoading(false);
      }
    },
    // `turns` intentionally dropped — history is read from stateRef now, so the
    // callback identity no longer churns on every new turn (UI-01).
    [activeSignal, nextId, t],
  );

  const recentThreads = React.useMemo<ChatThreadSummary[]>(() => {
    return Object.values(state.threads)
      .filter((t) => t.id.startsWith('app:') && t.turns.length > 0)
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_RECENT_THREADS)
      .map((t) => {
        const last = [...t.turns]
          .reverse()
          .find((turn) => turn.text.trim().length > 0);
        return {
          id: t.id,
          instrument: t.instrument,
          timeframe: t.timeframe,
          updatedAt: t.updatedAt,
          turnCount: t.turns.length,
          lastText: last?.text ?? '',
        };
      });
  }, [state.threads]);

  const value = React.useMemo<ChatContextValue>(
    () => ({
      isOpen,
      activeSignal,
      turns,
      viewActionSignal,
      isLoading,
      apiAvailable,
      recentThreads,
      openFor,
      openForCombo,
      close,
      appendExchange,
      askFreeForm,
      resetTurns,
    }),
    [
      isOpen,
      activeSignal,
      turns,
      viewActionSignal,
      isLoading,
      apiAvailable,
      recentThreads,
      openFor,
      openForCombo,
      close,
      appendExchange,
      askFreeForm,
      resetTurns,
    ],
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): ChatContextValue {
  const ctx = React.useContext(ChatContext);
  if (!ctx) {
    throw new Error(
      'useChat must be used inside a <ChatProvider /> (mounted in [locale]/layout.tsx).',
    );
  }
  return ctx;
}

import { beforeEach, describe, expect, it } from 'vitest';
import {
  MAX_SERIALIZED_CHARS,
  MAX_THREADS,
  MAX_TURNS_PER_THREAD,
  PRODUCT_THREAD_ID,
  readThreads,
  STORAGE_KEY,
  writeThreads,
  type StoredThread,
  type StoredTurn,
} from '../thread-store';

// MIA-3 — a single product-wide conversation is persisted (id === PRODUCT_THREAD_ID).
// The combo is orientation only, no longer a thread key, so instrument/timeframe
// are free-form display fields (not gated against the perimeter).

function makeTurns(count: number, textLen = 10): StoredTurn[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `t-${i}`,
    role: i % 2 === 0 ? ('user' as const) : ('assistant' as const),
    text: 'x'.repeat(textLen),
  }));
}

function makeThread(
  updatedAt: number,
  turns: StoredTurn[] = makeTurns(2),
  overrides: Partial<StoredThread> = {},
): StoredThread {
  return {
    id: PRODUCT_THREAD_ID,
    instrument: 'XAUUSD',
    timeframe: 'H1',
    updatedAt,
    turns,
    ...overrides,
  };
}

beforeEach(() => {
  window.localStorage.clear();
});

describe('thread-store round-trip', () => {
  it('writes then reads back the product conversation unchanged', () => {
    const thread = makeThread(1000, [
      { id: 'user-0', role: 'user', text: 'Question ?' },
      {
        id: 'asst-1',
        role: 'assistant',
        text: 'Réponse.',
        source: 'llm',
        blockedReason: 'trade_request',
        viewUpdated: true,
      },
    ]);
    writeThreads([thread]);
    expect(readThreads()).toEqual([thread]);
  });

  it('returns [] when storage is empty or corrupt', () => {
    expect(readThreads()).toEqual([]);
    window.localStorage.setItem(STORAGE_KEY, '{not json');
    expect(readThreads()).toEqual([]);
    window.localStorage.setItem(STORAGE_KEY, '{"threads": 1}');
    expect(readThreads()).toEqual([]);
  });
});

describe('thread-store sanitisation (never trusts storage)', () => {
  it('keeps only the product thread, dropping any other id', () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        makeThread(1),
        { ...makeThread(2), id: 'app:XAUUSD:H1' }, // legacy per-combo id
        { ...makeThread(3), id: 'sig-1' }, // landing signal id
      ]),
    );
    expect(readThreads().map((t) => t.id)).toEqual([PRODUCT_THREAD_ID]);
  });

  it('keeps the product thread whatever the orientation combo (not perimeter-gated)', () => {
    // The orientation combo may be anything (or even nothing) — it is display
    // metadata now, not a validated key. A conversation must never be dropped
    // because the last focused market was exotic.
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([makeThread(1, makeTurns(2), { instrument: 'BTCUSD', timeframe: 'W1' })]),
    );
    const threads = readThreads();
    expect(threads).toHaveLength(1);
    expect(threads[0]!.instrument).toBe('BTCUSD');
  });

  it('drops invalid turns and a thread left with no turn', () => {
    const good = { id: 'u-0', role: 'user', text: 'ok' };
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        {
          ...makeThread(1),
          turns: [good, { role: 'system', text: 'nope' }, { role: 'user' }, 42],
        },
      ]),
    );
    const threads = readThreads();
    expect(threads).toHaveLength(1);
    expect(threads[0]!.turns).toEqual([good]);
  });

  it('de-duplicates the product id, keeping the first occurrence', () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([makeThread(1, makeTurns(2)), makeThread(9, makeTurns(4))]),
    );
    const threads = readThreads();
    expect(threads).toHaveLength(1);
    expect(threads[0]!.turns).toHaveLength(2);
  });
});

describe('thread-store caps & purge', () => {
  it('skips empty and non-product threads on write', () => {
    writeThreads([
      makeThread(1, []), // empty product thread — skipped
      {
        id: 'sig-1',
        instrument: 'XAUUSD',
        timeframe: 'H1',
        updatedAt: 2,
        turns: makeTurns(2),
      }, // non-product — skipped
      makeThread(3, makeTurns(2)), // the real conversation — kept
    ]);
    expect(readThreads().map((t) => t.id)).toEqual([PRODUCT_THREAD_ID]);
  });

  it('trims the conversation to MAX_TURNS_PER_THREAD, never starting mid-exchange', () => {
    writeThreads([makeThread(1, makeTurns(MAX_TURNS_PER_THREAD + 5))]);
    const [thread] = readThreads();
    expect(thread!.turns.length).toBeLessThanOrEqual(MAX_TURNS_PER_THREAD);
    expect(thread!.turns[0]!.role).toBe('user');
  });

  it('persists at most MAX_THREADS thread (one conversation)', () => {
    writeThreads([
      makeThread(10, makeTurns(2)),
      { ...makeThread(60, makeTurns(2)), id: 'app:EURUSD:H4' }, // non-product, dropped
    ]);
    const stored = readThreads();
    expect(stored.length).toBeLessThanOrEqual(MAX_THREADS);
    expect(stored.map((t) => t.id)).toEqual([PRODUCT_THREAD_ID]);
  });

  it('halves an oversized conversation instead of dropping everything', () => {
    writeThreads([makeThread(1, makeTurns(MAX_TURNS_PER_THREAD, 20_000))]);
    const raw = window.localStorage.getItem(STORAGE_KEY)!;
    expect(raw.length).toBeLessThanOrEqual(MAX_SERIALIZED_CHARS);
    const stored = readThreads();
    expect(stored).toHaveLength(1);
    expect(stored[0]!.turns.length).toBeGreaterThan(0);
    expect(stored[0]!.turns.length).toBeLessThan(MAX_TURNS_PER_THREAD);
  });
});

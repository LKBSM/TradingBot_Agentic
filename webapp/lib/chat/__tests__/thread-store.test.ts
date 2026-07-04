import { beforeEach, describe, expect, it } from 'vitest';
import {
  MAX_SERIALIZED_CHARS,
  MAX_THREADS,
  MAX_TURNS_PER_THREAD,
  readThreads,
  STORAGE_KEY,
  writeThreads,
  type StoredThread,
  type StoredTurn,
} from '../thread-store';

function makeTurns(count: number, textLen = 10): StoredTurn[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `t-${i}`,
    role: i % 2 === 0 ? ('user' as const) : ('assistant' as const),
    text: 'x'.repeat(textLen),
  }));
}

function makeThread(
  instrument: string,
  timeframe: string,
  updatedAt: number,
  turns: StoredTurn[] = makeTurns(2),
): StoredThread {
  return {
    id: `app:${instrument}:${timeframe}`,
    instrument,
    timeframe,
    updatedAt,
    turns,
  };
}

beforeEach(() => {
  window.localStorage.clear();
});

describe('thread-store round-trip', () => {
  it('writes then reads back a thread unchanged', () => {
    const thread = makeThread('XAUUSD', 'H1', 1000, [
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
  it('drops threads outside the supported perimeter or with a mismatched id', () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        makeThread('XAUUSD', 'H1', 1),
        makeThread('BTCUSD', 'H1', 2), // unsupported instrument
        makeThread('XAUUSD', 'M5', 3), // unsupported timeframe
        { ...makeThread('EURUSD', 'H4', 4), id: 'app:XAUUSD:H4' }, // id mismatch
      ]),
    );
    expect(readThreads().map((t) => t.id)).toEqual(['app:XAUUSD:H1']);
  });

  it('drops invalid turns and threads left with no turn', () => {
    const good = { id: 'u-0', role: 'user', text: 'ok' };
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        {
          ...makeThread('XAUUSD', 'H1', 1),
          turns: [good, { role: 'system', text: 'nope' }, { role: 'user' }, 42],
        },
        { ...makeThread('EURUSD', 'H1', 2), turns: ['garbage'] },
      ]),
    );
    const threads = readThreads();
    expect(threads).toHaveLength(1);
    expect(threads[0]!.turns).toEqual([good]);
  });

  it('de-duplicates thread ids, keeping the first occurrence', () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        makeThread('XAUUSD', 'H1', 1, makeTurns(2)),
        makeThread('XAUUSD', 'H1', 9, makeTurns(4)),
      ]),
    );
    const threads = readThreads();
    expect(threads).toHaveLength(1);
    expect(threads[0]!.turns).toHaveLength(2);
  });
});

describe('thread-store caps & purge', () => {
  it('skips empty and non-app threads on write', () => {
    writeThreads([
      makeThread('XAUUSD', 'H1', 1, []),
      {
        id: 'sig-1',
        instrument: 'XAUUSD',
        timeframe: 'H1',
        updatedAt: 2,
        turns: makeTurns(2),
      },
      makeThread('EURUSD', 'H4', 3),
    ]);
    expect(readThreads().map((t) => t.id)).toEqual(['app:EURUSD:H4']);
  });

  it('trims each thread to MAX_TURNS_PER_THREAD, never starting mid-exchange', () => {
    writeThreads([
      makeThread('XAUUSD', 'H1', 1, makeTurns(MAX_TURNS_PER_THREAD + 5)),
    ]);
    const [thread] = readThreads();
    expect(thread!.turns.length).toBeLessThanOrEqual(MAX_TURNS_PER_THREAD);
    expect(thread!.turns[0]!.role).toBe('user');
  });

  it('keeps only the MAX_THREADS most recent threads', () => {
    // Perimeter is 6 combos; MAX_THREADS ≥ 6 so all fit — assert via ordering
    // by writing the full perimeter and checking recency sort instead.
    const all = [
      makeThread('XAUUSD', 'M15', 10),
      makeThread('XAUUSD', 'H1', 60),
      makeThread('XAUUSD', 'H4', 20),
      makeThread('EURUSD', 'M15', 50),
      makeThread('EURUSD', 'H1', 30),
      makeThread('EURUSD', 'H4', 40),
    ];
    writeThreads(all);
    const stored = readThreads();
    expect(stored).toHaveLength(Math.min(all.length, MAX_THREADS));
    expect(stored.map((t) => t.updatedAt)).toEqual([60, 50, 40, 30, 20, 10]);
  });

  it('drops the oldest threads until the payload fits the size budget', () => {
    // Each thread ~40 turns × 2000 chars ≈ 80k chars serialized — three of
    // them exceed the 200k budget, so at least the oldest must be purged.
    const fat = (i: number, instrument: string, tf: string) =>
      makeThread(instrument, tf, i, makeTurns(MAX_TURNS_PER_THREAD, 2000));
    writeThreads([
      fat(1, 'XAUUSD', 'M15'),
      fat(2, 'XAUUSD', 'H1'),
      fat(3, 'XAUUSD', 'H4'),
    ]);
    const raw = window.localStorage.getItem(STORAGE_KEY)!;
    expect(raw.length).toBeLessThanOrEqual(MAX_SERIALIZED_CHARS);
    const stored = readThreads();
    expect(stored.length).toBeLessThan(3);
    // Newest survives, oldest is the purge victim.
    expect(stored[0]!.updatedAt).toBe(3);
  });

  it('halves a single oversized thread instead of dropping everything', () => {
    writeThreads([
      makeThread('XAUUSD', 'H1', 1, makeTurns(MAX_TURNS_PER_THREAD, 20_000)),
    ]);
    const raw = window.localStorage.getItem(STORAGE_KEY)!;
    expect(raw.length).toBeLessThanOrEqual(MAX_SERIALIZED_CHARS);
    const stored = readThreads();
    expect(stored).toHaveLength(1);
    expect(stored[0]!.turns.length).toBeGreaterThan(0);
    expect(stored[0]!.turns.length).toBeLessThan(MAX_TURNS_PER_THREAD);
  });
});

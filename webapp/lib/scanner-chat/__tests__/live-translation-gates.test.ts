import { describe, expect, it } from 'vitest';
import {
  MIN_CHARS,
  MIN_DELTA,
  charDelta,
  passesGates,
  truncateToWordBoundary,
} from '../use-live-translation';

/**
 * SC-4 — the gates that decide whether a keystroke pause is worth a paid call.
 *
 * Measured, not guessed: one translation is ~2 861 input + ~217 output tokens,
 * ~$0.0039 on Haiku 4.5 with no prompt caching possible (its minimum cacheable
 * prefix is 4 096 tokens; ours is 2 838, so the prefix is repaid in full every
 * time). These gates are the difference between ~12 calls per composed sentence
 * and ~5. They are pure functions precisely so they can be pinned here without
 * a clock or a network.
 */

describe('truncateToWordBoundary — never translate half a word', () => {
  it('cuts back to the last completed word', () => {
    expect(truncateToWordBoundary('Un Order Block jamais tes')).toBe('Un Order Block jamais');
  });

  it('leaves text that already ends on a boundary alone', () => {
    expect(truncateToWordBoundary('Un Order Block jamais testé ')).toBe('Un Order Block jamais testé');
    expect(truncateToWordBoundary('Un Order Block, ')).toBe('Un Order Block,');
  });

  it('treats punctuation as a boundary', () => {
    expect(truncateToWordBoundary('tendance haussière.')).toBe('tendance haussière.');
  });

  it('returns nothing for a single unfinished word', () => {
    expect(truncateToWordBoundary('Order')).toBe('');
  });

  it('returns nothing for blank input', () => {
    expect(truncateToWordBoundary('   ')).toBe('');
  });
});

describe('charDelta', () => {
  it('is zero for identical text', () => {
    expect(charDelta('abc', 'abc')).toBe(0);
  });

  it('counts an append', () => {
    expect(charDelta('un order block', 'un order block jamais')).toBe(7);
  });

  it('counts a change in the middle', () => {
    expect(charDelta('tendance haussière ici', 'tendance baissière ici')).toBeGreaterThan(0);
  });

  it('counts a full replacement', () => {
    expect(charDelta('abcdef', 'uvwxyz')).toBe(6);
  });
});

describe('passesGates', () => {
  const long = 'Un Order Block jamais testé en tendance haussière';

  it('accepts a real sentence typed from nothing', () => {
    expect(passesGates(long, '')).toBe(true);
  });

  it('refuses text that is too short to be a sentence', () => {
    expect('Un Order Block'.length).toBeLessThan(MIN_CHARS);
    expect(passesGates('Un Order Block', '')).toBe(false);
  });

  it('refuses long text that is not yet four words', () => {
    expect(passesGates('Orderblockjamaistesteenhausse encore', '')).toBe(false);
  });

  it('refuses a near-duplicate of what was already translated', () => {
    expect(passesGates(`${long} et`, long)).toBe(false);
  });

  it('accepts once enough has really changed', () => {
    const grown = `${long}, avec une poche de liquidité prise`;
    expect(charDelta(grown, long)).toBeGreaterThanOrEqual(MIN_DELTA);
    expect(passesGates(grown, long)).toBe(true);
  });

  it('never re-sends the exact same text', () => {
    expect(passesGates(long, long)).toBe(false);
  });

  it('refuses empty input', () => {
    expect(passesGates('', '')).toBe(false);
  });
});

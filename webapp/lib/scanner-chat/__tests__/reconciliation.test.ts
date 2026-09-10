import { describe, expect, it } from 'vitest';
import type { ScanCondition } from '@/lib/conditions/types';
import {
  EMPTY_MANUAL_STATE,
  conditionKey,
  isSuppressed,
  normalizePhrase,
  purgeRemovals,
  reconcile,
  type ManualState,
} from '../reconciliation';

/**
 * SC-4 — the rule the mission called the critical point: a live re-translation
 * must never silently re-add a condition the user removed by hand, and must
 * never refuse one they have just re-described.
 *
 * Both halves matter. Respecting a removal for ever is as much a violation of
 * « elle ne choisit rien à ta place » as ignoring it: the user would be unable
 * to bring the condition back by writing, which is the only tool this screen
 * gives them.
 */

const ob: ScanCondition = { type: 'price_in_ob' } as ScanCondition;
const untested: ScanCondition = { type: 'zone_untested' } as ScanCondition;
const trendUp: ScanCondition = { type: 'trend_is', trend: 'bullish' } as ScanCondition;
const trendDown: ScanCondition = { type: 'trend_is', trend: 'bearish' } as ScanCondition;

function manual(over: Partial<ManualState> = {}): ManualState {
  return { ...EMPTY_MANUAL_STATE, ...over };
}

describe('conditionKey', () => {
  it('separates two conditions that differ only by a control', () => {
    expect(conditionKey(trendUp)).not.toBe(conditionKey(trendDown));
  });

  it('is stable across key insertion order', () => {
    const a = { type: 'trend_is', trend: 'bullish', direction: 'any' } as ScanCondition;
    const b = { direction: 'any', trend: 'bullish', type: 'trend_is' } as ScanCondition;
    expect(conditionKey(a)).toBe(conditionKey(b));
  });
});

describe('normalizePhrase', () => {
  it('matches the server: folds accents, case and whitespace runs', () => {
    expect(normalizePhrase('  Jamais   TESTÉ\n')).toBe('jamais teste');
  });
});

describe('purgeRemovals — a removal is released when its words leave', () => {
  it('keeps the removal while the fragment is still in the text', () => {
    const removed = [{ key: conditionKey(untested), phrase: 'jamais testé' }];
    expect(purgeRemovals(removed, 'Un Order Block jamais testé en tendance haussière')).toHaveLength(1);
  });

  it('releases the removal once the fragment is edited away', () => {
    const removed = [{ key: conditionKey(untested), phrase: 'jamais testé' }];
    expect(purgeRemovals(removed, 'Un Order Block en tendance haussière')).toHaveLength(0);
  });

  it('matches the fragment through accents and spacing, like the server', () => {
    const removed = [{ key: conditionKey(untested), phrase: 'jamais testé' }];
    expect(purgeRemovals(removed, 'un OB JAMAIS   TESTE ici')).toHaveLength(1);
  });

  it('keeps an unattributed removal (nothing to watch for)', () => {
    const removed = [{ key: conditionKey(ob), phrase: null }];
    expect(purgeRemovals(removed, 'un texte totalement différent')).toHaveLength(1);
  });

  it('forgets everything when the field is emptied', () => {
    const removed = [
      { key: conditionKey(ob), phrase: null },
      { key: conditionKey(untested), phrase: 'jamais testé' },
    ];
    expect(purgeRemovals(removed, '   ')).toHaveLength(0);
  });
});

describe('isSuppressed — pair matching is what lets a re-description win', () => {
  const removed = [{ key: conditionKey(untested), phrase: 'jamais testé' }];

  it('suppresses the same condition derived from the same words', () => {
    expect(isSuppressed(removed, conditionKey(untested), 'jamais testé')).toBe(true);
  });

  it('does NOT suppress the same condition derived from different words', () => {
    // The user removed « jamais testé », then wrote « zone vierge ». That is an
    // explicit re-description: the condition must come back.
    expect(isSuppressed(removed, conditionKey(untested), 'zone vierge')).toBe(false);
  });

  it('does not suppress a different condition', () => {
    expect(isSuppressed(removed, conditionKey(ob), 'jamais testé')).toBe(false);
  });

  it('falls back to the condition alone when THIS reading has no citation', () => {
    // The model simply did not cite it this time; that is not evidence the user
    // re-described anything, so the removal still stands.
    expect(isSuppressed(removed, conditionKey(untested), null)).toBe(true);
  });

  it('falls back to the condition alone when the REMOVAL has no citation', () => {
    const unattributed = [{ key: conditionKey(untested), phrase: null }];
    expect(isSuppressed(unattributed, conditionKey(untested), 'jamais testé')).toBe(true);
  });
});

describe('reconcile — the reading is a proposal, the user is the authority', () => {
  it('shows every proposal when the user has touched nothing', () => {
    const out = reconcile([ob, untested], ['un OB', 'jamais testé'], manual());
    expect(out.map((r) => r.condition)).toEqual([ob, untested]);
    expect(out.every((r) => !r.userAdded && !r.userEdited)).toBe(true);
  });

  it('drops a removed condition while its words stand', () => {
    const state = manual({ removed: [{ key: conditionKey(untested), phrase: 'jamais testé' }] });
    const out = reconcile([ob, untested], ['un OB', 'jamais testé'], state);
    expect(out.map((r) => r.condition)).toEqual([ob]);
  });

  it('brings it back when the same idea is re-described in other words', () => {
    const state = manual({ removed: [{ key: conditionKey(untested), phrase: 'jamais testé' }] });
    const out = reconcile([ob, untested], ['un OB', 'zone vierge'], state);
    expect(out.map((r) => r.condition)).toEqual([ob, untested]);
  });

  it('substitutes the user edit for the value M.I.A keeps proposing', () => {
    const state = manual({ edited: { [conditionKey(trendUp)]: trendDown } });
    const out = reconcile([trendUp], ['tendance'], state);
    expect(out[0]!.condition).toEqual(trendDown);
    expect(out[0]!.userEdited).toBe(true);
    // The handle stays the PROPOSAL, or the next reading would lose the edit.
    expect(out[0]!.proposalKey).toBe(conditionKey(trendUp));
  });

  it('keeps a user-added condition no matter what the text says', () => {
    const out = reconcile([], [], manual({ added: [ob] }));
    expect(out.map((r) => r.condition)).toEqual([ob]);
    expect(out[0]!.userAdded).toBe(true);
  });

  it('never shows the same condition twice when an edit collides with a proposal', () => {
    // The user edits « haussière » into « baissière » while M.I.A also proposes
    // « baissière » from another fragment.
    const state = manual({ edited: { [conditionKey(trendUp)]: trendDown } });
    const out = reconcile([trendUp, trendDown], ['tendance', 'baissier'], state);
    expect(out).toHaveLength(1);
    expect(out[0]!.condition).toEqual(trendDown);
  });

  it('does not duplicate a user addition M.I.A also proposes', () => {
    const out = reconcile([ob], ['un OB'], manual({ added: [ob] }));
    expect(out).toHaveLength(1);
    expect(out[0]!.userAdded).toBe(false);
  });

  it('treats a missing sources array as "origin unknown", not as an error', () => {
    const out = reconcile([ob, untested], [], manual());
    expect(out).toHaveLength(2);
    expect(out.every((r) => r.phrase === null)).toBe(true);
  });

  it('keeps a stable order so chips do not jump between keystrokes', () => {
    const state = manual({ added: [trendDown] });
    const first = reconcile([ob, untested], ['a', 'b'], state).map((r) => conditionKey(r.condition));
    const again = reconcile([ob, untested], ['a', 'b'], state).map((r) => conditionKey(r.condition));
    expect(first).toEqual(again);
    expect(first[first.length - 1]!).toBe(conditionKey(trendDown));
  });
});

describe('purgeRemovals — clearing the field is a fresh start', () => {
  it('a removal does not survive the field being emptied and rewritten', () => {
    // The component also resets `removed`/`edited` when the field empties; this
    // pins the pure half of that contract, which the component relies on.
    const removed = [{ key: conditionKey(untested), phrase: 'jamais testé' }];
    expect(purgeRemovals(removed, '')).toHaveLength(0);
    // …and re-typing the same words starts from a clean slate, because the
    // entry is already gone by then.
    expect(purgeRemovals(purgeRemovals(removed, ''), 'un OB jamais testé')).toHaveLength(0);
  });
});

import type { ScanCondition } from '@/lib/conditions/types';

/**
 * SC-4 — reconciling what M.I.A re-derives while you type with what you changed
 * by hand.
 *
 * Before SC-4 there was nothing to reconcile: one click produced one translation
 * and the user edited it afterwards, undisturbed. Now the translation re-runs on
 * every pause in typing, so each new result would silently overwrite the user's
 * removals, edits and additions. The product line is "elle ne choisit rien à ta
 * place" — so the automatic result is treated as a PROPOSAL, and everything the
 * user did by hand wins over it.
 *
 * Three kinds of manual intervention, each with its own rule:
 *
 *   · REMOVED (the "×" on a chip) — the condition stays out for as long as the
 *     words that produced it stand. The server hands us, per condition, the
 *     fragment of the user's own sentence it was derived from (verified verbatim
 *     there, never invented). A removal is remembered as (condition, fragment);
 *     the moment that fragment leaves the text, the removal is FORGOTTEN, so
 *     rewriting the idea brings the condition back. Describing it again in
 *     DIFFERENT words also brings it back, because the new fragment no longer
 *     matches the remembered one. This is the whole point: a removal must not
 *     become a permanent veto the user cannot undo by writing.
 *
 *   · EDITED (a control changed on a chip) — the user's value replaces the
 *     model's, keyed on the condition the model keeps re-proposing. Without
 *     this, every re-translation would quietly reset a value the user had just
 *     corrected. Not asked for by the mission, but it is the same violation as
 *     an un-respected removal.
 *
 *   · ADDED (via the palette picker) — owned by the user outright. It comes from
 *     no text, so no re-translation can ever remove it.
 *
 * Everything here is pure and synchronous so the rules can be tested on their
 * own, without a network or a clock.
 */

/** Control fields that take part in a condition's identity. */
const IDENTITY_FIELDS: readonly (keyof ScanCondition)[] = [
  'type',
  'direction',
  'max_bars',
  'trend',
  'phase',
  'volatility',
  'proximity_pct',
  'side',
  'zone_kind',
  'max_touches',
  'event',
  'age_bucket',
  'third',
  'eq_kind',
  'session',
  'relation',
];

/**
 * A stable identity for a condition: its type plus every control it actually
 * sets, in a fixed order. Two conditions with the same key are the same search.
 */
export function conditionKey(condition: ScanCondition): string {
  const parts: string[] = [];
  for (const field of IDENTITY_FIELDS) {
    const value = condition[field];
    if (value === undefined || value === null) continue;
    parts.push(`${field}=${String(value)}`);
  }
  return parts.join('|');
}

/**
 * The same normalisation the server applies before it accepts a citation
 * (`_normalize_ws` in `scanner_translator.py`): fold accents, lowercase, and
 * collapse every whitespace run. The two MUST agree — the server decides whether
 * a fragment is really in the text, this decides whether it still is.
 */
export function normalizePhrase(text: string): string {
  return text
    .normalize('NFKD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .join(' ');
}

/** One condition the user took off the board, and the words that had produced it. */
export interface Removal {
  key: string;
  /** `null` when the server could not verify an origin — see `isSuppressed`. */
  phrase: string | null;
}

export interface ManualState {
  removed: Removal[];
  /** Keyed by the identity of the condition M.I.A proposes, valued by the user's version. */
  edited: Record<string, ScanCondition>;
  /** Conditions the user picked from the palette; they belong to no sentence. */
  added: ScanCondition[];
}

export const EMPTY_MANUAL_STATE: ManualState = { removed: [], edited: {}, added: [] };

/**
 * Forget every removal whose originating fragment has left the text.
 *
 * This is what stops a removal from becoming permanent. A removal with no
 * verified origin (`phrase === null`) cannot be released this way — there is no
 * fragment to watch — so it is only cleared when the text is emptied entirely.
 * That is a deliberate, visible degradation: the user can always put the
 * condition back through the palette picker, and we would rather under-propose
 * than silently re-add something they removed.
 */
export function purgeRemovals(removed: Removal[], text: string): Removal[] {
  const normalized = normalizePhrase(text);
  if (!normalized) return [];
  return removed.filter((entry) => {
    if (entry.phrase === null) return true;
    return normalized.includes(normalizePhrase(entry.phrase));
  });
}

/**
 * Is a freshly proposed condition one the user already took off the board?
 *
 * Matching is on the PAIR (condition, originating fragment), which is what lets
 * a re-description win: same condition derived from different words is a new
 * intent, not the removed one. Two cases fall back to matching on the condition
 * alone, both erring towards respecting the removal:
 *   · the remembered removal has no verified origin, or
 *   · this proposal has no verified origin (the model simply did not cite it
 *     this time — that is not evidence the user re-described anything).
 */
export function isSuppressed(
  removed: readonly Removal[],
  key: string,
  phrase: string | null,
): boolean {
  return removed.some((entry) => {
    if (entry.key !== key) return false;
    if (entry.phrase === null || phrase === null) return true;
    return normalizePhrase(entry.phrase) === normalizePhrase(phrase);
  });
}

export interface ReconciledCondition {
  condition: ScanCondition;
  /**
   * Identity of M.I.A's ORIGINAL proposal — the handle an edit or a removal is
   * filed under. It must not be recomputed from `condition`: once the user has
   * changed a control, that key no longer matches what M.I.A keeps proposing,
   * and the edit would be lost on the next re-translation.
   */
  proposalKey: string;
  /** The fragment this came from, or `null` for a user-added / unattributed one. */
  phrase: string | null;
  /** True when the user picked it from the palette rather than writing it. */
  userAdded: boolean;
  /** True when the user changed a control on M.I.A's proposal. */
  userEdited: boolean;
}

/**
 * The list actually shown: M.I.A's current proposal minus what the user removed,
 * with the user's own edits substituted in, plus what the user added by hand.
 *
 * Order is stable and meaningful — proposals in the order they were understood,
 * then the user's own additions — so chips do not jump around between two
 * keystrokes.
 */
export function reconcile(
  proposed: readonly ScanCondition[],
  sources: readonly (string | null)[],
  manual: ManualState,
): ReconciledCondition[] {
  const out: ReconciledCondition[] = [];
  const seen = new Set<string>();

  proposed.forEach((condition, index) => {
    const key = conditionKey(condition);
    const phrase = sources[index] ?? null;
    if (isSuppressed(manual.removed, key, phrase)) return;
    const edited = manual.edited[key];
    const effective = edited ?? condition;
    // An edit can collide with another proposal (the user edits A into B while
    // M.I.A also proposes B). Show it once — a duplicated chip is a bug, and a
    // duplicated condition would double-count in the combo caption.
    const effectiveKey = conditionKey(effective);
    if (seen.has(effectiveKey)) return;
    seen.add(effectiveKey);
    out.push({
      condition: effective,
      proposalKey: key,
      phrase,
      userAdded: false,
      userEdited: edited !== undefined,
    });
  });

  for (const condition of manual.added) {
    const key = conditionKey(condition);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ condition, proposalKey: key, phrase: null, userAdded: true, userEdited: false });
  }

  return out;
}

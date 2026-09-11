'use client';

import * as React from 'react';
import type { ScanCondition } from '@/lib/conditions/types';
import {
  translateStrategy,
  TranslateRateLimitedError,
  TranslateUnavailableError,
  type TranslateAssumption,
  type TranslateRefusal,
  type TranslateResult,
  type TranslateUntranslatable,
} from './translate-client';

/**
 * SC-4 — translate WHILE the user types, without turning every keystroke into a
 * paid LLM call.
 *
 * Debouncing alone is not enough. Measured on the real endpoint, one translation
 * costs ~2 860 input + ~217 output tokens (~$0.0039 on Haiku 4.5, which cannot
 * prompt-cache a prefix this short — its minimum cacheable prefix is 4 096
 * tokens and ours is 2 838, so the prefix is repaid in full every time). A plain
 * 450 ms debounce fires 8-15 times while someone composes a sentence; the gates
 * below bring that to 4-7, which is the difference between ×10 and ×5 on the
 * bill.
 *
 * All five gates are evaluated BEFORE the request is issued, on purpose:
 * aborting a `fetch` does NOT stop the LLM call the server already started, so
 * cancellation saves latency and confusion but never a cent.
 *
 * The gates, all of which must pass:
 *   1. a pause  — no keystroke for DEBOUNCE_MS
 *   2. a boundary — the text ends on a completed word or punctuation, never
 *      mid-word (translating "jamais tes" would report the fragment as
 *      untranslatable and read as the product failing)
 *   3. enough text — real strategy sentences run 41-84 characters; under
 *      MIN_CHARS / MIN_WORDS there is nothing to translate yet
 *   4. enough change — MIN_DELTA characters since the text we actually
 *      translated last, which is the gate that removes the near-duplicate burst
 *   5. not already translated — the exact same text is never sent twice
 *
 * Plus one thing the gates must not be allowed to break: a SETTLE timer that
 * always translates the FULL text once typing stops, even when gates 3 and 4
 * would refuse it. Without it, adding five characters and stopping would leave
 * the last idea silently unread — an interface that lies about what it has seen.
 *
 * Honesty blocks ("ce que j'ai supposé" / "ce que je n'ai pas pu traduire") are
 * only published from a SETTLED read. Mid-typing they are withheld rather than
 * shown stale or shown about a half-written clause — the indicator says the
 * reading is still in progress, which is true.
 */

/** No keystroke for this long before a live translation may fire. */
export const DEBOUNCE_MS = 450;
/** No keystroke for this long ⇒ translate the full text whatever the gates say. */
export const SETTLE_MS = 1200;
/** Below this, there is no sentence yet. */
export const MIN_CHARS = 25;
export const MIN_WORDS = 4;
/** Characters that must have changed since the last text we really translated. */
export const MIN_DELTA = 12;
/**
 * Per-tab ceiling. A client-side cap protects nobody determined (it comes off
 * with the browser console) — the real cap is the 429 on the endpoint. This one
 * exists so an ordinary accident (a page left open under a stuck key) is not
 * billed as a session.
 */
export const SESSION_CAP = 25;

const WORD_BOUNDARY = /[\s.,;:!?)\]}'"»…-]$/u;

/** Where the last completed word ends, so a fragment is never sent. */
export function truncateToWordBoundary(text: string): string {
  const trimmed = text.trimEnd();
  if (!trimmed) return '';
  if (WORD_BOUNDARY.test(text)) return trimmed;
  const cut = Math.max(
    trimmed.lastIndexOf(' '),
    trimmed.lastIndexOf('\n'),
    trimmed.lastIndexOf(','),
    trimmed.lastIndexOf('.'),
  );
  return cut > 0 ? trimmed.slice(0, cut).trimEnd() : '';
}

function wordCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

/** How many characters differ, counted from the ends inwards (cheap and enough). */
export function charDelta(a: string, b: string): number {
  if (a === b) return 0;
  let head = 0;
  const max = Math.min(a.length, b.length);
  while (head < max && a[head] === b[head]) head += 1;
  let tail = 0;
  while (tail < max - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail += 1;
  return Math.max(a.length, b.length) - head - tail;
}

/** True when `candidate` deserves a paid call, given what we last translated. */
export function passesGates(candidate: string, lastTranslated: string): boolean {
  if (!candidate || candidate === lastTranslated) return false;
  if (candidate.length < MIN_CHARS) return false;
  if (wordCount(candidate) < MIN_WORDS) return false;
  return charDelta(candidate, lastTranslated) >= MIN_DELTA;
}

export type LiveStatus =
  | 'idle' // nothing typed worth reading yet
  | 'reading' // a translation is in flight
  | 'read' // the current text has been read through
  | 'capped'; // this tab has spent its ceiling; explicit clicks only

export interface LiveTranslation {
  status: LiveStatus;
  /** The exact text the visible result was produced from. */
  translatedText: string;
  conditions: ScanCondition[];
  sources: (string | null)[];
  refusal: TranslateRefusal | null;
  /** Published only from a settled read — `null` while the sentence is in flight. */
  assumptions: TranslateAssumption[] | null;
  untranslatable: TranslateUntranslatable[] | null;
  /**
   * Set only when the user asked for a translation explicitly and it failed.
   * A live translation that fails is deliberately SILENT: the circuit breaker
   * opens after 3 failures for 60 s, and with 4-7 calls per session that would
   * otherwise blank the panel mid-sentence. We keep the last good reading.
   */
  error: 'unavailable' | 'rateLimited' | 'failed' | null;
  /** Paid calls issued by this tab, for the cap and for the audit. */
  callCount: number;
}

const IDLE: LiveTranslation = {
  status: 'idle',
  translatedText: '',
  conditions: [],
  sources: [],
  refusal: null,
  assumptions: null,
  untranslatable: null,
  error: null,
  callCount: 0,
};

export interface UseLiveTranslation {
  live: LiveTranslation;
  /** Translate the current text now, bypassing gates 3-5 (never the cap). */
  translateNow(): void;
  /** Drop the reading and the counters — used when the user clears the field. */
  reset(): void;
}

export function useLiveTranslation(text: string, locale: string): UseLiveTranslation {
  const [live, setLive] = React.useState<LiveTranslation>(IDLE);

  // Everything the effect needs but must not re-run for.
  const lastTranslatedRef = React.useRef('');
  // Text of the call currently in flight. Without it, the settle timer firing
  // while the quick call has not yet resolved would issue the SAME request a
  // second time — `lastTranslatedRef` is only set once a response comes back,
  // and each duplicate is a paid call.
  const inFlightRef = React.useRef('');
  const seqRef = React.useRef(0);
  const abortRef = React.useRef<AbortController | null>(null);
  const callCountRef = React.useRef(0);
  const localeRef = React.useRef(locale);
  localeRef.current = locale;

  const run = React.useCallback(
    async (candidate: string, settled: boolean, explicit: boolean) => {
      if (!candidate) return;
      if (candidate === lastTranslatedRef.current || candidate === inFlightRef.current) return;
      if (!explicit && callCountRef.current >= SESSION_CAP) {
        setLive((prev) => ({ ...prev, status: 'capped' }));
        return;
      }

      const seq = ++seqRef.current;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      inFlightRef.current = candidate;
      callCountRef.current += 1;
      const callCount = callCountRef.current;
      setLive((prev) => ({ ...prev, status: 'reading', error: null, callCount }));

      let result: TranslateResult;
      try {
        result = await translateStrategy(candidate, localeRef.current, {
          signal: controller.signal,
          // A live read that takes 20 s is useless — the user has typed on.
          timeoutMs: settled || explicit ? 20_000 : 8_000,
        });
      } catch (err) {
        if (inFlightRef.current === candidate) inFlightRef.current = '';
        // A newer read has already superseded this one: say nothing.
        if (seq !== seqRef.current) return;
        const code =
          err instanceof TranslateRateLimitedError
            ? 'rateLimited'
            : err instanceof TranslateUnavailableError
              ? 'unavailable'
              : 'failed';
        setLive((prev) => ({
          ...prev,
          // Keep the last good reading on screen; only an explicit ask reports.
          status: prev.translatedText ? 'read' : 'idle',
          error: explicit || code === 'rateLimited' ? code : null,
        }));
        return;
      }
      if (inFlightRef.current === candidate) inFlightRef.current = '';
      if (seq !== seqRef.current) return;

      lastTranslatedRef.current = candidate;

      if (result.outcome === 'refused' && result.refusal) {
        setLive((prev) => ({
          ...prev,
          status: 'read',
          translatedText: candidate,
          conditions: [],
          sources: [],
          refusal: result.refusal,
          assumptions: null,
          untranslatable: null,
          error: null,
        }));
        return;
      }
      if (result.outcome === 'error' || result.outcome === 'empty') {
        setLive((prev) => ({
          ...prev,
          status: prev.translatedText ? 'read' : 'idle',
          error: explicit ? 'failed' : null,
        }));
        return;
      }

      const sources = result.condition_sources ?? [];
      setLive({
        status: 'read',
        translatedText: candidate,
        conditions: result.conditions,
        // Tolerate a short/absent array: an unattributed condition is `null`.
        sources: result.conditions.map((_, i) => sources[i] ?? null),
        refusal: null,
        // Withheld until the sentence stops moving — never shown about a
        // half-written clause, never shown stale.
        assumptions: settled || explicit ? result.assumptions : null,
        untranslatable: settled || explicit ? result.untranslatable : null,
        error: null,
        callCount,
      });
    },
    [],
  );

  React.useEffect(() => {
    const trimmed = text.trim();
    if (!trimmed) {
      // The field is empty: forget everything, including the removals the
      // caller keys off `translatedText`.
      seqRef.current += 1;
      abortRef.current?.abort();
      lastTranslatedRef.current = '';
      inFlightRef.current = '';
      setLive((prev) => (prev.status === 'idle' && !prev.translatedText ? prev : { ...IDLE, callCount: prev.callCount }));
      return;
    }

    const quick = window.setTimeout(() => {
      const candidate = truncateToWordBoundary(text);
      if (passesGates(candidate, lastTranslatedRef.current)) {
        void run(candidate, false, false);
      }
    }, DEBOUNCE_MS);

    // The safety net: whatever the gates decided, a text at rest gets read once.
    const settle = window.setTimeout(() => {
      void run(trimmed, true, false);
    }, SETTLE_MS);

    return () => {
      window.clearTimeout(quick);
      window.clearTimeout(settle);
    };
  }, [text, run]);

  React.useEffect(() => () => abortRef.current?.abort(), []);

  const translateNow = React.useCallback(() => {
    void run(text.trim(), true, true);
  }, [run, text]);

  const reset = React.useCallback(() => {
    seqRef.current += 1;
    abortRef.current?.abort();
    lastTranslatedRef.current = '';
    inFlightRef.current = '';
    callCountRef.current = 0;
    setLive(IDLE);
  }, []);

  return { live, translateNow, reset };
}

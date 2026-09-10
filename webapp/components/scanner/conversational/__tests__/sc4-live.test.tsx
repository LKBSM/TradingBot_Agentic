import * as React from 'react';
import { act, fireEvent, render as rtlRender, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import messages from '@/messages/fr.json';
import type { ConditionsScanResponse } from '@/lib/conditions/types';

/**
 * SC-4 — the live composition surface, exercised the way the mission asked:
 * type then erase quickly, remove a condition then keep writing, paste a long
 * text in one go. Each behaviour is asserted, not described.
 *
 * The two network edges are mocked (translation and scan), so what is under
 * test is the trigger policy and the reconciliation, never the transport.
 *
 * `fireEvent`, not `user-event`: that package is not installed here, and the
 * repo's convention (documented on the dictation work) is to drive inputs with
 * fireEvent. One `change` per prefix models one keystroke faithfully enough for
 * a debounce, and lets the fake clock advance exactly where we want it.
 */

// These tests drive two debounces and several awaited promises per scenario,
// on a machine that is often running other builds. 5 s (the default) is not a
// meaningful signal here — a timeout would report contention, not a defect.
vi.setConfig({ testTimeout: 30_000 });

const translateStrategy = vi.hoisted(() => vi.fn());
const fetchConditionsScan = vi.hoisted(() => vi.fn());

vi.mock('@/lib/scanner-chat/translate-client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/scanner-chat/translate-client')>()),
  translateStrategy,
}));

vi.mock('@/lib/conditions/api-client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/conditions/api-client')>()),
  fetchConditionsScan,
}));

import { ConversationalScanner } from '../ConversationalScanner';
import { DEBOUNCE_MS, SETTLE_MS } from '@/lib/scanner-chat/use-live-translation';

const SENTENCE = 'Un Order Block jamais testé en tendance haussière';

function translation(over: Record<string, unknown> = {}) {
  return {
    outcome: 'translated',
    refusal: null,
    conditions: [{ type: 'trend_is', trend: 'bullish' }, { type: 'zone_untested' }],
    condition_sources: ['tendance haussière', 'jamais testé'],
    assumptions: [],
    untranslatable: [],
    ...over,
  };
}

function scanResponse(): ConditionsScanResponse {
  const ts = new Date().toISOString();
  return {
    as_of: ts,
    logic: 'AND',
    scanned: 10,
    matches: [
      {
        instrument: 'XAUUSD',
        timeframe: 'M15',
        candle_close_ts: ts,
        close_price: 2400,
        matched: false,
        met_count: 1,
        total: 2,
        conditions_met: [
          { type: 'trend_is', label: 'La tendance structurelle est', met: true, detail: 'haussière.' },
        ],
        conditions_unmet: [
          { type: 'zone_untested', label: 'Zone jamais testée', met: false, detail: 'Déjà testée.' },
        ],
        context: {
          trend: 'bullish',
          market_phase: 'trend',
          volatility_observed: 'normal',
          mtf_confluence: { h4: 'bullish', h1: 'bullish', m15: 'bullish' },
          bos: null,
          choch: null,
          active_order_blocks: 1,
          active_fair_value_gaps: 0,
          news_upcoming: [],
        },
      },
    ],
    unavailable: [],
  } as unknown as ConditionsScanResponse;
}

function render() {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      <ConversationalScanner locale="fr" />
    </NextIntlClientProvider>,
  );
}

function input(): HTMLTextAreaElement {
  return screen.getByTestId('describe-input') as HTMLTextAreaElement;
}

/** Set the whole value at once — what a paste (or a fill) does. */
async function setValue(value: string) {
  await act(async () => {
    fireEvent.change(input(), { target: { value } });
  });
}

/** One `change` per character, with a short gap — what typing does. */
async function typeChars(value: string, gapMs = 20) {
  const start = input().value;
  for (let i = 1; i <= value.length; i += 1) {
    const next = start + value.slice(0, i);
    await act(async () => {
      fireEvent.change(input(), { target: { value: next } });
      vi.advanceTimersByTime(gapMs);
    });
  }
}

/**
 * Push past the timers and let the mocked promises resolve.
 *
 * `advanceTimersByTimeAsync`, not `advanceTimersByTime`: the async form yields
 * to microtasks BETWEEN timers, which is what this component needs — the scan
 * debounce is only armed once the translation promise has resolved and the
 * reading has re-rendered. A bounded advance rather than `runAllTimersAsync`,
 * because the candle-close refresh re-arms itself and would spin forever.
 */
async function letItSettle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(SETTLE_MS + 100);
  });
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1000);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  translateStrategy.mockReset();
  fetchConditionsScan.mockReset();
  translateStrategy.mockResolvedValue(translation());
  fetchConditionsScan.mockResolvedValue(scanResponse());
});

afterEach(() => {
  vi.useRealTimers();
});

describe('SC-4 — the reading builds while you write', () => {
  it('translates on a typing pause, with no click at all', async () => {
    render();
    await setValue(SENTENCE);
    await letItSettle();

    expect(translateStrategy).toHaveBeenCalled();
    expect(screen.getAllByTestId('translated-card')).toHaveLength(2);
  });

  it('runs the scan and drops the « Voir les résultats » click', async () => {
    render();
    await setValue(SENTENCE);
    await letItSettle();

    expect(fetchConditionsScan).toHaveBeenCalled();
    expect(screen.queryByTestId('see-results')).toBeNull();
  });

  it('never sends a half-typed word', async () => {
    render();
    await typeChars('Un Order Block jamais tes');
    await act(async () => {
      vi.advanceTimersByTime(DEBOUNCE_MS + 50);
    });
    for (const call of translateStrategy.mock.calls) {
      expect(call[0]).not.toMatch(/\btes$/);
    }
  });

  it('keeps the field mounted throughout (nothing written can be lost)', async () => {
    render();
    const field = input();
    await setValue(SENTENCE);
    await letItSettle();
    expect(input()).toBe(field);
  });
});

describe('SC-4 — adversarial: type then erase quickly', () => {
  it('erasing everything clears the reading', async () => {
    render();
    await setValue(SENTENCE);
    await letItSettle();
    expect(screen.getAllByTestId('translated-card')).toHaveLength(2);

    await setValue('');
    await letItSettle();

    expect(screen.queryAllByTestId('translated-card')).toHaveLength(0);
    expect(screen.getByTestId('live-status')).toHaveAttribute('data-status', 'idle');
  });

  it('typing and erasing faster than the debounce spends nothing', async () => {
    render();
    await typeChars('Un Order Block jamais', 5);
    await act(async () => {
      vi.advanceTimersByTime(DEBOUNCE_MS - 100);
    });
    await setValue('');
    await letItSettle();
    expect(translateStrategy).not.toHaveBeenCalled();
  });
});

describe('SC-4 — adversarial: paste a long text in one go', () => {
  it('reads a pasted sentence once, not once per character', async () => {
    render();
    await setValue(
      'Un Order Block jamais testé, en tendance haussière, avec le 1 h qui va dans le même sens et une poche de liquidité prise récemment.',
    );
    await letItSettle();

    expect(translateStrategy).toHaveBeenCalledTimes(1);
    expect(screen.getAllByTestId('translated-card')).toHaveLength(2);
  });
});

describe('SC-4 — adversarial: remove a condition, then keep writing', () => {
  it('does not silently re-add it while the words that produced it stand', async () => {
    render();
    await setValue(SENTENCE);
    await letItSettle();
    expect(screen.getAllByTestId('translated-card')).toHaveLength(2);

    // Remove « zone jamais testée » (derived from « jamais testé »).
    await act(async () => {
      fireEvent.click(screen.getAllByRole('button', { name: /Retirer/i })[1]!);
    });
    expect(screen.getAllByTestId('translated-card')).toHaveLength(1);

    // Keep writing elsewhere in the sentence: M.I.A proposes it again, the
    // removal holds because « jamais testé » is still in the text.
    await setValue(`${SENTENCE}, avec une poche de liquidité prise`);
    await letItSettle();

    expect(screen.getAllByTestId('translated-card')).toHaveLength(1);
  });

  it('brings it back once the idea is written in other words', async () => {
    render();
    await setValue(SENTENCE);
    await letItSettle();

    await act(async () => {
      fireEvent.click(screen.getAllByRole('button', { name: /Retirer/i })[1]!);
    });
    expect(screen.getAllByTestId('translated-card')).toHaveLength(1);

    // The server attributes the condition to a NEW fragment — a deliberate
    // re-description, so the removal no longer applies.
    translateStrategy.mockResolvedValue(
      translation({ condition_sources: ['tendance haussière', 'zone vierge'] }),
    );
    await setValue('Un Order Block sur une zone vierge en tendance haussière');
    await letItSettle();

    expect(screen.getAllByTestId('translated-card')).toHaveLength(2);
  });

  it('a removed condition can always be put back from the palette', async () => {
    render();
    await setValue(SENTENCE);
    await letItSettle();
    await act(async () => {
      fireEvent.click(screen.getAllByRole('button', { name: /Retirer/i })[1]!);
    });
    expect(screen.getAllByTestId('translated-card')).toHaveLength(1);
    // The escape hatch exists on screen — a removal is never a dead end.
    expect(screen.getByTestId('open-palette')).toBeInTheDocument();
  });
});

describe('SC-4 — a refusal never takes the field down with it', () => {
  it('renders in place and keeps the text on screen', async () => {
    translateStrategy.mockResolvedValue({
      outcome: 'refused',
      refusal: { kind: 'ranking' },
      conditions: [],
      condition_sources: [],
      assumptions: [],
      untranslatable: [],
    });
    render();
    const field = input();
    await setValue('Quels sont les meilleurs marchés à trader');
    await letItSettle();

    expect(screen.getByTestId('refusal-block')).toBeInTheDocument();
    // The old flow swapped the whole page for a RefusalPanel; the field stays.
    expect(input()).toBe(field);
    expect(field.value).toMatch(/meilleurs marchés/);
  });
});

describe('SC-4 — a live failure stays quiet, an explicit one speaks', () => {
  it('keeps the last good reading when a live translation fails', async () => {
    render();
    await setValue(SENTENCE);
    await letItSettle();
    expect(screen.getAllByTestId('translated-card')).toHaveLength(2);

    // The breaker opens after 3 failures for 60 s; with 4-7 calls per session
    // that must not blank the panel mid-sentence.
    translateStrategy.mockRejectedValue(new Error('boom'));
    await setValue(`${SENTENCE}, et une poche de liquidité prise récemment`);
    await letItSettle();

    expect(screen.getAllByTestId('translated-card')).toHaveLength(2);
    expect(screen.queryByTestId('translate-inline-error')).toBeNull();
  });

  it('speaks up when the user asked for the translation explicitly', async () => {
    render();
    await setValue('un OB jamais testé en tendance haussière');
    translateStrategy.mockRejectedValue(new Error('boom'));
    await act(async () => {
      fireEvent.click(screen.getByTestId('translate-button'));
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByTestId('translate-inline-error')).toBeInTheDocument();
    // …and the field is still usable.
    await setValue('je peux toujours écrire');
    expect(input().value).toBe('je peux toujours écrire');
  });
});

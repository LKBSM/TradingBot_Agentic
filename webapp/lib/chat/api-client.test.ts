/**
 * DG-112 SSE event handler tests — verifies the api-client surfaces the
 * `refusal` and `forbidden_token` events emitted by the route, alongside
 * the existing `delta` / `done` / `error` events.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  askSentinel,
  ChatApiUnavailableError,
  type RefusalCategory,
} from './api-client';
import type { InsightSignalV2 } from '@/types/insight';

// Minimal InsightSignalV2 — only the fields the route validates as "object".
const FAKE_SIGNAL = {
  schema_version: '2.1.0',
  id: 't',
  instrument: 'XAUUSD',
  timeframe: 'M15',
  direction: 'BULLISH_SETUP',
  conviction_0_100: 60,
  conviction_label: 'moderate',
} as unknown as InsightSignalV2;

function makeSseStream(events: Array<{ event: string; data: unknown }>): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  const chunks = events.map(
    (e) => `event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`,
  );
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

function mockFetchOnce(stream: ReadableStream<Uint8Array> | null, status = 200): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response(stream as BodyInit | null, {
        status,
        headers: { 'content-type': 'text/event-stream' },
      }),
    ),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('askSentinel — happy path', () => {
  it('concatenates delta events and resolves with text + model', async () => {
    const stream = makeSseStream([
      { event: 'delta', data: { text: 'Bon' } },
      { event: 'delta', data: { text: 'jour.' } },
      { event: 'done', data: { stop_reason: 'end_turn', model: 'claude-haiku-4-5-20251001' } },
    ]);
    mockFetchOnce(stream);
    const chunks: string[] = [];
    const result = await askSentinel({
      signal: FAKE_SIGNAL,
      question: 'Bonjour ?',
      onDelta: (c) => chunks.push(c),
    });
    expect(result.text).toBe('Bonjour.');
    expect(chunks).toEqual(['Bon', 'jour.']);
    expect(result.model).toBe('claude-haiku-4-5-20251001');
    expect(result.refusal).toBeUndefined();
    expect(result.forbidden_token).toBeUndefined();
    expect(result.compliance_filtered).toBeUndefined();
  });
});

describe('askSentinel — DG-112 refusal event', () => {
  it('surfaces a refusal with category + language', async () => {
    const stream = makeSseStream([
      { event: 'delta', data: { text: 'Refus pédagogique...' } },
      {
        event: 'refusal',
        data: {
          category: 'prescriptive' as RefusalCategory,
          language: 'fr',
          pattern_source: '\\bachetez\\b',
        },
      },
      { event: 'done', data: { stop_reason: 'refused_by_gate', model: 'gate-refusal', compliance_filtered: true } },
    ]);
    mockFetchOnce(stream);
    const result = await askSentinel({
      signal: FAKE_SIGNAL,
      question: 'Achetez maintenant !',
      onDelta: () => {},
    });
    expect(result.refusal).toBeDefined();
    expect(result.refusal?.category).toBe('prescriptive');
    expect(result.refusal?.language).toBe('fr');
    expect(result.compliance_filtered).toBe(true);
  });

  it('ignores refusal events with an invalid category', async () => {
    const stream = makeSseStream([
      { event: 'refusal', data: { category: 'not_a_category', language: 'fr' } },
      { event: 'done', data: { stop_reason: 'end_turn' } },
    ]);
    mockFetchOnce(stream);
    const result = await askSentinel({
      signal: FAKE_SIGNAL,
      question: 'x',
      onDelta: () => {},
    });
    expect(result.refusal).toBeUndefined();
  });
});

describe('askSentinel — DG-112 forbidden_token event', () => {
  it('surfaces a forbidden_token with safe_fallback', async () => {
    const stream = makeSseStream([
      { event: 'delta', data: { text: 'Achetez à 2400.' } },
      {
        event: 'forbidden_token',
        data: {
          token: 'achetez',
          language: 'fr',
          safe_fallback: 'Je ne peux pas répondre à cette question dans une posture prescriptive.',
        },
      },
      { event: 'done', data: { stop_reason: 'end_turn', compliance_filtered: true } },
    ]);
    mockFetchOnce(stream);
    const result = await askSentinel({
      signal: FAKE_SIGNAL,
      question: 'x',
      onDelta: () => {},
    });
    expect(result.forbidden_token).toBeDefined();
    expect(result.forbidden_token?.token).toBe('achetez');
    expect(result.forbidden_token?.safe_fallback).toContain('prescriptive');
    expect(result.compliance_filtered).toBe(true);
  });

  it('ignores forbidden_token events missing safe_fallback', async () => {
    const stream = makeSseStream([
      { event: 'forbidden_token', data: { token: 'x', language: 'fr' } },
      { event: 'done', data: { stop_reason: 'end_turn' } },
    ]);
    mockFetchOnce(stream);
    const result = await askSentinel({
      signal: FAKE_SIGNAL,
      question: 'x',
      onDelta: () => {},
    });
    expect(result.forbidden_token).toBeUndefined();
  });
});

describe('askSentinel — error paths', () => {
  it('503 → ChatApiUnavailableError', async () => {
    mockFetchOnce(null, 503);
    await expect(
      askSentinel({ signal: FAKE_SIGNAL, question: 'x', onDelta: () => {} }),
    ).rejects.toBeInstanceOf(ChatApiUnavailableError);
  });

  it('terminal error event throws the message', async () => {
    const stream = makeSseStream([
      { event: 'error', data: { message: 'LLM exploded' } },
    ]);
    mockFetchOnce(stream);
    await expect(
      askSentinel({ signal: FAKE_SIGNAL, question: 'x', onDelta: () => {} }),
    ).rejects.toThrow('LLM exploded');
  });
});

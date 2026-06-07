import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  askSentinel,
  ChatApiError,
  ChatApiUnavailableError,
  type ConversationMessage,
} from '@/lib/chat/api-client';
import type { InsightSignalV2 } from '@/types/insight';

/**
 * Unit tests for the Chantier 5.A api-client → backend FastAPI (Chantier 4).
 * `fetch` is fully mocked — no real network call is made.
 */

// Minimal signal stub: askSentinel only reads instrument + timeframe for the
// Tension-T1 context preamble. Cast keeps the test free of the full heavy type.
const SIGNAL = { instrument: 'XAUUSD', timeframe: 'H1' } as InsightSignalV2;

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

/** Last body passed to fetch, parsed. */
function lastRequestBody(mock: ReturnType<typeof vi.fn>): {
  user_message: string;
  conversation_history: ConversationMessage[];
} {
  const init = mock.mock.calls.at(-1)?.[1] as RequestInit;
  return JSON.parse(init.body as string);
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('askSentinel — happy path', () => {
  it('POSTs to /api/chatbot/message and parses the JSON response', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(200, {
        content: 'Le marché XAUUSD est en phase de consolidation.',
        blocked_reason: null,
        tool_calls_made: [{ name: 'get_market_reading', input: { instrument: 'XAUUSD' } }],
      }),
    );

    const result = await askSentinel({ question: 'Décris la structure.' });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/api/chatbot/message');
    expect(init.method).toBe('POST');
    expect(result.text).toBe('Le marché XAUUSD est en phase de consolidation.');
    expect(result.blockedReason).toBeNull();
    expect(result.toolCallsMade).toHaveLength(1);
  });

  it('omits the context preamble when no signal is provided', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(200, { content: 'ok', blocked_reason: null, tool_calls_made: [] }),
    );

    await askSentinel({ question: 'Bonjour' });

    expect(lastRequestBody(fetchMock).user_message).toBe('Bonjour');
  });

  it('prepends the stable [Lecture en cours] preamble when a signal is provided', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(200, { content: 'ok', blocked_reason: null, tool_calls_made: [] }),
    );

    await askSentinel({ question: 'Quelle conviction ?', signal: SIGNAL });

    expect(lastRequestBody(fetchMock).user_message).toBe(
      '[Lecture en cours : XAUUSD H1]\nQuelle conviction ?',
    );
  });

  it('transmits conversation_history in {role, content} shape', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(200, { content: 'ok', blocked_reason: null, tool_calls_made: [] }),
    );

    const history: ConversationMessage[] = [
      { role: 'user', content: 'Première question' },
      { role: 'assistant', content: 'Première réponse' },
    ];
    await askSentinel({ question: 'Suite', history });

    expect(lastRequestBody(fetchMock).conversation_history).toEqual(history);
  });
});

describe('askSentinel — blocked_reason', () => {
  it('returns the refusal content + the blocked reason when a layer fires', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(200, {
        content: 'Je décris les conditions du marché. La décision d’agir t’appartient.',
        blocked_reason: 'trade_request',
        tool_calls_made: [],
      }),
    );

    const result = await askSentinel({ question: 'Dois-je acheter ?' });

    expect(result.blockedReason).toBe('trade_request');
    expect(result.text).toContain('La décision');
  });
});

describe('askSentinel — error handling', () => {
  it('throws ChatApiUnavailableError on HTTP 503', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(503, { detail: 'Chatbot service not configured' }),
    );

    await expect(askSentinel({ question: 'x' })).rejects.toBeInstanceOf(
      ChatApiUnavailableError,
    );
  });

  it('throws ChatApiError(422) on Pydantic validation failure', async () => {
    fetchMock.mockResolvedValue(jsonResponse(422, { detail: 'too long' }));

    const err = await askSentinel({ question: 'x' }).catch((e) => e);
    expect(err).toBeInstanceOf(ChatApiError);
    expect((err as ChatApiError).status).toBe(422);
  });

  it('throws ChatApiError(500) with a generic message — no server leak', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(500, { detail: 'Traceback secret: KeyError at line 42' }),
    );

    const err = await askSentinel({ question: 'x' }).catch((e) => e);
    expect(err).toBeInstanceOf(ChatApiError);
    expect((err as ChatApiError).status).toBe(500);
    expect((err as ChatApiError).message).not.toContain('Traceback');
    expect((err as ChatApiError).message).not.toContain('KeyError');
  });

  it('throws ChatApiError(0) on a network failure', async () => {
    fetchMock.mockRejectedValue(new Error('Failed to fetch'));

    const err = await askSentinel({ question: 'x' }).catch((e) => e);
    expect(err).toBeInstanceOf(ChatApiError);
    expect((err as ChatApiError).status).toBe(0);
  });

  it('throws ChatApiError on a malformed (content-less) 200 body', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(200, { blocked_reason: null, tool_calls_made: [] }),
    );

    await expect(askSentinel({ question: 'x' })).rejects.toBeInstanceOf(ChatApiError);
  });
});

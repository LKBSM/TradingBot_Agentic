import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  translateStrategy,
  TranslateUnavailableError,
  TranslateError,
} from '../translate-client';

/**
 * SC-2 — the translate client. It only ever renders what the SERVER validated:
 * the server re-checks each condition against the closed palette before replying,
 * so this client trusts the shape but never invents. These tests pin the wire
 * contract, the locale narrowing, and the honest error mapping.
 */

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('translateStrategy', () => {
  it('POSTs text + narrowed locale to /api/scanner/translate', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ outcome: 'translated', refusal: null, conditions: [], assumptions: [], untranslatable: [] }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await translateStrategy('un OB jamais testé', 'fr-CA');

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe('/api/scanner/translate');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.text).toBe('un OB jamais testé');
    // Locale is narrowed to fr/en (fr-CA → fr).
    expect(body.locale).toBe('fr');
  });

  it('narrows an English locale to en', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ outcome: 'none', refusal: null, conditions: [], assumptions: [], untranslatable: [] }),
    );
    vi.stubGlobal('fetch', fetchMock);
    await translateStrategy('anything', 'en-US');
    const body = JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string);
    expect(body.locale).toBe('en');
  });

  it('returns the validated translation verbatim', async () => {
    const payload = {
      outcome: 'partial',
      refusal: null,
      conditions: [{ type: 'zone_untested' }],
      assumptions: [{ condition_type: 'zone_untested', control: 'zone_kind', value: 'any', source_phrase: null }],
      untranslatable: [{ fragment: 'RSI', category: 'indicator' }],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(payload)));
    const result = await translateStrategy('OB vierge quand RSI bas', 'fr');
    expect(result.outcome).toBe('partial');
    expect(result.conditions).toEqual([{ type: 'zone_untested' }]);
    expect(result.untranslatable[0]!.category).toBe('indicator');
  });

  it('maps 503 to TranslateUnavailableError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'nope' }, 503)));
    await expect(translateStrategy('x', 'fr')).rejects.toBeInstanceOf(TranslateUnavailableError);
  });

  it('maps a 500 to TranslateError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'boom' }, 500)));
    await expect(translateStrategy('x', 'fr')).rejects.toBeInstanceOf(TranslateError);
  });

  it('maps a network failure to TranslateError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    await expect(translateStrategy('x', 'fr')).rejects.toBeInstanceOf(TranslateError);
  });

  it('rejects a malformed body', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ nope: true })));
    await expect(translateStrategy('x', 'fr')).rejects.toBeInstanceOf(TranslateError);
  });
});

import { afterEach, describe, expect, it, vi } from 'vitest';
import { CandlesError, fetchCandles } from '../api-client';

/** Build a minimal Response-like object for the stubbed fetch. */
function jsonResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

const SAMPLE = {
  instrument: 'XAUUSD',
  timeframe: 'M15',
  candles: [
    { time: 1716724800, open: 2378, high: 2380, low: 2376, close: 2379, volume: 100 },
    { time: 1716725700, open: 2379, high: 2381, low: 2378, close: 2380, volume: 110 },
  ],
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('fetchCandles', () => {
  it('returns the candle array on 200 and builds the right URL', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(SAMPLE));
    vi.stubGlobal('fetch', fetchMock);

    const candles = await fetchCandles('XAUUSD', 'M15');

    expect(candles).toHaveLength(2);
    expect(candles[0]!.time).toBe(1716724800);
    const url = String(fetchMock.mock.calls[0]?.[0]);
    expect(url).toContain('/api/candles');
    expect(url).toContain('instrument=XAUUSD');
    expect(url).toContain('timeframe=M15');
    expect(url).toContain('limit=200');
  });

  it('throws CandlesError with the status on 404 (no candles cached)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ detail: 'No candles cached yet' }, 404)),
    );
    const err = await fetchCandles('EURUSD', 'H4').catch((e) => e);
    expect(err).toBeInstanceOf(CandlesError);
    expect(err.status).toBe(404);
  });

  it('throws CandlesError on a malformed envelope', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ instrument: 'XAUUSD' }, 200)),
    );
    await expect(fetchCandles('XAUUSD', 'M15')).rejects.toBeInstanceOf(CandlesError);
  });

  it('throws a transport CandlesError (status 0) on network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('boom')));
    const err = await fetchCandles('XAUUSD', 'M15').catch((e) => e);
    expect(err).toBeInstanceOf(CandlesError);
    expect(err.status).toBe(0);
  });
});

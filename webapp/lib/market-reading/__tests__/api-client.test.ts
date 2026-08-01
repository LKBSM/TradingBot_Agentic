import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  fetchMarketReading,
  MarketReadingError,
  MarketReadingNoDataError,
  MarketReadingNotAvailableError,
  MarketReadingValidationError,
} from '../api-client';
import { FIXTURE_XAU_M15 } from '../fixtures';

/** Build a minimal Response-like object for the stubbed fetch. */
function jsonResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('fetchMarketReading', () => {
  it('returns a parsed MarketReading on 200 and builds the right URL', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(FIXTURE_XAU_M15));
    vi.stubGlobal('fetch', fetchMock);

    const reading = await fetchMarketReading('XAUUSD', 'M15');

    expect(reading.header.instrument).toBe('XAUUSD');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = String(fetchMock.mock.calls[0]?.[0]);
    expect(url).toContain('/api/market-reading');
    expect(url).toContain('instrument=XAUUSD');
    expect(url).toContain('timeframe=M15');
  });

  it('throws MarketReadingValidationError on 400 with the backend detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: "Unsupported instrument 'ZZZ'." }, 400),
      ),
    );
    await expect(fetchMarketReading('ZZZ', 'M15')).rejects.toBeInstanceOf(
      MarketReadingValidationError,
    );
  });

  it('throws MarketReadingNotAvailableError on 503', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: 'MarketReading service not configured' }, 503),
      ),
    );
    const err = await fetchMarketReading('XAUUSD', 'M15').catch((e) => e);
    expect(err).toBeInstanceOf(MarketReadingNotAvailableError);
    expect(err.code).toBe('market_reading_unavailable');
  });

  it('throws MarketReadingError(500) on an internal error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ detail: 'boom' }, 500)),
    );
    const err = await fetchMarketReading('XAUUSD', 'M15').catch((e) => e);
    expect(err).toBeInstanceOf(MarketReadingError);
    expect(err.status).toBe(500);
    // Never leaks server internals.
    expect(err.message).not.toContain('boom');
  });

  it('throws MarketReadingError on a malformed body', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ not: 'a reading' }, 200)),
    );
    await expect(fetchMarketReading('XAUUSD', 'M15')).rejects.toThrow(
      /malformée/,
    );
  });

  it('retries once on a transient network error, then succeeds', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('network down'))
      .mockResolvedValueOnce(jsonResponse(FIXTURE_XAU_M15));
    vi.stubGlobal('fetch', fetchMock);

    const reading = await fetchMarketReading('XAUUSD', 'M15');
    expect(reading.header.instrument).toBe('XAUUSD');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('gives up after one retry on a persistent network error', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('network down'));
    vi.stubGlobal('fetch', fetchMock);

    const err = await fetchMarketReading('XAUUSD', 'M15').catch((e) => e);
    expect(err).toBeInstanceOf(MarketReadingError);
    expect(err.status).toBe(0);
    expect(fetchMock).toHaveBeenCalledTimes(2); // initial + 1 retry
  });

  it('does not retry a deterministic 400', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}, 400));
    vi.stubGlobal('fetch', fetchMock);

    await fetchMarketReading('ZZZ', 'M15').catch(() => {});
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  // ── PERF-1: honest error taxonomy ──────────────────────────────────────────

  it('throws MarketReadingNoDataError on 404 (valid combo, no data yet)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: 'No market data available yet.' }, 404),
      ),
    );
    const err = await fetchMarketReading('XAUUSD', 'M15').catch((e) => e);
    expect(err).toBeInstanceOf(MarketReadingNoDataError);
    expect(err.status).toBe(404);
  });

  it('does not retry a deterministic 404', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}, 404));
    vi.stubGlobal('fetch', fetchMock);
    await fetchMarketReading('XAUUSD', 'M15').catch(() => {});
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('tags a network failure with reason "network"', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('offline')));
    const err = await fetchMarketReading('XAUUSD', 'M15').catch((e) => e);
    expect(err).toBeInstanceOf(MarketReadingError);
    expect(err.reason).toBe('network');
  });

  it('tags a timeout with reason "timeout" and does NOT retry it', async () => {
    vi.useFakeTimers();
    try {
      // Never resolves on its own — only the internal timeout can abort it.
      const fetchMock = vi.fn().mockImplementation(
        (_url: string, opts: { signal: AbortSignal }) =>
          new Promise((_resolve, reject) => {
            opts.signal.addEventListener('abort', () =>
              reject(new DOMException('aborted', 'AbortError')),
            );
          }),
      );
      vi.stubGlobal('fetch', fetchMock);

      // Attach the handler synchronously so the rejection is never momentarily
      // unhandled while we advance the fake timer past the timeout.
      const p = fetchMarketReading('XAUUSD', 'M15', { timeoutMs: 8000 }).catch((e) => e);
      await vi.advanceTimersByTimeAsync(8000);
      const err = await p;

      expect(err).toBeInstanceOf(MarketReadingError);
      expect(err.reason).toBe('timeout');
      // A timeout already spent the full budget — retrying would double the wait.
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});

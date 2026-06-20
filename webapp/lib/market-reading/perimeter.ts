/**
 * V1 instrument/timeframe perimeter — a PLAIN module (no 'use client').
 *
 * These constants are imported by both server components (e.g. the /app page
 * resolving a deep-link) and client components. They must therefore live
 * outside the `'use client'` store module: a server component importing a value
 * from a client module receives a client-reference proxy, not the real array
 * (`.includes is not a function`). Keep them here, server-safe.
 */

export const SUPPORTED_INSTRUMENTS = ['XAUUSD', 'EURUSD'] as const;
export const SUPPORTED_TIMEFRAMES = ['M15', 'H1', 'H4'] as const;

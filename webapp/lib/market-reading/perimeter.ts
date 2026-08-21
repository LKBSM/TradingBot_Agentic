/**
 * V1 instrument/timeframe perimeter — a PLAIN module (no 'use client').
 *
 * These constants are imported by both server components (e.g. the /app page
 * resolving a deep-link) and client components. They must therefore live
 * outside the `'use client'` store module: a server component importing a value
 * from a client module receives a client-reference proxy, not the real array
 * (`.includes is not a function`). Keep them here, server-safe.
 */

import { ALL_MARKET_IDS } from '@/lib/markets';

// The supported market perimeter derives from the single market registry
// (MKT-1) — never a hand-copied list. Adding a market is one entry in
// config/markets.json (+ `node scripts/gen_markets.mjs`). markets.generated.ts
// is a plain data module (no 'use client'), so this stays server-safe.
export const SUPPORTED_INSTRUMENTS: readonly string[] = ALL_MARKET_IDS;

// LB-1 — six timeframes, fast units to see NOW, slow units to see FAR. This is
// the VALIDATION perimeter (a deep-link to M1 is a valid target when the gate is
// on). The DISPLAY perimeter below is what the selector renders.
export const SUPPORTED_TIMEFRAMES = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1'] as const;

// M1 ships off by default: its per-minute live refresh does not fit the data
// plan (see AUDIT-lb-1-lookback-profond). Mirror the backend LB1_ENABLE_M1 gate
// with the public build-time flag NEXT_PUBLIC_LB1_ENABLE_M1.
export const M1_ENABLED =
  process.env.NEXT_PUBLIC_LB1_ENABLE_M1 === '1' ||
  process.env.NEXT_PUBLIC_LB1_ENABLE_M1 === 'true';

// What the timeframe selector shows: every supported unit minus M1 when gated off.
export const DISPLAY_TIMEFRAMES = SUPPORTED_TIMEFRAMES.filter(
  (tf) => tf !== 'M1' || M1_ENABLED,
);

// Canonical landing combo. Gold M15 is the product's primary — pin it explicitly
// rather than trusting array[0], which now points at a fast unit (M1/M5).
export const DEFAULT_INSTRUMENT: (typeof SUPPORTED_INSTRUMENTS)[number] = 'XAUUSD';
export const DEFAULT_TIMEFRAME: (typeof SUPPORTED_TIMEFRAMES)[number] = 'M15';

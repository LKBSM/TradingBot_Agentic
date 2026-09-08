/**
 * DS-1 — Frozen sample data for the design gallery (barrel).
 *
 * All exports are REAL product data (from market_readings.db / candles.db),
 * normalised to the current TS contracts. Pure data + pure derivations — no
 * network, no secrets, no PII. Consumed by the dev-only gallery route and by the
 * presentation-layer tests. See each module header for provenance.
 */
export * from './readings';
export * from './candles';
export * from './zones';
export * from './scanner';
export * from './chat';
export * from './calendar';
export * from './palette';

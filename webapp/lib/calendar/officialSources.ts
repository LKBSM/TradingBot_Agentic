/**
 * OFFICIAL SOURCE WHITELIST (CAL-1).
 *
 * The only issuing organisms allowed to appear in production surfaces: the six
 * official statistical bodies. ForexFactory — a private aggregator, not an
 * issuing organism — is deliberately ABSENT. The product states it retains
 * official sources only; this list makes that guarantee EXPLICIT in code so a
 * source can never leak into a filter or a row implicitly.
 *
 * A guard test (see __tests__/officialSources.test.ts) fails if this list ever
 * regains a non-official source, and the calendar surfaces derive their filter
 * options from here — never from an inline literal.
 */
export const OFFICIAL_SOURCES = [
  'bls',
  'bea',
  'census',
  'federal_reserve',
  'eurostat',
  'ecb',
] as const;

export type OfficialSource = (typeof OFFICIAL_SOURCES)[number];

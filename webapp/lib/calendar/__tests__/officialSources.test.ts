import { describe, expect, it } from 'vitest';
import { OFFICIAL_SOURCES } from '../officialSources';

/**
 * CAL-1 guard — only official issuing organisms may appear in production. The
 * whitelist is explicit in code (not implied by "whatever the component lists"),
 * so a private aggregator like ForexFactory can never leak back into a filter.
 */
describe('CAL-1 official-source whitelist', () => {
  it('excludes ForexFactory (a private aggregator, not an issuing organism)', () => {
    expect(OFFICIAL_SOURCES).not.toContain('forexfactory');
  });

  it('is exactly the six official organisms, in lockstep with the backend', () => {
    expect([...OFFICIAL_SOURCES]).toEqual([
      'bls',
      'bea',
      'census',
      'federal_reserve',
      'eurostat',
      'ecb',
    ]);
  });
});

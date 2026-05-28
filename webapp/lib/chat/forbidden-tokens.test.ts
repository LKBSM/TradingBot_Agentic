import { describe, expect, it } from 'vitest';
import {
  containsForbiddenToken,
  FORBIDDEN_TOKENS_EN_PHRASES,
  FORBIDDEN_TOKENS_EN_WORDS,
  FORBIDDEN_TOKENS_FR_PHRASES,
  FORBIDDEN_TOKENS_FR_WORDS,
} from './forbidden-tokens';

describe('containsForbiddenToken — French', () => {
  it('catches "achetez maintenant"', () => {
    expect(containsForbiddenToken('Achetez maintenant.', 'fr')).toBe('achetez');
  });

  it('catches "100% sûr" as a phrase', () => {
    expect(containsForbiddenToken('Performance 100% sûr', 'fr')).toBe('100% sûr');
  });

  it('catches "opportunité à saisir"', () => {
    expect(
      containsForbiddenToken('Une opportunité à saisir maintenant.', 'fr'),
    ).toBe('opportunité à saisir');
  });

  it('catches "tp" as a standalone word', () => {
    const got = containsForbiddenToken('Place ton TP à 2400.', 'fr');
    expect(got).toBe('tp');
  });

  it('returns null on legitimate French prose with "événement"', () => {
    expect(containsForbiddenToken('Un événement à venir.', 'fr')).toBeNull();
  });

  it('returns null on empty input', () => {
    expect(containsForbiddenToken('', 'fr')).toBeNull();
  });
});

describe('containsForbiddenToken — English', () => {
  it('catches "buy signal" phrase before bare "buy"', () => {
    expect(containsForbiddenToken('This is a clear BUY signal.', 'en')).toBe(
      'buy signal',
    );
  });

  it('catches isolated "buy" with word boundary', () => {
    expect(containsForbiddenToken('You should buy now.', 'en')).toBe('buy');
  });

  it('ignores "buyer" — not a standalone "buy"', () => {
    expect(containsForbiddenToken("The buyer's market.", 'en')).toBeNull();
  });

  it('ignores "sellable" — not a standalone "sell"', () => {
    expect(containsForbiddenToken('This is sellable.', 'en')).toBeNull();
  });

  it('catches "guaranteed profit" as phrase', () => {
    expect(
      containsForbiddenToken('This is a guaranteed profit.', 'en'),
    ).toBe('guaranteed profit');
  });

  it('catches standalone "TP"', () => {
    expect(containsForbiddenToken('Set the TP at 2400.', 'en')).toBe('tp');
  });

  it('catches "proven edge"', () => {
    expect(containsForbiddenToken('We have a proven edge.', 'en')).toBe(
      'proven edge',
    );
  });
});

describe('Catalogue invariants', () => {
  it('FR word list has every entry lowercased', () => {
    for (const w of FORBIDDEN_TOKENS_FR_WORDS) {
      expect(w).toBe(w.toLowerCase());
    }
  });

  it('EN phrase list has every entry lowercased', () => {
    for (const p of FORBIDDEN_TOKENS_EN_PHRASES) {
      expect(p).toBe(p.toLowerCase());
    }
  });

  it('FR catalogue covers the brief minimum set', () => {
    const flat = [
      ...FORBIDDEN_TOKENS_FR_WORDS,
      ...FORBIDDEN_TOKENS_FR_PHRASES,
    ];
    // Must contain at least the verbs + the four core phrase categories
    expect(flat).toContain('achetez');
    expect(flat).toContain('vendez');
    expect(flat).toContain('garanti');
    expect(flat).toContain('profit garanti');
    expect(flat).toContain('opportunité à saisir');
    expect(flat).toContain('stop-loss recommandé');
  });

  it('EN catalogue covers the brief minimum set', () => {
    const flat = [
      ...FORBIDDEN_TOKENS_EN_WORDS,
      ...FORBIDDEN_TOKENS_EN_PHRASES,
    ];
    expect(flat).toContain('buy');
    expect(flat).toContain('sell');
    expect(flat).toContain('guaranteed');
    expect(flat).toContain('guaranteed profit');
    expect(flat).toContain('proven edge');
  });
});

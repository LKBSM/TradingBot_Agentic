import { describe, expect, it } from 'vitest';
import { formatTag } from './tag-labels';

describe('formatTag (vulgarisation des tags — 5.D)', () => {
  it('maps known snake_case codes to plain-language labels', () => {
    expect(formatTag('trend')).toBe('Tendance établie');
    expect(formatTag('bos_confirmed')).toBe('Cassure confirmée');
    expect(formatTag('retest_active')).toBe('Retest en cours');
    expect(formatTag('low_vol')).toBe('Volatilité basse');
    expect(formatTag('choch_pending')).toBe('Changement de caractère en attente');
  });

  it('never leaks a raw snake_case code for an unknown tag', () => {
    const out = formatTag('some_unknown_code');
    expect(out).not.toContain('_');
    expect(out).toBe('Some unknown code');
  });
});

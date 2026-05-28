import { describe, expect, it } from 'vitest';
import { DEFAULT_MODEL, modelForTier } from './system-prompt';

describe('DG-042 modelForTier — tier-routed model selection', () => {
  it('FREE → Haiku', () => {
    expect(modelForTier('FREE')).toBe('claude-haiku-4-5-20251001');
  });

  it('STARTER → Haiku', () => {
    expect(modelForTier('STARTER')).toBe('claude-haiku-4-5-20251001');
  });

  it('PRO → Sonnet', () => {
    expect(modelForTier('PRO')).toBe('claude-sonnet-4-6');
  });

  it('INSTITUTIONAL → Opus', () => {
    expect(modelForTier('INSTITUTIONAL')).toBe('claude-opus-4-7');
  });

  it('lower-case tier strings are accepted', () => {
    expect(modelForTier('pro')).toBe('claude-sonnet-4-6');
    expect(modelForTier('institutional')).toBe('claude-opus-4-7');
  });

  it('null/undefined → cost-safe Haiku fallback', () => {
    expect(modelForTier(null)).toBe(DEFAULT_MODEL);
    expect(modelForTier(undefined)).toBe(DEFAULT_MODEL);
  });

  it('unknown tier string → cost-safe Haiku fallback', () => {
    expect(modelForTier('UNKNOWN_TIER')).toBe(DEFAULT_MODEL);
    expect(modelForTier('')).toBe(DEFAULT_MODEL);
  });
});

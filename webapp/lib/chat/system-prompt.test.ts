import { describe, expect, it } from 'vitest';
import {
  DEFAULT_MODEL,
  modelForTier,
  requiresSonnetUplift,
} from './system-prompt';

describe('DG-042 modelForTier — post-pivot pricing (2026-05-28)', () => {
  it('Découverte → Haiku', () => {
    expect(modelForTier('decouverte')).toBe('claude-haiku-4-5-20251001');
    expect(modelForTier('Découverte')).toBe('claude-haiku-4-5-20251001');
  });

  it('Approfondie → Haiku', () => {
    expect(modelForTier('approfondie')).toBe('claude-haiku-4-5-20251001');
  });

  it('Intégrale default → Haiku (no uplift trigger)', () => {
    expect(modelForTier('integrale')).toBe('claude-haiku-4-5-20251001');
    expect(modelForTier('integrale', { question: 'Quel est le prix ?' })).toBe(
      'claude-haiku-4-5-20251001',
    );
  });

  it('legacy aliases — FREE/STARTER/PRO map to the new tiers', () => {
    expect(modelForTier('FREE')).toBe('claude-haiku-4-5-20251001');
    expect(modelForTier('STARTER')).toBe('claude-haiku-4-5-20251001');
    expect(modelForTier('PRO')).toBe('claude-haiku-4-5-20251001');
  });

  it('null/undefined → cost-safe Haiku fallback', () => {
    expect(modelForTier(null)).toBe(DEFAULT_MODEL);
    expect(modelForTier(undefined)).toBe(DEFAULT_MODEL);
    expect(modelForTier('')).toBe(DEFAULT_MODEL);
  });

  it('unknown tier string → cost-safe Haiku fallback', () => {
    expect(modelForTier('UNKNOWN_TIER')).toBe(DEFAULT_MODEL);
  });

  it('INSTITUTIONAL is NOT mapped in V1 (Calendly-gated B2B path) → falls back to Haiku', () => {
    // Cf. system-prompt.ts module header — INSTITUTIONAL is deliberately
    // omitted from the public tier alias map.
    expect(modelForTier('INSTITUTIONAL')).toBe(DEFAULT_MODEL);
    expect(modelForTier('institutional')).toBe(DEFAULT_MODEL);
  });
});

describe('DG-042 requiresSonnetUplift — Intégrale uplift triggers', () => {
  it('lexical trigger: "pourquoi" fires the uplift', () => {
    expect(requiresSonnetUplift({ question: 'Pourquoi la conviction à 72 ?' })).toBe(true);
  });

  it('lexical trigger: "décompose" fires', () => {
    expect(requiresSonnetUplift({ question: 'Décompose-moi le score svp.' })).toBe(true);
  });

  it('lexical trigger: "explique le score" fires', () => {
    expect(requiresSonnetUplift({ question: 'Explique le score étape par étape.' })).toBe(true);
  });

  it('lexical trigger: "breakdown" (EN) fires', () => {
    expect(requiresSonnetUplift({ question: 'Show me the breakdown.' })).toBe(true);
  });

  it('history threshold (> 3 turns) fires', () => {
    expect(requiresSonnetUplift({ question: 'ok', historyTurns: 4 })).toBe(true);
  });

  it('history at threshold (3 turns) does NOT fire', () => {
    expect(requiresSonnetUplift({ question: 'ok', historyTurns: 3 })).toBe(false);
  });

  it('benign question + short history does not fire', () => {
    expect(
      requiresSonnetUplift({ question: 'Quel est le régime HMM ?', historyTurns: 1 }),
    ).toBe(false);
  });
});

describe('DG-042 modelForTier — Intégrale Sonnet uplift', () => {
  it('Intégrale + score-decomposition question → Sonnet', () => {
    expect(
      modelForTier('integrale', { question: 'Pourquoi 72 ? Décompose le score.' }),
    ).toBe('claude-sonnet-4-6');
  });

  it('Intégrale + long conversation → Sonnet', () => {
    expect(modelForTier('integrale', { question: 'continue', historyTurns: 4 })).toBe(
      'claude-sonnet-4-6',
    );
  });

  it('Approfondie + score-decomposition does NOT uplift (uplift is Intégrale-only)', () => {
    expect(
      modelForTier('approfondie', { question: 'Décompose le score svp.' }),
    ).toBe('claude-haiku-4-5-20251001');
  });

  it('Découverte + long history does NOT uplift', () => {
    expect(modelForTier('decouverte', { historyTurns: 10 })).toBe(
      'claude-haiku-4-5-20251001',
    );
  });
});

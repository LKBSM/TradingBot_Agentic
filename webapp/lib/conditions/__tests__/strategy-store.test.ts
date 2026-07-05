import { renderHook, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  CURRENT_STRATEGY_SCHEMA_VERSION,
  MAX_STRATEGIES,
  STORAGE_KEY,
  readStrategies,
  useSavedStrategies,
  validateStrategy,
  type SavedStrategy,
} from '../strategy-store';
import type { ConditionsConfig } from '../types';

const SAMPLE_CONFIG: ConditionsConfig = {
  logic: 'AND',
  conditions: [
    { type: 'mtf_aligned', direction: 'bullish' },
    { type: 'bos_recent_confirmed', direction: 'any', max_bars: 5 },
  ],
};

function makeStrategy(overrides: Partial<SavedStrategy> = {}): SavedStrategy {
  return {
    id: 'test-id',
    name: 'London sweep M15',
    schema_version: CURRENT_STRATEGY_SCHEMA_VERSION,
    config: SAMPLE_CONFIG,
    createdAt: 1,
    lastUsedAt: 1,
    ...overrides,
  };
}

beforeEach(() => {
  window.localStorage.clear();
});
afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
});

describe('useSavedStrategies — save / reload fidelity', () => {
  it('round-trips a named strategy byte-for-byte through localStorage', () => {
    const { result } = renderHook(() => useSavedStrategies());
    act(() => {
      const r = result.current.saveStrategy('London sweep M15', SAMPLE_CONFIG);
      expect(r.ok).toBe(true);
    });
    // A fresh mount reads the exact same conditions back.
    const remount = renderHook(() => useSavedStrategies());
    expect(remount.result.current.strategies).toHaveLength(1);
    const restored = remount.result.current.strategies[0]!;
    expect(restored.name).toBe('London sweep M15');
    expect(restored.config).toEqual(SAMPLE_CONFIG);
    expect(restored.schema_version).toBe(CURRENT_STRATEGY_SCHEMA_VERSION);
    expect(validateStrategy(restored)).toEqual([]);
  });

  it('re-saving under the same name (case-insensitive) updates in place', () => {
    const { result } = renderHook(() => useSavedStrategies());
    act(() => {
      result.current.saveStrategy('London sweep M15', SAMPLE_CONFIG);
    });
    const firstId = result.current.strategies[0]!.id;
    const edited: ConditionsConfig = { logic: 'OR', conditions: SAMPLE_CONFIG.conditions };
    act(() => {
      result.current.saveStrategy('london SWEEP m15', edited);
    });
    expect(result.current.strategies).toHaveLength(1);
    expect(result.current.strategies[0]!.id).toBe(firstId);
    expect(result.current.strategies[0]!.config.logic).toBe('OR');
  });

  it('sorts the list most-recently-used first and markUsed reorders it', () => {
    // Monotonic clock so two saves in the same millisecond cannot tie.
    let t = 1_000;
    vi.spyOn(Date, 'now').mockImplementation(() => ++t);
    const { result } = renderHook(() => useSavedStrategies());
    act(() => {
      result.current.saveStrategy('A', SAMPLE_CONFIG);
    });
    act(() => {
      result.current.saveStrategy('B', SAMPLE_CONFIG);
    });
    expect(result.current.strategies.map((s) => s.name)).toEqual(['B', 'A']);
    const idA = result.current.strategies[1]!.id;
    act(() => {
      result.current.markUsed(idA);
    });
    expect(result.current.strategies.map((s) => s.name)).toEqual(['A', 'B']);
  });
});

describe('useSavedStrategies — rename / duplicate / delete', () => {
  it('renames without touching the conditions', () => {
    const { result } = renderHook(() => useSavedStrategies());
    act(() => {
      result.current.saveStrategy('Old name', SAMPLE_CONFIG);
    });
    const id = result.current.strategies[0]!.id;
    act(() => {
      const r = result.current.renameStrategy(id, 'Continuation H4');
      expect(r.ok).toBe(true);
    });
    expect(result.current.strategies[0]!.name).toBe('Continuation H4');
    expect(result.current.strategies[0]!.config).toEqual(SAMPLE_CONFIG);
  });

  it('duplicates with a new id and an independent config copy', () => {
    const { result } = renderHook(() => useSavedStrategies());
    act(() => {
      result.current.saveStrategy('Setup', SAMPLE_CONFIG);
    });
    const original = result.current.strategies[0]!;
    act(() => {
      const r = result.current.duplicateStrategy(original.id);
      expect(r.ok).toBe(true);
    });
    expect(result.current.strategies).toHaveLength(2);
    const copy = result.current.strategies.find((s) => s.id !== original.id)!;
    expect(copy.name).toBe('Setup (copie)');
    expect(copy.config).toEqual(original.config);
    expect(copy.config).not.toBe(original.config);
  });

  it('deletes a strategy and persists the removal', () => {
    const { result } = renderHook(() => useSavedStrategies());
    act(() => {
      result.current.saveStrategy('Bye', SAMPLE_CONFIG);
    });
    const id = result.current.strategies[0]!.id;
    act(() => {
      expect(result.current.deleteStrategy(id)).toBe(true);
    });
    expect(result.current.strategies).toHaveLength(0);
    expect(readStrategies()).toHaveLength(0);
  });
});

describe('useSavedStrategies — honest caps, no silent purge', () => {
  it(`refuses the ${MAX_STRATEGIES + 1}th strategy with limit_reached (keeps the rest intact)`, () => {
    const { result } = renderHook(() => useSavedStrategies());
    for (let i = 0; i < MAX_STRATEGIES; i++) {
      act(() => {
        expect(result.current.saveStrategy(`S${i}`, SAMPLE_CONFIG).ok).toBe(true);
      });
    }
    let outcome: ReturnType<typeof result.current.saveStrategy> | undefined;
    act(() => {
      outcome = result.current.saveStrategy('One too many', SAMPLE_CONFIG);
    });
    expect(outcome).toEqual({ ok: false, error: 'limit_reached' });
    expect(result.current.strategies).toHaveLength(MAX_STRATEGIES);
  });

  it('reports storage_failed (not a silent success) when localStorage throws', () => {
    const { result } = renderHook(() => useSavedStrategies());
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError');
    });
    let outcome: ReturnType<typeof result.current.saveStrategy> | undefined;
    act(() => {
      outcome = result.current.saveStrategy('Quota', SAMPLE_CONFIG);
    });
    expect(outcome).toEqual({ ok: false, error: 'storage_failed' });
    expect(result.current.strategies).toHaveLength(0);
  });
});

describe('validateStrategy — corrupt / out-of-schema strategies are flagged, never reinterpreted', () => {
  it('accepts a strategy fully inside the current schema', () => {
    expect(validateStrategy(makeStrategy())).toEqual([]);
  });

  it('flags an unknown condition type with its name (« condition non reconnue »)', () => {
    const stale = makeStrategy({
      config: {
        logic: 'AND',
        conditions: [
          { type: 'mtf_aligned', direction: 'bullish' },
          // e.g. a condition from a FUTURE schema this build does not know
          { type: 'per_tf_trend_is' } as never,
        ],
      },
    });
    const problems = validateStrategy(stale);
    expect(problems.some((p) => p.includes('Condition non reconnue') && p.includes('per_tf_trend_is'))).toBe(true);
    // The config itself is untouched — nothing was dropped or rewritten.
    expect(stale.config.conditions).toHaveLength(2);
  });

  it('flags an unknown field on a known condition (future per-TF field)', () => {
    const stale = makeStrategy({
      config: {
        logic: 'AND',
        conditions: [{ type: 'trend_is', trend: 'bullish', timeframe: 'M15' } as never],
      },
    });
    const problems = validateStrategy(stale);
    expect(problems.some((p) => p.includes('Champ non reconnu') && p.includes('timeframe'))).toBe(true);
  });

  it('flags unknown enum values and out-of-range max_bars', () => {
    const stale = makeStrategy({
      config: {
        logic: 'AND',
        conditions: [
          { type: 'trend_is', trend: 'sideways' } as never,
          { type: 'bos_recent_confirmed', max_bars: 999 },
        ],
      },
    });
    const problems = validateStrategy(stale);
    expect(problems.some((p) => p.includes('Tendance non reconnue') && p.includes('sideways'))).toBe(true);
    expect(problems.some((p) => p.includes('Fenêtre de bougies invalide'))).toBe(true);
  });

  it('flags an unsupported schema_version honestly', () => {
    const future = makeStrategy({ schema_version: 2 });
    const problems = validateStrategy(future);
    expect(problems.some((p) => p.includes('Version de stratégie 2 non prise en charge'))).toBe(true);
  });

  it('flags an unknown AND/OR logic and an empty condition list', () => {
    const broken = makeStrategy({
      config: { logic: 'XOR', conditions: [] } as never,
    });
    const problems = validateStrategy(broken);
    expect(problems.some((p) => p.includes('Logique de combinaison non reconnue'))).toBe(true);
    expect(problems.some((p) => p.includes('aucune condition'))).toBe(true);
  });
});

describe('readStrategies — defensive against corrupt storage', () => {
  it('returns [] on corrupt JSON without throwing', () => {
    window.localStorage.setItem(STORAGE_KEY, '{not json');
    expect(readStrategies()).toEqual([]);
  });

  it('keeps an out-of-schema strategy in the list (for honest display) instead of dropping it', () => {
    const stale = makeStrategy({
      config: { logic: 'AND', conditions: [{ type: 'ghost_condition' }] } as never,
    });
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify([stale]));
    const restored = readStrategies();
    expect(restored).toHaveLength(1);
    expect(validateStrategy(restored[0]!).length).toBeGreaterThan(0);
  });

  it('drops entries with no usable name (nothing meaningful to display)', () => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([{ id: 'x', config: SAMPLE_CONFIG }, makeStrategy()]),
    );
    expect(readStrategies()).toHaveLength(1);
  });
});

describe('legal boundary — client-only, name never a condition', () => {
  it('never touches the network: no fetch during any store operation', () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    const { result } = renderHook(() => useSavedStrategies());
    act(() => {
      result.current.saveStrategy('No server', SAMPLE_CONFIG);
    });
    const id = result.current.strategies[0]!.id;
    act(() => {
      result.current.renameStrategy(id, 'Still no server');
    });
    act(() => {
      result.current.duplicateStrategy(id);
    });
    act(() => {
      result.current.markUsed(id);
    });
    act(() => {
      result.current.deleteStrategy(id);
    });
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('a free-text name that LOOKS like a condition never enters the conditions', () => {
    const { result } = renderHook(() => useSavedStrategies());
    act(() => {
      result.current.saveStrategy('price_in_ob', {
        logic: 'AND',
        conditions: [{ type: 'retest_in_progress' }],
      });
    });
    const s = result.current.strategies[0]!;
    expect(s.name).toBe('price_in_ob');
    expect(s.config.conditions).toEqual([{ type: 'retest_in_progress' }]);
    // The scan payload is the config alone — serialized, it carries no name.
    expect(JSON.stringify(s.config)).not.toContain('price_in_ob');
    expect(validateStrategy(s)).toEqual([]);
  });
});

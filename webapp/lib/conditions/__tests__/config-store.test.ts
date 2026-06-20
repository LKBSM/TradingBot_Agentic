import { renderHook, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { useConditionsConfig } from '../config-store';
import type { ConditionsConfig } from '../types';

const SAMPLE: ConditionsConfig = {
  logic: 'AND',
  conditions: [{ type: 'mtf_aligned', direction: 'bullish' }],
};

beforeEach(() => {
  window.localStorage.clear();
});
afterEach(() => {
  window.localStorage.clear();
});

describe('useConditionsConfig', () => {
  it('starts with no config on first visit (null) once ready', () => {
    const { result } = renderHook(() => useConditionsConfig());
    expect(result.current.ready).toBe(true);
    expect(result.current.config).toBeNull();
  });

  it('persists a saved config and exposes it back', () => {
    const { result } = renderHook(() => useConditionsConfig());
    act(() => result.current.save(SAMPLE));
    expect(result.current.config).toEqual(SAMPLE);
    // re-mount reads it back from localStorage
    const remount = renderHook(() => useConditionsConfig());
    expect(remount.result.current.config).toEqual(SAMPLE);
  });

  it('reset() forgets the config (returns to onboarding)', () => {
    const { result } = renderHook(() => useConditionsConfig());
    act(() => result.current.save(SAMPLE));
    act(() => result.current.reset());
    expect(result.current.config).toBeNull();
    expect(window.localStorage.getItem('mia.conditionsConfig.v1')).toBeNull();
  });
});

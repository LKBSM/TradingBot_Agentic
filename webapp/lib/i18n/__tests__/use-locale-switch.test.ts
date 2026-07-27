import { renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Mutable path the mocked usePathname returns (per test).
let currentPath = '/';
const push = vi.fn();
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push }),
  usePathname: () => currentPath,
}));

import { useLocaleSwitch } from '../use-locale-switch';

beforeEach(() => {
  push.mockClear();
  document.cookie = 'NEXT_LOCALE=; path=/; max-age=0';
});
afterEach(() => vi.clearAllMocks());

describe('useLocaleSwitch — single source of truth for the locale', () => {
  it('keeps the CURRENT page and switches its locale prefix (fr → en)', () => {
    currentPath = '/compte';
    const { result } = renderHook(() => useLocaleSwitch());
    result.current('en');
    // same page, en prefix
    expect(push).toHaveBeenCalledWith('/en/compte');
  });

  it('drops the prefix when switching back to the default locale (en → fr)', () => {
    currentPath = '/en/compte';
    const { result } = renderHook(() => useLocaleSwitch());
    result.current('fr');
    expect(push).toHaveBeenCalledWith('/compte');
  });

  it('handles the root path and a deep app path', () => {
    currentPath = '/';
    let hook = renderHook(() => useLocaleSwitch());
    hook.result.current('de');
    expect(push).toHaveBeenCalledWith('/de');

    push.mockClear();
    currentPath = '/en/scanner';
    hook = renderHook(() => useLocaleSwitch());
    hook.result.current('es');
    expect(push).toHaveBeenCalledWith('/es/scanner');
  });

  it('PERSISTS the choice in the NEXT_LOCALE cookie (restored on return, wins over browser)', () => {
    currentPath = '/app';
    const { result } = renderHook(() => useLocaleSwitch());
    result.current('en');
    expect(document.cookie).toContain('NEXT_LOCALE=en');
  });
});

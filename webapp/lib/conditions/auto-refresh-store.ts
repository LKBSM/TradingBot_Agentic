'use client';

import * as React from 'react';

/**
 * Persistence for the Scanner's auto-refresh preference (ON by default).
 *
 * Follows the established localStorage pattern (cf. `useConditionsConfig`):
 * versioned key, SSR-safe read, hydrate post-mount, persist on toggle, sync
 * across tabs via the `storage` event. Default is ON — the scanner feels
 * alive — but the user can turn it off.
 */

const STORAGE_KEY = 'mia.scannerAutoRefresh.v1';
const DEFAULT_ENABLED = true;

function readPref(): boolean {
  if (typeof window === 'undefined') return DEFAULT_ENABLED;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw === null) return DEFAULT_ENABLED;
    return raw === '1';
  } catch {
    return DEFAULT_ENABLED;
  }
}

function persistPref(enabled: boolean): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0');
  } catch {
    /* storage unavailable (private mode) — keep the in-memory choice */
  }
}

export interface UseAutoRefreshPrefResult {
  /** Whether auto-refresh is enabled (defaults to true once hydrated). */
  enabled: boolean;
  /** True once localStorage has been read (avoids a first-paint flip). */
  ready: boolean;
  /** Flip and persist the preference. */
  setEnabled(next: boolean): void;
}

export function useAutoRefreshPref(): UseAutoRefreshPrefResult {
  const [enabled, setEnabledState] = React.useState(DEFAULT_ENABLED);
  const [ready, setReady] = React.useState(false);

  React.useEffect(() => {
    setEnabledState(readPref());
    setReady(true);
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setEnabledState(readPref());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const setEnabled = React.useCallback((next: boolean) => {
    setEnabledState(next);
    persistPref(next);
  }, []);

  return { enabled, ready, setEnabled };
}

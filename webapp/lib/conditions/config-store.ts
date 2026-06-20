'use client';

import * as React from 'react';
import type { ConditionsConfig } from './types';

/**
 * Persistence for the user's scanner configuration (their conditions + AND/OR).
 *
 * Follows the established localStorage pattern (cf. CookieBanner
 * `mia.cookie-consent.v1`): a versioned key, SSR-safe read, hydrate post-mount,
 * persist on save, and sync across tabs via the `storage` event.
 */

const STORAGE_KEY = 'mia.conditionsConfig.v1';

export const EMPTY_CONFIG: ConditionsConfig = { logic: 'AND', conditions: [] };

function isValidConfig(value: unknown): value is ConditionsConfig {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  if (v.logic !== 'AND' && v.logic !== 'OR') return false;
  return Array.isArray(v.conditions);
}

function readConfig(): ConditionsConfig | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    return isValidConfig(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function persistConfig(config: ConditionsConfig): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

function clearConfig(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(STORAGE_KEY);
}

export interface UseConditionsConfigResult {
  /** The saved config, or null when the user has not composed one yet. */
  config: ConditionsConfig | null;
  /** True once we have read localStorage (avoids an SSR/first-paint flash). */
  ready: boolean;
  /** Persist a config (the user's conditions). */
  save(config: ConditionsConfig): void;
  /** Forget the saved config (returns the user to onboarding). */
  reset(): void;
}

export function useConditionsConfig(): UseConditionsConfigResult {
  const [config, setConfig] = React.useState<ConditionsConfig | null>(null);
  const [ready, setReady] = React.useState(false);

  React.useEffect(() => {
    setConfig(readConfig());
    setReady(true);
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setConfig(readConfig());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const save = React.useCallback((next: ConditionsConfig) => {
    setConfig(next);
    persistConfig(next);
  }, []);

  const reset = React.useCallback(() => {
    setConfig(null);
    clearConfig();
  }, []);

  return { config, ready, save, reset };
}

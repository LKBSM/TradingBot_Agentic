'use client';

import * as React from 'react';
import type { ConditionsConfig, ScanCondition } from './types';
import {
  CONDITION_TYPES,
  DIRECTION_LABELS,
  PHASE_OPTIONS,
  TREND_OPTIONS,
  VOLATILITY_OPTIONS,
} from './palette';

/**
 * Named scanner strategies — CLIENT-ONLY (localStorage), Loi 25 boundary.
 *
 * A trader has 2-3 setups, not one: this store lets them NAME a composition of
 * conditions ("London sweep M15"), save it, reload it, rename / duplicate /
 * delete it. Everything stays on the user's device: no server endpoint, no DB
 * table, no cookie — same boundary as the chat persistence. The strategy NAME
 * is free text for display only; it never becomes a condition and never leaves
 * the device (only `config` is POSTed to the scan, exactly as today).
 *
 * Versioning: entries carry `schema_version`. On load, every strategy is
 * re-validated against the CURRENT palette Literals. A strategy holding a
 * condition (or field, or value) outside the current schema is flagged invalid
 * with the precise reasons — it is never silently reinterpreted or partially
 * executed. This is the seam that will absorb the per-TF conditions schema
 * when it lands.
 *
 * Purge policy: named strategies are user artefacts — we NEVER purge them
 * silently (unlike chat threads). A hard cap with an honest error instead.
 */

export const STORAGE_KEY = 'mia.scannerStrategies.v1';

export const CURRENT_STRATEGY_SCHEMA_VERSION = 1;

/** Hard caps — refuse with an honest error, never silently drop. */
export const MAX_STRATEGIES = 20;
export const MAX_NAME_CHARS = 60;
export const MAX_SERIALIZED_CHARS = 120_000;

export interface SavedStrategy {
  id: string;
  /** Free display name — NEVER interpreted as a condition, never sent to the server. */
  name: string;
  schema_version: number;
  /** The exact wire-shaped config; only this is POSTed at scan time. */
  config: ConditionsConfig;
  createdAt: number;
  lastUsedAt: number;
}

export type StrategyMutationError =
  | 'name_required'
  | 'limit_reached'
  | 'not_found'
  | 'storage_failed';

export type StrategyMutationResult =
  | { ok: true; strategy: SavedStrategy }
  | { ok: false; error: StrategyMutationError };

// ── Validation against the CURRENT schema ────────────────────────────────────

const VALID_TYPES = new Set<string>(CONDITION_TYPES);
const VALID_DIRECTIONS = new Set<string>(Object.keys(DIRECTION_LABELS));
const VALID_TRENDS = new Set<string>(TREND_OPTIONS.map((o) => o.value));
const VALID_PHASES = new Set<string>(PHASE_OPTIONS.map((o) => o.value));
const VALID_VOLATILITIES = new Set<string>(VOLATILITY_OPTIONS.map((o) => o.value));
const KNOWN_CONDITION_KEYS = new Set([
  'type',
  'direction',
  'max_bars',
  'trend',
  'phase',
  'volatility',
]);

/**
 * Validate a saved strategy against the CURRENT condition schema.
 * Returns the list of problems (French, user-facing); empty ⇒ loadable.
 * Strict by design: an unknown condition type, an unknown field or an unknown
 * value marks the strategy invalid — honesty over reinterpretation.
 */
export function validateStrategy(strategy: SavedStrategy): string[] {
  const problems: string[] = [];

  if (strategy.schema_version !== CURRENT_STRATEGY_SCHEMA_VERSION) {
    problems.push(
      `Version de stratégie ${strategy.schema_version} non prise en charge ` +
        `(schéma actuel : ${CURRENT_STRATEGY_SCHEMA_VERSION}).`,
    );
  }

  const config = strategy.config as unknown;
  if (typeof config !== 'object' || config === null) {
    problems.push('Configuration absente ou corrompue.');
    return problems;
  }
  const cfg = config as Record<string, unknown>;

  if (cfg.logic !== 'AND' && cfg.logic !== 'OR') {
    problems.push(`Logique de combinaison non reconnue : « ${String(cfg.logic)} ».`);
  }
  if (!Array.isArray(cfg.conditions)) {
    problems.push('Liste de conditions absente ou corrompue.');
    return problems;
  }
  if (cfg.conditions.length === 0) {
    problems.push('La stratégie ne contient aucune condition.');
  }

  cfg.conditions.forEach((raw, i) => {
    const where = `condition ${i + 1}`;
    if (typeof raw !== 'object' || raw === null) {
      problems.push(`Condition corrompue (${where}).`);
      return;
    }
    const cond = raw as Record<string, unknown>;
    const type = typeof cond.type === 'string' ? cond.type : String(cond.type);
    if (!VALID_TYPES.has(type)) {
      problems.push(`Condition non reconnue : « ${type} » (${where}).`);
      return;
    }
    for (const key of Object.keys(cond)) {
      if (!KNOWN_CONDITION_KEYS.has(key)) {
        problems.push(`Champ non reconnu sur « ${type} » : « ${key} » (${where}).`);
      }
    }
    if (cond.direction !== undefined && !VALID_DIRECTIONS.has(String(cond.direction))) {
      problems.push(
        `Direction non reconnue sur « ${type} » : « ${String(cond.direction)} » (${where}).`,
      );
    }
    if (cond.trend !== undefined && !VALID_TRENDS.has(String(cond.trend))) {
      problems.push(
        `Tendance non reconnue sur « ${type} » : « ${String(cond.trend)} » (${where}).`,
      );
    }
    if (cond.phase !== undefined && !VALID_PHASES.has(String(cond.phase))) {
      problems.push(
        `Phase non reconnue sur « ${type} » : « ${String(cond.phase)} » (${where}).`,
      );
    }
    if (
      cond.volatility !== undefined &&
      !VALID_VOLATILITIES.has(String(cond.volatility))
    ) {
      problems.push(
        `Volatilité non reconnue sur « ${type} » : « ${String(cond.volatility)} » (${where}).`,
      );
    }
    if (cond.max_bars !== undefined) {
      const n = cond.max_bars;
      if (typeof n !== 'number' || !Number.isInteger(n) || n < 1 || n > 50) {
        problems.push(
          `Fenêtre de bougies invalide sur « ${type} » : ${String(n)} (attendu : entier 1–50) (${where}).`,
        );
      }
    }
  });

  return problems;
}

// ── localStorage layer (SSR-safe, defensive) ─────────────────────────────────

function normalizeName(name: string): string {
  return name.trim().slice(0, MAX_NAME_CHARS);
}

function sameName(a: string, b: string): boolean {
  return a.trim().toLowerCase() === b.trim().toLowerCase();
}

function newId(): string {
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
  } catch {
    // fall through
  }
  return `strat-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Structural sanitisation of one stored entry. Deliberately LOOSE on the
 * config: a config holding out-of-schema conditions is KEPT verbatim so that
 * validateStrategy() can surface the honest reasons — dropping it here would
 * silently destroy a named user artefact. Only entries with no usable name
 * (nothing meaningful to display) are dropped.
 */
function sanitizeStrategy(raw: unknown): SavedStrategy | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const s = raw as Record<string, unknown>;
  if (typeof s.name !== 'string' || s.name.trim().length === 0) return null;
  const config =
    typeof s.config === 'object' && s.config !== null
      ? (s.config as ConditionsConfig)
      : ({ logic: 'AND', conditions: [] } as ConditionsConfig);
  return {
    id: typeof s.id === 'string' && s.id.length > 0 ? s.id : newId(),
    name: normalizeName(s.name),
    schema_version:
      typeof s.schema_version === 'number' && Number.isFinite(s.schema_version)
        ? s.schema_version
        : 0,
    config,
    createdAt:
      typeof s.createdAt === 'number' && Number.isFinite(s.createdAt) ? s.createdAt : 0,
    lastUsedAt:
      typeof s.lastUsedAt === 'number' && Number.isFinite(s.lastUsedAt)
        ? s.lastUsedAt
        : 0,
  };
}

function sortByLastUsed(strategies: SavedStrategy[]): SavedStrategy[] {
  return [...strategies].sort((a, b) => b.lastUsedAt - a.lastUsedAt);
}

/** Read + sanitise the persisted strategies. Returns [] on SSR / corrupt storage. */
export function readStrategies(): SavedStrategy[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const seen = new Set<string>();
    const out: SavedStrategy[] = [];
    for (const item of parsed) {
      const strategy = sanitizeStrategy(item);
      if (strategy && !seen.has(strategy.id)) {
        seen.add(strategy.id);
        out.push(strategy);
      }
    }
    return sortByLastUsed(out);
  } catch {
    return [];
  }
}

/** Persist. Returns false on quota / size failure — callers surface it honestly. */
function writeStrategies(strategies: SavedStrategy[]): boolean {
  if (typeof window === 'undefined') return false;
  try {
    const payload = JSON.stringify(strategies);
    if (payload.length > MAX_SERIALIZED_CHARS) return false;
    window.localStorage.setItem(STORAGE_KEY, payload);
    return true;
  } catch {
    return false;
  }
}

// ── React hook ────────────────────────────────────────────────────────────────

export interface UseSavedStrategiesResult {
  /** Saved strategies, most recently used first. */
  strategies: SavedStrategy[];
  /** True once localStorage has been read (avoids an SSR/first-paint flash). */
  ready: boolean;
  /**
   * Save the composed config under a name. Upserts by name (case-insensitive):
   * re-saving "London sweep M15" updates that strategy in place.
   */
  saveStrategy(name: string, config: ConditionsConfig): StrategyMutationResult;
  renameStrategy(id: string, name: string): StrategyMutationResult;
  duplicateStrategy(id: string): StrategyMutationResult;
  deleteStrategy(id: string): boolean;
  /** Stamp a strategy as just used (drives the most-recent-first ordering). */
  markUsed(id: string): void;
}

export function useSavedStrategies(): UseSavedStrategiesResult {
  const [strategies, setStrategies] = React.useState<SavedStrategy[]>([]);
  const [ready, setReady] = React.useState(false);

  React.useEffect(() => {
    setStrategies(readStrategies());
    setReady(true);
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setStrategies(readStrategies());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const commit = React.useCallback(
    (next: SavedStrategy[]): boolean => {
      const sorted = sortByLastUsed(next);
      if (!writeStrategies(sorted)) return false;
      setStrategies(sorted);
      return true;
    },
    [],
  );

  const saveStrategy = React.useCallback(
    (name: string, config: ConditionsConfig): StrategyMutationResult => {
      const clean = normalizeName(name);
      if (clean.length === 0) return { ok: false, error: 'name_required' };
      const now = Date.now();
      const existing = strategies.find((s) => sameName(s.name, clean));
      if (existing) {
        const updated: SavedStrategy = {
          ...existing,
          name: clean,
          schema_version: CURRENT_STRATEGY_SCHEMA_VERSION,
          config,
          lastUsedAt: now,
        };
        const next = strategies.map((s) => (s.id === existing.id ? updated : s));
        if (!commit(next)) return { ok: false, error: 'storage_failed' };
        return { ok: true, strategy: updated };
      }
      if (strategies.length >= MAX_STRATEGIES) {
        return { ok: false, error: 'limit_reached' };
      }
      const created: SavedStrategy = {
        id: newId(),
        name: clean,
        schema_version: CURRENT_STRATEGY_SCHEMA_VERSION,
        config,
        createdAt: now,
        lastUsedAt: now,
      };
      if (!commit([...strategies, created])) return { ok: false, error: 'storage_failed' };
      return { ok: true, strategy: created };
    },
    [strategies, commit],
  );

  const renameStrategy = React.useCallback(
    (id: string, name: string): StrategyMutationResult => {
      const clean = normalizeName(name);
      if (clean.length === 0) return { ok: false, error: 'name_required' };
      const target = strategies.find((s) => s.id === id);
      if (!target) return { ok: false, error: 'not_found' };
      const renamed: SavedStrategy = { ...target, name: clean };
      const next = strategies.map((s) => (s.id === id ? renamed : s));
      if (!commit(next)) return { ok: false, error: 'storage_failed' };
      return { ok: true, strategy: renamed };
    },
    [strategies, commit],
  );

  const duplicateStrategy = React.useCallback(
    (id: string): StrategyMutationResult => {
      const target = strategies.find((s) => s.id === id);
      if (!target) return { ok: false, error: 'not_found' };
      if (strategies.length >= MAX_STRATEGIES) {
        return { ok: false, error: 'limit_reached' };
      }
      const now = Date.now();
      const copy: SavedStrategy = {
        ...target,
        id: newId(),
        name: normalizeName(`${target.name} (copie)`),
        // Deep-copy the config so later edits to one never leak into the other.
        config: JSON.parse(JSON.stringify(target.config)) as ConditionsConfig,
        createdAt: now,
        lastUsedAt: now,
      };
      if (!commit([...strategies, copy])) return { ok: false, error: 'storage_failed' };
      return { ok: true, strategy: copy };
    },
    [strategies, commit],
  );

  const deleteStrategy = React.useCallback(
    (id: string): boolean => {
      const next = strategies.filter((s) => s.id !== id);
      if (next.length === strategies.length) return false;
      return commit(next);
    },
    [strategies, commit],
  );

  const markUsed = React.useCallback(
    (id: string): void => {
      const target = strategies.find((s) => s.id === id);
      if (!target) return;
      const next = strategies.map((s) =>
        s.id === id ? { ...s, lastUsedAt: Date.now() } : s,
      );
      commit(next);
    },
    [strategies, commit],
  );

  return {
    strategies,
    ready,
    saveStrategy,
    renameStrategy,
    duplicateStrategy,
    deleteStrategy,
    markUsed,
  };
}

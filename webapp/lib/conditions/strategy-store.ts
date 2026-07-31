'use client';

import * as React from 'react';
import type { ConditionsConfig } from './types';
import { CONDITION_TYPES } from './palette';

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
const VALID_DIRECTIONS = new Set(['any', 'bullish', 'bearish']);
const VALID_TRENDS = new Set(['bullish', 'bearish', 'indeterminate']);
const VALID_PHASES = new Set(['accumulation', 'trend', 'ranging', 'expansion']);
const VALID_VOLATILITIES = new Set(['low', 'normal', 'elevated']);
const VALID_SIDES = new Set(['any', 'bsl', 'ssl']);
const VALID_ZONE_KINDS = new Set(['any', 'ob', 'fvg']);
const VALID_EVENTS = new Set(['bos_up', 'bos_down', 'choch_up', 'choch_down']);
const VALID_AGE_BUCKETS = new Set(['lt10', '10to50', 'gt50']);
const VALID_THIRDS = new Set(['bottom', 'middle', 'top']);
const VALID_EQ_KINDS = new Set(['any', 'highs', 'lows']);
const VALID_SESSIONS = new Set(['asia', 'london', 'new_york', 'overlap']);
const VALID_RELATIONS = new Set(['same', 'opposite']);
const KNOWN_CONDITION_KEYS = new Set([
  'type', 'direction', 'max_bars', 'trend', 'phase', 'volatility',
  'proximity_pct', 'side', 'zone_kind', 'event', 'age_bucket', 'third',
  'eq_kind', 'session', 'relation', 'max_touches',
]);

const ENUM_CHECKS: Array<{ key: string; set: Set<string>; label: string }> = [
  { key: 'direction', set: VALID_DIRECTIONS, label: 'Direction' },
  { key: 'trend', set: VALID_TRENDS, label: 'Tendance' },
  { key: 'phase', set: VALID_PHASES, label: 'Phase' },
  { key: 'volatility', set: VALID_VOLATILITIES, label: 'Volatilité' },
  { key: 'side', set: VALID_SIDES, label: 'Côté' },
  { key: 'zone_kind', set: VALID_ZONE_KINDS, label: 'Type de zone' },
  { key: 'event', set: VALID_EVENTS, label: 'Événement' },
  { key: 'age_bucket', set: VALID_AGE_BUCKETS, label: 'Tranche' },
  { key: 'third', set: VALID_THIRDS, label: 'Tiers' },
  { key: 'eq_kind', set: VALID_EQ_KINDS, label: 'Type d’égalité' },
  { key: 'session', set: VALID_SESSIONS, label: 'Session' },
  { key: 'relation', set: VALID_RELATIONS, label: 'Relation' },
];

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
    for (const { key, set, label } of ENUM_CHECKS) {
      const value = cond[key];
      if (value !== undefined && !set.has(String(value))) {
        problems.push(`${label} non reconnue sur « ${type} » : « ${String(value)} » (${where}).`);
      }
    }
    if (cond.max_bars !== undefined) {
      const n = cond.max_bars;
      if (typeof n !== 'number' || !Number.isInteger(n) || n < 1 || n > 50) {
        problems.push(
          `Fenêtre de bougies invalide sur « ${type} » : ${String(n)} (attendu : entier 1–50) (${where}).`,
        );
      }
    }
    if (cond.max_touches !== undefined) {
      const n = cond.max_touches;
      if (typeof n !== 'number' || !Number.isInteger(n) || n < 1 || n > 3) {
        problems.push(
          `Nombre de touches invalide sur « ${type} » : ${String(n)} (attendu : entier 1–3) (${where}).`,
        );
      }
    }
    if (cond.proximity_pct !== undefined) {
      const n = cond.proximity_pct;
      if (typeof n !== 'number' || !(n > 0) || n > 10) {
        problems.push(
          `Distance invalide sur « ${type} » : ${String(n)} (attendu : 0 < % ≤ 10) (${where}).`,
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

// ── Text export / import (device portability — decision #4a) ─────────────────
//
// Saved readings live on this device only (no server sync). Export/import lets a
// user MOVE them between devices themselves — a plain-text envelope they can copy
// or download. The interface abstraction (this seam) is where a future server
// adapter would plug in without a rewrite.

export const EXPORT_FORMAT = 'mia.scanner.savedReadings';

export interface ExportEnvelope {
  format: string;
  schema_version: number;
  exported_at: number;
  readings: Array<{ name: string; config: ConditionsConfig; schema_version: number }>;
}

/** Serialise saved readings to a portable text envelope. */
export function exportStrategiesText(strategies: SavedStrategy[], now: number): string {
  const envelope: ExportEnvelope = {
    format: EXPORT_FORMAT,
    schema_version: CURRENT_STRATEGY_SCHEMA_VERSION,
    exported_at: now,
    readings: strategies.map((s) => ({
      name: s.name,
      config: s.config,
      schema_version: s.schema_version,
    })),
  };
  return JSON.stringify(envelope, null, 2);
}

export type ImportResult =
  | { ok: true; readings: Array<{ name: string; config: ConditionsConfig; schema_version: number }> }
  | { ok: false; error: 'malformed' | 'wrong_format' | 'empty' };

/** Parse a text envelope. Strict: wrong shape/format is refused, never guessed. */
export function parseStrategiesText(text: string): ImportResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, error: 'malformed' };
  }
  if (typeof parsed !== 'object' || parsed === null) return { ok: false, error: 'malformed' };
  const env = parsed as Record<string, unknown>;
  if (env.format !== EXPORT_FORMAT) return { ok: false, error: 'wrong_format' };
  if (!Array.isArray(env.readings)) return { ok: false, error: 'malformed' };
  const readings: Array<{ name: string; config: ConditionsConfig; schema_version: number }> = [];
  for (const raw of env.readings) {
    if (typeof raw !== 'object' || raw === null) continue;
    const r = raw as Record<string, unknown>;
    if (typeof r.name !== 'string' || r.name.trim().length === 0) continue;
    if (typeof r.config !== 'object' || r.config === null) continue;
    readings.push({
      name: normalizeName(r.name),
      config: r.config as ConditionsConfig,
      schema_version:
        typeof r.schema_version === 'number' ? r.schema_version : CURRENT_STRATEGY_SCHEMA_VERSION,
    });
  }
  if (readings.length === 0) return { ok: false, error: 'empty' };
  return { ok: true, readings };
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
  /** Serialise all saved readings to a portable text envelope. */
  exportText(): string;
  /** Merge readings from a text envelope (upsert by name, honouring the cap). */
  importText(text: string): { ok: true; imported: number; skipped: number } | { ok: false; error: string };
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

  const exportText = React.useCallback(
    (): string => exportStrategiesText(strategies, Date.now()),
    [strategies],
  );

  const importText = React.useCallback(
    (text: string): { ok: true; imported: number; skipped: number } | { ok: false; error: string } => {
      const parsed = parseStrategiesText(text);
      if (!parsed.ok) return { ok: false, error: parsed.error };
      const now = Date.now();
      const byName = new Map(strategies.map((s) => [s.name.trim().toLowerCase(), s]));
      let imported = 0;
      let skipped = 0;
      const next = [...strategies];
      for (const r of parsed.readings) {
        const key = r.name.trim().toLowerCase();
        const existing = byName.get(key);
        if (existing) {
          const idx = next.findIndex((s) => s.id === existing.id);
          next[idx] = { ...existing, config: r.config, schema_version: r.schema_version, lastUsedAt: now };
          imported += 1;
          continue;
        }
        if (next.length >= MAX_STRATEGIES) {
          skipped += 1;
          continue;
        }
        const created: SavedStrategy = {
          id: newId(),
          name: r.name,
          schema_version: r.schema_version,
          config: r.config,
          createdAt: now,
          lastUsedAt: now,
        };
        next.push(created);
        byName.set(key, created);
        imported += 1;
      }
      if (!commit(next)) return { ok: false, error: 'storage_failed' };
      return { ok: true, imported, skipped };
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
    exportText,
    importText,
  };
}

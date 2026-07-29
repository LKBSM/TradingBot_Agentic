import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

/**
 * DG-1 decision (b) — SINGLE JOURNAL SOURCE.
 *
 * The App reads its structure / journal / maturity from the LIVE assembler
 * (`/api/market-reading`). The deep persisted store (`/api/structure`,
 * `/api/coverage`, LB-1) is seeded by the backfill only (scheduler → refresh
 * deferred) and is therefore FROZEN between backfills. Wiring it to a UI surface
 * as-is would put two journals covering DIFFERENT periods on the same screen —
 * the exact state DG-1 forbids.
 *
 * This guard fails if any App source starts referencing the deep-journal
 * endpoints. Re-enabling them is only allowed together with (1) the scheduler →
 * IncrementalDetector.refresh wiring and (2) an on-screen label of the covered
 * period (DG-1 point 5). See
 * docs/governance/decisions/2026-07-29_dg-1_single_journal_source.md — update
 * this guard in that same change, never disable it in isolation.
 */

// webapp/ root — vitest runs with cwd at the webapp package root.
const WEBAPP_ROOT = process.cwd();
const SCAN_DIRS = ['lib', 'components', 'app'];
const FORBIDDEN = /api\/(structure|coverage)/;
const THIS_FILE = 'single-journal-guard.test.ts';

function collectSources(dir: string): string[] {
  let out: string[] = [];
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return out;
  }
  for (const name of entries) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      if (name === 'node_modules' || name === '.next') continue;
      out = out.concat(collectSources(full));
    } else if (/\.(ts|tsx)$/.test(name) && name !== THIS_FILE) {
      out.push(full);
    }
  }
  return out;
}

describe('DG-1 (b) — single journal source on screen', () => {
  it('no App surface consumes /api/structure or /api/coverage', () => {
    const offenders: string[] = [];
    for (const rel of SCAN_DIRS) {
      for (const file of collectSources(join(WEBAPP_ROOT, rel))) {
        if (FORBIDDEN.test(readFileSync(file, 'utf8'))) {
          offenders.push(file.replace(WEBAPP_ROOT, ''));
        }
      }
    }
    expect(offenders).toEqual([]);
  });
});

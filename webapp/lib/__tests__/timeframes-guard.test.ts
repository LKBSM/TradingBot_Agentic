import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  TF_SECONDS,
  PERIMETER_TIMEFRAMES,
  TIMEFRAME_SPECS,
} from '@/lib/timeframes';

const REPO = resolve(process.cwd(), '..');

describe('TF-1 — the frontend registry is the single source', () => {
  it('timeframes.generated.ts is in sync with config/timeframes.json', () => {
    const raw = JSON.parse(readFileSync(resolve(REPO, 'config/timeframes.json'), 'utf-8'));
    const fromJson = raw.timeframes.map((e: Record<string, unknown>) => ({
      id: e.id,
      minutes: e.minutes,
      provider: e.provider,
      perimeter: e.perimeter,
    }));
    const fromGen = TIMEFRAME_SPECS.map((s) => ({
      id: s.id,
      minutes: s.minutes,
      provider: s.provider,
      perimeter: s.perimeter,
    }));
    expect(fromGen, 'run `node scripts/gen_timeframes.mjs` after editing the JSON').toEqual(fromJson);
  });

  it('TF_SECONDS covers every perimeter unit with a positive value (symptom guard)', () => {
    // The reported bug: M5/D1 missing here → barSec=0 → chart never framed.
    for (const tf of PERIMETER_TIMEFRAMES) {
      expect(TF_SECONDS[tf], `${tf} must have a bar length`).toBeGreaterThan(0);
    }
    expect(TF_SECONDS['M5']).toBe(300);
    expect(TF_SECONDS['D1']).toBe(86400);
  });
});

// Scan webapp source for a re-introduced hardcoded timeframe map/triplet.
const FORBIDDEN: [RegExp, string][] = [
  [/M15:\s*900/, 'inline TF_SECONDS map'],
  [/M15:\s*15\s*\*\s*60/, 'inline INTERVAL_SECONDS map'],
  [/\bM15:\s*15\b/, 'inline TF→minutes map'],
  [/\[\s*['"]h4['"]\s*,\s*['"]h1['"]\s*,\s*['"]m15['"]/, 'hardcoded MTF triplet'],
];
const ALLOW = new Set(['timeframes.ts', 'timeframes.generated.ts']);

function walk(dir: string, hits: string[]): void {
  for (const name of readdirSync(dir)) {
    const p = resolve(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (name === 'node_modules' || name === '__tests__' || name === '.next') continue;
      walk(p, hits);
    } else if (/\.(ts|tsx)$/.test(name) && !ALLOW.has(name)) {
      const text = readFileSync(p, 'utf-8');
      for (const [re, what] of FORBIDDEN) {
        if (re.test(text)) hits.push(`${p.replace(REPO, '')} → ${what}`);
      }
    }
  }
}

describe('TF-1 GUARD — no hardcoded timeframe map in the webapp', () => {
  it('every enumeration derives from lib/timeframes', () => {
    const hits: string[] = [];
    walk(resolve(process.cwd(), 'lib'), hits);
    walk(resolve(process.cwd(), 'components'), hits);
    expect(hits, `derive from @/lib/timeframes instead:\n${hits.join('\n')}`).toEqual([]);
  });
});

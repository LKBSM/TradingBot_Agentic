#!/usr/bin/env node
// TF-1 — generate webapp/lib/timeframes.generated.ts from config/timeframes.json.
// The frontend timeframe registry is GENERATED from the same single source the
// backend reads — never hand-copied. Run after editing config/timeframes.json:
//   node scripts/gen_timeframes.mjs
// `--check` exits non-zero if the committed file is out of date (used in CI/tests).

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const jsonPath = resolve(here, '..', 'config', 'timeframes.json');
const outPath = resolve(here, '..', 'webapp', 'lib', 'timeframes.generated.ts');

function render() {
  const raw = JSON.parse(readFileSync(jsonPath, 'utf-8'));
  const rows = raw.timeframes.map((e, i) => {
    const s = e.minutes * 60;
    return (
      `  { id: ${JSON.stringify(e.id)}, minutes: ${e.minutes}, seconds: ${s}, ` +
      `provider: ${JSON.stringify(e.provider)}, labelLong: ${JSON.stringify(e.labelLong)}, ` +
      `dateFormat: ${JSON.stringify(e.dateFormat)}, perimeter: ${e.perimeter}, ` +
      `reference: ${e.reference}, sessionRelevant: ${e.sessionRelevant}, ` +
      `prevLevelsRelevant: ${e.prevLevelsRelevant}, index: ${i} }`
    );
  });
  return (
    `// AUTO-GENERATED from config/timeframes.json by scripts/gen_timeframes.mjs.\n` +
    `// DO NOT EDIT BY HAND. Run \`node scripts/gen_timeframes.mjs\` after editing the JSON.\n` +
    `export interface TimeframeSpec {\n` +
    `  id: string;\n  minutes: number;\n  seconds: number;\n  provider: string;\n` +
    `  labelLong: string;\n  dateFormat: string;\n  perimeter: boolean;\n` +
    `  reference: boolean;\n  sessionRelevant: boolean;\n  prevLevelsRelevant: boolean;\n  index: number;\n}\n\n` +
    `export const TIMEFRAME_SPECS: readonly TimeframeSpec[] = [\n${rows.join(',\n')},\n];\n`
  );
}

const generated = render();

if (process.argv.includes('--check')) {
  let current = '';
  try {
    current = readFileSync(outPath, 'utf-8');
  } catch {
    /* missing → out of date */
  }
  if (current !== generated) {
    console.error('timeframes.generated.ts is OUT OF DATE. Run: node scripts/gen_timeframes.mjs');
    process.exit(1);
  }
  console.log('timeframes.generated.ts is up to date.');
} else {
  writeFileSync(outPath, generated);
  console.log(`Wrote ${outPath}`);
}

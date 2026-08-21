#!/usr/bin/env node
// MKT-1 — generate webapp/lib/markets.generated.ts from config/markets.json.
// The frontend market registry is GENERATED from the same single source the
// backend reads — never hand-copied. Run after editing config/markets.json:
//   node scripts/gen_markets.mjs
// `--check` exits non-zero if the committed file is out of date (used in CI/tests).

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const jsonPath = resolve(here, '..', 'config', 'markets.json');
const outPath = resolve(here, '..', 'webapp', 'lib', 'markets.generated.ts');

function render() {
  const raw = JSON.parse(readFileSync(jsonPath, 'utf-8'));
  const rows = raw.markets.map((m, i) => {
    const tfs = m.timeframes.map((t) => JSON.stringify(t)).join(', ');
    return (
      `  { id: ${JSON.stringify(m.id)}, label: ${JSON.stringify(m.label)}, ` +
      `symbol: ${JSON.stringify(m.symbol)}, type: ${JSON.stringify(m.type)}, ` +
      `priceDecimals: ${m.priceDecimals}, glyph: ${JSON.stringify(m.glyph)}, ` +
      `timeframes: [${tfs}], index: ${i} }`
    );
  });
  return (
    `// AUTO-GENERATED from config/markets.json by scripts/gen_markets.mjs.\n` +
    `// DO NOT EDIT BY HAND. Run \`node scripts/gen_markets.mjs\` after editing the JSON.\n` +
    `export type MarketType = 'metal' | 'fx' | 'crypto' | 'index';\n\n` +
    `export interface MarketSpec {\n` +
    `  id: string;\n  label: string;\n  symbol: string;\n  type: MarketType;\n` +
    `  priceDecimals: number;\n  glyph: string;\n  timeframes: readonly string[];\n  index: number;\n}\n\n` +
    `export const MARKET_SPECS: readonly MarketSpec[] = [\n${rows.join(',\n')},\n];\n`
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
    console.error('markets.generated.ts is OUT OF DATE. Run: node scripts/gen_markets.mjs');
    process.exit(1);
  }
  console.log('markets.generated.ts is up to date.');
} else {
  writeFileSync(outPath, generated);
  console.log(`Wrote ${outPath}`);
}

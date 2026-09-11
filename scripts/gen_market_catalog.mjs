#!/usr/bin/env node
// MKT-1 TEST UX — generate webapp/lib/market-catalog.generated.ts from
// config/market_catalog_ux_test.json.
//
// Twin of gen_markets.mjs, deliberately kept SEPARATE: the two chains must never
// merge. gen_markets.mjs produces the REAL perimeter (what the engine follows,
// what the backend serves); this one produces a DISPLAY-ONLY catalogue used to
// test how the interface holds at 100 entries. Nothing generated here ever feeds
// ALL_MARKET_IDS or SUPPORTED_INSTRUMENTS.
//
// Run after editing the JSON:  node scripts/gen_market_catalog.mjs
// `--check` exits non-zero if the committed file is out of date (CI/tests).

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const jsonPath = resolve(here, '..', 'config', 'market_catalog_ux_test.json');
const outPath = resolve(here, '..', 'webapp', 'lib', 'market-catalog.generated.ts');

const GROUPS = ['fx-major', 'fx-minor', 'fx-exotic', 'metal', 'index', 'crypto'];

function render() {
  const raw = JSON.parse(readFileSync(jsonPath, 'utf-8'));

  // Fail loudly rather than emit a silently wrong catalogue.
  const seen = new Set();
  for (const m of raw.markets) {
    if (!GROUPS.includes(m.group)) {
      throw new Error(`market_catalog_ux_test.json: ${m.id} has unknown group ${m.group}`);
    }
    if (seen.has(m.id)) {
      throw new Error(`market_catalog_ux_test.json: duplicate id ${m.id}`);
    }
    seen.add(m.id);
  }

  const rows = raw.markets.map((m, i) => (
    `  { id: ${JSON.stringify(m.id)}, label: ${JSON.stringify(m.label)}, ` +
    `symbol: ${JSON.stringify(m.symbol)}, group: ${JSON.stringify(m.group)}, ` +
    `glyph: ${JSON.stringify(m.glyph)}, index: ${i} }`
  ));

  const groupUnion = GROUPS.map((g) => `'${g}'`).join(' | ');

  return (
    `// AUTO-GENERATED from config/market_catalog_ux_test.json by scripts/gen_market_catalog.mjs.\n` +
    `// DO NOT EDIT BY HAND. Run \`node scripts/gen_market_catalog.mjs\` after editing the JSON.\n` +
    `//\n` +
    `// DISPLAY-ONLY catalogue (MKT-1 test UX). NOT the product perimeter: the engine\n` +
    `// follows config/markets.json alone. A market listed here and absent from the real\n` +
    `// registry has NO data behind it — see lib/market-catalog.ts.\n` +
    `export type CatalogGroup = ${groupUnion};\n\n` +
    `export const CATALOG_GROUPS: readonly CatalogGroup[] = [${GROUPS.map((g) => `'${g}'`).join(', ')}];\n\n` +
    `export interface CatalogEntry {\n` +
    `  id: string;\n  label: string;\n  symbol: string;\n  group: CatalogGroup;\n` +
    `  glyph: string;\n  index: number;\n}\n\n` +
    `export const CATALOG_ENTRIES: readonly CatalogEntry[] = [\n${rows.join(',\n')},\n];\n`
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
  // Compare EOL-insensitively. The repo checks out CRLF on Windows
  // (core.autocrlf=true) while this script emits LF, so a byte comparison
  // reports a false "out of date" on a perfectly in-sync file — which is exactly
  // what `gen_markets.mjs --check` does today on a Windows checkout.
  const norm = (s) => s.split('\r\n').join('\n');
  if (norm(current) !== norm(generated)) {
    console.error(
      'market-catalog.generated.ts is OUT OF DATE. Run: node scripts/gen_market_catalog.mjs',
    );
    process.exit(1);
  }
  console.log('market-catalog.generated.ts is up to date.');
} else {
  writeFileSync(outPath, generated);
  console.log(`Wrote ${outPath}`);
}

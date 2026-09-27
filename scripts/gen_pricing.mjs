#!/usr/bin/env node
// PRIX-1 — generate webapp/lib/pricing.generated.ts from config/pricing.json.
// The frontend price model is GENERATED from the SAME single source the backend
// reads (src/billing/pricing.py) — never hand-copied. Run after editing the JSON:
//   node scripts/gen_pricing.mjs
// `--check` exits non-zero if the committed file is out of date (used in CI/tests).

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const jsonPath = resolve(here, '..', 'config', 'pricing.json');
const outPath = resolve(here, '..', 'webapp', 'lib', 'pricing.generated.ts');

function render() {
  const raw = JSON.parse(readFileSync(jsonPath, 'utf-8'));
  const currency = raw.currency;
  const monthly = raw.plans.monthly.amount;
  const annualPerYear = raw.plans.annual.amountPerYear;

  // annualPerMonth is DERIVED — never authored. Amounts carry cents since the
  // 2026-09-27 go-live, so the check is done in CENTS (integers): the annual
  // total's cents must divide by 12 exactly, otherwise the displayed "soit N,NN
  // $ par mois" would be a rounded figure presented as an exact one.
  const annualCents = Math.round(annualPerYear * 100);
  if (annualCents % 12 !== 0) {
    console.error(
      `annualPerYear (${annualPerYear}) is not divisible by 12 to the cent ` +
        `(${annualCents} cents / 12 = ${annualCents / 12}) — the monthly ` +
        `equivalent could not be shown exactly. Adjust config/pricing.json.`,
    );
    process.exit(1);
  }
  // Rebuilt from the integer cents so no binary-float residue reaches the UI.
  const annualPerMonth = annualCents / 12 / 100;

  return (
    `// AUTO-GENERATED from config/pricing.json by scripts/gen_pricing.mjs.\n` +
    `// DO NOT EDIT BY HAND. Run \`node scripts/gen_pricing.mjs\` after editing the JSON.\n` +
    `// This is the ONLY place amounts reach the frontend — no price is hard-coded in\n` +
    `// any component. \`annualPerMonth\` is derived (annualPerYear / 12, to the cent).\n` +
    `// Amounts carry cents: never render one raw (\`{PRICING.monthly}\` would print an\n` +
    `// English dot in every locale) — pass it through \`formatAmount\` from\n` +
    `// \`@/lib/pricing\`, which shows exactly two decimals with the locale separator.\n` +
    `export interface PricingModel {\n` +
    `  /** ISO 4217 currency code — USD everywhere, including Canadian customers. */\n` +
    `  currency: string;\n` +
    `  /** Monthly cadence, billed every month. */\n` +
    `  monthly: number;\n` +
    `  /** Annual cadence, billed once per year. */\n` +
    `  annualPerYear: number;\n` +
    `  /** Monthly equivalent of the annual cadence (derived: annualPerYear / 12). */\n` +
    `  annualPerMonth: number;\n` +
    `}\n\n` +
    `export const PRICING: PricingModel = {\n` +
    `  currency: ${JSON.stringify(currency)},\n` +
    `  monthly: ${monthly},\n` +
    `  annualPerYear: ${annualPerYear},\n` +
    `  annualPerMonth: ${annualPerMonth},\n` +
    `} as const;\n`
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
    console.error('pricing.generated.ts is OUT OF DATE. Run: node scripts/gen_pricing.mjs');
    process.exit(1);
  }
  console.log('pricing.generated.ts is up to date.');
} else {
  writeFileSync(outPath, generated);
  console.log(`Wrote ${outPath}`);
}

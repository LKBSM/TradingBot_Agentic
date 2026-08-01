/**
 * PRIX-1 — guards for the pricing single source of truth.
 *
 * The product sells honesty, so pricing has hard invariants:
 *  - amounts live in ONE place (config/pricing.json → pricing.generated.ts);
 *  - the ONLY sanctioned amounts are 39 (monthly), 348 (annual total), 29
 *    (derived monthly equivalent) — no legacy price (49.99 / 39.99 / 479 …)
 *    survives anywhere in the rendered frontend;
 *  - every price is shown with an explicit US-dollar currency;
 *  - no tax is added or mentioned; no struck-through price, discount or promo;
 *  - the four mandatory mentions are present in both fr and en.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { resolve, relative, extname } from 'node:path';
import { describe, expect, it } from 'vitest';
import { PRICING } from '@/lib/pricing.generated';

const WEBAPP = process.cwd();
const REPO = resolve(WEBAPP, '..');

// ---------------------------------------------------------------------------
// 1. Single source of truth
// ---------------------------------------------------------------------------
describe('PRIX-1 — pricing.generated.ts is the single source', () => {
  it('is in sync with config/pricing.json', () => {
    const cfg = JSON.parse(readFileSync(resolve(REPO, 'config/pricing.json'), 'utf-8'));
    expect(PRICING.currency).toBe(cfg.currency);
    expect(PRICING.monthly).toBe(cfg.plans.monthly.amount);
    expect(PRICING.annualPerYear).toBe(cfg.plans.annual.amountPerYear);
    // Derived, exact.
    expect(PRICING.annualPerMonth).toBe(cfg.plans.annual.amountPerYear / 12);
  });

  it('holds the retained prices: $39 / $348 (i.e. $29) in USD', () => {
    expect(PRICING.currency).toBe('USD');
    expect(PRICING.monthly).toBe(39);
    expect(PRICING.annualPerYear).toBe(348);
    expect(PRICING.annualPerMonth).toBe(29);
  });
});

// ---------------------------------------------------------------------------
// Source scan helpers
// ---------------------------------------------------------------------------
const SCANNED_DIRS = ['components', 'app', 'lib', 'messages'] as const;
const SCANNED_EXT = new Set(['.ts', '.tsx', '.json']);
// pricing.generated.ts is the ONLY place literal amounts may appear.
const ALLOW_BASENAMES = new Set(['pricing.generated.ts']);

function walk(dir: string, acc: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (entry === 'node_modules' || entry === '.next' || entry === '__tests__') continue;
    const full = resolve(dir, entry);
    if (statSync(full).isDirectory()) walk(full, acc);
    else if (SCANNED_EXT.has(extname(entry)) && !ALLOW_BASENAMES.has(entry)) acc.push(full);
  }
  return acc;
}

// Read every scanned file ONCE up front (the walk + reads are the slow part;
// re-reading per pattern made the suite time out under load).
const FILES: { rel: string; text: string }[] = SCANNED_DIRS.flatMap((d) =>
  walk(resolve(WEBAPP, d)),
).map((f) => ({ rel: relative(REPO, f), text: readFileSync(f, 'utf-8') }));

// ---------------------------------------------------------------------------
// 2. No legacy / hard-coded price anywhere (unambiguous decimal price forms)
// ---------------------------------------------------------------------------
describe('PRIX-1 — no stale or hard-coded price in the frontend', () => {
  it('scans a non-empty perimeter', () => {
    expect(FILES.length).toBeGreaterThan(50);
  });

  // Decimal-cents amounts are unambiguously prices (never pixels/ids/timestamps).
  const STALE = [/49[.,]99/, /39[.,]99/, /479[.,]88/, /29[.,]99/, /19[.,]99/, /9[.,]99/];
  it.each(STALE.map((re) => [re.source, re] as const))(
    'no occurrence of /%s/',
    (_src, re) => {
      const offenders = FILES.filter((f) => re.test(f.text)).map((f) => f.rel);
      expect(offenders).toEqual([]);
    },
  );
});

// ---------------------------------------------------------------------------
// 3. Message-level invariants (parsed) — every locale
// ---------------------------------------------------------------------------
const LOCALES = ['fr', 'en', 'de', 'es', 'it', 'nl', 'pl', 'pt', 'ar'] as const;

function loadPricingCopy(locale: string) {
  const d = JSON.parse(readFileSync(resolve(WEBAPP, 'messages', `${locale}.json`), 'utf-8'));
  return { pricing: d.landing.pricing as Record<string, string>, billing: d.billing as Record<string, string> };
}

// Only the price-bearing keys — features legitimately contain digits (M15, 4 h).
const PRICING_PRICE_KEYS = [
  'currency', 'perMonth', 'perYear', 'annualBilling', 'monthlyBilling',
  'mentionCancel', 'mentionCurrency', 'mentionRenewal', 'mentionEducational', 'mentionRisk',
] as const;
const BILLING_PRICE_KEYS = ['currency', 'planMonthly', 'planAnnual'] as const;

describe.each(LOCALES)('PRIX-1 — %s pricing copy', (locale) => {
  const { pricing, billing } = loadPricingCopy(locale);
  const priceBlob =
    PRICING_PRICE_KEYS.map((k) => pricing[k] ?? '').join(' ') +
    ' ' +
    BILLING_PRICE_KEYS.map((k) => billing[k] ?? '').join(' ');

  it('currency is explicit (not a bare "$")', () => {
    expect(pricing.currency).toBeTruthy();
    expect(pricing.currency).not.toBe('$');
    expect(billing.currency).toBeTruthy();
    expect(billing.currency).not.toBe('$');
  });

  it('mentions no tax', () => {
    expect(/\b(tax|taxe|tva|tps|tvq)\b/i.test(priceBlob)).toBe(false);
  });

  it('shows no discount / struck price / promo', () => {
    // No discount badge left, no percentage, no "−20" in the price copy.
    expect('discountBadge' in pricing).toBe(false);
    expect(priceBlob.includes('%')).toBe(false);
    expect(priceBlob.includes('−20') || priceBlob.includes('-20')).toBe(false);
  });

  it('carries no literal price amount (amounts arrive via placeholders)', () => {
    // A standalone 2+ digit run in the price copy would be a price leak — the
    // numbers are injected from PRICING at render. "18" (age) lives in the risk
    // mention only, so we strip it before checking.
    const withoutAge = priceBlob.replace(/18/g, '');
    expect(/\d{2,}/.test(withoutAge)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 4. The four mandatory mentions — present in fr AND en (visible, no click)
// ---------------------------------------------------------------------------
const MANDATORY = [
  'mentionCancel',
  'mentionCurrency',
  'mentionEducational',
  'mentionRisk',
] as const;

describe.each(['fr', 'en'] as const)('PRIX-1 — mandatory mentions (%s)', (locale) => {
  const { pricing } = loadPricingCopy(locale);
  it.each(MANDATORY)('%s is present and non-empty', (key) => {
    expect(typeof pricing[key]).toBe('string');
    expect((pricing[key] ?? '').trim().length).toBeGreaterThan(0);
  });
  it('renewal notice is present (Quebec consumer law)', () => {
    expect((pricing.mentionRenewal ?? '').trim().length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 5. Surfaces consume the single source (no amount hard-coded in components)
// ---------------------------------------------------------------------------
describe('PRIX-1 — pricing components import the generated source', () => {
  it.each([
    'components/landing/PricingSection.tsx',
    'components/seo/JsonLd.tsx',
    'components/billing/SubscriptionPanel.tsx',
  ])('%s imports @/lib/pricing.generated', (rel) => {
    const src = readFileSync(resolve(WEBAPP, rel), 'utf-8');
    expect(src.includes("@/lib/pricing.generated")).toBe(true);
  });
});

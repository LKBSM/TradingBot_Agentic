/**
 * PRIX-1 — guards for the pricing single source of truth.
 *
 * The product sells honesty, so pricing has hard invariants:
 *  - amounts live in ONE place (config/pricing.json → pricing.generated.ts);
 *  - the ONLY sanctioned amounts are the live Stripe ones — 39.99 (monthly),
 *    359.88 (annual total), 29.99 (derived monthly equivalent) — and no LEGACY
 *    price survives anywhere in the rendered frontend. The pre-go-live pair
 *    (39 / 348, i.e. 29) is now itself a legacy price: it is forbidden in every
 *    price-shaped form, so a half-finished amount change fails here instead of
 *    being discovered by a customer charged one figure and shown another;
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
    // Derived, exact — rebuilt from integer cents, like the generator does.
    expect(PRICING.annualPerMonth).toBe(
      Math.round(cfg.plans.annual.amountPerYear * 100) / 12 / 100,
    );
  });

  it('holds the live Stripe prices: $39.99 / $359.88 (i.e. $29.99) in USD', () => {
    // These are the amounts the LIVE Stripe prices charge (go-live 2026-09-27):
    //   price_1UKOUBFiM5Kf1kQcGwJyVv0W   3999 cents / month
    //   price_1UKOVNFiM5Kf1kQcizm0JiYB  35988 cents / year
    // If this test fails, the site is quoting a price Stripe does not charge.
    expect(PRICING.currency).toBe('USD');
    expect(PRICING.monthly).toBe(39.99);
    expect(PRICING.annualPerYear).toBe(359.88);
    expect(PRICING.annualPerMonth).toBe(29.99);
  });

  it('the annual total divides by 12 to the exact cent', () => {
    // 35988 / 12 = 2999. The page says "soit 29,99 $ par mois"; that has to be
    // the real quotient, not a rounded one dressed up as exact.
    const annualCents = Math.round(PRICING.annualPerYear * 100);
    expect(annualCents % 12).toBe(0);
    expect(annualCents / 12 / 100).toBe(PRICING.annualPerMonth);
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
  // 39.99 / 359.88 / 29.99 are the CURRENT prices and so are absent from this
  // list — they are asserted present in pricing.generated.ts above, and that file
  // is the one place allowed to hold a literal amount.
  //
  // Each pattern is anchored with `(?<![\d.,])`: unanchored, /9[.,]99/ matches
  // inside "39.99" and "29.99", so the CURRENT prices would fail the guard for
  // containing a legacy one as a substring. The anchor makes each amount match
  // only when it starts the number.
  const STALE = [
    /(?<![\d.,])49[.,]99/,
    /(?<![\d.,])479[.,]88/,
    /(?<![\d.,])19[.,]99/,
    /(?<![\d.,])9[.,]99/,
  ];
  it.each(STALE.map((re) => [re.source, re] as const))(
    'no occurrence of /%s/',
    (_src, re) => {
      const offenders = FILES.filter((f) => re.test(f.text)).map((f) => f.rel);
      expect(offenders).toEqual([]);
    },
  );

  // The pre-go-live amounts (39 / 348 / 29) are whole numbers, so a bare "39"
  // is ambiguous — it is also a pixel size, an opacity, a colour channel. Only
  // PRICE-SHAPED occurrences are forbidden: the amount glued to a currency
  // marker, or written with explicit .00 cents. That is precise enough to catch
  // a missed "39 $ US" and loose enough never to fire on geometry.
  const LEGACY = [
    { name: '$39 / US$348 / $29 (symbol before)', re: /(?:US\$|\$|USD)\s*(?:&nbsp;)?\s*(?:39|348|29)(?![\d.,])/ },
    { name: '39 $ / 348 USD (symbol after)', re: /(?<![\d.,])(?:39|348|29)(?:&nbsp;|\s)*(?:US\$|\$|USD)/ },
    { name: '39.00 / 348,00 (explicit cents)', re: /(?<![\d.,])(?:39|348|29)[.,]00(?![\d])/ },
  ] as const;
  it.each(LEGACY.map((l) => [l.name, l.re] as const))(
    'no legacy pre-go-live price, %s',
    (_name, re) => {
      const offenders = FILES.filter((f) => re.test(f.text)).map((f) => f.rel);
      expect(offenders).toEqual([]);
    },
  );
});

// ---------------------------------------------------------------------------
// 3. Message-level invariants (parsed) — every locale
// ---------------------------------------------------------------------------
const LOCALES = ['fr', 'en'] as const;

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

// AUTO-GENERATED from config/pricing.json by scripts/gen_pricing.mjs.
// DO NOT EDIT BY HAND. Run `node scripts/gen_pricing.mjs` after editing the JSON.
// This is the ONLY place amounts reach the frontend — no price is hard-coded in
// any component. `annualPerMonth` is derived (annualPerYear / 12, to the cent).
// Amounts carry cents: never render one raw (`{PRICING.monthly}` would print an
// English dot in every locale) — pass it through `formatAmount` from
// `@/lib/pricing`, which shows exactly two decimals with the locale separator.
export interface PricingModel {
  /** ISO 4217 currency code — USD everywhere, including Canadian customers. */
  currency: string;
  /** Monthly cadence, billed every month. */
  monthly: number;
  /** Annual cadence, billed once per year. */
  annualPerYear: number;
  /** Monthly equivalent of the annual cadence (derived: annualPerYear / 12). */
  annualPerMonth: number;
}

export const PRICING: PricingModel = {
  currency: "USD",
  monthly: 39.99,
  annualPerYear: 359.88,
  annualPerMonth: 29.99,
} as const;

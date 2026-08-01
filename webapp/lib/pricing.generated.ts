// AUTO-GENERATED from config/pricing.json by scripts/gen_pricing.mjs.
// DO NOT EDIT BY HAND. Run `node scripts/gen_pricing.mjs` after editing the JSON.
// This is the ONLY place amounts reach the frontend — no price is hard-coded in
// any component. `annualPerMonth` is derived (annualPerYear / 12).
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
  monthly: 39,
  annualPerYear: 348,
  annualPerMonth: 29,
} as const;

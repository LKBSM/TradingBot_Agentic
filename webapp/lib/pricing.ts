/**
 * Pricing presentation — the ONLY way an amount becomes a string.
 *
 * Amounts have carried cents since the 2026-09-27 go-live (39.99 / month,
 * 359.88 / year, i.e. 29.99 / month). A raw `{PRICING.monthly}` in JSX prints
 * the JavaScript number literal — "39.99" with an English dot — in all nine
 * locales, which is wrong everywhere but English. Every price therefore goes
 * through `formatAmount`, which renders exactly two decimals with the viewer's
 * own separator ("39,99" in fr, "39.99" in en).
 *
 * The amounts themselves stay in `./pricing.generated` (single source:
 * config/pricing.json). Nothing here invents, rounds or discounts a price.
 */
import { PRICING } from './pricing.generated';

/** Both decimals always shown: a price is never "39,9" nor "40". */
const FRACTION_DIGITS = 2;

/**
 * Format an amount for display in `locale` — two decimals, locale separator,
 * no currency symbol (the copy supplies the explicit "$ US" / "USD" itself, so
 * the currency is never reduced to a bare "$").
 */
export function formatAmount(amount: number, locale: string): string {
  // `-u-nu-latn` pins Western digits. Without it, ICU picks the locale's default
  // numbering system, and an Arabic locale would render "٣٩٫٩٩" next to a Latin
  // "$ US" — a mix that varies with the ICU build the browser ships, so the same
  // page would show different digits to different visitors. The amount is a USD
  // figure paired with a Latin currency label; the SEPARATOR is what has to be
  // local, not the digits.
  return new Intl.NumberFormat(`${locale}-u-nu-latn`, {
    minimumFractionDigits: FRACTION_DIGITS,
    maximumFractionDigits: FRACTION_DIGITS,
  }).format(amount);
}

/**
 * Machine-readable form (schema.org `Offer.price`, HTML `content` attributes):
 * a dot decimal separator and two digits, never localised.
 */
export function machineAmount(amount: number): string {
  return amount.toFixed(FRACTION_DIGITS);
}

/**
 * What a year on the annual cadence costs less than twelve monthly charges.
 * Computed in integer cents so the subtraction cannot surface a float residue
 * (39.99 × 12 − 359.88 must be 120, not 120.00000000000006).
 */
export function annualSavingPerYear(): number {
  const cents = Math.round(PRICING.monthly * 100) * 12 - Math.round(PRICING.annualPerYear * 100);
  return cents / 100;
}

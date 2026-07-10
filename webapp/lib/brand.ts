/**
 * Brand constants — single source of truth for the wordmark and its baseline.
 *
 * The baseline stays in ENGLISH across every locale on purpose: "MIA" is an
 * acronym of "Multi-asset Intelligence Assistant", and a localized expansion
 * would spell a different acronym in each language (e.g. FR "Assistant
 * d'intelligence multi-actifs" → AIM, not MIA). Keeping the source words in
 * English is the standard move for brand baselines and lets the acronym stay
 * legible everywhere. "Markets" already lives in the wordmark, so the baseline
 * carries only the three source words.
 */
export const BRAND_NAME = 'MIA Markets';
export const BRAND_BASELINE = 'Multi-asset Intelligence Assistant';

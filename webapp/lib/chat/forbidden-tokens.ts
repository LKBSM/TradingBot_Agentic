/**
 * Forbidden-token validator for chatbot output.
 *
 * Mirrors `src/intelligence/score_calibration.py::contains_forbidden_token`.
 * The two implementations MUST stay in sync — if you add a phrase here,
 * also add it in the Python module (or vice versa). The Python tests in
 * `tests/test_score_calibration.py` lock the surface; this file is the
 * webapp-side defence so a streamed LLM response cannot slip a
 * prescriptive verb past the post-process check.
 *
 * Two match modes:
 *   - "phrase": substring match (multi-word; only appears in prescriptive
 *     contexts).
 *   - "word":   word-boundary match so we don't flag "buyer" as "buy".
 */

export const FORBIDDEN_TOKENS_FR_WORDS = [
  'achetez',
  'vendez',
  'achète',
  'vends',
  'garanti',
  'garantie',
  'garanties',
  'tp',
  'sl',
] as const;

export const FORBIDDEN_TOKENS_FR_PHRASES = [
  '100% sûr',
  '100 % sûr',
  'edge prouvé',
  'edge prouvee',
  "signal d'achat",
  'signal de vente',
  "signal d'entrée",
  'stop-loss recommandé',
  'stop loss recommandé',
  'objectif de prix',
  'profit garanti',
  'opportunité à saisir',
  'ouvrez une position',
  'fermez votre position',
] as const;

export const FORBIDDEN_TOKENS_EN_WORDS = [
  'buy',
  'sell',
  'guaranteed',
  'tp',
  'sl',
] as const;

export const FORBIDDEN_TOKENS_EN_PHRASES = [
  '100% sure',
  '100 % sure',
  'proven edge',
  'buy signal',
  'sell signal',
  'entry signal',
  'recommended stop-loss',
  'recommended stop loss',
  'price target',
  'guaranteed profit',
  'opportunity to seize',
  'open a long',
  'open a short',
  'close your position',
] as const;

export type Language = 'fr' | 'en';

/**
 * Return the first forbidden token found in `text`, or `null`.
 *
 * Phrases are checked before bare words — the more specific match is more
 * useful to the operator audit when a token slips through.
 */
export function containsForbiddenToken(
  text: string,
  language: Language = 'fr',
): string | null {
  if (!text) return null;
  const lowered = text.toLowerCase();
  const isFr = language.toLowerCase().startsWith('fr');
  const phrases = isFr ? FORBIDDEN_TOKENS_FR_PHRASES : FORBIDDEN_TOKENS_EN_PHRASES;
  const words = isFr ? FORBIDDEN_TOKENS_FR_WORDS : FORBIDDEN_TOKENS_EN_WORDS;

  for (const phr of phrases) {
    if (lowered.includes(phr)) return phr;
  }
  for (const w of words) {
    // \b in JS regex uses ASCII word semantics; for the French accented
    // forms we use a custom boundary that allows non-word chars (or
    // string boundary) on either side. `(?<![\\p{L}\\p{N}])w(?![\\p{L}\\p{N}])`
    // is the Unicode-correct version (Node 18+ supports `u` flag).
    const re = new RegExp(
      `(?<![\\p{L}\\p{N}])${escapeRegex(w)}(?![\\p{L}\\p{N}])`,
      'u',
    );
    if (re.test(lowered)) return w;
  }
  return null;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Safe fallback paragraph used to replace the assistant's reply when a
 * forbidden token slips through. Mirrors the system-prompt's pedagogical
 * refusal stance.
 */
export const SAFE_FALLBACK_FR =
  "Je ne peux pas répondre à cette question dans une posture prescriptive (achat, vente, taille de position, garantie). Mon rôle est de décrire la structure de marché — pose-moi une question sur l'un des chiffres du contexte (conviction, IC conformel, composantes, régime, volatilité) et je l'explique.";

export const SAFE_FALLBACK_EN =
  "I can't answer that in a prescriptive way (buy, sell, position-sizing, guarantees). My job is to describe the market structure — ask me about any number from the context (conviction, conformal CI, components, regime, volatility) and I'll explain it.";

/**
 * System prompt for the Sentinel chatbot. Designed for prompt caching:
 * the static portion (compliance rules, persona, style) is identical
 * across all calls so Anthropic can cache it across requests within
 * the 5-minute window — bringing the marginal cost on repeated questions
 * down to ~10 % of the uncached rate.
 *
 * LEGAL-PENDING: the compliance wording below is a placeholder pending the
 * legal terminal review (UE 2024/2811 finfluencer + MiFID II March 2026).
 * Replace with the finalised legal text in the integration pass — keep the
 * SAME structure so the cache key stays stable.
 */
export const SENTINEL_SYSTEM_PROMPT = `Tu es Sentinel, l'assistant quantitatif conversationnel de MIA Markets — un indicateur de marché pour XAU/USD et le forex. Tu réponds aux questions des utilisateurs sur leur lecture de marché en cours, en français par défaut.

# Rôle et posture

Tu décris l'état du marché et tu calibres des probabilités. Tu n'es PAS un service de signaux ni un conseiller en investissement.

Le produit MIA Markets est un INDICATEUR éducatif — pas un service de recommandations personnalisées. Tu adoptes une posture pédagogique, honnête sur l'incertitude, et tu refuses systématiquement les demandes prescriptives.

# Règles compliance non négociables (UE 2024/2811 finfluencer)

1. **Aucune instruction d'achat, de vente, de timing ou de taille de position.** Si l'utilisateur demande "dois-je acheter/vendre/sortir ?", tu refuses pédagogiquement : tu expliques que ton rôle est de décrire la structure, qu'il est le seul à pouvoir décider en fonction de sa gestion du risque, de son horizon et de sa situation personnelle.

2. **Vocabulaire interdit dans un contexte prescriptif** : "achetez", "vendez", "garanti", "edge prouvé", "signal d'entrée", "stop-loss recommandé", "objectif", "TP", "SL", "profit garanti", "opportunité à saisir".

3. **Mention systématique de l'incertitude** : quand tu cites une probabilité ou une statistique historique, mentionne l'intervalle de confiance (IC 95 %), la marge d'erreur conformelle, ou que la performance passée ne garantit pas la performance future.

   *Comment expliquer un intervalle conformel (Gibbs-Candès / ACI)* : "La conviction affichée à X repose sur un intervalle conformel [lo–hi] à 90 %. Cela veut dire que sur les setups historiques similaires, la conviction réelle est tombée dans cet intervalle dans environ 90 % des cas. Plus l'intervalle est large, moins le modèle est sûr du score ponctuel."

   *Comment expliquer le profit factor + IC95 bootstrap* : "Sur N setups similaires (fenêtre F), le profit factor observé est PF, avec un intervalle de confiance bootstrap à 95 % de [lo–hi]. Si la borne basse est ≤ 1, on ne peut pas dire que le système a un avantage prouvé sur l'échantillon — c'est compatible avec du hasard."

4. **edge_claim=false** : MIA Markets ne revendique pas d'edge prouvé tant que les critères empiriques (PF > 1.20 sur 12 mois rolling, DSR > 1.0, PBO < 0.5, walk-forward 2+ ans hors-sample) ne sont pas franchis. Tu peux décrire l'historique observé, jamais le présenter comme une promesse de performance.

5. **Démonstration paper-trading** : l'utilisateur regarde une démonstration éducative, pas un track-record live commercialisé. Mentionne-le si la question implique un engagement réel.

# Style de réponse

- Français naturel, ton respectueux et pédagogique.
- Réponses concises : 100-300 mots typiquement. 400 mots maximum sauf si l'utilisateur demande explicitement un développement long.
- Cite des nombres précis du contexte signal quand pertinent (PF, IC, posterior HMM, cp_prob, ATR, etc.).
- Pas de markdown gras / italique. Tu peux utiliser des sauts de ligne pour la lisibilité et des listes à puces sobres.
- Tu peux dire "je ne sais pas" si la question sort du champ de l'indicateur.
- Si l'utilisateur écrit dans une autre langue, réponds dans cette langue tout en respectant les règles compliance.

# Limites du système (à mentionner si demandé)

- Pas d'accès au marché en temps réel — tu lis la lecture algorithmique fournie en contexte.
- Pas d'historique des conversations précédentes (chaque session est indépendante).
- Pas de personnalisation au profil de l'utilisateur (tier, taille de compte, etc.).
- Pas de prédiction de prix — tu décris l'état actuel et l'historique de setups similaires.

Tu commences directement par la réponse, sans préambule du type "Bonjour" ou "Voici ma réponse :".`;

/**
 * DG-042 — NARRATIVE_MODE tier-routed model selection (révision 2026-05-28
 * post-pivot positioning 2026-05-27).
 *
 * Grille pricing publique V1 (3 tiers, INSTITUTIONAL retiré → Calendly) :
 *
 *   Découverte  (gratuit) → Haiku 4.5
 *   Approfondie (9 €/mo)  → Haiku 4.5
 *   Intégrale   (19 €/mo) → Haiku 4.5 par défaut, Sonnet 4.6 sur uplift
 *
 * Inactif V1 (gardé en commentaire pour activation future B2B) :
 *
 *   // Institutional (Calendly) → Opus 4.7
 *
 * Pourquoi Haiku par défaut sur les 3 tiers : Haiku tient bien sur le
 * narratif court conditionné par un signal_summary structuré (~2840
 * tokens de context). La différence Sonnet/Opus ne devient visible que
 * sur les chaînes de raisonnement longues. Payer Sonnet 5×Haiku sur
 * toutes les requêtes Intégrale ferait exploser la marge.
 *
 * Pourquoi un uplift Sonnet conditionnel sur Intégrale uniquement :
 * deux situations bénéficient nettement de la capacité Sonnet :
 *
 *  1. Décomposition de score (waterfall des 8 composantes argumentée).
 *     Détection lexicale via ``requiresSonnetUplift`` : la question
 *     contient un trigger ("pourquoi", "décompose", "explique le score",
 *     "breakdown", etc.). Sonnet sait tenir une chaîne argumentaire de
 *     7-8 étapes sans dériver — Haiku rate ~15 % du temps sur ce type.
 *
 *  2. Conversation longue (cohérence multi-turn). Dès que l'historique
 *     dépasse 3 turns, on bascule Sonnet pour préserver la cohérence
 *     contextuelle. Haiku tend à oublier les contraintes établies tôt
 *     dans la session.
 *
 * Les deux triggers sont gratuits à évaluer (regex + count) et ne
 * laissent pas fuiter de signal personnel — pas de profilage.
 */
export type SentinelModel =
  | 'claude-haiku-4-5-20251001'
  | 'claude-sonnet-4-6'
  | 'claude-opus-4-7';

/** Public V1 tier identifiers. Use the canonical lowercase form. */
export type SentinelTier = 'decouverte' | 'approfondie' | 'integrale';

/** Aliases — legacy / Stripe metadata strings the route may receive. */
const TIER_ALIASES: Record<string, SentinelTier> = {
  // Public canonical
  decouverte: 'decouverte',
  approfondie: 'approfondie',
  integrale: 'integrale',
  // Legacy English names from earlier auth/tier_manager schemas
  free: 'decouverte',
  starter: 'approfondie',
  pro: 'integrale',
  // INSTITUTIONAL is INTENTIONALLY NOT mapped here — public V1 grid no
  // longer surfaces it (post-pivot 2026-05-27). B2B access is via
  // Calendly and uses a dedicated server-side router (not this map).
};

const TIER_BASE_MODEL: Record<SentinelTier, SentinelModel> = {
  decouverte: 'claude-haiku-4-5-20251001',
  approfondie: 'claude-haiku-4-5-20251001',
  integrale: 'claude-haiku-4-5-20251001',
};

export const DEFAULT_MODEL: SentinelModel = 'claude-haiku-4-5-20251001';
export const DEFAULT_TIER: SentinelTier = 'decouverte';

// ───────────────────────────────────────────────────────────────────────────
// Commented out: future B2B Institutional path (Calendly-gated).
// When re-enabled, route through a server-side proxy that re-checks the
// Calendly-issued JWT before applying this mapping.
//
//   const TIER_INSTITUTIONAL_MODEL: SentinelModel = 'claude-opus-4-7';
//
// ───────────────────────────────────────────────────────────────────────────

/** Trigger lexicon for the Haiku→Sonnet uplift (Intégrale tier only). */
const SCORE_DECOMP_RE =
  /\b(?:pourquoi|décompose|decompose|explique\s+(?:le|la|ce)\s+(?:score|conviction|note)|breakdown|décomposition|repartition)/i;

/** History-length threshold for the cohérence multi-turn uplift. */
const HISTORY_UPLIFT_TURNS = 3;

export interface ModelSelectionContext {
  /** Free-form user question text — used for the lexical uplift trigger. */
  question?: string;
  /** Count of prior user/assistant turns (excludes the current one). */
  historyTurns?: number;
}

function normaliseTier(raw: string | null | undefined): SentinelTier | null {
  if (!raw) return null;
  const lower = raw.toLowerCase().trim();
  return TIER_ALIASES[lower] ?? null;
}

export function requiresSonnetUplift(ctx: ModelSelectionContext): boolean {
  const q = ctx.question ?? '';
  const turns = ctx.historyTurns ?? 0;
  if (turns > HISTORY_UPLIFT_TURNS) return true;
  if (SCORE_DECOMP_RE.test(q)) return true;
  return false;
}

/**
 * Resolve the LLM model for a given subscriber tier + request context.
 *
 * Unknown / missing tiers fall back to ``DEFAULT_MODEL`` (Haiku — cost
 * safe). Intégrale tier may get a Sonnet uplift when the question or
 * conversation length triggers it. See module header for the rationale.
 */
export function modelForTier(
  tier: string | null | undefined,
  ctx: ModelSelectionContext = {},
): SentinelModel {
  const canonical = normaliseTier(tier);
  if (!canonical) return DEFAULT_MODEL;
  const base = TIER_BASE_MODEL[canonical];
  if (canonical === 'integrale' && requiresSonnetUplift(ctx)) {
    return 'claude-sonnet-4-6';
  }
  return base;
}

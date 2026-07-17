/**
 * Contenu de la page /methodology (Chantier 5.D) — documentation algorithmique
 * transparente.
 *
 * Data-driven : la page mappe ces structures sur des composants génériques
 * (MethodologySection / ConceptCard / ScoreFormula) pour rester légère.
 *
 * Source de vérité partagée : les termes et définitions courtes proviennent du
 * glossaire central `lib/glossary.ts` (mêmes ancres que les tooltips ⓘ de l'UI).
 *
 * Niveau 1.5 strict : tout est DESCRIPTIF. On explique ce que l'indicateur
 * mesure et comment il le détecte — jamais une prédiction, jamais une promesse,
 * jamais une prétention de supériorité.
 */

import { GLOSSARY, type GlossaryKey } from '@/lib/glossary';

/** Un concept de structure SMC : sa définition + comment le moteur le détecte. */
export interface MethodologyConcept {
  /** id d'ancrage (sans #) — doit matcher l'ancre du glossaire. */
  id: string;
  /** Clé glossaire (terme + définition courte réutilisés). */
  glossaryKey: GlossaryKey;
  /** Comment le moteur détecte ce concept, en langage simple. */
  detection: string;
}

/** Un « score » affiché : ce qu'il décrit et les variables qui le composent. */
export interface MethodologyFormula {
  /** id d'ancrage (sans #). */
  id: string;
  title: string;
  /** Ce que l'indicateur décrit (niveau 1.5, pas de prédiction). */
  description: string;
  /** Les variables prises en compte, en langage simple. */
  variables: ReadonlyArray<string>;
}

// ── Section 2 : concepts de structure SMC ─────────────────────────────────────

export const SMC_CONCEPTS: ReadonlyArray<MethodologyConcept> = [
  {
    id: 'order-block',
    glossaryKey: 'order_block',
    detection:
      'On repère la dernière bougie de sens opposé juste avant un mouvement marqué. C’est la zone de prix d’où le mouvement est parti — souvent là où des ordres importants se sont accumulés. Le moteur en suit l’état : actif, mitigé (déjà retouché) ou invalidé.',
  },
  {
    id: 'fvg',
    glossaryKey: 'fvg',
    detection:
      'On détecte un « trou » de prix laissé par trois bougies consécutives où la première et la troisième ne se chevauchent pas : un mouvement trop rapide pour que tous les échanges aient lieu. Le moteur mesure la taille de ce trou et suit s’il reste ouvert, partiellement comblé ou comblé.',
  },
  {
    id: 'bos',
    glossaryKey: 'bos',
    detection:
      'On compare le prix aux sommets et creux récents (les « swings »). Quand il franchit nettement le dernier sommet (ou creux) dans le sens de la tendance, c’est une cassure de structure : la tendance en cours se confirme.',
  },
  {
    id: 'choch',
    glossaryKey: 'choch',
    detection:
      'Même logique que la cassure de structure, mais dans le sens OPPOSÉ à la tendance récente. C’est le premier signe possible d’un retournement de structure — décrit comme tel, sans en déduire une direction future.',
  },
  {
    id: 'retest',
    glossaryKey: 'retest',
    detection:
      'Après une cassure, le moteur surveille si le prix revient toucher le niveau franchi. Il suit l’état du retour : pas encore revenu, en cours de test, confirmé (le niveau tient) ou invalidé (le prix l’a traversé en sens inverse).',
  },
  {
    id: 'mtf',
    glossaryKey: 'mtf',
    detection:
      'La même lecture est calculée sur plusieurs unités de temps (15 min, 1 h, 4 h, 1 jour…). On affiche le biais de chacune côte à côte : quand plusieurs durées pointent dans le même sens, la structure est plus nette. Aucune n’est pondérée pour produire un score directif.',
  },
];

// ── Section 3 : comment nous décrivons les éléments affichés ───────────────────

export const SCORE_FORMULAS: ReadonlyArray<MethodologyFormula> = [
  {
    id: 'order-block',
    title: 'Order Block',
    description:
      'Une zone laissée par une bougie de déséquilibre avant un mouvement marqué. On l’affiche à sa formation avec sa fourchette de prix et son état présent (active, testée, mitigée, invalidée) — sans note ni classement de qualité, aucune probabilité de réussite.',
    variables: [
      'la fourchette de prix de la zone',
      'sa taille rapportée à la volatilité récente',
      'son état courant : active, testée, mitigée ou invalidée',
    ],
  },
  {
    id: 'fvg-force',
    title: 'Statut d’un Fair Value Gap',
    description:
      'Décrit l’état d’une zone de déséquilibre : active, partiellement comblée ou comblée. La taille relative du trou indique son ampleur, sans présager du moment où il sera comblé.',
    variables: [
      'la taille du trou rapportée à la volatilité moyenne récente',
      'le nombre de bougies écoulées depuis sa création',
      'la part déjà comblée par le prix',
    ],
  },
  {
    id: 'phase-de-marche',
    title: 'Phase de marché',
    description:
      'Trois libellés descriptifs côte à côte — tendance, volatilité, phase (accumulation / tendance / distribution / range / expansion). Une photo de l’état observé, jamais une instruction.',
    variables: [
      'la direction et la pente des sommets/creux récents',
      'l’amplitude moyenne des bougies (volatilité)',
      'l’alignement entre unités de temps',
    ],
  },
  {
    id: 'incertitude',
    title: 'Plage d’incertitude observée',
    description:
      'L’écart de prix typiquement observé récemment autour de la lecture. C’est une mesure d’incertitude affichée honnêtement — pas une fourchette de prévision ni un objectif.',
    variables: [
      'la dispersion récente des prix, mesurée par l’ATR (Average True Range, la moyenne standard de l’amplitude des bougies)',
      'l’unité de temps de la lecture',
    ],
  },
];

// ── Section 4 : source de données ─────────────────────────────────────────────

export const DATA_SOURCE = {
  provider: 'Twelve Data',
  detail:
    'Données de marché via l’API Twelve Data (palier gratuit, 800 requêtes par jour en V1).',
  coverage: 'XAU/USD et EUR/USD, sur 15 minutes, 1 heure et 4 heures.',
  refresh: 'Mise à jour à chaque clôture de bougie.',
} as const;

// ── Section 5 : ce que nous ne faisons pas ────────────────────────────────────

export const NEVER_DO: ReadonlyArray<string> = [
  'Prédire le sens ou l’ampleur d’un mouvement.',
  'Émettre un signal de trade, un point d’entrée ou de sortie.',
  'Calculer un objectif de prix, un stop-loss ou un take-profit.',
  'Afficher un score de probabilité de réussite.',
  'Promettre un gain, un rendement ou une performance.',
];

/** Citation imposée (lock 2) — réutilisée depuis la landing pour cohérence. */
export const ENGAGEMENT_QUOTE =
  'Aucun indicateur de marché ne devrait promettre des gains. Nous n’en faisons pas. Ce que nous offrons, c’est une compréhension augmentée du marché — pas une performance financière.';

/** Helper : le glossaire reste la source unique pour term + short. */
export function conceptTerm(key: GlossaryKey): string {
  return GLOSSARY[key].term;
}
export function conceptShort(key: GlossaryKey): string {
  return GLOSSARY[key].short;
}

/**
 * Glossaire central — source de vérité UNIQUE pour la vulgarisation des termes
 * techniques (Chantier 5.D).
 *
 * Consommé par :
 *   · `InfoTooltip` (tooltips ⓘ dans l'UI market-reading)
 *   · la page `/methodology` (documentation algorithmique transparente)
 *
 * Convention par entrée (validée founder 2026-06-07) :
 *   · term   — le terme technique tel qu'affiché (ex: "Order Block")
 *   · short  — définition courte, vulgarisée, niveau 1.5 (affichée dans le tooltip)
 *   · anchor — ancre vers /methodology (ex: "#order-block")
 *
 * Niveau 1.5 strict : chaque définition DÉCRIT un concept observé. Aucun verbe
 * directif (acheter / vendre), aucune promesse de résultat, aucune prétention
 * prédictive. On explique ce que l'indicateur mesure, jamais ce qu'il prédit.
 */

export interface GlossaryEntry {
  /** Terme technique tel qu'affiché dans l'UI. */
  term: string;
  /** Définition courte vulgarisée — contenu du tooltip. */
  short: string;
  /** Ancre vers la page /methodology (ex: "#order-block"). */
  anchor: string;
}

/** Clés stables — référencées par les composants et la page /methodology. */
export type GlossaryKey =
  | 'bos'
  | 'choch'
  | 'order_block'
  | 'fvg'
  | 'retest'
  | 'liquidity'
  | 'mtf'
  | 'volatility'
  | 'atr'
  | 'market_phase'
  | 'uncertainty';

export const GLOSSARY: Record<GlossaryKey, GlossaryEntry> = {
  bos: {
    term: 'Cassure de structure (BOS)',
    short:
      'Le prix franchit un sommet ou un creux récent : la tendance en cours se confirme.',
    anchor: '#bos',
  },
  choch: {
    term: 'Changement de caractère (CHOCH)',
    short:
      'Le prix casse dans le sens opposé à la tendance récente : un possible retournement de structure.',
    anchor: '#choch',
  },
  order_block: {
    term: 'Order Block',
    short:
      'La dernière bougie avant un mouvement marqué : une zone de prix où des ordres importants se sont accumulés.',
    anchor: '#order-block',
  },
  fvg: {
    term: 'Fair Value Gap (FVG)',
    short:
      'Un « trou » laissé par un mouvement rapide, où peu d’échanges ont eu lieu : le prix revient souvent le combler.',
    anchor: '#fvg',
  },
  retest: {
    term: 'Retest',
    short:
      'Après une cassure, le prix revient toucher le niveau franchi pour le tester : la zone confirme ou s’invalide.',
    anchor: '#retest',
  },
  liquidity: {
    term: 'Liquidité externe (BSL / SSL)',
    short:
      'Des niveaux où des ordres en attente s’accumulent — au-dessus de sommets égaux (BSL) ou sous des creux égaux (SSL). On indique si le niveau est intact, pris (mèche puis retour — il tient encore) ou cassé.',
    anchor: '#liquidite-externe',
  },
  mtf: {
    term: 'Confluence multi-timeframe',
    short:
      'On compare la lecture sur plusieurs durées (15 min, 1 h, 4 h…) : quand elles pointent dans le même sens, la structure est plus nette.',
    anchor: '#mtf',
  },
  volatility: {
    term: 'Volatilité',
    short:
      'L’amplitude moyenne récente des bougies : à quel point le marché bouge en ce moment, comparé à son habitude.',
    anchor: '#volatilite',
  },
  atr: {
    term: 'ATR',
    short:
      'Average True Range : la mesure standard de la volatilité moyenne récente du marché.',
    anchor: '#volatilite',
  },
  market_phase: {
    term: 'Phase de marché',
    short:
      'L’état observé du marché — accumulation, tendance, distribution ou range — décrit, jamais une instruction.',
    anchor: '#phase-de-marche',
  },
  uncertainty: {
    term: 'Plage d’incertitude observée',
    short:
      'L’écart de prix typiquement observé récemment autour de la lecture : une mesure d’incertitude affichée, pas une prévision.',
    anchor: '#incertitude',
  },
};

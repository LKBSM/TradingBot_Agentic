/**
 * DG-114-REDUCED — 3 contextual suggested questions for the chatbot panel.
 *
 * The questions are *derived from the live InsightSignalV2*, so the user
 * always has at least three concrete entry points to start a conversation
 * that's relevant to *this specific* setup. They are NOT prescriptive —
 * each one asks the chatbot to explain a number, never to recommend.
 *
 * Rules (per brief):
 *   Q1 → depends on conviction tier (weak / moderate / strong / institutional).
 *   Q2 → depends on the top-contributing component in the breakdown.
 *   Q3 → if an event is ≤ 4 hours away, ask about event impact;
 *        else if conviction ≥ 70, ask about the conformal CI;
 *        else fall back to the regime gate decision.
 *
 * All three are returned in the user's preferred display language.
 */

import type { InsightSignalV2, ConvictionLabel, ComponentBreakdown } from '@/types/insight';

export type SuggestedLanguage = 'fr' | 'en';

export interface SuggestedQuestion {
  /** Human-readable text shown to the user. */
  text: string;
  /** Telemetry slot to know which rule fired. */
  source: 'q1_conviction' | 'q2_top_component' | 'q3_event' | 'q3_ci' | 'q3_regime';
}

// ---------------------------------------------------------------------------
// Q1 — conviction-tier
// ---------------------------------------------------------------------------

const Q1_FR: Record<ConvictionLabel, string> = {
  weak: "Pourquoi la conviction est-elle aussi basse ?",
  moderate:
    "Quels composants tirent la conviction vers le haut ou vers le bas ?",
  strong:
    "Quel composant pèse le plus dans cette conviction forte ?",
  institutional:
    "Pourquoi ce setup atteint-il le label institutional ?",
};

const Q1_EN: Record<ConvictionLabel, string> = {
  weak: 'Why is the conviction this low?',
  moderate:
    'Which components are pulling the conviction up or down?',
  strong: 'Which component contributes most to this strong conviction?',
  institutional:
    'Why does this setup reach the institutional label?',
};

function pickQ1(
  conviction: ConvictionLabel,
  lang: SuggestedLanguage,
): SuggestedQuestion {
  const text = lang === 'fr' ? Q1_FR[conviction] : Q1_EN[conviction];
  return { text, source: 'q1_conviction' };
}

// ---------------------------------------------------------------------------
// Q2 — top-component
// ---------------------------------------------------------------------------

const Q2_FR_BY_COMPONENT: Record<string, string> = {
  bos: "Que veut dire la cassure de structure (BOS) ici ?",
  smc_structure: "Que dit la structure SMC sur ce setup ?",
  order_block: "Pourquoi l'Order Block est-il aussi influent ?",
  fvg: "Que m'indique la Fair Value Gap ?",
  retest: "Le retest est dans quel état exactement ?",
  regime: "Pourquoi le régime actuel pèse autant ?",
  momentum_rsi_div: "Que m'apporte la divergence RSI ici ?",
  news: "Comment le contexte news affecte cette lecture ?",
  vol_forecast: "Que dit la prévision de volatilité ?",
  htf: "Le contexte HTF est-il aligné avec le setup ?",
};

const Q2_EN_BY_COMPONENT: Record<string, string> = {
  bos: 'What does the Break of Structure (BOS) mean here?',
  smc_structure: 'What does the SMC structure say about this setup?',
  order_block: 'Why is the Order Block this influential?',
  fvg: 'What does the Fair Value Gap tell me?',
  retest: 'What state is the retest in exactly?',
  regime: 'Why does the current regime weigh this much?',
  momentum_rsi_div: 'What does the RSI divergence add here?',
  news: 'How does the news context affect this reading?',
  vol_forecast: 'What does the volatility forecast say?',
  htf: 'Is the HTF context aligned with the setup?',
};

const Q2_FR_FALLBACK =
  "Décris-moi la contribution des 8 composantes en détail.";
const Q2_EN_FALLBACK = 'Describe the 8-component contribution in detail.';

function pickQ2(
  components: ReadonlyArray<ComponentBreakdown>,
  lang: SuggestedLanguage,
): SuggestedQuestion {
  if (!components || components.length === 0) {
    return {
      text: lang === 'fr' ? Q2_FR_FALLBACK : Q2_EN_FALLBACK,
      source: 'q2_top_component',
    };
  }
  const top = components.reduce((a, b) =>
    (b.contribution ?? 0) > (a.contribution ?? 0) ? b : a,
  );
  const map = lang === 'fr' ? Q2_FR_BY_COMPONENT : Q2_EN_BY_COMPONENT;
  const text = map[top.name] ?? (lang === 'fr' ? Q2_FR_FALLBACK : Q2_EN_FALLBACK);
  return { text, source: 'q2_top_component' };
}

// ---------------------------------------------------------------------------
// Q3 — event / CI / regime gate (in priority order)
// ---------------------------------------------------------------------------

function pickQ3(
  signal: InsightSignalV2,
  lang: SuggestedLanguage,
): SuggestedQuestion {
  const eventIn = signal.event_readout?.next_event_in_minutes;
  if (eventIn !== null && eventIn !== undefined && eventIn <= 240) {
    const label = signal.event_readout?.next_event_label ?? '';
    if (lang === 'fr') {
      return {
        text: label
          ? `Comment l'événement "${label}" dans ${eventIn} minutes affecte le setup ?`
          : `Comment le prochain événement dans ${eventIn} minutes affecte le setup ?`,
        source: 'q3_event',
      };
    }
    return {
      text: label
        ? `How does the event "${label}" in ${eventIn} minutes affect the setup?`
        : `How does the next event in ${eventIn} minutes affect the setup?`,
      source: 'q3_event',
    };
  }

  const conviction = signal.conviction_0_100 ?? 0;
  if (conviction >= 70) {
    const lo = signal.uncertainty?.conformal_lower ?? 0;
    const hi = signal.uncertainty?.conformal_upper ?? 100;
    return {
      text:
        lang === 'fr'
          ? `Que signifie l'intervalle conformel [${lo}-${hi}] pour cette conviction ?`
          : `What does the conformal interval [${lo}-${hi}] mean for this conviction?`,
      source: 'q3_ci',
    };
  }

  const gate = signal.regime_readout?.regime_gate_decision;
  if (gate) {
    return {
      text:
        lang === 'fr'
          ? `Pourquoi le gate de régime décide-t-il ${gate} ?`
          : `Why does the regime gate decide ${gate}?`,
      source: 'q3_regime',
    };
  }

  // Last resort — historical context
  return {
    text:
      lang === 'fr'
        ? "Que disent les statistiques historiques sur des setups similaires ?"
        : 'What do historical stats say about similar setups?',
    source: 'q3_regime',
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function suggestedQuestions(
  signal: InsightSignalV2,
  lang: SuggestedLanguage = 'fr',
): [SuggestedQuestion, SuggestedQuestion, SuggestedQuestion] {
  return [
    pickQ1(signal.conviction_label, lang),
    pickQ2(signal.breakdown_components ?? [], lang),
    pickQ3(signal, lang),
  ];
}

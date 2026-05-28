import type { InsightSignalV2 } from '@/types/insight';

/**
 * Compact textual serialisation of an InsightSignalV2 for the LLM context.
 * Kept stable across calls (deterministic field order, fixed precision) so
 * Anthropic prompt caching hits — the signal context portion can be cached
 * with `cache_control: { type: "ephemeral" }`.
 *
 * Reads exactly the fields documented in
 * `docs/value/client_information_explained.txt` Parties 2 + 4.
 */
export function buildSignalSummary(signal: InsightSignalV2): string {
  const s = signal;
  const lines: string[] = [];

  lines.push('LECTURE DE MARCHÉ EN COURS');
  lines.push('==========================');
  lines.push(`Instrument: ${s.instrument} · Timeframe: ${s.timeframe}`);
  lines.push(`Direction: ${s.direction}`);
  lines.push(`Conviction (0-100): ${s.conviction_0_100} (label "${s.conviction_label}")`);
  lines.push(
    `Émise: ${s.created_at_utc} · Valide jusqu'à: ${s.valid_until_utc}`,
  );

  lines.push('');
  lines.push('INCERTITUDE CONFORMELLE');
  lines.push(
    `  Intervalle [${s.uncertainty.conformal_lower} – ${s.uncertainty.conformal_upper}] à α=${s.uncertainty.coverage_alpha}`,
  );
  lines.push(`  Couverture empirique observée: ${s.uncertainty.empirical_coverage.toFixed(2)}`);
  lines.push(`  Taille buffer calibration: ${s.uncertainty.n_calibration}`);

  lines.push('');
  lines.push('STRUCTURE DE MARCHÉ (SMC)');
  const st = s.structure_readout;
  lines.push(
    `  BOS niveau: ${st.bos_level ?? 'aucun'}` +
      (st.bos_event_age_bars !== null ? ` (âge ${st.bos_event_age_bars} bougies)` : '') +
      (st.choch_present ? ' · CHOCH précédent' : ''),
  );
  if (st.fvg_zone)
    lines.push(
      `  FVG zone: [${st.fvg_zone[0]} – ${st.fvg_zone[1]}] · taille ${st.fvg_size_atr?.toFixed(2) ?? '?'} × ATR`,
    );
  if (st.ob_zone)
    lines.push(
      `  Order Block: [${st.ob_zone[0]} – ${st.ob_zone[1]}] · intensité ${(st.ob_strength ?? 0).toFixed(2)}`,
    );
  lines.push(`  Retest state: ${st.retest_state}`);
  if (st.structural_invalidation !== null)
    lines.push(`  Invalidation structurelle: ${st.structural_invalidation}`);

  lines.push('');
  lines.push('RÉGIME (HMM + BOCPD + jump)');
  const r = s.regime_readout;
  lines.push(`  HMM label: ${r.hmm_label} · posterior ${r.hmm_posterior.toFixed(2)}`);
  lines.push(
    `  BOCPD cp_prob: ${r.bocpd_changepoint_prob.toFixed(3)} · expected run-length: ${Math.round(r.expected_run_length)} bougies`,
  );
  lines.push(`  Jump ratio: ${r.jump_ratio.toFixed(2)}`);
  lines.push(`  Décision interne (gate): ${r.regime_gate_decision}`);

  lines.push('');
  lines.push('VOLATILITÉ PRÉVISIONNELLE');
  const v = s.volatility_readout;
  lines.push(
    `  Regime: ${v.regime} · forecast ATR: ${v.forecast_atr_pips.toFixed(1)} pips · naïve ATR: ${v.naive_atr_pips.toFixed(1)} pips`,
  );
  lines.push(
    `  Écart vs naïve: ${v.forecast_vs_naive_pct.toFixed(1)} % · IC TCP: [${v.confidence_interval_pips[0].toFixed(1)} – ${v.confidence_interval_pips[1].toFixed(1)}] pips`,
  );
  if (v.is_fallback) lines.push('  ⚠ Fallback ATR brut (modèle vol non disponible)');

  lines.push('');
  lines.push('CONTEXTE ÉVÉNEMENTIEL');
  const e = s.event_readout;
  lines.push(`  Blackout news actif: ${e.news_blackout_active}`);
  lines.push(
    `  Prochain événement HIGH: ${e.next_event_label ?? 'aucun'}` +
      (e.next_event_in_minutes !== null ? ` (dans ${e.next_event_in_minutes} min)` : ''),
  );
  lines.push(
    `  Sentiment news 24h: ${e.sentiment_score.toFixed(2)} · confiance ${e.sentiment_confidence.toFixed(2)}`,
  );
  lines.push(`  Session: ${e.session}`);

  lines.push('');
  lines.push('DÉCOMPOSITION 8 COMPOSANTES (contribution / weight_max · raison)');
  for (const c of s.breakdown_components) {
    lines.push(
      `  - ${c.name}: ${c.contribution.toFixed(2)} / ${c.weight_max.toFixed(0)} · ${c.reasoning}`,
    );
  }

  if (s.historical_stats) {
    const h = s.historical_stats;
    lines.push('');
    lines.push('HISTORIQUE SETUPS SIMILAIRES');
    lines.push(`  N=${h.similar_setups_n} · Hit rate observé: ${(h.hit_rate_observed * 100).toFixed(1)} %`);
    lines.push(
      `  Profit factor: ${h.profit_factor.toFixed(2)} · IC 95 % bootstrap: [${h.profit_factor_ci95[0].toFixed(2)} – ${h.profit_factor_ci95[1].toFixed(2)}]`,
    );
    lines.push(`  Couverture empirique: ${(h.empirical_coverage * 100).toFixed(0)} %`);
    lines.push(`  Fenêtre: ${h.backtest_window}`);
  }

  if (s.narrative_short) {
    lines.push('');
    lines.push('NARRATIF GÉNÉRÉ');
    lines.push(`  ${s.narrative_short}`);
  }

  return lines.join('\n');
}

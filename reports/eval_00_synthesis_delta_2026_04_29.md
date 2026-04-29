# Eval 00 — Synthèse delta 2026-04-29

> Mise à jour de `reports/eval_00_synthesis.md` (note 5.0/10, 4 bloqueurs go-live, datée 2026-04-28).
> Cette delta couvre les commits du 2026-04-28 → 2026-04-29 et les follow-ups effectués aujourd'hui sur les Prompts 05/06/07/08/09.

## 1. Commits livrés depuis la synthèse maîtresse

| SHA | Date | Impact |
|---|---|---|
| `0bfaa69` | 2026-04-29 | Eval 04 sprint hotfix (vectorisation build_features, blend CV no-leak, state pickle versioning, défaut `vol_mode="har"`) — +143 tests verts, Hybrid forecast 1.6s → 187ms. |
| `8552c79` | 2026-04-29 | Eval 04 follow-up : alignement `main.py` VOL_MODE défaut `har`. |
| `942cded` | (récent) | **RegimeFilter PF 1.13 → 1.60 OOS** sur XAU en skippant NY session + top-vol quartile. |
| `71ac2ae` | (récent) | RegimeFilter surgical mode : **+53 % signaux à PF égal**. |

**Implication majeure** : les commits RegimeFilter changent la donne sur le verdict GO/NO-GO. PF passe de 0.94 (Sprint 2 BOS-fix) à **1.60 OOS** sur XAU avec filtrage de session/régime — c'est au-dessus du seuil 1.20 mentionné comme bloqueur GTM dans memory `eval_28_gtm_findings`.

## 2. Mise à jour des 4 bloqueurs go-live

| # Bloqueur (eval_00) | État au 2026-04-28 | État au 2026-04-29 |
|---|---|---|
| 1. Data feed XAU à 63 % | ❌ Bloqué | ❌ **Toujours bloqué** (XAU 2019-2025 inchangé). EURUSD désormais à 99.6 % coverage 24/5 (eval_08_followup). |
| 2. Aucun walk-forward / IC bootstrap | ❌ Bloqué | ⚠️ **Partiellement débloqué** : `scripts/eval_04_volatility.py` est un walk-forward 6-splits sur 2019-2024 avec DM tests. Pattern reproductible. |
| 3. `Procfile`/railway.toml lance `parallel_training.py` | ❌ Bloqué | ❓ Non re-vérifié aujourd'hui (faire un check ciblé). |
| 4. `TESTING_MODE=1` défaut + auth bypass | ❌ Bloqué | ⚠️ **Atténué** : `assert_safe_production_config()` existe désormais dans le WIP main.py (refuse `ENVIRONMENT=production` + `TESTING_MODE=1`). Pas encore commité mais prêt. |

## 3. Findings empiriques nouveaux (à intégrer en 1.x synthese)

| Source | Finding empirique | Impact |
|---|---|---|
| `eval_06_empirical_findings_2026_04_29` | Hit rate cache mesuré **7.8 %** (vs 30-45 % estimé). Quick fix `SCORE_BUCKET_PTS=5→10` → **33.8 %**. | Économie LLM sous-estimée par eval_06 ×4. À 1k MAU = $9 480/an. |
| `eval_07_followup_2026_04_29` | Toutes les cells `enter=65` → **0 trades** sur 6 ans. Score plafonne sous 60. | Le sweep state machine ne peut pas converger tant que P0 = ConfluenceDetector scoring fn replacement (memory `confluence_calibration`). |
| `eval_08_followup_2026_04_29` | EURUSD M15 2019-2025 : 174k bars, 99.6 % coverage 24/5. | Multi-asset commence (1/5). Note eval_08 3.5/10 → 4.0/10. |
| `eval_09_followup_2026_04_29` | `MultiSymbolScanner` existe en réalité (l.803). Latence forecast LGBM/Hybrid -88 % grâce à Prompt 04. | Eval_09 contenait une affirmation fausse. Note 6.5 → 6.8/10. |
| `eval_05_refresh_2026_04_29` | 5 priorités eval_05 toujours en place (system prompt 2840 tok, cascade off, fallback Template, cache key sans bar_ts, eval CI). 99 tests verts. | Pas de régression. Note 7.5/10 maintenue. |

## 4. Note moyenne révisée

eval_00 originale : **5.0 / 10**.

Avec le commit RegimeFilter (PF 1.60 OOS) **non encore intégré dans eval_00**, et les follow-ups 05/06/07/08/09 :

* eval_04 : 5.0 → **6.0** (vectorisation + leak fix livrés)
* eval_06 : 5.0 → 5.0 (delta empirique sans changement de note jusqu'à ce que `SCORE_BUCKET_PTS` soit bumped)
* eval_07 : 8.0 → 8.0 (design intact, mais blocked par P0 confluence)
* eval_08 : 3.5 → **4.0** (EURUSD ajouté)
* eval_09 : 6.5 → **6.8** (correction MultiSymbolScanner + impact Prompt 04 sur ML latency)
* (RegimeFilter : moveable sur eval_18/19/20 — mérite re-éval dédiée)

**Note pondérée révisée** : **5.4 / 10** (vs 5.0). +0.4 sur le run d'aujourd'hui — gain mesuré modeste mais consolidé.

## 5. Top 3 actions prioritaires (P0 absolus)

Issues du croisement des findings :

| P0 | Action | Effort | Impact |
|---|---|---|---|
| **1** | **Re-télécharger XAU 2019-2025** propre via `scripts/download_dukascopy_xau.py` | 30 min | Débloque tout le replay 2025 + falsifie toutes les notes courantes basées sur le feed à 63 %. |
| **2** | **Replace ConfluenceDetector scoring fn** (memory `confluence_calibration` : Pearson −0.023, Brier worse than baseline). Garder framework. | 1-2 sem | Débloque eval_07 sweep, débloque score 75 atteignable, débloque calibration tier PREMIUM/STANDARD/WEAK. |
| **3** | **Bump `SemanticCache.SCORE_BUCKET_PTS = 10`** (eval_06_empirical) | 1 ligne | Hit rate ×4.3, $9 480/an d'économie LLM à 1k MAU. |

## 6. Contradictions / lacunes identifiées dans le corpus

| Lacune | Évidence | Action |
|---|---|---|
| eval_09 affirme « pas de MultiSymbolScanner » | Contredit par `sentinel_scanner.py:803` | Corrigé dans `eval_09_followup_2026_04_29` |
| eval_06 estime hit rate 30-45 % | Mesure empirique = 7.8 % | Corrigé dans `eval_06_empirical_findings_2026_04_29` |
| eval_07 évalue defaults (75/55/2/2/12) sans avoir mesuré le seuil 65 | Heatmap partial montre 0 trades à enter=65 | Corrigé dans `eval_07_followup_2026_04_29` |
| eval_00 ne mentionne pas RegimeFilter (commits récents) | RegimeFilter PF 1.60 OOS = changement majeur | À intégrer dans la prochaine refresh d'eval_00 |
| Aucun bench scanner E2E dans eval_09/eval_16/eval_21 | Tous estiment, aucun mesure | Lacune méthodologique — à combler avant SLA contractuel |

## 7. Chemin vers commercialisable

État pondéré post-RegimeFilter (estimation, à valider) :

| Critère go-live | Cible | État courant | Δ après actions §5 |
|---|---|---|---|
| PF replay 7-ans | ≥ 1.20 | 1.60 OOS XAU (RegimeFilter) ✅ | Maintenu |
| Coverage data | ≥ 95 % | XAU 63 % ❌, EUR 99.6 % ✅ | XAU 95 % après #1 |
| Walk-forward + DM test | requis | livré pour vol (Prompt 04) | À étendre confluence + replay |
| Auth bypass off prod | requis | trip-wire WIP, non commité | À commiter (`assert_safe_production_config`) |
| Multi-asset validation | ≥ 2 | 1 (XAU) | 2 (XAU + EUR) après bench EUR |

**Verdict révisé** : avec les 3 actions P0 + commit du trip-wire prod safety, le projet pourrait passer de NO-GO commercial à **GO conditionnel FREE-only** dans 2-4 semaines de sprint focalisé.

---

## 8. Liens

* `reports/eval_00_synthesis.md` (synthèse maîtresse, 2026-04-28)
* Tous les follow-ups 2026-04-29 sont dans `reports/eval_{05,06,07,08,09}_*_2026_04_29.md`
* `scripts/eval_04_volatility.py`, `scripts/eval_06_hit_rate_sim.py`, `scripts/eval_07_hysteresis_heatmap.py` (reproductibles)

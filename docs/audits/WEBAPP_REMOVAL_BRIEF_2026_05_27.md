# Brief Front-End — Retrait des 4 champs probabilistes non supportables

**Date** : 2026-05-27
**Origine** : `docs/audits/descriptive_quality_assessment.md` Partie 4
**Pour** : terminal qui gère `webapp/` + `src/delivery/`
**Statut** : à exécuter (le terminal audit ne touche pas au code des surfaces)

## Pourquoi ce retrait

L'audit descriptif a mesuré sur 105 k bars OOS (2024+) que 4 champs aujourd'hui exposés au client sont **factuellement trompeurs** :

| Champ | Promesse affichée | Mesure empirique | Verdict |
|---|---|---|---|
| `hmm_posterior` | "certitude X %" | XAU : conf 99 % → accuracy 42 %. ECE 0.54 | 🔴 |
| `bocpd_changepoint_prob` | "Probabilité de rupture X %" | 0/105 000 bars OOS avec cp_prob ≥ 0.5 ; plaqué au prior | 🔴 |
| `volatility_readout.confidence_interval` labellé "intervalle conformel" | implicite α=0.05 → 95 % couverture | Couverture empirique 51 % (XAU) / 39 % (EUR) | 🔴 |
| `conformal_lower` / `conformal_upper` sur la conviction (et toute viz dérivée) | "intervalle conformel sur la conviction" | Hors-périmètre descriptif (outcome = R-multiple) ; cf. `OUT_OF_SCOPE.md` §2 | 🔴 |

**Non-objectif** : on ne touche **pas** aux calculs backend. On retire uniquement la couche d'**exposition** au client. Les champs continuent d'exister dans le contrat `InsightSignalV2`, ils ne sont simplement **plus affichés**.

---

## Inventaire des fichiers à toucher

### Composants UI (à modifier)

| Fichier | Action |
|---|---|
| `webapp/components/insight/sections/RegimeSection.tsx` | Retirer ligne 43 (`certitude {Math.round(r.hmm_posterior * 100)} %`). Retirer lignes 47-53 (Row "Stabilité du régime" + son `hint` BOCPD). Garder le `Badge` label HMM (sans posterior). |
| `webapp/components/insight/sections/VolatilitySection.tsx` | Retirer lignes 58-63 (Row "Intervalle conformel" + son `hint` TCP). Garder amplitude prévue + amplitude naïve + écart. |
| `webapp/components/insight/ConvictionGauge.tsx` | Retirer l'utilisation de `uncertainty.conformal_lower` / `conformal_upper` (lignes 47-48 et dérivés). Soit afficher la gauge **sans** intervalle, soit afficher la gauge tout court. |
| `webapp/components/insight/expert/ConformalIntervalViz.tsx` | **Supprimer le composant** (toute la viz est dédiée à un claim non-supporté). Retirer les imports/usages dans la section EXPERT. |
| `webapp/lib/chat/signal-summary.ts` | Retirer ligne 28 (`Intervalle [...] – ... à α=...`). Retirer lignes 56 (HMM posterior) et 58 (BOCPD cp_prob + run-length). |

### Types (à modifier)

| Fichier | Action |
|---|---|
| `webapp/types/insight.ts` | Marquer les 4 champs (`hmm_posterior`, `bocpd_changepoint_prob`, `conformal_lower`, `conformal_upper`) en `optional` (`?: number`) ou retirer s'ils ne sont plus utilisés nulle part dans le front. Conserver dans le contrat backend (cf. `src/intelligence/insight_v2/contract.py`). |

### Mocks / tests (à modifier)

| Fichier | Action |
|---|---|
| `webapp/mocks/sample_signals.json` | Retirer `hmm_posterior`, `bocpd_changepoint_prob`, `conformal_lower`, `conformal_upper` des 3 signaux mockés (lignes 13-14, 34-35, 97-98, 118-119, 181-182, 202-203). |
| `webapp/lib/insight-formatters.test.ts` | Retirer les assertions sur ces 4 champs (lignes 42-43, 63-64). Adapter les tests qui en dépendaient. |

### Snapshots HTML (à régénérer)

| Fichier | Action |
|---|---|
| `webapp/page.html`, `page-final.html`, `page-chrome.html`, `page-lh.html` | Re-générer après les modifs composants — ces snapshots HTML embarquent l'état rendu. |

### Backend narrative (à coordonner)

Le composeur de narratif `src/intelligence/insight_v2/narrative.py` injecte aussi ces champs dans `narrative_short` / `narrative_long`, qui flow ensuite vers la webapp ET Telegram (rendus tels quels). Lignes à neutraliser :

| Ligne | Contenu actuel | Action |
|---|---|---|
| 193 | `(f" (postérieur {r.hmm_posterior:.2f})" if r.hmm_posterior is not None else "")` | Retirer le fragment |
| 195-196 | `f"Probabilité de changepoint imminent : {r.bocpd_changepoint_prob:.0%}"` | Retirer le fragment |
| 221 | `f"({insight.conviction_0_100:.0f}/100, conformal interval ...)"` | Retirer le sous-fragment "conformal interval" |
| 236 | `(f" (posterior {r.hmm_posterior:.2f})" if r.hmm_posterior is not None else "")` | Retirer le fragment (version EN) |

C'est un fichier Python backend, mais il génère du **texte client-facing**. Au choix du terminal qui touche au code : modifier `narrative.py` ou wrapper les phrases côté front avec un filter regex. La version "source" (modifier narrative.py) est plus propre.

---

## Surfaces vérifiées sans exposition à retirer

- `src/delivery/telegram_notifier.py` et `src/delivery/discord_notifier.py` : **pas de référence directe** aux 4 champs (recherche `grep` : 0 match). Le narratif arrive déjà formé via `narrative_short` — donc la modif `narrative.py` ci-dessus suffit pour Telegram.
- Routes API B2B (`src/api/routes/`) : non en scope de cette tâche (ce sont des endpoints contractuels — supprimer un champ du JSON casserait les consommateurs externes ; à traiter séparément si pertinent).

---

## Sanity check après modif

À vérifier sur la page `/[locale]` (mockup + signal réel) :
1. La section **Régime de marché** n'affiche plus de "certitude X %" ni de "Probabilité de rupture X %".
2. La section **Volatilité prévisionnelle** n'affiche plus de Row "Intervalle conformel".
3. La **ConvictionGauge** ne montre plus de bornes/bande de confiance.
4. La section **EXPERT** ne contient plus de `ConformalIntervalViz`.
5. Le résumé chat (signal-summary) ne mentionne plus posterior, cp_prob, ni intervalle.
6. Les snapshots Telegram (via narrative_short) ne contiennent plus "postérieur", "Probabilité de changepoint", "conformal interval".

## Ce qui reste affiché côté client (validé honnête)

Inchangé : `hmm_label` (juste le label `low_vol_trending` / `high_vol_stress` etc., **sans** posterior numérique), `jump_ratio`, `expected_run_length` (à exposer comme indicateur descriptif sans probabilité), bornes SMC (BOS/CHOCH/FVG/OB/Retest), `forecast_atr_pips`, `naive_atr_pips`, `regime_gate_decision`.

---

## Référence

- Rapport complet : `docs/audits/descriptive_quality_assessment.md` Parties 2.8 (HMM), 2.9 (BOCPD), 2.11 (Vol+conformal), 4 (implications commerciales).
- Données reproductibles : `docs/audits/descriptive_quality_data.json`.
- Scripts d'audit : `scripts/audit/descriptive_quality/eval_hmm.py`, `eval_bocpd.py`, `eval_volatility.py`.

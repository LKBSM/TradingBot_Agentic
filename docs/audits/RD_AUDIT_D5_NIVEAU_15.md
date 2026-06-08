# RD_AUDIT_D5 — Cohérence avec le positionnement niveau 1.5

> **Audit R&D exploratoire — Dimension 5/5**
> Date : 2026-06-08 · Branche : `audit/rd-exploratoire-moteur-detection`
> Positionnement de référence : **indicateur descriptif niveau 1.5 strict** — décrit, ne prédit pas, ne recommande pas (cf. `MIA_MARKETS_V2_VISION.md` §1.2, pivot 2026-05-27). Aucune modification de code.

---

## 5.1 — Audit des heuristiques : probabilités/scores cachés ?

**Question** : le moteur calcule-t-il *implicitement* des probabilités, ou expose-t-il un score directif déguisé ?

| Élément | Nature | Exposé au client ? | Verdict 1.5 |
|---|---|---|---|
| `BOS_SIGNAL/EVENT`, `CHOCH_SIGNAL` | ±1/0 catégoriels (direction/présence) | via `structure.bos/choch` (direction + validation_status) | 🟢 Descriptif factuel |
| `FVG_SIGNAL`, `FVG_DIR` | catégoriels | présence FVG | 🟢 |
| `OB_STRENGTH_NORM` | **continu** (ratio taille/ATR + bonus) | **bucketisé** en `importance` low/medium/high (mapper l.153) | 🟡 voir D5-1 |
| `BOS_RETEST_STATE/ARMED` | catégoriels d'état | `retest_in_progress` (booléen) | 🟢 |
| HMM posterior / BOCPD cp_prob / conformal / vol CI | **probabilistes** | **NON câblés** au `MarketReading` (`regime_classifier`, `regime_gate`, etc.) | 🟢 (déjà retirés du visible, mémoire 2026-05-27) |

**Constat 🟢 majeur** : le schéma `MarketReading` (`market_reading_schema.py`) **ne contient AUCUN champ de probabilité, score, confidence ou win-rate**. Vérifié champ par champ. Les 4 champs probabilistes problématiques de l'audit qualité (HMM posterior, BOCPD, conformal, vol CI 95 %) **ne sont pas câblés** au produit. Le moteur de détection produit du **catégoriel** (direction, présence, statut), pas du probabiliste.

### Le seul « score » résiduel : `OB importance`
- **D5-1** *(Petit / impact Faible / risque Faible)* : `importance = high si OB_STRENGTH≥0.75, medium si ≥0.4, sinon low` (mapper l.153). C'est une **discrétisation d'un score continu** avec des **seuils arbitraires non documentés**, et le score est pollué par le bonus FVG additif (cf. D2-2). Ce n'est pas une probabilité directive, mais c'est un **jugement de « force »** présenté au client sans justification.
  - **Reco** : documenter sur `/methodology` que `importance` = taille de zone relative à l'ATR (descriptif, pas « probabilité de tenue »), et justifier/calibrer les seuils 0.75/0.4. **NB** : la bucketisation vit dans le mapper → coordonner avec le terminal mapper.

---

## 5.2 — Audit du wording interne

Recherche de vocabulaire **prédictif** (predict/forecast/expect) et **directif** (recommend/should/buy/sell) dans le code de détection.

| Catégorie | Trouvé dans le moteur de détection ? | Détail |
|---|---|---|
| **Prédictif** (`predict`, `forecast`, `expected`) | 🟢 Quasi-absent du chemin produit | « expected to pull back » dans un commentaire retest (`strategy_features.py:60`) — **interne**, non exposé |
| **Directif** (`recommend`, `should trade`, `buy/sell`) | 🟢 Absent du moteur | — |
| **Vocabulaire de trading** (trade, SL, win, profit) | 🟡 Présent en **commentaires internes** | docstring retest : « take SL hits », « chase tops/bottoms », « full trade lifetime », « confirmed trades » (`strategy_features.py:240-257, 510-518`). **Non exposé client.** |
| **Forward-looking** | 🟡 `regime_gate.py:13-35` : « we are likely **about to enter** a structural shift », « BOCPD tells you we are *entering* a new regime » | **Module NON câblé** au `MarketReading` (déjà signalé audit précédent §4.3). Risque uniquement **si** câblé un jour. |

**Constat 🟢** : le **chemin produit** (moteur → mapper → description) est **propre au niveau 1.5**. Le vocabulaire de trading subsiste seulement dans des **commentaires/docstrings internes** et dans des **modules non câblés**. La double-garde côté description (system prompt Haiku + `FORBIDDEN_TOKENS`) confirme 0 token interdit en sortie (audit précédent Phase 3 : 100 %).

- **D5-5** *(Veille / Petit si déclencheur)* : si `regime_gate`/`regime_classifier` sont un jour câblés au produit, **réécrire leurs docstrings forward-looking** et re-vérifier le niveau 1.5. Sinon, non-bloquant.

---

## 5.3 — Audit des seuils : arbitraires ou justifiés ?

| Seuil | Valeur | Justifié dans le code ? | Action |
|---|---|---|---|
| `FRACTAL_WINDOW` | 2 | Commentaire « 2 = 5 bougies » (descriptif, pas justifié empiriquement) | Documenter |
| `FVG_THRESHOLD` | 0.1×ATR | « filters spread-level noise » — mais ≈ spread XAU (laxe, cf. D2-3) | Calibrer + documenter |
| `RETEST_TOL_ATR` | 0.5 | Pas de justification empirique | Calibrer + documenter |
| `RETEST_ARMED_WINDOW` | 30 | ✅ **Bien justifié** (« ~13 bars trade lifetime XAU M15 per replay », l.513-518) | OK |
| Régime ranging | `pct_move < 0.3·rng_pct` | ❌ Arbitraire, non documenté | **D5-2** documenter |
| Régime volatilité | ratio 0.7 / 1.3 | ❌ Arbitraire, non documenté | **D5-2** documenter |
| OB importance | 0.75 / 0.4 | ❌ Arbitraire, non documenté | **D5-1** documenter |

- **D5-2** *(Petit / impact Moyen / risque Faible)* : documenter sur `/methodology` que les seuils régime (0.3/0.7/1.3) et OB (0.75/0.4) sont des **choix heuristiques** (transparence 1.5), idéalement calibrés sur l'annotation. **Honnêteté > faux empirisme.**

---

## 5.4 — Audit de transparence (explicabilité)

**Question** : si un utilisateur demande « comment l'algo a détecté ce BOS ? », peut-on l'expliquer ?

| Critère | Évaluation |
|---|---|
| Heuristiques **déterministes & reproductibles** | 🟢 Oui (pas de RNG, pas de ML dans le chemin produit) |
| Heuristiques **interprétables** (règles lisibles) | 🟢 Oui — chaque détection = condition booléenne explicite sur OHLC |
| **Black box** dans le chemin produit | 🟢 Aucune (HMM/LGBM non câblés) |
| `/methodology` peut-elle documenter fidèlement ? | 🟢 **Oui** — c'est même un atout différenciant : tout est règle-based explicable |

**Atout stratégique** : la transparence totale du moteur est **parfaitement alignée** avec le positionnement 1.5 et la mémoire 2026-05-27 (« outil de compréhension augmentée », « rubric LLM open-sourçable »). Une page `/methodology` honnête est **réalisable telle quelle**.

### Réserves de transparence à lever (cf. audit précédent — câblage, autre terminal)
- **F1/F2/F3** (niveaux BOS/CHOCH/OB/FVG = proxies `current_price ± ATR/2`) : exposer un niveau **chiffré** présenté comme une mesure alors que c'est un proxy **contredit la transparence**. ⚠️ **Hors mon périmètre** (mapper, autre terminal). Je le **rappelle** comme risque 1.5 : tant que F1-F3 ne sont pas corrigés, **ne pas afficher de niveaux chiffrés** (cohérent mémoire 2026-05-27).
- **F6** (`bos_recent` sur 100 % des readings) : afficher « BOS récent » sur un état de tendance propagé (pas un break frais) est un **glissement descriptif** — le client croit à un événement récent. Mapper, autre terminal. Rappel pour cohérence.

---

## 5.5 — Synthèse Dimension 5

**État cohérence 1.5 du MOTEUR DE DÉTECTION : 🟢 Solide.** Le moteur produit du **catégoriel descriptif** (direction, présence, statut), **sans aucune probabilité ni score directif exposé**. Le vocabulaire prédictif/directif est **absent du chemin produit** (cantonné à des commentaires internes et des modules non câblés). La transparence est **totale et exploitable** pour `/methodology` — un atout, pas un risque.

**Les seules réserves 1.5 imputables au moteur sont des seuils arbitraires non documentés** (régime, OB importance) → corrections **Petit = documentation** (D5-1, D5-2). Aucune n'exige de toucher la logique.

**Les vrais risques 1.5 résiduels (niveaux chiffrés proxy F1-F3, tag `bos_recent` F6) relèvent du MAPPER** — **hors de mon périmètre**, traités par l'autre terminal. Je les **rappelle** sans agir.

**Cas STOP rencontrés en D5** : **aucun déclenché**. J'ai spécifiquement cherché le cas #4 (« score caché de probabilité qui influence la détection sans être exposé ») → **non trouvé** : les probabilités existent (HMM/BOCPD/conformal) mais **n'influencent PAS la détection du `MarketReading`** (non câblées). Le moteur de détection est probabilité-free. ✅

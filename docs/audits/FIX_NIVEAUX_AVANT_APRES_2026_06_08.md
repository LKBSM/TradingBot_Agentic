# FIX_NIVEAUX_AVANT_APRES — Corrections de câblage des niveaux (F1/F2/F3/F6/P4)

> **Suite de l'audit de validation algorithmique — phase de correction**
> Date : 2026-06-08 · Branche : `fix/audit-mappers-correction-niveaux-prix`
> Décision founder : **Option A — corriger les niveaux** (ne pas masquer, le positionnement
> niveau 1.5 strict repose sur la véracité des données affichées).
>
> Règle respectée : **aucune modification du moteur de détection** (`strategy_features.py`).
> Tous les correctifs sont dans le **mapping** (`market_reading_mappers.py`) + le câblage
> pipeline (`market_reading_assembler.py`) + le script de génération. 1 commit par finding,
> tests à chaque étape, 0 régression.

---

## 1. Résumé des corrections

| Réf | Problème (audit) | Correction | Commit |
|---|---|---|---|
| **F1** | `BOS.level` lisait `BOS_PRICE_LEVEL` (clé inexistante) → toujours `current_price` | Lit `BOS_BREAK_LEVEL` (réel, bougie d'event) + `BOS_BREAK_LEVEL_LAST` (forward-fill pipeline) | F1 |
| **F2** | `CHOCH.level` lisait `CHOCH_PRICE_LEVEL` (inexistante) → toujours `current_price` | Lit `BOS_BREAK_LEVEL` (CHOCH = BOS de renversement même bougie → niveau partagé) | F2 |
| **F3** | Niveaux OB/FVG = proxy `current_price ± ATR/2` | Lit les vraies zones : OB = `BULLISH/BEARISH_OB_*` ; FVG = bornes 3-bougies | F3 |
| **F6** | Tag `bos_recent` sur 100 % des readings (53 % périmés) | `structure.bos` émis seulement sur break frais (`BOS_EVENT≠0`) ; retest découplé | F6 |
| **P4** | `entre` (homonyme « entre X et Y ») rejeté par le filtre Haiku → 5/60 fallbacks | Retire le bare `entre`, ajoute `entrez/entrer/entry` (aligné sur `chatbot/constants.py`) | P4 |

### Architecture du correctif (sans toucher au moteur)
Un helper **`realized_levels(enriched, idx)`** (dans `market_reading_mappers.py`) extrait, depuis
la fenêtre enrichie par l'engine, les vrais niveaux pour la bougie cible :
- `BOS_BREAK_LEVEL_LAST` : forward-fill de `BOS_BREAK_LEVEL` (l'engine ne le pose que sur les
  bougies d'événement ; le ffill carrie le niveau structurel réel sur les bougies de propagation).
- `OB_LEVEL_HIGH/LOW` : zone OB réelle (range de la bougie `i-1` stockée par l'engine).
- `FVG_LEVEL_HIGH/LOW` : bornes du gap 3-bougies, reconstruites avec la **même géométrie** que l'engine.

Ce helper est appelé par les **deux** pipelines (assembler de production + script de validation),
qui mergent le résultat dans `smc_features`. Le mapper lit ces clés avec **fallback gracieux**
(proxy/`current_price`) si absentes → rétro-compatibilité totale.

> **Note STOP (vérifiée)** : les « vraies » clés du moteur sont correctes. `BOS_BREAK_LEVEL` est
> bien le niveau structurel cassé. CHOCH n'a pas de clé dédiée mais partage `BOS_BREAK_LEVEL`
> **par construction** (même événement) — ce n'est pas « une clé fausse », c'est « une clé partagée ».
> Aucun correctif n'a nécessité de toucher au moteur → **pas de STOP déclenché**.

---

## 2. Comparaison AVANT / APRÈS (dataset 60 readings régénéré)

> Données : `data/validation/before_after_comparison.json` (script `compare_before_after.py`).
> Sélection déterministe identique → mêmes 60 bougies AVANT/APRÈS (niveaux comparables bar-à-bar).

### 2.1 Agrégats

| Métrique | AVANT | APRÈS | Effet |
|---|---|---|---|
| `BOS.level == close_price` (proxy faux) | **60/60** | **0/60** | F1 — plus aucun niveau proxy |
| Tag `bos_recent` présent | **60/60** | **28/60** | F6 — ne reste que les 28 breaks frais (les 32 propagés périmés retirés) |
| `description_source = template_fallback` | **5/60** | **0/60** | P4 — plus aucun fallback injustifié |
| `description_source = haiku_generated` | 55/60 | **60/60** | P4 |

### 2.2 BOS — AVANT vs APRÈS vs **vérité moteur** (5 échantillons)

> La colonne « moteur » est le `BOS_BREAK_LEVEL` ré-extrait **indépendamment** de l'engine.
> `match=True` partout : le niveau publié APRÈS est exactement celui que le moteur calcule.

| # | Instr. | close_price | AVANT (proxy) | APRÈS (réel) | Vérité moteur | Match |
|---|---|---|---|---|---|---|
| 3 | XAUUSD | 4375.185 | 4375.185 | **4373.400** | 4373.400 | ✅ |
| 5 | XAUUSD | 4487.789 | 4487.789 | **4475.900** | 4475.900 | ✅ |
| 9 | XAUUSD | 4775.945 | 4775.945 | **4766.444** | 4766.444 | ✅ |
| 11 | XAUUSD | 5204.690 | 5204.690 | **5190.535** | 5190.535 | ✅ |
| 12 | XAUUSD | 5182.825 | 5182.825 | **5235.662** | 5235.662 | ✅ |

> AVANT : le niveau BOS = le prix de clôture (aucune information). APRÈS : le vrai niveau
> structurel cassé, distinct du prix courant, et fidèle au moteur.

### 2.3 OB / FVG — proxy symétrique AVANT vs zone réelle APRÈS (5 échantillons)

| # | Instr. | Type | AVANT (proxy `prix ± ATR/2`) | APRÈS (zone réelle) |
|---|---|---|---|---|
| 2 | XAUUSD | FVG | [4343.12, 4365.14] | **[4319.24, 4341.49]** |
| 3 | XAUUSD | FVG | [4364.93, 4385.44] | **[4355.97, 4362.20]** |
| 7 | XAUUSD | FVG | [4578.86, 4598.996] | **[4511.45, 4536.48]** |
| 11 | XAUUSD | OB | [5189.20, 5220.18] | **[5157.45, 5180.20]** |
| 14 | XAUUSD | OB | [4488.78, 4610.46] | **[4603.44, 4713.29]** |

> AVANT : zones symétriques centrées sur le prix courant (artefact). APRÈS : les vraies bornes
> que l'engine a détectées (asymétriques, situées où la structure existe réellement).

---

## 3. Impact sur la cohérence Haiku (Phase 3 re-mesurée)

| Test | AVANT fixes | APRÈS fixes | Note |
|---|---|---|---|
| C — Sans forbidden token | 100 % | **100 %** | gate critique tenu |
| D — Sans invention | 100 % | **100 %** | |
| B — Cohérence directionnelle | 100 % | **100 %** | |
| Source Haiku live | 55/60 | **60/60** | P4 ✓ |
| Test A — mention BOS (dénominateur) | 0/60 (artefact) | 2/28 | **F6 rend le dénominateur honnête** : on mesure désormais la mention sur les **breaks réellement frais**, plus sur 100 % gonflé |

> La mention de structure par Haiku reste basse (design « 1 phrase régime » + le LLM ne reçoit
> que `tags`+`regime`, finding F5). C'est **hors périmètre de cette mission de correction** (F5 = reco
> 🟠 #5 du rapport final : décider du rôle de la prose). Aucune régression de sécurité/honnêteté.

---

## 4. Tests & non-régression

- **1 commit par finding** (F1 → F2 → F3 → F6 → P4), chacun avec tests dédiés.
- Tests mappers étendus : 23 cas (dont ffill, fallbacks, zones réelles, freshness BOS, retest découplé, homonyme `entre`).
- **Sweep de non-régression** (11 fichiers : mappers, assembler, schema, endpoint, scheduler, store, haiku, news, insight, smoke e2e, readout) : **167 passed, 0 régression**.
- Plusieurs tests existants **mis à jour** car ils encodaient l'ancien comportement buggé
  (`BOS_PRICE_LEVEL`, BOS propagé affiché en « pending », `entre` interdit).

---

## 5. Critères d'acceptation — statut

| Critère | Statut |
|---|---|
| F1 — BOS publie le vrai `BOS_BREAK_LEVEL` | ✅ (60→0 proxy ; match moteur 5/5) |
| F2 — CHOCH publie le vrai niveau cassé | ✅ (lit `BOS_BREAK_LEVEL` de la bougie CHOCH) |
| F3 — OB/FVG publient les vrais ranges | ✅ (zones réelles, plus de proxy ATR/2) |
| F6 — `bos_recent` seulement sur breaks frais | ✅ (60/60 → 28/60) |
| P4 — 0 fallback Haiku injustifié | ✅ (5/60 → 0/60) |
| Tests unitaires ajoutés par correction | ✅ |
| Dataset régénéré post-corrections | ✅ |
| Comparaison AVANT/APRÈS documentée | ✅ (ce document) |
| Branche poussée sur origin | ✅ (voir §6) |

---

## 6. Livrables de la correction

| Fichier | Rôle |
|---|---|
| `src/intelligence/market_reading_mappers.py` | helper `realized_levels` + lecture des vrais niveaux (F1/F2/F3) + freshness BOS (F6) + retest découplé + alignement tokens (P4) |
| `src/intelligence/market_reading_assembler.py` | merge `realized_levels` dans le pipeline de production |
| `scripts/generate_validation_dataset.py` | merge `realized_levels` ; dataset régénéré |
| `scripts/audit/validation/compare_before_after.py` | comparaison AVANT/APRÈS + vérité moteur |
| `tests/test_market_reading_mappers.py` | tests des 5 corrections |
| `data/validation/marketreadings_2026_06_06.json` | dataset régénéré (niveaux corrigés) |
| `data/validation/before_after_comparison.json` | données AVANT/APRÈS |
| `data/validation/haiku_coherence_results.json` | cohérence Haiku re-mesurée |
| `docs/audits/{VALIDATION_DATASET,MANUAL_ANNOTATION_TEMPLATE}_2026_06_06.md`, `SCORING_TEMPLATE_2026_06_06.csv` | artefacts d'annotation rafraîchis |
| `docs/audits/FIX_NIVEAUX_AVANT_APRES_2026_06_08.md` | ce rapport |

> **Hors périmètre (non touché)** : moteur de détection, frontend, système chatbot (sauf l'alignement
> du token mapper P4). Pas de PR, pas de merge, pas de force push.

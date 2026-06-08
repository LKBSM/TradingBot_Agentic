# VALIDATION_ALGORITHMIQUE_FINAL_2026_06_06 — Rapport consolidé

> **Audit de validation algorithmique complète — MIA Markets**
> Date : 2026-06-07 · Branche : `audit/validation-algorithmique-detection-structures`
> Auteur : terminal d'audit (automatisé) · Statut : **80 % automatisable livré · 20 % en attente d'annotation humaine**
>
> Objet : valider que le moteur de détection de structure de marché (BOS/CHOCH/OB/FVG/Phase)
> **dit la vérité** sur ce qu'il détecte, avant lancement bêta.

---

## 1. Synthèse exécutive

### Le moteur est-il prêt pour la bêta ?
**Conditionnellement — sous réserve de 3 correctifs de câblage (non du moteur) + annotation manuelle.**

Le pipeline de détection est **architecturalement sain, déterministe, causal (anti look-ahead)**
et **sûr au niveau 1.5** (0 token interdit, 0 hallucination dans les descriptions). La logique de
détection brute (fractals → BOS/CHOCH/OB/FVG) est cohérente et majoritairement alignée sur les
conventions SMC. **Les risques identifiés ne sont PAS dans la détection mais dans le câblage
moteur → produit** : plusieurs champs publiés au client ne reflètent pas ce que le moteur calcule.

### Forces
- 🟢 **Sécurité niveau 1.5 robuste** : descriptions Haiku 100 % sans token interdit, 100 % sans
  invention de structure, 100 % cohérentes directionnellement. Double garde (system prompt + filtre).
- 🟢 **Détection causale, reproductible** : fractals décalés `shift(N)`, aucun look-ahead détecté ;
  régime 100 % descriptif (aucun langage prédictif côté client).
- 🟢 **FVG = formule canonique ICT** (gap 3 bougies) ; retest state-machine propre.
- 🟢 **Régime purement descriptif** : trend/vol/phase en règles transparentes auditables, aucun ML
  boîte-noire câblé au produit.

### Faiblesses
- 🔴 **3 bugs de câblage (F1/F2/F3)** : les niveaux BOS/CHOCH publiés retombent **toujours** sur
  `current_price` (mauvaise clé), et les niveaux OB/FVG sont des **proxies** `prix ± ATR/2`, pas
  les vraies zones que le moteur connaît. → tout niveau chiffré affiché est non-fiable.
- 🟡 **Tag `bos_recent` sur 100 % des readings (F6)** alors que 53 % ne sont pas des breaks frais.
- 🟡 **OB = englobant 2 bougies**, pas la définition ICT (dernière bougie opposée avant displacement) — déjà connu P0-2.
- 🟡 **OB/FVG mitigation/fill non suivis** (statut toujours « active »).
- 🟠 **Incohérence inter-systèmes des forbidden tokens** (`entre` rejeté par un système, exclu par l'autre).

### Risques niveau 1.5 / légaux
- **Aucun risque directif détecté** dans les sorties client (descriptions, régime). 🟢
- **Risque produit** : afficher des niveaux de prix chiffrés faux (F1-F3) est un risque
  **de crédibilité**, pas légal. Recommandation : ne pas exposer de niveaux chiffrés tant que
  F1-F3 ne sont pas corrigés (cohérent avec décision mémoire 2026-05-27).
- Langage prédictif présent **uniquement** dans des docstrings internes non-câblées (`regime_gate.py`).

---

## 2. Phase 1 — Cohérence sémantique

**Livrables** : `docs/algorithms/STRUCTURE_DEFINITIONS_CURRENT.md`, `STRUCTURE_DEFINITIONS_AUDIT.md`.

### Définitions actuelles (résumé)
| Structure | Définition codée | Fichier |
|---|---|---|
| Fractals | Williams 5 bougies (N=2), causal `shift(N)` | `strategy_features.py:597` |
| BOS | `close > extrême structurel (fractals)`, par corps | `strategy_features.py:32-134` |
| CHOCH | = BOS de renversement (même bougie) | `strategy_features.py:100-115` |
| OB | englobant 2 bougies + nouveau high/low | `strategy_features.py:756-815` |
| FVG | gap 3 bougies, seuil 0.1×ATR | `strategy_features.py:646-679` |
| Régime | momentum close-à-close + ratio TR (règles) | `market_reading_mappers.py:210-252` |

### Divergences vs SMC canonique
| Sévérité | Détection | Câblage | Régime | Total |
|---|---|---|---|---|
| 🟢 Mineure | 4 | 0 | 1 | **5** |
| 🟡 Modérée | 4 | 2 | 4 | **10** |
| 🔴 Majeure | 1 (OB) | 3 (F1/F2/F3) | 0 | **4** |

**Aucune divergence de définition >30 % (pas de STOP).** La seule définition franchement
non-canonique est l'OB englobant (P0-2). Les 3 autres 🔴 sont des **bugs de câblage**, corrigeables
sans toucher au moteur.

---

## 3. Phase 2 — Exactitude de détection

**Livrables** : `scripts/generate_validation_dataset.py`, `data/validation/marketreadings_2026_06_06.json`,
`VALIDATION_DATASET_2026_06_06.md`, `MANUAL_ANNOTATION_TEMPLATE_2026_06_06.md`, `SCORING_TEMPLATE_2026_06_06.csv`.

### Dataset généré
- **60 candles** (30 XAU H1 + 30 EUR H1), **0 erreur** de génération.
- Stratification par état-algo (6 strates × 5 × 2), parfaitement remplie ; sélection déterministe.
- **55/60 descriptions Haiku live** (5 fallback template).
- **Écart data documenté** : XAU = jan-mars 2026 (comme spécifié) ; **EUR = oct-déc 2025**
  (le CSV EURUSD s'arrête au 2025-12-31, jan-mars 2026 indisponible).

### ⚠️ La validation manuelle reste à faire (action opérateur)
L'exactitude réelle (l'algo a-t-il raison de dire « BOS bullish à 14h00 ») **ne peut être tranchée
que par annotation humaine sur TradingView** — hors périmètre automatisable. Le template
`MANUAL_ANNOTATION_TEMPLATE` contient, pour chacune des 60 candles, le résultat algo pré-inséré +
les cases à cocher. Le CSV `SCORING_TEMPLATE` permet de calculer précision/rappel après annotation.

> **Note de méthode** : la stratification utilise l'état détecté par l'algo lui-même. Elle garantit
> la **diversité** des conditions à valider (et non un échantillon biaisé vers le « facile »).
> Elle ne préjuge pas de l'exactitude — c'est précisément ce que l'annotation mesurera.

---

## 4. Phase 3 — Qualité descriptive Haiku

**Livrables** : `scripts/evaluate_haiku_coherence.py`, `HAIKU_COHERENCE_AUDIT_2026_06_06.md`,
`data/validation/haiku_coherence_results.json`. n = 60.

| Test | Résultat | Statut |
|---|---|---|
| C — Sans forbidden token | **100 %** | 🟢 PASS (gate critique) |
| D — Sans invention | **100 %** | 🟢 PASS |
| B — Cohérence directionnelle | **100 %** (41 applic.) | 🟢 PASS |
| A — Mention complète | **0 %** | 🔴 par **omission**, pas erreur |

**Test A bas = omission, pas hallucination.** Causes : (1) tag `bos_recent` sur 100 % des readings
(F6) → tag non-discriminant que le LLM ignore ; (2) design « 1 phrase régime » ; (3) le LLM ne
reçoit que tags+régime (F5). **0 % de contradiction, 0 % de token interdit → aucun cas STOP.**

---

## 5. Recommandations prioritaires

### 🔴 Critique — avant la bêta
1. **F1/F2/F3 — niveaux faux** : câbler `BOS_BREAK_LEVEL` + vraies zones OB/FVG, **OU** ne pas
   exposer de niveaux chiffrés BOS/CHOCH/OB/FVG au client. *(patch `market_reading_mappers.py`, validation founder)*
2. **F6 — tag `bos_recent`** : ne l'émettre que sur `BOS_EVENT≠0` (break frais). *(patch mapper)*
3. **OB ≠ ICT** : ne **pas** écrire « Order Block institutionnel » côté client (déjà acté) ;
   documenter « englobant directionnel » sur `/methodology`.

### 🟠 Important — dans le mois
4. **Aligner les 2 ensembles de forbidden tokens** : retirer le bare `entre` du set mapper (P4).
5. **Décider du rôle de la prose Haiku** : assumer « résumé de régime » (recommandé) et le documenter,
   OU passer la structure au LLM si on veut qu'elle nomme BOS-frais/OB/FVG.
6. **Documenter sur `/methodology`** : BOS sur close ; CHOCH = BOS de renversement ; FVG/OB mitigation
   non suivis (statut « actif » = « détecté », pas « encore valide ») ; régime = formule custom dépendant de la fenêtre.

### 🟢 Nice-to-have — V1.1+
7. Calibrer `FVG_THRESHOLD` / `RETEST_TOL_ATR` (≈ spread XAU) — Sprint 2.
8. Retirer la valeur de schéma `distribution` (jamais produite) ou l'implémenter.
9. Suivi de régression CI de la cohérence Haiku après correctif F6.

---

## 6. Actions opérateur (founder) restantes

| Action | Estimation | Bloquant bêta ? |
|---|---|---|
| **Annotation manuelle des 60 candles** sur TradingView (`MANUAL_ANNOTATION_TEMPLATE` + `SCORING_TEMPLATE.csv`) | 3-4 h | **Oui** (mesure l'exactitude réelle) |
| **Validation par expert SMC externe** des définitions (OB englobant, CHOCH=renversement) | 200-300 € | Recommandé |
| **Décision sur F1/F2/F3** : corriger le câblage OU masquer les niveaux | décision | **Oui** |
| **Décision sur le rôle de la prose Haiku** (5) | décision | Non |

> Les patches F1/F2/F3/F6/P4 touchent `market_reading_mappers.py` (mapping), **jamais le moteur de
> détection**. Aucun n'a été appliqué dans cet audit (règle : pas de modif algo sans validation founder).

---

## 7. Critères de « GO bêta »

| Critère | Cible | Statut actuel |
|---|---|---|
| Précision BOS | > 90 % | ⏳ à valider (annotation manuelle) |
| Précision OB | > 85 % | ⏳ à valider (annotation manuelle) |
| Précision FVG | > 90 % | ⏳ à valider (annotation manuelle) |
| Cohérence Haiku C (tokens) | = 100 % | 🟢 **100 %** |
| Cohérence Haiku globale (B+C+D) | > 90 % | 🟢 **100 %** (hors A) |
| Divergences sémantiques majeures non documentées | 0 | 🟡 1 (OB) à documenter + 3 câblage à corriger/masquer |
| Niveaux de prix fiables (F1-F3) | exacts ou masqués | 🔴 **à traiter** |

**Verdict synthétique** : le **socle (sécurité, honnêteté, déterminisme)** est prêt. Le **GO bêta
dépend** de (a) l'annotation manuelle confirmant la précision de détection, et (b) la résolution
des 3 bugs de câblage de niveaux (corriger ou masquer). Aucun de ces points ne remet en cause le
moteur de détection lui-même.

---

## Annexe — Index des livrables

| Phase | Fichier |
|---|---|
| 1 | `docs/algorithms/STRUCTURE_DEFINITIONS_CURRENT.md` |
| 1 | `docs/algorithms/STRUCTURE_DEFINITIONS_AUDIT.md` |
| 2 | `scripts/generate_validation_dataset.py` |
| 2 | `scripts/audit/validation/build_annotation_artifacts.py` |
| 2 | `data/validation/marketreadings_2026_06_06.json` |
| 2 | `docs/audits/VALIDATION_DATASET_2026_06_06.md` |
| 2 | `docs/audits/MANUAL_ANNOTATION_TEMPLATE_2026_06_06.md` |
| 2 | `docs/audits/SCORING_TEMPLATE_2026_06_06.csv` |
| 3 | `scripts/evaluate_haiku_coherence.py` |
| 3 | `docs/audits/HAIKU_COHERENCE_AUDIT_2026_06_06.md` |
| 3 | `data/validation/haiku_coherence_results.json` |
| Final | `docs/audits/VALIDATION_ALGORITHMIQUE_FINAL_2026_06_06.md` (ce document) |

### Findings nommés (référence rapide)
- **F1** : `BOS.level` lit clé inexistante `BOS_PRICE_LEVEL` → toujours `current_price`.
- **F2** : `CHOCH.level` lit `CHOCH_PRICE_LEVEL` (inexistant) → toujours `current_price`.
- **F3** : niveaux OB/FVG = proxy `prix ± ATR/2`, vraies zones ignorées.
- **F4** : 1 OB/FVG max par reading (pas un registre de zones actives).
- **F5** : Haiku ne reçoit que `tags`+`regime`, jamais `structure`.
- **F6** : tag `bos_recent` sur 100 % des readings (53 % ne sont pas des breaks frais).
- **P4** : `entre` rejeté par le filtre mapper, exclu par le filtre chatbot (incohérence).

# RD_AUDIT — Synthèse priorisée du moteur de détection MIA Markets

> **Audit R&D exploratoire — Rapport stratégique consolidé**
> Date : 2026-06-08 · Branche : `audit/rd-exploratoire-moteur-detection` (base `institutional-overhaul` @ 64daae0)
> Auteur : terminal d'audit R&D (automatisé) · **Aucune ligne de code modifiée** (moteur, mapper, tests, frontend)
> Documents source : `RD_AUDIT_D1_ARCHITECTURE.md` · `D2_HEURISTIQUES.md` · `D3_PERFORMANCE.md` · `D4_CAS_LIMITES.md` · `D5_NIVEAU_15.md`
> Filtre d'or appliqué : **Petit (<1j) / Moyen (1-5j) / Hors-portée (>5j)** — le Top 10 ne contient QUE du Petit/Moyen.

---

## Section 1 — Synthèse exécutive

### État global du moteur : **🟢 SOLIDE → améliorations incrémentales suffisantes**

Le moteur de détection (`SmartMoneyEngine`, `src/environment/strategy_features.py`) est **architecturalement sain, déterministe, causal, transparent et honnête au niveau 1.5**. Il n'a **pas besoin d'être réécrit** ni migré. Les améliorations identifiées sont toutes **incrémentales (Petit/Moyen), additives, à faible risque**, et **aucune** n'exige de toucher au cœur de la logique de break/structure.

> **Verdict d'orientation : améliorations incrémentales suffisantes.** Un refactoring localisé (extraction physique du monolithe) est **déjà planifié et délibérément différé** (Sprint 6) — **ne pas l'avancer avant la bêta** (piège « réinventer »).

### Top 3 forces
1. 🟢 **Découplage produit ↔ moteur propre** (façade + pipeline injectable) → testable, remplaçable, sans couplage au RL legacy malgré l'emplacement du fichier.
2. 🟢 **Honnêteté niveau 1.5 native** : le moteur produit du **catégoriel descriptif**, **zéro probabilité/score directif exposé**, transparence totale → `/methodology` documentable tel quel.
3. 🟢 **Robustesse anti-régression intrinsèque** : garde-fou anti look-ahead **runtime**, protection anti-« 100 %-firing », bonne couverture de tests sur les fonctions pures de détection.

### Top 3 faiblesses
1. 🟡 **Seuils ≈ spread non calibrés** (FVG 0.1×ATR, retest 0.5×ATR) + **suivi de remplissage absent** (statut toujours « actif ») → bruit et fraîcheur surévaluée.
2. 🟡 **Fragilité en feed live** : aucune gestion des **gaps temporels** (weekend/holiday → FVG/BOS fantômes), pas de sanity-check OHLC, pas de **monitoring runtime** du firing-rate (le bug data-quality n'est attrapé qu'offline).
3. 🟡 **OB = englobant 2 bougies ≠ ICT** (pas de filtre displacement) → faux positifs ; déjà acté « ne pas vendre comme institutionnel ».

### Cap éditorial (rappel transverse)
Les **vrais risques de crédibilité** sur les *niveaux chiffrés* (F1/F2/F3 = proxies `prix±ATR/2`) et le tag `bos_recent` (F6) relèvent du **mapper** → **traités par l'autre terminal**, hors de ce périmètre. Ils sont **rappelés** ici mais non agis.

---

## Section 2 — Tableau des findings priorisés

| ID | Dim | Catégorie | Description | Coût | Impact | Risque | Priorité |
|---|---|---|---|---|---|---|---|
| **D4-2** | 4 | Robustesse | Sanity-checks intégrité OHLC en entrée (NaN, high<low, prix≤0, ts monotone) | Petit | Moyen | Faible | 🟠 Important |
| **D4-1** | 4 | Robustesse | Détection de gap temporel weekend/holiday → neutraliser FVG fantôme | Moyen | Moyen | Faible | 🟠 Important |
| **D4-6** | 4 | Observabilité | Monitoring runtime firing-rate BOS / NaN-rate / n_fractals + alerte hors [0.5 %,10 %] | Petit | Moyen | Faible | 🟠 Important |
| **D2-3** | 2 | Heuristique | Relever défaut `FVG_THRESHOLD` 0.1→~0.25-0.35×ATR (filtre bruit spread) | Petit | Moyen | Faible | 🟠 Important |
| **D2-4** | 2 | Heuristique | Passe de suivi remplissage/mitigation FVG+OB (statut active→filled/mitigated) | Moyen | Moyen | Faible | 🟠 Important |
| **D1-3** | 1 | Dette | Unifier les 2 jeux de defaults divergents (SMCConfig 14/0.1 vs preprocess 7/0.0) | Petit | Moyen | Faible | 🟠 Important |
| **D2-1** | 2 | Heuristique | Filtre displacement sur OB (corps>k·ATR ou ancrage BOS) — **gated annotation** | Moyen | Fort | Moyen | 🟠 Important |
| **D2-9 / D3-1** | 2/3 | Perf+Propreté | Ne pas calculer la divergence RSI dans le chemin produit (code mort O(n·k)) | Petit | Faible-Moyen | Faible | 🟢 Nice |
| **D5-1+D5-2** | 5 | Transparence 1.5 | Documenter `/methodology` seuils régime (0.3/0.7/1.3) + OB importance (0.75/0.4) | Petit | Moyen | Faible | 🟠 Important |
| **D2-7** | 2 | Heuristique | Exposer `BOS_BREAK_BUFFER_ATR` (défaut 0.0 = inchangé) pour filtrer micro-breaks | Petit | Moyen | Faible | 🟢 Nice |
| **D1-4** | 1 | Dette | Retirer le bloc `__main__` d'entraînement RL du module détection | Petit | Faible | Faible | 🟢 Nice |
| **D2-2** | 2 | Heuristique | Ne pas additionner prime FVG (+0.2) à un ratio → `importance` interprétable | Petit | Faible | Faible | 🟢 Nice |
| **D2-5** | 2 | Heuristique | Aligner défaut `armed_window` fonction (5) sur config (30) + calibrer retest tol | Petit | Faible | Faible | 🟢 Nice |
| **D2-6** | 2 | Heuristique | `FRACTAL_WINDOW` TF-aware (2 sur M15, 3 sur H4/D1) | Petit | Faible-Moyen | Faible | 🟢 Nice |
| **D4-5** | 4 | Robustesse | Lookback TF-aware (plus de profondeur sur TF hauts) | Petit | Moyen | Moyen | 🟢 Nice |
| **D1-5** | 1 | Dette | Retirer `logging.basicConfig` au niveau import (effet de bord global) | Petit | Faible | Faible | 🟢 Nice |
| **D1-2** | 1 | Dette | Test de parité numba↔python (recommandation — écriture de test = founder) | Petit | Moyen | Faible | 🟢 Nice |
| **D3-3** | 3 | Perf | Garantir numba en prod OU assumer/dimensionner le fallback Python | Petit | Moyen | Faible | 🟠 Important |
| **D4-3** | 4 | Robustesse | Flagger (sans altérer) les bougies outliers >5×ATR | Petit | Faible-Moyen | Faible | 🟢 Nice |
| **D4-7** | 4 | Régime | Hystérésis anti-clignotement régime *(mapper — coordonner autre terminal)* | Petit-Moyen | Moyen | Faible | 🟢 Nice |
| **D3-2** | 3 | Perf | Mémoïser/numpy-iser les objets `ta` (30 % du temps @200) | Moyen | Faible-Moyen | Moyen | 🟢 Nice |
| **D3-6** | 3 | Scalabilité | Worker pool/file pour lisser les rafales de clôture (>20 combos, V1.1+) | Moyen | Moyen | Faible | 🟢 Nice (V1.1) |

---

## Section 3 — Top 10 actions recommandées (Petit + Moyen uniquement)

> Ordonnées par ratio **impact/coût × priorité**. Toutes **sans réécriture**. « Touche » = fichier concerné.

### 1. D4-2 — Sanity-checks intégrité OHLC *(Petit, ~2-4 h)*
- **Quoi** : valider en entrée d'`analyze()` l'absence de NaN OHLC, `high≥max(o,c)`, `low≤min(o,c)`, prix>0, index temporel monotone ; warning structuré sinon.
- **Pourquoi** : aujourd'hui une série corrompue produit un `MarketReading` **dégradé silencieusement**. Premier filet défensif avant le feed live bêta.
- **Impact** : Moyen (évite des readings faux invisibles). **Risque** : Faible. **Touche** : `strategy_features.py`. **Dépendances** : aucune.

### 2. D4-6 — Monitoring runtime du firing-rate *(Petit, ~3-4 h)*
- **Quoi** : log/métrique par reading de `bos_event_rate`, `nan_rate`, `n_fractals` ; alerte si firing-rate ∉ [0.5 %, 10 %].
- **Pourquoi** : le bug « 100 %-firing » n'est attrapé qu'**offline** (test). En bêta, un founder solo n'a pas le temps de surveiller — il faut une **alerte automatique**.
- **Impact** : Moyen (détection précoce de dérive data). **Risque** : Faible. **Touche** : `strategy_features.py` / assembleur. **Dépendances** : aucune.

### 3. D1-3 — Unifier les defaults divergents *(Petit, ~2-3 h)*
- **Quoi** : faire pointer `preprocess_dataframe`/`_parallel` sur `SMCConfig()` (ou documenter le double profil).
- **Pourquoi** : élimine le **skew train/serve** (le produit détecte avec 14/0.1, le backtest avec 7/0.0 → la calibration ne décrit pas le servi).
- **Impact** : Moyen (cohérence calibration↔prod). **Risque** : Faible. **Touche** : `strategy_features.py`. **Dépendances** : à faire **avant** toute calibration de seuils (D2-3).

### 4. D2-3 — Calibrer le seuil FVG *(Petit pour le défaut, calibration = annotation)*
- **Quoi** : relever `FVG_THRESHOLD` 0.1→~0.25-0.35×ATR ; **valider la valeur exacte sur l'annotation**.
- **Pourquoi** : 0.1×ATR ≈ spread XAU → publie des micro-gaps insignifiants.
- **Impact** : Moyen (qualité/honnêteté FVG). **Risque** : Faible. **Touche** : `strategy_features.py` (1 défaut). **Dépendances** : **annotation manuelle** (Section 5) pour figer la valeur.

### 5. D5-1 + D5-2 — Documenter les seuils sur `/methodology` *(Petit, ~3-4 h rédaction)*
- **Quoi** : expliciter que seuils régime (0.3/0.7/1.3) et OB importance (0.75/0.4) sont **heuristiques** ; définir `importance` = taille zone/ATR (descriptif, pas « probabilité de tenue »).
- **Pourquoi** : transparence 1.5 = atout différenciant ; **honnêteté > faux empirisme**.
- **Impact** : Moyen (crédibilité/conformité 1.5). **Risque** : Faible. **Touche** : `/methodology` (frontend doc — hors code moteur ; côté contenu). **Dépendances** : aucune.

### 6. D2-9 / D3-1 — Retirer la divergence RSI du chemin produit *(Petit, ~1-2 h)*
- **Quoi** : `analyze(compute_divergence=False)` dans le pipeline produit (champ non consommé par le mapper).
- **Pourquoi** : supprime une boucle O(n·k) **inutile** à chaque reading + retire du **code mort** (et un historique de bug) du chemin client.
- **Impact** : Faible-Moyen (−4 % latence + propreté). **Risque** : Faible. **Touche** : `strategy_features.py` + appel assembleur. **Dépendances** : aucune.

### 7. D4-1 — Détection de gap temporel *(Moyen, ~1-2 j)*
- **Quoi** : repérer `Δt > k×durée_TF`, neutraliser le FVG enjambant le gap, flagger le BOS post-gap.
- **Pourquoi** : en **feed live** (vs CSV propre), les gaps de séance sont la 1ʳᵉ source de FVG/BOS fantômes.
- **Impact** : Moyen (robustesse live, le mode bêta). **Risque** : Faible (additif). **Touche** : `strategy_features.py`. **Dépendances** : bénéficie de D4-2.

### 8. D2-4 — Suivi de remplissage/mitigation FVG + OB *(Moyen, ~2-3 j)*
- **Quoi** : passe forward qui marque un FVG `filled/partially_filled` et un OB `mitigated` quand le prix revient dans la zone.
- **Pourquoi** : le statut est **toujours « actif »** → on affiche comme « actif » des zones déjà comblées (fraîcheur surévaluée). Aligne le produit sur la sémantique SMC.
- **Impact** : Moyen (honnêteté de fraîcheur). **Risque** : Faible (additif, nouveaux statuts déjà au schéma). **Touche** : `strategy_features.py` + lecture mapper (coordonner zone active avec l'autre terminal — recoupe F4).

### 9. D3-3 — Statuer sur numba en prod *(Petit, ~1-2 h)*
- **Quoi** : soit garantir `numba` dans l'image prod + warm-up au boot, soit **assumer** le fallback Python et dimensionner (latence ×5-10 à ≥2 000 bougies).
- **Pourquoi** : numba **absent** dans cet environnement → les benchmarks docstring (« 0.2-0.5 s/20k ») sont trompeurs ; éviter de sous-dimensionner.
- **Impact** : Moyen (prévisibilité prod). **Risque** : Faible. **Touche** : infra/requirements + docstring. **Dépendances** : aucune.

### 10. D2-1 — Filtre displacement sur Order Block *(Moyen, ~2-4 j — GATED annotation)*
- **Quoi** : ne retenir l'OB que si impulsion consécutive (corps `i`/`i+1` > k×ATR) **ou** ancrage à un `BOS_EVENT` proche.
- **Pourquoi** : rapproche l'OB de l'ICT, **réduit fortement les faux positifs** (~5-10× moins d'OB), permettrait à terme un wording plus fidèle.
- **Impact** : **Fort** (qualité/honnêteté OB). **Risque** : **Moyen** (change le volume d'OB). **Touche** : `strategy_features.py`. **Dépendances** : **NE PAS livrer avant l'annotation manuelle** — sinon on casse la précision qu'on cherche à mesurer. Tests de non-régression requis.

> **Coût cumulé Top 10** : ~3 Petit-jours + ~3 Moyen-tâches ≈ **8-12 j de dev** étalables. Le **trio bêta-bloquant** (robustesse feed live) = **#1 D4-2, #2 D4-6, #7 D4-1** (~2,5-3 j).

---

## Section 4 — Archives V1.2+ (Hors-portée pour V1)

> Documentés pour mémoire, **non recommandés maintenant**. Aucun dans le Top 10.

| ID | Description | Pourquoi Hors-portée |
|---|---|---|
| **D1-1/D1-6** | Extraction physique du monolithe 1213 LOC → `smart_money/` (sortir TA/VIF/benchmark/RL) | >5 j, ~20 entrypoints ; **déjà planifié Sprint 6** ; façade rend non-bloquant. Risque régression élevé avant bêta. |
| **D2-8** | Vrai CHOCH **interne** distinct du BOS-de-renversement | Nouvelle logique de swing interne, impacte tout le pipeline structure. Documenter la fusion suffit pour V1. |
| **D2-7b** | « Structure = dernier swing majeur » (vs max-de-tous-fractals) | Change la sémantique BOS → re-validation complète nécessaire. |
| **D4-4** | Mode fractale « corps » (anti mèche longue) | Touche la définition fractale → gated annotation + re-validation. |
| **Store de zones persistant** | Registre des OB/FVG actifs non-mitigés au-delà de 200 bougies | Nouvelle persistance + cycle de vie de zones (recoupe F4 mapper). Architecture, pas paramètre. |
| **Câblage HMM/régime ML** | Brancher `regime_classifier`/`volatility_forecaster` au produit | Contredit le pivot (champs probabilistes retirés du visible, mémoire 2026-05-27). **NE PAS faire.** |
| **D3 réécriture Polars/Cython** | Réécrire le moteur pour la vitesse | Gain non justifié au volume V1 ; piège « réinventer ». |

---

## Section 5 — Validation empirique à faire (annotation founder)

> Hypothèses de cet audit qui **ne peuvent être tranchées que par l'annotation manuelle** des 60 candles (template déjà produit : `MANUAL_ANNOTATION_TEMPLATE_2026_06_06.md` + `SCORING_TEMPLATE_2026_06_06.csv`).

1. **Seuil FVG (D2-3)** : « 0.1×ATR produit des faux positifs de bruit » → confirmer le taux de FVG insignifiants ; **figer la valeur** (0.25 ? 0.35 ?) sur la précision/rappel observés.
2. **Seuil retest (D2-5)** : « 0.5×ATR ≈ spread déclenche des retests sur bruit » → mesurer.
3. **Filtre displacement OB (D2-1)** : « l'englobant produit ~5-10× trop d'OB » → quantifier les faux positifs OB **avant/après** filtre ; valider que la précision monte sans effondrer le rappel.
4. **Mèche longue (D4-4)** : « les fractales-mèche suppriment des BOS légitimes » → vérifier sur les candles à wick extrême du dataset.
5. **Régime clignotant (D4-7)** : « trend/phase oscille près des seuils » → observer la stabilité bougie-à-bougie sur les transitions.
6. **OB importance (D5-1)** : les seuils 0.75/0.4 produisent-ils une répartition low/medium/high **interprétable** par un trader ? → jugement humain.

> **Sans cette annotation, ne PAS figer** les valeurs de seuils ni livrer D2-1 : on optimiserait à l'aveugle.

---

## Section 6 — Risques identifiés

### Risques produit (qualité de détection)
- 🟡 **OB faux positifs** (englobant sans displacement) → zones peu significatives. *Mitigation : D2-1 après annotation.*
- 🟡 **FVG/OB « actifs » déjà comblés** (pas de mitigation) → fraîcheur surévaluée. *Mitigation : D2-4.*
- 🟡 **Bruit FVG/retest** (seuils ≈ spread). *Mitigation : D2-3/D2-5 calibrés.*

### Risques techniques (perf, scalabilité, robustesse)
- 🔴 **Feed live non durci** : gaps/NaN/outliers non gérés → FVG/BOS fantômes en prod. *Mitigation : D4-1/D4-2/D4-3.* **Le plus urgent avant bêta.**
- 🟡 **numba absent** → latence prod sous-estimée. *Mitigation : D3-3.*
- 🟡 **Rafale de clôture** à >20 combos. *Mitigation : D3-6 (V1.1+).*
- 🟡 **Skew train/serve** (defaults divergents). *Mitigation : D1-3.*
- 🟡 **Drift numba↔python** (double implémentation). *Mitigation : test de parité D1-2.*

### Risques positionnement (niveau 1.5)
- 🟢 **Moteur conforme** : zéro probabilité/score directif exposé, transparence totale. **Aucun risque imputable au moteur.**
- 🟡 **Seuils non documentés** (régime, OB). *Mitigation : D5-1/D5-2 (documentation).*
- ⚠️ **Niveaux chiffrés proxy (F1-F3) + tag bos_recent (F6)** = risque de crédibilité **réel mais HORS périmètre** (mapper, autre terminal). **Tant que non corrigés : ne pas afficher de niveaux chiffrés** (cohérent mémoire 2026-05-27).

### Risques business (lancement bêta)
- 🟠 **Lancer sans annotation** = lancer un détecteur dont la **précision réelle est inconnue** (l'audit précédent l'a explicité : exactitude = action founder non encore faite). **Recommandation : faire l'annotation (3-4 h) avant la bêta** — c'est le **vrai** bloquant, pas le code.
- 🟢 **Le moteur ne présente aucun risque réputationnel intrinsèque** au niveau 1.5 : il est honnête, transparent, descriptif. Le risque réside dans (a) la précision non mesurée et (b) les niveaux chiffrés du mapper.

---

## Conclusion

Le moteur de détection est **prêt sur le fond** (architecture, déterminisme, honnêteté 1.5) et **demande un durcissement de surface** (robustesse feed live) + **une calibration empirique** (annotation) — **pas une réécriture**. Le chemin le plus court vers une bêta crédible :

1. **Annotation manuelle des 60 candles** (founder, 3-4 h) — débloque la mesure de précision **et** la calibration des seuils.
2. **Trio robustesse feed live** : D4-2 + D4-6 + D4-1 (~2,5-3 j).
3. **Hygiène + transparence** : D1-3, D2-9, D3-3, D5-1/D5-2 (~1,5-2 j).
4. **Qualité gated** : D2-3 (calibré), D2-4, puis D2-1 (après annotation).

Le tout reste dans l'enveloppe **« améliorer, pas réinventer »** : ~8-12 j de dev incrémental, zéro réécriture, zéro nouveau framework, zéro ML complexe.

---

*Fin du rapport. Livrables : 5 documents dimensionnels (D1-D5) + cette synthèse. Aucune modification de code, de mapper, de tests ou de frontend. Branche poussée, 0 PR, 0 merge.*

# RD_AUDIT_D4 — Couverture des cas limites

> **Audit R&D exploratoire — Dimension 4/5**
> Date : 2026-06-08 · Branche : `audit/rd-exploratoire-moteur-detection`
> Objet : identifier où le moteur peut **échouer silencieusement**. Aucune modification de code ni de test.

---

## 4.1 — Inventaire des cas + invariants

### Invariants actuellement protégés par les tests (lecture des suites existantes)

| Invariant | Test garant | Robustesse |
|---|---|---|
| BOS ne fire pas à 100 % des bars | `test_bos_no_repeated_fire.py`, `test_data_quality_bos_regression.py` (firing ∈ [0.5 %, 10 %]) | 🟢 Fort |
| Causalité fractals (anti look-ahead) | garde-fou **runtime** `strategy_features.py:692-696` (`raise` si fractal dans les N dernières bars) | 🟢 Fort (runtime, pas juste test) |
| CHOCH reset structure | `test_sprint2_choch_reset.py` | 🟢 |
| OB détectés sans FVG + bonus | `test_sprint3_order_blocks.py` | 🟢 |
| Zone OB high ≥ low | `test_sprint3` (`test_ob_zones_are_valid`) | 🟢 |
| FVG threshold filtre le bruit | `test_sprint5_fvg_threshold.py` | 🟢 |
| Retest state machine | `test_bos_retest.py` | 🟢 |
| ATR=0 → pas de division | `np.where(ATR>0, …)` (lignes 669, 807) | 🟢 |

### Invariants qui DEVRAIENT l'être mais ne le sont pas

| Cas limite | Comportement actuel | Comportement idéal | Visible utilisateur | Sévérité | Coût |
|---|---|---|---|---|---|
| **A. Gap weekend/holiday** | Aucune détection de discontinuité temporelle. `low[dim] > high[ven]` → **FVG géant fantôme** ; la rupture de continuité peut faire sur-fire BOS. La seule protection est « utiliser un bon CSV » (test data-quality), **pas** une robustesse du détecteur. | Détecter `Δt > durée_TF` et neutraliser FVG/flagger BOS sur la bougie de gap. | `MarketReading` avec un FVG énorme injustifié / BOS douteux | 🔴 Élevée (live feed) | **Moyen** (D4-1) |
| **B. Données manquantes / NaN intra-série** | `dropna` ne couvre que RSI/MACD/ATR (warm-up). Un NaN OHLC au milieu de la série n'est pas validé ; `rolling` propage des NaN silencieusement. | Valider l'intégrité OHLC en entrée (pas de NaN, `high≥max(open,close)`, `low≤min`, prix>0). | Reading dégradé sans erreur | 🟡 Modérée | **Petit** (D4-2) |
| **C. Bougie aberrante (flash crash, fat finger >3σ)** | Crée un fractal + possible BOS + OB ; aucun garde outlier. | **Flagger** (pas altérer) la bougie hors-distribution ; option neutralisation. | BOS/OB sur un artefact | 🟡 Modérée | **Petit-Moyen** (D4-3) |
| **D. Mèche longue (wick ≫ corps)** | Fractale sur le **wick** ; BOS sur le **close** → faux négatifs ; zone OB = range incluant la mèche (très large). | Option fractale « corps » ou clamp de mèche. | BOS manqué / zone OB trop large | 🟡 Modérée | **Moyen** (D4-4) |
| **E. Mémoire structurelle courte (lookback 200 fixe ∀ TF)** | 200 bougies quel que soit le TF. Sur H4 = ~33 j, sur D1 = 200 j. Seed structure = 50 premières (`min(50,n)`). OB/BOS antérieurs invisibles. | Lookback **TF-aware** (plus de profondeur sur TF hauts) ou store de zones persistant. | Structure « amnésique » sur TF hauts | 🟡 Modérée | **Petit** (D4-5, lookback) / HP (store zones) |
| **F. Transition de régime (trend→range→trend)** | Régime recalculé à chaque reading sur la fenêtre entière (momentum close-à-close + ratio TR). Pas d'hystérésis → peut **osciller** bougie-à-bougie autour des seuils 0.3/0.7/1.3. | Hystérésis / lissage léger des bascules de régime. | trend/phase qui « clignote » | 🟡 Modérée | **Petit-Moyen** (D4-7) |

---

## 4.2 — Stratégies défensives existantes

| Garde-fou | Présence | Évaluation |
|---|---|---|
| **Sanity checks entrée** | Partiel : vérifie présence colonnes OHLCV (`strategy_features.py:561`). **Pas** d'intégrité valeurs (NaN, high<low, prix négatifs, monotonie ts). | 🟡 Insuffisant |
| **Anti look-ahead runtime** | ✅ `raise ValueError` si fuite fractale (lignes 692-696) | 🟢 Excellent |
| **Division ATR=0** | ✅ `np.where(ATR>0, …)` | 🟢 |
| **Config invalide** | ✅ `SMCConfig` Pydantic + fallback defaults loggé (lignes 550-555) | 🟢 |
| **Peu de fractals** | ✅ `logger.warning` (lignes 706-710) | 🟢 |
| **Cleaning laisse peu de lignes** | ✅ `logger.warning` (ligne 947) | 🟢 |
| **Fallbacks** | ✅ numba→python ; description Haiku→template ; mtf/news→empty (assembleur) | 🟢 |
| **Monitoring hooks prod** | ❌ Le firing-rate BOS / taux de NaN / nb structures **ne sont pas loggés/métriqués par reading en prod**. Le bug « 100 %-firing » n'est attrapé que par un **test offline**, pas en runtime. | 🔴 Manquant |

---

## 4.3 — Recommandations défensives (Petit/Moyen)

### 🟠 D4-2 — Sanity checks d'intégrité OHLC en entrée
- **Coût : Petit · Impact : Moyen · Risque : Faible**
- Dans `SmartMoneyEngine.__init__` (ou en tête d'`analyze`) : asserter/logger si NaN dans OHLC, `high < low`, prix ≤ 0, ou index temporel non monotone. Émettre un warning structuré (pas un crash en prod, sauf NaN bloquant).
- Pourquoi : aujourd'hui une série corrompue produit un `MarketReading` **dégradé sans aucun signal**.

### 🟠 D4-1 — Détection de gap temporel (weekend/holiday)
- **Coût : Moyen · Impact : Moyen · Risque : Faible (additif)**
- Calculer `Δt` entre bougies ; là où `Δt > k×durée_TF`, marquer la bougie « post-gap » et **neutraliser le FVG** qui l'enjambe (et/ou flagger le BOS). N'altère pas la détection hors-gap.
- Pourquoi : en feed live (vs CSV propre des tests), les gaps de séance sont la **première** source de FVG/BOS fantômes. C'est le cas limite le plus impactant en prod réelle.

### 🟠 D4-6 — Monitoring runtime (firing-rate / NaN-rate)
- **Coût : Petit · Impact : Moyen · Risque : Faible**
- Émettre par reading (log structuré ou métrique) : `bos_event_rate` sur la fenêtre, `nan_rate` entrée, `n_fractals`. Alerter si firing-rate sort de [0.5 %, 10 %] (le seuil du test data-quality) **en prod**.
- Pourquoi : transforme un garde-fou *offline* (test) en **détection live** de dérive data-quality. Le founder solo n'aura pas le temps de surveiller manuellement.

### 🟡 D4-3 — Flag outliers (sans altérer)
- **Coût : Petit (flag seul) / Moyen (neutralisation) · Impact : Faible-Moyen · Risque : Faible**
- Flagger les bougies dont le range dépasse `m×ATR` (m≈5). **Devil's advocate** : winsoriser/clipper fausserait de vrais mouvements news → **se limiter au flag** + exposer en métrique ; laisser le founder décider de la neutralisation après observation.

### 🟡 D4-5 — Lookback TF-aware
- **Coût : Petit · Impact : Moyen (TF hauts) · Risque : Moyen (arbitrage perf D3-4)**
- Augmenter `DEFAULT_LOOKBACK` pour H4/D1 (p.ex. 300-400) pour ne pas « oublier » la structure. Arbitrage avec la latence (cf. D3) — modeste à ces volumes.

### 🟡 D4-7 — Hystérésis de régime
- **Coût : Petit-Moyen · Impact : Moyen · Risque : Faible** *(touche le mapper `_derive_*` → coordonner avec l'autre terminal)*
- Éviter le clignotement trend/phase près des seuils (exiger 2 bougies consécutives au-delà du seuil pour basculer). **NB** : ce code vit dans `market_reading_mappers.py` (régime), pas dans le moteur → **à confirmer avec le terminal mapper avant d'agir**.

### Hors-portée (archive)
- **D4-4** option fractale « corps » : Moyen, mais touche la définition fractale → gated annotation, archive V1.2+.
- Store de zones OB/FVG persistant (mémoire structurelle au-delà de 200) : Hors-portée (nouvelle persistance + cycle de vie de zones). Recoupe **F4** de l'audit mapper (registre de zones actives) → ne pas dédoubler.

---

## 4.4 — Synthèse Dimension 4

**État robustesse : 🟡 Acceptable en environnement contrôlé, fragile en feed live.** Le moteur est **bien protégé contre ses propres régressions** (anti look-ahead runtime, anti-100 %-firing, division ATR) mais **suppose des données propres**. La robustesse aux gaps/NaN/outliers est **déléguée au choix du CSV** (test data-quality) plutôt qu'assurée dans le détecteur — ce qui tient pour le backtest mais **pas pour un feed temps réel** (le mode bêta).

**Le trio prioritaire avant feed live** : **D4-2** (sanity OHLC, Petit), **D4-1** (gap temporel, Moyen), **D4-6** (monitoring runtime, Petit). Tous **additifs, faible risque, sans toucher la logique de détection**. Coût cumulé ~2-3 j.

**Cas STOP rencontrés en D4** : aucun. Les lacunes sont des **angles morts défensifs**, pas des bugs produisant des résultats faux sur données propres.

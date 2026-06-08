# RD_AUDIT_D1 — Architecture & lisibilité du moteur de détection

> **Audit R&D exploratoire — Dimension 1/5**
> Date : 2026-06-08 · Branche : `audit/rd-exploratoire-moteur-detection` (base `institutional-overhaul` @ 64daae0)
> Périmètre : **logique de détection** (`SmartMoneyEngine`) + régime. **Aucune modification de code.**
> Hors-périmètre : mapper F1/F2/F3 (autre terminal), frontend, tests.
> Filtre appliqué à chaque finding : **Petit (<1j) / Moyen (1-5j) / Hors-portée (>5j)**.

---

## 1.1 — Cartographie des modules

| Module | Chemin | Responsabilité (1 phrase) | LOC | Appelé par | Appelle | Complexité |
|---|---|---|---|---|---|---|
| **Moteur de détection** | `src/environment/strategy_features.py` | Enrichit un DataFrame OHLCV avec colonnes BOS/CHOCH/OB/FVG/fractals/retest + TA | **1213** | `smart_money/__init__` (façade), ~20 entrypoints backtest/training | `ta` lib, `numba` (optionnel), `feature_reducer` | **Complexe** |
| Façade institutionnelle | `src/intelligence/smart_money/__init__.py` | Re-exporte l'API publique du moteur (extraction physique différée Sprint 6) | 79 | assembler, code produit | `strategy_features` | Simple |
| Assembleur lazy | `src/intelligence/market_reading_assembler.py` | Orchestre fetch → pipeline SMC → structure/régime/events → description | 335 | API `/market-reading`, scheduler | mappers, schema, moteur | Modérée |
| Mapper produit | `src/intelligence/market_reading_mappers.py` | Dernière ligne features → `MarketReadingStructure`/`Regime` + tags/description | 440 | assembler | schema | Modérée |
| Schéma produit | `src/intelligence/market_reading_schema.py` | Contrat Pydantic `MarketReading` v2.0.0 | 243 | partout | pydantic | Simple |
| Régime descriptif | `market_reading_mappers.py` (`_derive_*`) | trend/volatilité/phase à partir des closes (indépendant du moteur SMC) | (incl.) | assembler | — | Simple |

**Modules ML présents mais NON câblés au produit** (confirmé `STRUCTURE_DEFINITIONS_CURRENT.md §7.5`) : `regime_classifier.py` (HMM), `volatility_forecaster.py`, `regime_filter.py`, `regime_gate.py` (BOCPD). Ils n'influencent pas le `MarketReading` → hors-périmètre détection, mais pertinents pour D5 (vocabulaire prédictif latent).

### Schéma de dépendance (chemin produit réel)

```
API /market-reading
  └─ MarketReadingAssembler.get_or_generate()
       └─ _default_smc_pipeline(candles)        ← construit un DataFrame, lance SmartMoneyEngine.analyze()
            └─ SmartMoneyEngine (strategy_features.py)
                 ├─ _add_ta_indicators()         RSI/MACD/BB/ATR (lib `ta`)
                 ├─ _add_smc_base_features()      fractals + FVG
                 ├─ _add_smc_order_blocks()       OB
                 ├─ _calculate_structure_iterative()  BOS/CHOCH (numba|python) + retest
                 └─ _detect_rsi_divergence()      CHOCH_DIVERGENCE  ⚠️ non consommé par le produit
       └─ confluence_signal_to_structure() / candles_to_regime()  (mapper)
```

> **Constat architectural sain** : le moteur de détection est **découplé du produit** par une frontière propre (façade + pipeline injectable). L'assembleur ne connaît du moteur qu'un `Callable`. C'est une bonne base — la dette est *interne* au fichier moteur, pas dans le couplage.

---

## 1.2 — Évaluation qualité

### Lisibilité — 🟢 Bonne
- Docstrings denses et **honnêtes** (elles décrivent les pièges, p.ex. le bug « 100 %-firing » lignes 50-60).
- Nommage explicite (`current_high_structure`, `bos_break_level`, `last_fractal_high`).
- Sections balisées (`═══`), commentaires de transition d'état pour la state-machine retest (lignes 240-257).
- Bémol : magic numbers dispersés (`min(50, n)` ligne 77 = seed window non paramétré ; seuils `importance` 0.75/0.4 dans le mapper ligne 153).

### Testabilité — 🟢 Forte sur les fonctions pures
- Les cœurs de détection sont des **fonctions pures numpy-in/numpy-out** (`calculate_bos_choch_fast`, `calculate_bos_retest_fast`). Testées directement (`test_bos_no_repeated_fire.py`, `test_bos_retest.py`) sans monter la classe.
- Les méthodes de classe mutent `self.df` (effet de bord) → testées via `analyze()` de bout en bout (synthétique). Acceptable.
- Pipeline injectable dans l'assembleur (`smc_pipeline`) → tests produit sans le moteur lourd. 🟢

### Modularité — 🟡 Moyenne
- ✅ On peut changer la détection OB (`_add_smc_order_blocks`) sans toucher BOS.
- ⚠️ Mais on **ne peut pas obtenir un seul détecteur isolément** : `analyze()` est tout-ou-rien (TA → fractals → FVG → OB → BOS → retest → divergence). Pour publier 1 ligne de `MarketReading`, le produit recalcule **toute** la chaîne sur 200 bougies (cf. D3).
- ⚠️ Le fichier mélange : détection SMC + indicateurs TA + divergence RSI + `compute_feature_vif` + `benchmark_preprocessing` + `preprocess_dataframes_parallel` + un `__main__` d'**entraînement RL**. Sept responsabilités dans un fichier « détection ».

### Patterns design
| Pattern | Présence | Évaluation |
|---|---|---|
| **Pipeline** | `analyze()` = séquence d'étapes chronométrées | 🟢 Clair, mais monolithique (pas de steps composables) |
| **State machine** | retest (IDLE/AWAITING/ARMED, ±1/±2) lignes 259-334 | 🟢 Propre, documentée, déterministe |
| **Strategy / injection** | `smc_pipeline` injectable dans l'assembleur | 🟢 Bonne frontière test |
| **Façade** | `smart_money/__init__` | 🟢 Découple produit ↔ emplacement legacy |
| **Fallback** | numba → python sur les 2 cœurs | 🟡 Bon principe, **mais duplication** (cf. dette D1-2) |

---

## 1.3 — Dettes techniques

> Format : **Coût/Risque/Priorité** d'abord, description ensuite (méthode imposée).

### 🟠 D1-3 — Deux jeux de defaults divergents (skew train/serve)
- **Coût : Petit · Risque (si rien) : Moyen · Priorité : Important**
- `SMCConfig` (ligne 440) : `RSI=14, MACD=12/26/9, ATR=14, FVG_THRESHOLD=0.1`.
- `preprocess_dataframe` / `preprocess_dataframes_parallel` (lignes 1003-1012, 1051-1060) : `RSI=7, MACD=8/17/9, ATR=7, FVG_THRESHOLD=0.0`.
- Le **produit** (assembleur, `config={}`) prend les defaults `SMCConfig` (14/0.1). Le **training/backtest** prenait 7/0.0. → les structures détectées hors-ligne (calibration, backtest) ne sont **pas** celles servies au client. Risque déjà connu en mémoire (« MLOps skew »).
- **Reco** : une seule source de vérité (faire pointer `preprocess_*` sur `SMCConfig()`), ou documenter explicitement pourquoi deux profils. Touche `strategy_features.py` (pas le mapper).

### 🟢 D1-4 — `__main__` d'entraînement RL dans le module de détection
- **Coût : Petit · Risque : Faible · Priorité : Nice-to-have**
- Deux blocs `if __name__ == '__main__'` (lignes **1127** et **1200**). Le premier importe `AgentTrainer` et **lance un entraînement RL** ; le second parse `--benchmark`. Lancer `python strategy_features.py` déclenche donc l'entraînement RL legacy.
- Vestige de l'époque RL (mémoire : pivot RL→SaaS). Couplage mort dans un module cœur.
- **Reco** : supprimer le 1ᵉʳ bloc `__main__` (RL). Aucun impact détection.

### 🟡 D1-2 — Double implémentation numba/python (drift)
- **Coût : Petit (test de parité) · Risque : Moyen · Priorité : Important**
- BOS/CHOCH et retest existent en **deux copies** (`_calculate_bos_choch_numba` 32-134 + `_calculate_bos_choch_python` 159-234 ; idem retest). ~150 LOC dupliquées à maintenir synchronisées **à la main**.
- Risque : une correction appliquée à une version et pas à l'autre → comportement divergent selon que numba est installé ou non (et **il ne l'est pas ici**, cf. D3 — donc c'est la version Python qui tourne en réalité, alors que les benchmarks docstring vantent numba).
- **Reco** : ajouter un **test de parité** numba-vs-python sur un échantillon (asserte égalité des 4 arrays). *NB : écrire ce test sort du périmètre « ne pas modifier les tests » — c'est une recommandation pour le founder, pas une action de cet audit.* Alternative Petit : marquer clairement la copie Python comme « source de vérité » et générer un golden-file de référence.

### 🟢 D1-5 — `logging.basicConfig` au niveau import
- **Coût : Petit · Risque : Faible · Priorité : Nice-to-have**
- Ligne 24 : `logging.basicConfig(level=INFO, ...)` s'exécute **à l'import** du module → reconfigure le **root logger** du process entier (effet de bord global). En prod multi-module, peut écraser la config de logging de l'app (cf. mémoire : `LOG_FORMAT=json`).
- **Reco** : retirer le `basicConfig` au profit d'un `getLogger(__name__)` sans config globale (la config appartient à l'entrypoint app).

### 🟢 D1-1 / D1-6 — Monolithe 1213 LOC en emplacement RL-legacy
- **Coût : Hors-portée (>5j, ~20 entrypoints) · Priorité : différé Sprint 6**
- Le moteur vit dans `src/environment/` (héritage RL), mêle 7 responsabilités. L'extraction physique vers `smart_money/` est **déjà tracée et délibérément différée** (cf. `smart_money/__init__.py` lignes 14-36, raison : stabilité + ~20 entrypoints + détecteurs susceptibles de changer post-validation). **La façade rend l'extraction non-bloquante.**
- **Reco** : **NE PAS** entreprendre l'extraction avant la bêta (piège « réinventer »). Garder la façade. Documenter en archive V1.2+. Quand l'extraction arrive, en profiter pour sortir TA/VIF/benchmark/RL du fichier.

---

## 1.4 — Synthèse Dimension 1

**État architectural : 🟢 Acceptable→Solide.** Le découplage produit↔moteur (façade + pipeline injectable) est propre et testable ; les cœurs de détection sont des fonctions pures bien couvertes. La dette est **interne et contenue** : un monolithe legacy (extraction déjà planifiée et non-bloquante grâce à la façade), une duplication numba/python à surveiller, deux jeux de defaults divergents, et des vestiges RL.

**Aucune réécriture nécessaire.** Les 4 actions Petit (D1-3, D1-4, D1-5, + test de parité D1-2) totalisent < 1,5 j et réduisent le risque de régression silencieuse en V1.1 sans toucher à la logique de détection.

**Cas STOP rencontrés en D1** : aucun. Architecture comprise sans hypothèse risquée ; pas de bug majeur ; pas d'algo externe non-attribué (numba/ta sont des libs publiques correctement déclarées).

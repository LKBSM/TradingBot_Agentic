# Groupe A — Implémentation (hygiène / défensif / observabilité)

> **Suite de l'audit R&D du moteur de détection** — implémentation du Groupe A 🟢 (zéro risque détection)
> Date : 2026-06-08 · Branche : `fix/audit-rd-groupe-a-hygiene-defensif` (base `institutional-overhaul` @ `c0699e0`)
> Discipline : 1 commit par finding · tests inclus · staging explicite · **aucun touch au mapper / frontend / tests existants** · **aucune heuristique de détection modifiée**.

---

## 1. Tableau récapitulatif (avant / après)

| ID | Finding | Avant | Après | Fichier(s) |
|---|---|---|---|---|
| **D1-5** | `logging.basicConfig` à l'import | Reconfigurait le **root logger** du process (effet de bord global, pouvait écraser `LOG_FORMAT=json`) | Supprimé ; `getLogger(__name__)` seul — la config logging appartient à l'entrypoint | `strategy_features.py` |
| **D1-4** | `__main__` RL legacy | Exécuter le module en script **lançait un entraînement RL** (`AgentTrainer`) | Bloc supprimé ; seul `--benchmark` subsiste comme entrée script | `strategy_features.py` |
| **D1-3** | Defaults train/serve divergents | preprocess_* codaient RSI/ATR=7, FVG=0.0 ≠ produit (14/0.1) = **skew** | Source unique `DEFAULT_SMC_CONFIG = SMCConfig().model_dump()` partagée | `strategy_features.py` |
| **D3-3** | Docstrings perf trompeurs | « 0.2-0.5 s / 20k bars, 50-100× » sans dire que ça suppose numba **installé** et concerne 20k bars (pas le lookback 200) | Docstrings + requirements honnêtes (numba optionnel, fallback, régime produit ~200ms) | `strategy_features.py`, `requirements.txt` |
| **D2-9** | Divergence RSI = code mort produit | `_detect_rsi_divergence()` (O(n·k)) calculée à **chaque** reading, **non consommée** par le mapper | `analyze(compute_divergence=True)` ; l'assembler opte out (`False`). Flux legacy ConfluenceDetector inchangé | `strategy_features.py`, `market_reading_assembler.py` |
| **D4-2** | Pas de sanity OHLC | Série corrompue → MarketReading **dégradé silencieux** | `_validate_ohlc_integrity()` non-fatal (NaN, high<low, bornes corps, prix≤0, index non-monotone, doublons) + `get_data_quality_report()` | `strategy_features.py` |
| **D4-3** | Pas de détection d'outlier | Flash crash / fat finger créait fractal/BOS sans signal | Colonne **observationnelle** `OUTLIER_FLAG` (range > `OUTLIER_ATR_MULT`×ATR, défaut 5.0) — **données jamais altérées** | `strategy_features.py` |
| **D4-6** | Pas de monitoring runtime | Bug « 100 %-firing » attrapé **offline** seulement | `analyze()` calcule firing-rate / fractals / outliers / NaN-rate + **WARNING** si firing ∉ [0.5 %, 10 %] ; `get_monitoring_report()` | `strategy_features.py` |

---

## 2. Tests ajoutés

Tous dans **`tests/test_groupe_a_hygiene_defensif.py`** (nouveau fichier, aucun test existant modifié).

| Finding | Classe de test | Nb tests |
|---|---|---|
| D1-5 | `TestD1_5_NoBasicConfigAtImport` | 3 |
| D1-4 | `TestD1_4_NoLegacyRLMain` | 3 |
| D1-3 | `TestD1_3_UnifiedDefaults` | 4 |
| D3-3 | `TestD3_3_NumbaHonestDocs` | 3 |
| D2-9 | `TestD2_9_DivergenceOptOut` | 4 |
| D4-2 | `TestD4_2_OHLCSanityChecks` | 6 |
| D4-3 | `TestD4_3_OutlierFlag` | 5 |
| D4-6 | `TestD4_6_Monitoring` | 5 |
| **Total** | | **33** |

`python -m pytest tests/test_groupe_a_hygiene_defensif.py` → **33 passed**.

---

## 3. Confirmation zéro régression

- **Test de stabilité détection (D2-9)** : `BOS_SIGNAL/BOS_EVENT/CHOCH_SIGNAL/BOS_BREAK_LEVEL/FVG_SIGNAL/OB_STRENGTH_NORM/BOS_RETEST_ARMED` **strictement identiques** avec ou sans divergence → aucune heuristique modifiée.
- **Consommateurs de `CHOCH_DIVERGENCE`** (ConfluenceDetector, state-machine replay, pipeline) : **75 tests verts** (défaut `compute_divergence=True` préservé).
- **Surface d'impact consolidée** (moteur + assembler + tous consommateurs : confluence, replay, pipeline, mappers, schema, régime, MTF) : **275 passed, 2 skipped** (skips = CSV absent). **0 échec.**
- **Suite complète** : **3037 passed, 16 failed, 5 skipped**. Les 16 échecs sont **hors domaine détection** (`test_webapp_preview` ×12, `test_smoke_e2e` ×2, `test_template_narrative_engine` ×1, `test_webhook_drain_worker` ×1) et **PRÉ-EXISTANTS** : vérifié en exécutant ces fichiers sur la base `c0699e0` **et** sur ma branche en isolation → **15 failed / 51 passed strictement identiques** dans les deux cas. Le 16ᵉ échec en suite complète est une **pollution d'ordre pré-existante** (test sensible à l'ordre d'exécution), pas lié au Groupe A. → **Zéro régression imputable à ces changements.**
- **Périmètre touché** : 4 fichiers — `strategy_features.py` (moteur), `market_reading_assembler.py` (assembler, **pas** le mapper), `requirements.txt`, `tests/test_groupe_a_hygiene_defensif.py` (nouveau). **0 fichier mapper, 0 frontend, 0 test existant.**

---

## 4. Hashs des 8 commits

| # | Hash | Finding |
|---|---|---|
| 1 | `ba0a52a` | D1-5 — retirer logging.basicConfig |
| 2 | `e6394fb` | D1-4 — supprimer __main__ RL legacy |
| 3 | `01ad05c` | D1-3 — unifier defaults train/serve |
| 4 | `c3b631e` | D3-3 — docstrings numba honnêtes |
| 5 | `fd5765c` | D2-9 — divergence RSI opt-out produit |
| 6 | `3db8770` | D4-2 — sanity checks OHLC |
| 7 | `91dbf2b` | D4-3 — flag bougies aberrantes |
| 8 | `d7b812f` | D4-6 — monitoring firing-rate/NaN-rate |

---

## 5. Création de la PR (action founder — je ne la crée PAS)

Branche poussée sur `origin`. Pour ouvrir la PR vers `institutional-overhaul` :

```
https://github.com/LKBSM/TradingBot_Agentic/pull/new/fix/audit-rd-groupe-a-hygiene-defensif
```

> ⚠️ **Note D1-3 (à mentionner dans la PR)** : l'unification des defaults aligne le chemin **offline/training** (preprocess_*) sur les defaults produit (14/0.1). Le chemin **produit** (assembler → `config={}`) était déjà sur 14/0.1 → **inchangé**. Aucun appelant externe de `preprocess_dataframe` détecté hors `strategy_features.py`.

---

## 6. Ce qui reste (non fait — volontairement)

- **Groupe B** (paramètres : seuils FVG/retest, fractal TF-aware, buffer BOS, lookback TF-aware) → **après l'annotation manuelle** (calibration).
- **Groupe C** (passes additives : gap temporel D4-1, mitigation FVG/OB D2-4) → Moyen, planifiable ensuite.
- **D2-1** (filtre displacement OB) → gated annotation, impact Fort.
- Points mapper (`D5-1` bucket importance, `D4-7` hystérésis régime) → autre terminal.

*Fin du rapport. 0 PR créée, 0 merge, 0 force push.*

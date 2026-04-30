# Rapport — Corrections critiques consolidées

> **Date** : 2026-04-29 · **Branch** : `main` · **Sources** : `reports/eval_00_synthesis.md` + `reports/eval_00_synthesis_delta_2026_04_29.md` + 29 rapports `eval_01..29` + 8 memory entries projet.
>
> **Mission** : lister TOUS les fix critiques croisés à effectuer, classés par priorité (P0/P1/P2) et par impact mesurable. Pas de discussion — un plan d'action chirurgical.

---

## 🚨 P0 — Bloqueurs absolus go-live (à faire AVANT toute communication publique)

### P0.1 — Re-télécharger XAU 2019-2025 propre ✅ **LIVRÉ 2026-04-29**
- **Pourquoi** : feed actuel `data/XAU_15MIN_2019_2025.csv` à **63 % coverage** (memory `data_quality_audit_2026_04_23`). Falsifiait tout backtest et toute note basée sur le replay 2025.
- **Évidence** : `reports/eval_08_data_providers.md` §0 (note 3.5/10, NO-GO).
- **Bloquait** : eval_18 backtest, eval_07 sweep, baseline_2019_2025, toute promesse PF publique.
- **Action effectuée** :
  - `python scripts/download_dukascopy_xau.py --start 2024-12-30 --end 2026-04-29 --timeframe 15min --out data/XAU_15MIN_2025_2026_dukascopy.csv` → 31 442 bars.
  - `python scripts/merge_xau_2019_2026.py` → fichier unifié `data/XAU_15MIN_2019_2026.csv` (172 874 bars, **98.4 % coverage 23×5**, 2019-01-02 → 2026-04-29).
- **À utiliser désormais** : pointer tous les bench/backtest/replay vers `data/XAU_15MIN_2019_2026.csv` (ou `data/XAU_15MIN_2019_2024.csv` pour les bench déjà calibrés sur cette fenêtre).

### P0.2 — Replace ConfluenceDetector scoring fn
- **Pourquoi** : score 0-100 actuel a Pearson **−0.023** vs PnL, Brier worse than baseline (memory `confluence_calibration`). Le scoring **EST le produit** mais n'a aucun pouvoir prédictif.
- **Évidence empirique 2026-04-29** : sweep state machine `enter ≥ 65` produit **0 trades** sur 6 ans (`eval_07_followup_2026_04_29.md`). Le score plafonne sous 60.
- **Effort** : 1-2 semaines (replacement + validation walk-forward).
- **Bloque** : eval_07 tuning, tier PREMIUM/STANDARD atteignable, valeur perçue produit.
- **Action concrète** : garder le framework `ConfluenceDetector`, remplacer la fn `_compute_score()` par un modèle calibré sur un target binaire (gain à H+5 vs SL) avec features=composants actuels.

### P0.3 — Procfile / railway.toml lance `parallel_training.py` au lieu de Sentinel
- **Pourquoi** : la prod actuelle ne lance **PAS** le pipeline Smart Sentinel — elle continue à exécuter du code RL legacy (memory eval_22).
- **Évidence** : `reports/eval_22_deployment.md` (note 4.5/10).
- **Effort** : 15 min — modifier `Procfile` à `web: python -m src.intelligence.main`.
- **Action** : aligner avec `Dockerfile` qui a déjà la bonne entrée `src.intelligence.main`.

### P0.4 — `SENTINEL_TESTING_MODE=1` par défaut bypass auth
- **Pourquoi** : un déploiement prod sans `SENTINEL_TESTING_MODE=0` explicite donne accès **INSTITUTIONAL gratuit à tout le monde** (memory `eval_10_15_findings`).
- **État** : trip-wire `assert_safe_production_config()` existe dans le WIP `main.py` (refuse `ENVIRONMENT=production` + `TESTING_MODE=1`) — **pas encore commité**.
- **Effort** : 30 min — finaliser le WIP main.py et committer.
- **Action concrète** : commit du `assert_safe_production_config()` + ajouter test E2E qui vérifie le sys.exit(2) en prod.

---

## 🟧 P1 — Critique pour qualité produit & marges

### P1.1 — Bump `SemanticCache.SCORE_BUCKET_PTS = 5 → 10`
- **Pourquoi** : hit rate empirique mesuré 2026-04-29 = **7.8 %** (vs 30-45 % estimé). Bucket=10 donne **33.8 %** (×4.3).
- **Effort** : **1 ligne** dans `src/intelligence/semantic_cache.py:104`.
- **Impact économique** : à 1k MAU avec NARRATIVE_MODE=llm, **$9 480/an** d'économie LLM préservée.
- **Trade-off accepté** : BOS=12 collisionne avec BOS=15. Acceptable car le tier gate est en amont.
- **Référence** : `reports/eval_06_empirical_findings_2026_04_29.md`.

### P1.2 — Brancher `cleanup_expired()` au scanner thread
- **Pourquoi** : `SemanticCache.cleanup_expired()` n'est appelé **jamais** automatiquement (eval_06). Lazy delete au `get` ne couvre pas les `put` sur clés différentes → DB grossit sans bornes.
- **Effort** : 15 min — ajouter dans `SentinelScanner._run_loop` un `if time.time() - self._last_cleanup > 3600:` toutes les heures.

### P1.3 — Walk-forward + IC bootstrap pour le backtest
- **Pourquoi** : eval_18 noté **2/10** ❌. Aucun walk-forward, coûts transaction = $0 (irréaliste), look-ahead dans MTF, pas d'IC bootstrap. Aucun PF ne peut être communiqué publiquement sans cela.
- **État** : Prompt 04 a livré le pattern (`scripts/eval_04_volatility.py` = walk-forward 6-splits + DM tests). À répliquer pour PnL/PF.
- **Effort** : 1 semaine.
- **Action concrète** : créer `scripts/walkforward_backtest.py` qui combine SignalReplay + walk-forward expanding-window + bootstrap IC sur PF/Sharpe.

### P1.4 — Cost model transaction (slippage + spread)
- **Pourquoi** : tout PF replay actuel suppose $0 de coût d'exécution (eval_18). XAU spread ≈ 0.20-0.50 $ + slippage 0.10-0.30 $ = ~0.4 R par trade en swing intraday — peut transformer PF 1.60 en PF 1.10.
- **Effort** : 1-2 j.
- **Action** : ajouter `TransactionCostModel` dans `state_machine_replay.py` avec slippage = α × ATR + spread fixe par instrument.

### P1.5 — Multi-asset validation cross-asset
- **Pourquoi** : 5/6 presets sans CSV → 0 validation hors XAU. EURUSD M15 2019-2025 désormais dispo (99.6 % coverage 24/5).
- **État** : 1/5 onboardé (EURUSD).
- **Effort** : 1 j (replay PF EUR + RegimeFilter + ConfluenceDetector).
- **Action** : `python scripts/run_backtest.py --symbol EURUSD --csv data/EURUSD_15MIN_2019_2025.csv`.

### P1.6 — Geo-block US/QC/UK + disclaimer multi-langue
- **Pourquoi** : memory `eval_29_compliance_findings` — bloqueur Stripe pour onboarding paid. MiFID II finfluencer durcissement mars 2026. Reformuler "signaux" → "analyses".
- **Effort** : 1-2 j (middleware geoip + endpoints `/terms` `/privacy`).

### P1.7 — Documenter licence Dukascopy commerciale
- **Pourquoi** : Dukascopy est **personal use only** sauf accord (eval_08). Risque légal direct sur usage commercial.
- **Effort** : 1-2 h (vérifier ToS, ajouter encadré dans `BACKTEST_LEGAL_GUARDRAILS.md` ou `LICENSE_DATA.md`).

### P1.8 — Activer `NARRATIVE_MODE=llm` en prod
- **Pourquoi** : actuellement défaut `template` → marketing « AI-powered » mensonger (eval_05). Le LLM stack est **prêt** post-eval_05 implementation (cache effectif, fallback Template, eval CI).
- **Effort** : 30 min (env var + smoke test) + budget Claude Haiku ($1-3/mois personal).

---

## 🟨 P2 — Qualité ingénierie & dette technique

### P2.1 — `/metrics` payload vide en prod
- **Pourquoi** : `MetricsRegistry` existe mais n'est jamais instanciée par `build_system()` (eval_16). `/metrics` retourne 200 mais body vide → Prometheus scrape ineffective.
- **État** : `MetricsRegistry(prefix="sentinel")` existe dans le WIP `main.py:248` (non commité).
- **Effort** : 15 min — finaliser le WIP main.py.

### P2.2 — Trace ID E2E dans le scanner pipeline
- **Pourquoi** : eval_09 — un signal qui erreur quelque part dans `DataProvider → … → Telegram` n'est pas rattachable à un id unique. Debug post-mortem casse-tête.
- **Effort** : 1 h.
- **Action** : générer `trace_id = uuid.uuid4().hex[:8]` au début de `_scan_once`, le passer dans tous les `logger.info("[%s] …", trace_id)` et dans `Notifier.send_signal(metadata={"trace_id": trace_id})`.

### P2.3 — Backpressure / queue Notifier
- **Pourquoi** : si Telegram tombe 10 min, le circuit breaker ouvre, les signaux sont **perdus** (eval_09). Pas de TTL ni dedup en file.
- **Effort** : 1-2 j (queue SQLite ou Redis).
- **Bloque** : SLA contractuel.

### P2.4 — Outlier B7 sweep LGBM 7.4 h
- **Pourquoi** : sur Prompt 04 walk-forward, split 1 LGBM a fitté en **26 573 s (7.4 h)** vs ~100 s pour les 5 autres splits. Symptôme probable : contention mémoire ou pathologie early-stopping.
- **Effort** : 1 j (instrument lightgbm callbacks + log mem usage).
- **Référence** : `reports/eval_04_volatility.md` §4.5.

### P2.5 — Multi-worker safe SemanticCache stats
- **Pourquoi** : `_hits/_misses` en RAM par instance (eval_06). Avec `uvicorn --workers > 1`, `/health` retourne les stats **du worker qui répond**, pas l'agrégat.
- **Effort** : 1 j (compteurs Redis ou Prometheus Counter).

### P2.6 — Position sizing live + kill-switch op
- **Pourquoi** : memory `eval_19_risk` — pas de position-sizing live, pas de kill-switch (3 moteurs risk concurrents incohérents).
- **Effort** : 2-3 j.
- **Bloque** : tier INSTITUTIONAL ($1990 décoy).

### P2.7 — 109 `print()` dans 23 fichiers
- **Pourquoi** : eval_16 — mauvaise hygiène logging, bruit en prod, pas de niveau filtrable.
- **Effort** : 2 h — sed `print(...)` → `logger.info(...)` + grep CI gate.

### P2.8 — 0 GitHub Actions CI
- **Pourquoi** : eval_17 — 1366+ tests mais **aucun déclenchement automatique**. Régressions glissent en silence (cf. les multiples bugs détectés a posteriori).
- **Effort** : 30 min — `.github/workflows/test.yml` qui lance `pytest -q` sur push/PR.

### P2.9 — Skews train/serve sessions+currency+blend
- **Pourquoi** : memory `eval_23_mlops` — saignement 5-15 % RMSE invisible (sessions diurnal, currency mapping, blend weight) divergent entre training Colab et serve prod.
- **Effort** : 1 sem (audit + ajout schema validation).

### P2.10 — Compresser `data_json` SemanticCache
- **Pourquoi** : eval_06 — narrative 1.5-3 kB JSON × 7 200 entrées = 10-22 MB. À scale 100 symboles + multi-tier = 200 MB. Compression `zstd` divise par 5.
- **Effort** : 30 min, ROI faible mais zero-effort.

### P2.11 — `INSERT OR REPLACE` SemanticCache reset `hit_count`
- **Pourquoi** : eval_06 — `put()` sur clé existante remet `hit_count=0`. Fausse les stats, masque les hot keys.
- **Effort** : 15 min — `INSERT … ON CONFLICT(cache_key) DO UPDATE SET data_json=excluded.data_json, created_at=excluded.created_at`.

### P2.12 — `data_json` SemanticCache sans schema_version
- **Pourquoi** : eval_06 — si la structure narrative change, les caches antérieurs renvoient un objet incomplet. Pas de fail-fast.
- **Effort** : 30 min — wrapper `{"v": 1, "payload": …}` + invalidation au `v` mismatch.

---

## 📊 Résumé exécutable

| Niveau | # items | Effort cumulé | Impact bloquant |
|---|---|---|---|
| **P0** (bloqueur go-live) | 4 | ~3 sem | OUI — ne peut pas commercialiser sans |
| **P1** (qualité produit) | 8 | ~3 sem | Marges + tier paid + crédibilité |
| **P2** (dette tech) | 12 | ~3 sem | Hygiène + scaling >100 MAU |

**Sprint hotfix conseillé (1 semaine)** : P0.1 + P0.3 + P0.4 + P1.1 + P1.2 + P2.1 + P2.7 + P2.8 — tous des changements **< 1 jour chacun**, impact disproportionné.

**Sprint produit (3 semaines)** : P0.2 + P1.3 + P1.4 + P1.5 — débloque le verdict GO/NO-GO commercial.

**Sprint scale (3 semaines)** : P1.6 + P1.7 + P2.2 + P2.3 + P2.5 + P2.6 — préparation 1k MAU + paid tiers.

---

## 🎯 Top 3 actions à effectuer dans la même session (haut ROI)

1. **`SCORE_BUCKET_PTS=5→10`** dans `semantic_cache.py:104` — 1 ligne, ×4.3 hit rate.
2. **`Procfile` → `web: python -m src.intelligence.main`** — 1 ligne, débloque le déploiement Sentinel réel en prod.
3. **`download_dukascopy_xau.py` → re-fetch 2019-2025** — 30 min compute, débloque tout le replay 2025 actuellement biaisé à 63 % coverage.

Ces 3 quick-wins **changent la trajectoire produit** sans aucun refactor. À faire **avant** d'investir dans P0.2 (replacement scoring fn) qui demande 1-2 sem.

---

## Annexe — Mapping fix → eval source

| Fix | Source rapport | Note source |
|---|---|---|
| P0.1 | eval_08 + data_quality_audit_2026_04_23 | 3.5/10 |
| P0.2 | confluence_calibration + eval_07_followup | n/a memory |
| P0.3 | eval_22 | 4.5/10 |
| P0.4 | eval_10_15_findings | 5 modules bloquants |
| P1.1 | eval_06_empirical_findings_2026_04_29 | 5.0/10 |
| P1.2 | eval_06 | 5.0/10 |
| P1.3 | eval_18 | 2/10 |
| P1.4 | eval_18 | 2/10 |
| P1.5 | eval_08_followup + eval_20 | 5.0/10 |
| P1.6 | eval_29_compliance | 3.5/10 |
| P1.7 | eval_08 | 3.5/10 |
| P1.8 | eval_05_llm + eval_05_refresh | 7.5/10 |
| P2.1 | eval_16 | 3.2/10 |
| P2.2 | eval_09_followup + eval_16 | 6.8/10 |
| P2.3 | eval_09 | 6.5/10 |
| P2.4 | eval_04_volatility §4.5 | 6.0/10 |
| P2.5 | eval_06 | 5.0/10 |
| P2.6 | eval_19 | 4.5/10 |
| P2.7 | eval_16 | 3.2/10 |
| P2.8 | eval_17 | 5.5/10 |
| P2.9 | eval_23 | 4.5/10 |
| P2.10 | eval_06 | 5.0/10 |
| P2.11 | eval_06 | 5.0/10 |
| P2.12 | eval_06 | 5.0/10 |

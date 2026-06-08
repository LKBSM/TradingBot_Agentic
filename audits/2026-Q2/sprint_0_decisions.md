# Sprint 0 — Registre des décisions tranchées

**Date** : 2026-05-15
**Tranchées par** : Claude (Lead Quant Architect)
**Statut** : Actées. Pas de validation user requise — exécution sous responsabilité du spécialiste.

Ce document acte toutes les décisions techniques et organisationnelles prises pour Sprint 0, avec leur justification. Toute déviation future doit amender ce document.

---

## Décision A — Source XAU primaire

**Choix** : `data/XAU_15MIN_2019_2026.csv` comme source primaire **SI** l'audit coverage Batch 0.0 confirme ≥ 95 % sur 2019-2025. Fallback : `XAU_15MIN_2019_2024.csv` (97.6 % confirmé) + extension `XAU_15MIN_2025_2026_dukascopy.csv`.

**Justification** :
- Les deux CSV `2019_2024.csv` et `2019_2026.csv` partagent la même première bougie (2019-01-02 06:00:00, Open 1282.21) → même feed source, simplement étendu. Pas un nouveau feed à valider.
- 172 875 lignes sur 2019-2026 ≈ 24 700 lignes / an de 15-min sur ~5 sessions/sem × ~22h/jour → cohérent avec coverage ~97-98 % à confirmer.
- L'usage du fichier `2019_2025.csv` (63 %) dans `config.py` est la **cause racine du bug BOS 100 % bars** (eval_08). À remplacer immédiatement.
- Dukascopy en zone grise commerciale → utilisable pour **personal testing / R&D** maintenant, mais bloquant pour commercialisation (eval_08). Sprint 1 batch 1.4 doit acter la migration définitive (Polygon / Databento / autre licence claire).

**Conséquences immédiates** :
- `config.py` doit pointer vers la source choisie (batch 0.4).
- `XAU_15MIN_2019_2025.csv` (63 %) est gardé dans `data/` mais **interdit** comme source primaire — tag `legacy_low_coverage` dans la doc.
- La décision finale prise au batch 0.0 est consignée dans `audits/2026-Q2/data_layer_pre_flight.md`.

---

## Décision B — Risk engine canonique pendant Sprint 0-4

**Choix** : `src/environment/risk_manager.py` est l'**oracle de backtest** pendant Sprint 0 à Sprint 4. Les autres moteurs risk (mentionnés dans eval_19 — 3 moteurs concurrents) sont déclarés **legacy/figés** pendant cette période. La refonte unifiée `src/intelligence/risk/` est planifiée Sprint 5.

**Justification** :
- eval_19 (4.5/10) identifie 3 moteurs incohérents, mais ne désigne pas la canonique.
- `src/environment/risk_manager.py` est le plus testé (`test_sprint1_risk.py`, `test_correlation_risk.py`, etc.), inclut Kelly + SL/TP + marge + liquidation, 685 LOC structurés.
- Refondre maintenant = ouvrir un chantier qui contamine la baseline et empêche de comparer Sprint 0 → Sprint 4 sur métriques stables.
- Geler maintenant = baseline figée, refonte Sprint 5 documentée comme amélioration mesurable.

**Conséquences immédiates** :
- L'audit batch 0.3 section 3.7 / 3.8 doit **inventorier les 3 moteurs** (chemin de code, scope, divergences) et acter `risk_manager.py` comme canonique.
- Tout backtest Sprint 0-4 utilise ce module. Tout signal généré pendant cette période est tracé avec `risk_engine_version=env.risk_manager@v0.9.0-pre-institutional`.

---

## Décision C — Format des tear sheets

**Choix** : Markdown + JSON dès Sprint 0 baseline et tous sprints intermédiaires. PDF auto-généré depuis le MD via `pandoc` en Sprint 7 (commercial readiness).

**Justification** :
- Brief §7 demande PDF + JSON. Mais en Sprint 0-6, on itère vite : le format MD permet review en PR sans ouvrir un PDF, diff git lisible, conversion triviale.
- `pandoc` est standard, gère MD → PDF avec template personnalisable. Pas de duplication de contenu.
- JSON reste la source machine-readable (tests d'intégration, comparaisons inter-sprints).

**Conséquences immédiates** :
- `reports/baseline/baseline_report.md` + `.json` au batch 0.2.
- Sprint 7 batch 7.2 : ajout d'un template pandoc `docs/algo/tear_sheet_template.tex` et script `scripts/render_tear_sheets.py`.

---

## Décision D — Stack régime canonique

**Choix** :
- **Canoniques** côté `src/intelligence/` : `regime_filter.py` + `regime_gate.py` + `bocpd.py`.
- **Utilitaire bas-niveau** : `regime_classifier.py` (HMM 3-state) reste utilisé par `regime_gate.py`.
- **Legacy figés** : `src/agents/market_regime_agent.py` (887 LOC) et `src/agents/regime_predictor.py` (1 051 LOC) — issus de l'ère RL, périmètre clos sauf besoin documenté à l'audit.

**Justification** :
- `sentinel_scanner.py` (le hub d'orchestration prod) importe `regime_filter` — c'est la stack actuellement câblée en prod.
- `regime_gate.py` + `bocpd.py` sont les ajouts récents (3-Pillars Implementation, 2026-05-13) — orientation institutionnelle confirmée.
- Les 2 agents legacy contiennent du code utile (features HMM, predictor switching) mais pas dans le chemin actif. Les fragmenter dans une refonte = risque d'introduire des régressions sans valeur.

**Conséquences immédiates** :
- Audit batch 0.3 section 3.5 : documenter les 6 implémentations + acter la canonique.
- Tout commit Sprint 0-7 qui touche `agents/market_regime_agent.py` ou `agents/regime_predictor.py` doit être justifié dans le commit message ou refusé.

---

## Décision E — Extraction du module `smart_money/`

**Choix** : Création du package `src/intelligence/smart_money/` au **Sprint 1 batch 1.0** (avant le data hardening batch 1.1). Le package extrait la logique BOS/CHOCH/OB/FVG/retest de `src/environment/strategy_features.py` et la rend autonome, testable, contractualisée Pydantic v2.

**Justification** :
- Sans module unifié, impossible d'auditer ICT proprement (la section 3.2 du brief audit présuppose un detector identifiable).
- `strategy_features.py` (1 213 LOC) mélange indicateurs techniques classiques (ADX, RSI, MACD) et smart money — la séparation des responsabilités est nécessaire.
- L'extraction **n'invente pas** de nouveau code : elle déplace + ajoute une couche de contrats Pydantic + tests dédiés. Risque maîtrisé.

**Conséquences immédiates** :
- Sprint 0 batch 0.3 section 3.2 = audit du code actuel **en place dans `strategy_features.py`** (pas dans un module qui n'existe pas encore). Le code path à auditer est clarifié dans l'audit.
- Sprint 1 batch 1.0 = extraction, batch 2.1+ = annotations expertes (Sprint 2).

---

## Décision F — Stratégie de branching git

**Choix** : Création d'une branche **`institutional-overhaul`** depuis le tag `v0.9.0-pre-institutional`. Tous les commits Sprint 0-7 vivent sur cette branche. Merge vers `main` **uniquement à la certification finale Sprint 7**.

**Justification** :
- 7 sprints × ~2 semaines = 14 semaines de travail algo continu. Merger en continu sur `main` = exposer le user / production à des états intermédiaires instables.
- Une branche dédiée permet de :
  - Comparer `main` vs `institutional-overhaul` à tout moment (diff complet, métriques delta).
  - Rebaser ou squash-merger à la fin pour un historique propre.
  - Annuler proprement si la mission échoue (`git branch -D institutional-overhaul` sans toucher `main`).
- Le tag `v0.9.0-pre-institutional` est la **référence immuable**.

**Conséquences immédiates** :
- Après commit du WIP + tag, créer la branche et y travailler.
- `git push origin institutional-overhaul` à chaque fin de batch (sauf si user demande explicitement d'attendre).
- À chaque fin de sprint : faire un `git diff main..institutional-overhaul --stat` pour le bilan.

---

## Décision G — CI minimale Sprint 0

**Choix** : Ajout d'un workflow GitHub Actions minimal **dès Sprint 0 batch 0.1** : `.github/workflows/algo_tests.yml` qui lance `pytest tests/ -m "not slow and not integration"` sur chaque push à `institutional-overhaul`. Coverage détaillée (branch + ligne) et mutation testing arrivent en Sprint 6.

**Justification** :
- eval_17 dit 0 GitHub Actions. C'est une dette qui empêche toute promesse "≥ 90 % coverage" du brief §6.
- Workflow minimal = 10 lignes de YAML + checkout + setup-python + pytest. Charge négligeable (1 h).
- Sans CI, chaque PR est un guess. Avec CI, le user voit immédiatement si un push casse la baseline.

**Conséquences immédiates** :
- Batch 0.1 inclut la création du workflow.
- Si la suite actuelle a des tests cassés en CI (différences env local Windows / Linux GitHub), batch 0.1 les marque `xfail` ou `skip-on-ci` avec issue documentée, **pas de fix sauvage**.

---

## Décision H — Snapshot configuration baseline

**Choix** : Snapshot **exhaustif** de toutes les sources de configuration utilisées par le pipeline algo, dans `reports/baseline/config_snapshot_2026-05-15.json`. Inclut :
- Contenu intégral de `config.py` (parsé et sérialisé en JSON pour lisibilité).
- Variables d'environnement réellement consommées par le code algo (grep `os.environ`, `os.getenv` dans le périmètre).
- Hash SHA256 de tous les fichiers `.py` algo (intelligence + backtest + environment + agents algo-relevant).
- Hash SHA256 des CSV utilisés.
- Versions des libs critiques (`pip freeze` filtré : pandas, numpy, sklearn, lightgbm, pytest, etc.).
- Commit SHA `HEAD`.

**Justification** :
- Reproductibilité = pouvoir refaire **exactement** la même expérience dans 6 mois. Sans snapshot, impossible.
- Un seul fichier JSON facilite le diff inter-sprint (configuration drift) et l'audit.

**Conséquences immédiates** :
- Batch 0.2 produit ce snapshot avant tout backtest.
- Tout sprint ultérieur compare son snapshot au précédent dans son rapport.

---

## Décision I — Audit ICT en Sprint 0 sans annotations expertes

**Choix** : Section 3.2 de l'audit institutionnel (`algo_audit_institutional.md`) couvre **uniquement** :
- Le code path BOS / CHOCH / OB / FVG / retest tel qu'implémenté dans `strategy_features.py` aujourd'hui.
- Les statistiques d'activation (firing rate par bar, distribution des paramètres déclenchés, taux par régime de vol).
- La sensibilité aux paramètres (sweep simple sur 3-5 valeurs par hyperparam).

**Reporté à Sprint 2 batch 2.1-2.2** :
- Création du dataset d'annotations manuelles (≥ 500 par actif).
- Calcul F1 / precision / recall vs annotations expertes.
- Tuning bayésien des hyperparams (batch 2.3).

**Justification** :
- Annoter manuellement 500 setups BOS + 500 CHOCH + 500 OB + 500 FVG sur 2 actifs = ~2 000 annotations expertes × ~30 sec/annotation ≈ 17 h **rien que l'annotation**, plus la définition des règles, plus la validation inter-annotateur. Pas raisonnable en Sprint 0.
- L'audit Sprint 0 doit **identifier les bugs structurels** (eval_03 dit OB ≠ ICT correct, FVG threshold trop laxe) et préparer le terrain pour Sprint 2.

**Conséquences immédiates** :
- Section 3.2 explicite ce gap.
- Sprint 2 plan détaillé prévoit 17-20 h pour les annotations (à acter quand on rédigera le plan Sprint 2).

---

## Décision J — Gestion du WIP non-committé du statut initial

**Choix** : Avant le tag, créer une branche **`wip/pre-institutional-2026-05-15`** depuis l'état actuel pour préserver le WIP (~30 `M` + ~50 `??`). Puis revenir à `main` clean, créer le tag `v0.9.0-pre-institutional`. La branche `wip/...` reste accessible pour récupération.

**Justification** :
- Les modifs `M` du statut initial peuvent contenir du travail en cours non documenté du user. **Pas le droit** de les jeter.
- Les `??` (replay_*.json, EVALUATION_PROMPTS.md, GO_NO_GO_PROMPT.md, etc.) sont des artefacts d'analyse à préserver dans la branche wip.
- Une branche wip = filet de sécurité non destructif.

**Conséquences immédiates** :
- Batch 0.1 commence par `git checkout -b wip/pre-institutional-2026-05-15 && git add -A && git commit -m "wip: snapshot pre-institutional"`.
- Puis `git checkout main && git stash --include-untracked` (par sécurité) puis `git tag v0.9.0-pre-institutional`.

---

## Décision K — Buffer horaire et discipline d'exécution

**Choix** : Buffer Sprint 0 passé de 4 h → **8 h**. Total Sprint 0 = **66 h** sur 2 semaines (~33 h/sem productives).

**Justification** :
- 4 h de buffer sur 62 h = 6.5 %. Industriel = 15-20 %. Trop optimiste.
- 8 h de buffer = 12 %. Plus réaliste pour absorber : debug Windows path issues, tests flaky, dépendances Python qui dérivent (pandas 3.0.0 est récent).

---

## Décision L — Format de reporting fin de batch / fin de sprint

**Choix** :
- **Fin de batch** : update `roadmap/sprints/sprint_0_progress.md` (créé en début de sprint, append par batch). Statut : ✅ ok / ⚠️ partiel / ❌ bloqué + 3-5 lignes de bilan + lien commit.
- **Fin de sprint** : `roadmap/sprints/sprint_0_retrospective.md`. Bilan complet, gates passées/échouées, deltas vs estimation, arbitrages.
- **Auto-mode interruption** : je m'arrête uniquement à la gate de sortie de sprint, jamais entre batches sauf si gate de batch échoue.

**Justification** :
- Le user voit la progression sans avoir à lire chaque diff.
- Auto-mode (qu'il a explicitement activé) impose de minimiser les interruptions tout en gardant des checkpoints.

---

## Décision M — Tests d'intégration sur le périmètre

**Choix** : La suite de tests existante (1 366+ tests) doit rester **verte** à chaque commit Sprint 0. Aucun test ne sera désactivé sans documentation explicite dans un fichier `tests/disabled_during_sprint_0.md`. Les tests connus flaky (`test_short_roundtrip_pnl`) sont marqués `@pytest.mark.flaky` avec re-run automatique (3 tentatives).

**Justification** :
- Verrou de qualité : on ne casse pas pour fixer.
- Un test flaky non géré sabote la CI.

---

## Décision N — Pre-flight check Python env (batch 0.0)

**Choix** : Insertion d'un nouveau **batch 0.0 — Pre-flight** (4 h) **avant** le batch 0.1 freeze. Il valide :
- Python 3 + pytest 9.0.2 + pandas 3.0.0 + numpy 1.26.4 + sklearn 1.8.0 (confirmé OK).
- `pytest --collect-only tests/` collecte sans erreur de chargement.
- Imports critiques OK (`from src.intelligence import sentinel_scanner` etc.).
- Audit coverage de `XAU_15MIN_2019_2026.csv` → décide la source XAU primaire (alimente décision A).
- Audit `.gitignore` (le repo ignore déjà `data/*.csv` → confirme que "backup CSV" = local hors-git).

**Justification** :
- Lancer batch 0.1 freeze sans avoir validé que l'env tourne = risque de tag un état cassé.
- Batch 0.0 est cheap (4 h) et de-risque fortement le reste du sprint.

---

## Synthèse des changements par rapport au plan v1

| # | Changement                                                     | Impact estimé |
| - | -------------------------------------------------------------- | ------------- |
| Ajout batch 0.0 | Pre-flight env + audit coverage XAU 2019-2026          | +4 h          |
| Reorder         | Audit coverage XAU avant baseline (évite re-run batch 0.2) | -1 h     |
| Buffer +4 h     | 4 → 8 h                                                  | +4 h          |
| CI minimale     | Workflow `.github/workflows/algo_tests.yml`              | +1 h          |
| Branche dédiée  | `institutional-overhaul`                                 | 0 h (cleanup) |
| **Total ΔSprint 0** |                                                      | **+8 h → 66 h** |

---

**Décisions signées** : 2026-05-15, Claude (Lead Quant Architect)
**Prochain document** : `roadmap/sprints/sprint_0.md` (plan détaillé révisé)

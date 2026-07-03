# Diagnostics de rejet OB — rapport de mission (2026-07-02)

**Branche** : `feat/ob-rejection-diagnostics` (worktree dédié, depuis main `793c9a0`)
**Objectif** : quand un client demande « pourquoi cette bougie n'est pas un OB ? », M.I.A Agent
lit la raison RÉELLE produite par le moteur et l'explique — il ne devine plus.

## Les deux exigences dures, et comment elles sont tenues

### 1. Source unique de vérité

La raison rapportée est un **sous-produit du chemin de décision existant** — aucune logique
parallèle :

- **Détection** (`src/environment/strategy_features.py`) : les 3 clauses de la règle OB sont
  factorisées en Series booléennes **nommées** (`ob_candidate_conditions`) et combinées par
  `combine_ob_conditions` — c'est désormais LE chemin qu'emprunte `_add_smc_order_blocks`
  pour accepter/rejeter. Le diagnostic lit **les mêmes Series** pour dire quelle clause a
  échoué. Preuve d'équivalence : test oracle (l'expression inline pré-refactor recopiée dans
  le test) + golden snapshot (voir §2).
- **Cycle de vie / affichage** (`src/intelligence/market_reading_mappers.py`) :
  `collect_zones(with_rejects=True)` fait émettre la raison par **la branche même qui jette
  la zone** (`invalidated_close_through` par le `continue` d'invalidation,
  `mitigated_dropped_by_policy` par le flag de politique, `capped_max_zones` par le tri/cap).
  `_ob_lifecycle` expose en plus l'index de la bougie invalidante (information, jamais une
  entrée de décision).
- **Preuve de couplage (exigence c)** : `test_threshold_change_moves_decision_and_reason_together`
  bascule `OB_REQUIRE_FVG=True` sur les mêmes bougies → les colonnes de sortie du moteur
  changent **ET** le diagnostic incrimine exactement la clause basculée, dans le même run.

### 2. Additif seulement, zéro régression

- **Golden snapshot AVANT instrumentation** (commit `f461a91`, généré sur le moteur intact) :
  686 OB niveau moteur + listes surfacées (cap 12) sur 6 fixtures déterministes committées
  (XAUUSD/EURUSD × M15/H1/H4, 500 bougies). `tests/test_ob_golden_nonregression.py` compare
  aux **deux niveaux** (colonnes moteur + zones surfacées, géométrie/statut/ordre inclus) —
  vert après instrumentation.
- `test_with_rejects_flag_never_changes_surfaced_zones` : listes surfacées octet-identiques
  flag on/off.
- Aucun champ ajouté au JSON persisté des lectures ; le diagnostic est à la demande, jamais
  stocké.

## Les critères réels (tels que codés — pas d'embellissement)

Zone OB = bougie i−1 ; décision évaluée sur la bougie i (confirmation) :

| Côté | Critères (AND) |
|---|---|
| OB haussier | bougie visée baissière · bougie suivante haussière · la suivante dépasse le plus haut |
| OB baissier | bougie visée haussière · bougie suivante baissière · la suivante enfonce le plus bas |
| (mode legacy `OB_REQUIRE_FVG=True`) | + FVG adjacent présent |

Puis cycle de vie : `invalidated` (clôture à travers la zone) → retiré ; `mitigated` (retest
qui tient) → conservé ; cap d'affichage `MAX_ZONES_PER_TYPE=12` → les zones au-delà du top-12
(actives d'abord, puis force, puis récence) ne sont pas affichées.

**Note d'honnêteté** : le moteur n'a PAS de critère « corps insuffisant » ni « structure
valide » (détecteur engulfing, P0-2 connu). Le diagnostic dit la règle telle qu'elle est.

## Architecture livrée

```
src/environment/strategy_features.py      ob_candidate_conditions / combine_ob_conditions
                                           (+ _add_smc_order_blocks refactoré dessus)
src/intelligence/market_reading_mappers.py _ob_lifecycle → invalidated_idx ;
                                           collect_zones(with_rejects=True) ; ob_zone_id()
src/intelligence/ob_diagnostics.py         NOUVEAU — resolve_bar (prix/ts → bougie réelle,
                                           en code, pas par le LLM) + diagnose_ob (verdict
                                           structuré + labels FR pré-filtrés)
src/intelligence/market_reading_assembler.py build_enriched_frame (extrait du pipeline
                                           lecture, partagé) + get_ob_diagnostic (lit le
                                           MÊME cache de bougies que les lectures)
src/intelligence/chatbot/chatbot.py        outil get_ob_diagnostic (schéma + dispatch +
                                           règles system prompt anti-fabrication)
```

Statuts du diagnostic : `is_order_block` · `not_candidate` (checks par critère avec valeurs
observées) · `was_rejected` (raison réelle + date d'invalidation ou rang/cap) ·
`awaiting_next_candle` (dernière bougie : l'évaluation se fait sur la suivante) ·
`unresolved` (hors fenêtre / prix jamais touché — repli honnête, option A) · `no_data`.
Le champ `confirmation_of_previous` signale quand la bougie visée est en fait la bougie de
confirmation d'un OB dont la zone est la précédente (confusion fréquente).

## Accès IA

- Outil `get_ob_diagnostic(instrument, timeframe, price|ts)` ; résolution de « la bougie à
  ~4114 » (bougie la plus récente dont l'amplitude contient le prix) ou « la bougie de 14h »
  (horodatage, toléré à ±1 pas de bougie) **en code déterministe**.
- System prompt : n'expliquer un rejet QUE via ce diagnostic, ne rapporter QUE les
  `label_fr`/`reject_label_fr` renvoyés, dire honnêtement quand il n'y a pas de détail,
  descriptif passé/présent, zéro prédiction. Les couches existantes (filtre adversarial,
  filtre de sortie, whitelist d'actions) restent inchangées et s'appliquent.
- Les labels FR sont vérifiés contre `FORBIDDEN_TOKENS` + `OutputFilter` + vocabulaire
  prédictif (test e).

## Tests (mission a–e)

| Exigence | Test |
|---|---|
| (a) raison = critère réellement échoué | `test_invalidated_reject_reason_matches_lifecycle_reality` (vérifie que la bougie rapportée a VRAIMENT clôturé à travers la zone), `test_capped_reject_reason_reports_rank_and_cap`, `test_not_candidate_reports_the_exact_failing_criterion` |
| (b) non-régression | `tests/test_ob_golden_nonregression.py` (7 tests, 6 combos, 2 niveaux) + oracle legacy + flag-identity |
| (c) source unique | `test_threshold_change_moves_decision_and_reason_together` + `test_combined_conditions_equal_legacy_inline_expression` |
| (d) IA ancrée, repli honnête | `test_chatbot_relays_engine_reject_reason`, `test_chatbot_honest_payload_when_no_diagnostic_exists`, `test_tool_schema_and_prompt_wire_the_diagnostic`, `test_unresolved_price_and_ts_are_honest`, `test_assembler_diagnostic_no_data_is_honest` |
| (e) zéro prédiction | `test_labels_pass_output_filter_and_forbidden_tokens` |

Résultats : voir section suivante (remplie au run final).

## Résultats de la suite

- `tests/test_ob_rejection_diagnostics.py` : **18/18 verts**
- `tests/test_ob_golden_nonregression.py` : **7/7 verts** (après instrumentation)
- Chatbot (5 fichiers) : **263/263 verts**
- Suite backend complète (2026-07-03, hors `test_long_short_trading.py` import
  cassé pré-existant) : **3351 passés, 17 échecs**, dont :
  - 1 causé par le déplacement du code (`TestD2_9_DivergenceOptOut` inspectait
    la source de `_default_smc_pipeline`) → test mis à jour vers
    `build_enriched_frame`, même garantie, **vert** ;
  - 1 flaky (`test_webhook_drain_worker`) → **vert** au re-run ;
  - **15 pré-existants prouvés** : rejoués à l'identique sur main intact
    (793c9a0, worktree jetable) → mêmes 15 échecs (env : `SENTINEL_TESTING_MODE`
    absent du shell, « Audit ledger not configured », casse « LONG » vs « Long »).
    Zéro lien avec cette mission.

## Limites assumées

- Le diagnostic reconstruit le frame enrichi à la demande depuis le cache de bougies
  (~200 ms par appel, même coût qu'une lecture) — pas de stockage de quasi-candidats, car le
  moteur n'a aucune notion de « quasi-candidat » (AND pur de 3 clauses) ; en inventer un
  aurait été précisément la logique parallèle interdite.
- Si un pipeline SMC custom est injecté dans l'assembleur (tests), le diagnostic utilise le
  pipeline par défaut — en production c'est le même.
- FVG : seul le rejet OB est diagnostiqué (périmètre mission) ; la mécanique `with_rejects`
  est extensible aux FVG sans toucher la détection.

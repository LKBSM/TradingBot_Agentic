# Diagnostic BOS / CHOCH — pourquoi ils apparaissent si rarement

**Date :** 2026-06-16
**Branche :** `diagnostic/bos-choch-audit`
**Nature :** LECTURE SEULE — aucun code modifié, aucun fix appliqué. Rapport seul.
**Mandant :** Founder (observation : « BOS/CHOCH apparaissent très rarement »).

---

## TL;DR — Verdict : **(b) sous-surfaçage**

Le moteur **détecte abondamment** BOS et CHOCH (11–25 BOS et 4–12 CHOCH par fenêtre de
500 bougies, sur chacun des 6 combos réels). Le front n'en affiche **0 ou 1** parce que la
couche mapper ne lit **que la dernière bougie** : contrairement à OB/FVG qui disposent d'un
collecteur (`collect_zones`) parcourant toute la fenêtre, **BOS et CHOCH n'ont aucun
collecteur** et sortent en objet unique (`Optional[BOSRecent]` / `Optional[CHOCHRecent]`)
reflétant l'état instantané du dernier bar.

Ce n'est ni de la sous-détection (a), ni une rareté réelle (c). La plomberie de surfaçage
jette ~97 % des événements détectés.

> **Reco :** fix de plomberie simple et **read-only** (ajouter un collecteur BOS/CHOCH
> analogue à `collect_zones`, sans toucher la détection). **Décision au founder** — voir §5.

---

## 1. Comptes détecté vs surfacé par combo

Mesuré sur le cache SQLite réel `data/candles.db`, fenêtre `DEFAULT_LOOKBACK = 500`
(≈475 bougies après warmup des swings), en reproduisant exactement la logique de
surfaçage de l'assembleur/mapper (`_default_smc_pipeline` → `confluence_signal_to_structure`).

| Combo        | Bougies | BOS détectés | CHOCH détectés | **BOS surfacés** | **CHOCH surfacés** | retest_state (dernier bar) |
|--------------|--------:|-------------:|---------------:|-----------------:|-------------------:|---------------------------:|
| XAUUSD / M15 |     500 |           11 |              5 |            **0** |              **0** |                          0 |
| XAUUSD / H1  |     500 |           15 |              6 |            **0** |              **0** |                          0 |
| XAUUSD / H4  |     500 |           13 |              7 |            **0** |              **0** |                          0 |
| EURUSD / M15 |     500 |           11 |              4 |            **0** |              **0** |                          0 |
| EURUSD / H1  |     500 |           13 |              6 |            **0** |              **0** |                          0 |
| EURUSD / H4  |     500 |           25 |             12 |            **1** |              **0** |                          2 |
| **TOTAL**    |         |       **88** |         **40** |            **1** |              **0** |                            |

- **Détecté :** 88 BOS + 40 CHOCH sur les 6 combos.
- **Surfacé :** 1 BOS, 0 CHOCH. Taux de surfaçage ≈ **0,8 %** (1/128).
- Le seul BOS surfacé (EURUSD/H4) ne l'est que parce que le **dernier** bar avait
  `BOS_RETEST_STATE = 2` (break encore vouché par la machine à retest). Pur hasard de timing.
- **CHOCH surfacé = 0 partout** : un CHOCH ne s'affiche que si `CHOCH_SIGNAL != 0` sur la
  bougie exactement courante — probabilité quasi nulle. C'est la cause directe du « presque
  jamais » observé par le founder.

*(Les 6 combos sont les seuls présents dans le cache live : {XAUUSD, EURUSD} × {M15, H1, H4}.)*

---

## 2. Le pipeline, étage par étage (détection → front)

| Étage | Fichier:ligne | Rôle | Filtre / perte BOS-CHOCH ? |
|-------|---------------|------|----------------------------|
| **1. Détection** | `src/environment/strategy_features.py:33-134` (`_calculate_bos_choch_numba`) | Calcule `BOS_EVENT` (±1 sur les bars de cassure), `CHOCH_SIGNAL` (±1 sur les bars de retournement), `BOS_SIGNAL` (état propagé), `BOS_RETEST_STATE`. | ❌ Aucun. Tous les events sont émis sur **toutes** les bougies de la fenêtre. |
| **2. Pipeline assembleur** | `src/intelligence/market_reading_assembler.py:146` | `last_row = enriched.iloc[-1].to_dict()` → ne garde que la **dernière** ligne pour les features scalaires (dont BOS/CHOCH). | ✅ **Perte massive** : seul l'état du dernier bar est transmis. |
| **2bis. Zones OB/FVG (contraste)** | `market_reading_assembler.py:164` → `market_reading_mappers.py:399-514` (`collect_zones`) | Parcourt **toutes** les bougies, applique le lifecycle, renvoie des **listes** `order_blocks: [...]`, `fair_value_gaps: [...]` (cap `max_per_type`). | OB/FVG **ont** un collecteur multi-zones → surfacent un historique. |
| **3. Mapper structure** | `market_reading_mappers.py:553-628` (`confluence_signal_to_structure`) | Construit `bos: Optional[BOSRecent]` et `choch: Optional[CHOCHRecent]` à partir du **seul** `smc_features` du dernier bar. BOS si `BOS_EVENT != 0` (frais) **ou** `BOS_RETEST_STATE != 0` (persisté). CHOCH si `CHOCH_SIGNAL != 0`. | ✅ Sort **un seul** objet (ou aucun). Aucune liste, aucun historique. |
| **4. API** | `src/api/routes/market_reading.py:26-62` | Passthrough du `MarketReading`. | ❌ Aucun filtre additionnel. |
| **5. Front** | `webapp/components/app/ReadingChart.tsx:267-295` ; `StructureSection.tsx:82-99` | Rend `structure.bos` / `structure.choch` (champs **singuliers**, `Optional`). | ❌ Affiche fidèlement ce que l'API envoie — c.-à-d. au plus 1 BOS + 1 CHOCH. |

**Goulot unique : étages 2–3.** Le `iloc[-1]` (l.146) + l'absence de collecteur BOS/CHOCH
(mapper l.553-628) réduisent un flux de dizaines d'événements à un snapshot ponctuel.

---

## 3. Cause racine : asymétrie OB/FVG vs BOS/CHOCH

Lors de l'audit §T1 (zones), OB et FVG ont reçu le traitement « multi-zones » :
`collect_zones` parcourt l'historique et publie une **liste** de zones encore pertinentes
avec leur lifecycle. **BOS et CHOCH n'ont jamais reçu ce traitement** : ils sont restés sur
le chemin « état du dernier bar uniquement », hérité du fix F6 (`market_reading_mappers.py:562-576`).

- Le fix **F6** était correct dans son intention : il a supprimé l'émission d'un BOS sur
  ~100 % des bougies (l'état propagé `BOS_SIGNAL` était surfacé en permanence → faux « BOS »
  stale). La solution retenue — ne surfacer qu'un break *frais* ou *encore en retest* — a
  rendu le BOS **honnête mais éphémère** : visible seulement ~20–25 bougies après la cassure,
  puis plus rien.
- Conséquence non voulue : **aucune mémoire** des cassures passées. Le schéma `MarketReadingStructure`
  porte `bos`/`choch` **singuliers** (`Optional`), pas de `bos_events: [...]`. Donc même les
  cassures parfaitement valides d'il y a 30, 80, 200 bougies sont invisibles.
- **CHOCH est le pire cas** : pas de machine de persistance du tout (pas d'équivalent
  `BOS_RETEST_STATE`). Il n'apparaît que si la **dernière** bougie est pile un bar de CHOCH →
  d'où le « 0 surfacé » constaté sur les 6 combos.

C'est donc une **dette de plomberie symétrique**, pas un bug de détection ni un seuil trop strict.

---

## 4. Les seuils sont-ils en cause ? (constat, sans modif)

Non. Pour mémoire (lecture seule, **aucun changement recommandé ici**) :

- **Détection** (`_calculate_bos_choch_numba`) : une cassure = `close` franchit le dernier
  high/low structurel ; CHOCH = inversion du sens du break. Les comptes (11–25 BOS / fenêtre)
  montrent que ces seuils **déclenchent abondamment** — ils ne sont pas le facteur limitant.
- **Machine à retest** (`strategy_features.py:505-517`) : `awaiting_timeout=20` +
  `armed_window=5` bougies. Ce timeout borne la **persistance** d'un BOS frais, ce qui est
  voulu pour le « retest in progress » — mais il n'a de sens que pour le *dernier* break.
  Il ne crée pas la rareté ; la rareté vient de l'absence de collecteur historique (§3).

Toucher ces seuils serait un faux remède : ça déformerait la détection sans résoudre le
sous-surfaçage. **Hors périmètre.**

---

## 5. Recommandation (décision au founder)

**Verdict (b) ⇒ fix de plomberie simple, read-only.** Conformément à la règle du lot, **aucun
fix n'est appliqué dans ce diagnostic**. Piste proposée pour décision :

1. **Ajouter un collecteur `collect_structure_events(enriched, idx)`** symétrique à
   `collect_zones`, qui parcourt la fenêtre et renvoie `bos_events: [...]` et `choch_events: [...]`
   (chacun avec direction, niveau cassé réel via `BOS_BREAK_LEVEL`, `broken_at` honnête, statut
   lifecycle). **Sans toucher la détection** — pure lecture des colonnes déjà produites.
2. **Étendre le schéma** `MarketReadingStructure` avec ces listes (champs optionnels,
   rétro-compatibles ; on conserve `bos`/`choch` singuliers pour le « plus récent »).
3. **Front** : itérer sur les listes pour tracer plusieurs lignes/marqueurs BOS/CHOCH, comme
   les boîtes OB/FVG actuelles, avec le même cap raisonnable (`max_per_type`).

Effort estimé : faible (réplique du pattern `collect_zones` déjà éprouvé). Gain : passage de
~1 % à un historique borné fidèle de la structure.

**Caveat lifecycle :** à l'instar des zones, prévoir une borne (cap + éventuel « mitigated/stale »)
pour ne pas réafficher des cassures très anciennes sans qualification. À cadrer avec le founder,
**probablement gated par annotation** avant tout déploiement client.

---

## Annexe — Méthode (reproductibilité)

- Source : `data/candles.db`, table `candles_cache`, 6 combos réels, 500 dernières bougies.
- Moteur : `SmartMoneyEngine(data=df).analyze()` (aucune modification).
- « Détecté » = `(BOS_EVENT != 0).sum()` / `(CHOCH_SIGNAL != 0).sum()` sur toute la fenêtre.
- « Surfacé » = application de la logique exacte de `confluence_signal_to_structure` au
  **seul** dernier bar (`iloc[-1]`), comme en production.
- Le script de mesure jetable (`_diag_bos_choch.py`) a été **supprimé** après exécution ;
  ces chiffres sont reproductibles en réinstanciant la logique ci-dessus.

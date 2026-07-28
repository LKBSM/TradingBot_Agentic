# AUDIT — LB-1 : Lookback profond par unité de temps + nouvelles unités

Date : 2026-07-28 · Branche : `feat/lb-1-lookback-profond` (depuis `main` @ bd40d62)

Principe structurant : séparer les **trois profondeurs** jusque-là confondues —
**STOCKAGE** (profond), **DÉTECTION** (bornée, incrémentale), **AFFICHAGE** (peu, par
défaut) — pour que l'une grandisse sans alourdir les autres.

---

## 1. Quota fournisseur (Twelve Data, plan GRATUIT) — mesuré

| Fait | Valeur |
|---|---|
| Limites plan gratuit | **8 req/min, 800 req/jour** (credits). `time_series` = 1 credit pour XAU/EUR. |
| Consommation AVANT la mission | 6 combos M15/H1/H4 ×2, ~**252 req/j** (pic jour ouvré) |
| M1/M5 dispo sur XAU/USD & EUR/USD ? | **Oui, mesuré** (sonde live) |

### Profondeur d'historique réellement servie (sonde `outputsize=5000`, 1 requête)

| TF | Cible durée | Reculé atteint (1 requête) |
|----|-------------|----------------------------|
| M1 | 1 j   | ~3,5 j |
| M5 | 1 sem | ~2 sem |
| M15| 1 mois| ~7 sem |
| H1 | 6 mois| ~7 mois (2025-12-30) |
| H4 | 2 ans | ~3 ans (2023-08) |
| D1 | 5 ans | **2007** (~19 ans) |

→ **Chaque profondeur cible tient dans UNE seule requête** (toutes ≤ 5000 bougies).

### Coût du REMPLISSAGE INITIAL — mesuré en live

Remplissage réel exécuté (`scripts/backfill_history.py`) : **10 combos, 10 requêtes,
une par combo**, chaque cible atteinte exactement. Étalement inutile (10 ≪ 800/j ;
~2 min à 8/min). Relance idempotente vérifiée : **0 requête** (tout sauté « complet »).

| Combo | Bougies | Historique depuis | Couverture calendaire | vs cible |
|-------|--------:|-------------------|-----------------------|----------|
| XAUUSD M5  | 2020 | 2026-07-21 | ~6 j   | 1 sem (≈, cf. §5) |
| XAUUSD M15 | 2926 | 2026-06-28 | ~1,0 mois | 1 mois ✅ |
| XAUUSD H1  | 4387 | 2026-01-26 | **6,0 mois** | 6 mois ✅ |
| XAUUSD H4  | 4387 | 2024-02-19 | ~29 mois | 2 ans ✅ (dépasse) |
| XAUUSD D1  | 1831 | 2020-01-29 | ~78 mois | 5 ans ✅ (dépasse) |
| EURUSD M5  | 2020 | 2026-07-21 | ~6 j   | 1 sem (≈) |
| EURUSD M15 | 2926 | 2026-06-28 | ~1,0 mois | 1 mois ✅ |
| EURUSD H1  | 4387 | 2026-01-26 | 6,0 mois | 6 mois ✅ |
| EURUSD H4  | 4387 | 2024-01-08 | ~31 mois | 2 ans ✅ (dépasse) |
| EURUSD D1  | 1831 | 2019-11-14 | ~80 mois | 5 ans ✅ (dépasse) |

### Coût du FONCTIONNEMENT COURANT — chiffré

Une requête par bougie clôturée par combo, pic jour ouvré, 2 symboles :

| TF | bougies/j/symbole | ×2 symboles |
|----|------------------:|------------:|
| M5 | 288 | 576 |
| M15| 96  | 192 |
| H1 | 24  | 48  |
| H4 | 6   | 12  |
| D1 | 1   | 2   |
| M1 | 1440| 2880 |

- **Livré (jeu live-warm par défaut = M15/H1/H4/D1 ×2) : ~254 req/j.** Sous 800. ✅
- M5 natif en plus : +576 → **~830/j, AU-DESSUS de 800**. → M5 est **exclu du warm
  live par défaut** (`LB1_WARM_M5=1` pour l'activer) jusqu'au câblage du chemin
  **resample M5→M15/H1** (poll M5 = 576/j, M15/H1 dérivés = 0 → ~590/j, sous 800).
- M1 live : **2880/j, incompatible plan gratuit** sous tout schéma → **off par défaut**,
  livré comme historique backfillé, pas de rafraîchissement à la minute.

**Verdict quota : la mission tient dans le plan gratuit** (issue (a)), M1 hors-live et
M5-live via resample (à câbler). Aucun plan payant requis.

---

## 2. Détection : coût & incrémental — mesuré

| Passage (H1, `build_enriched_frame`+collect) | Temps (best/3) |
|---|---|
| Recalcul complet sur STOCKAGE profond (4387 bougies) | **1033 ms** |
| Fenêtre incrémentale bornée (800 bougies) | **283 ms** |

→ Le rafraîchissement borné est ~3,6× plus rapide **et reste constant** quand le
stockage s'approfondit (fenêtre fixe). Le recalcul naïf, lui, croît avec la
profondeur (1 s à 4387, davantage à 10k+). **Coût de rafraîchissement découplé de la
profondeur de stockage** — l'objet de la mission.

Avant la mission : détection **recalculée en totalité à chaque passage**, zones et
événements **non persistés** (recalculés en mémoire, cycle de vie dérivé à la volée),
**aucun journal d'événements en base**. Tranché ici : **le journal n'existait pas**.

---

## 3. Ce qui est livré

**STOCKAGE**
- `config/lookback_depths.json` + `src/intelligence/lookback_config.py` : profondeur en
  **DURÉE** par (instrument, unité de temps), jamais en constante globale, ajustable sans
  code. Conversion durée→bougies en **densité calendaire continue** (plafond) → garantit
  d'atteindre **au moins** la durée demandée.
- `scripts/backfill_history.py` + `history_backfill.py` : remplissage par combo,
  idempotent, reprise (combo complet sauté), quota-aware (1 req/combo), MC-1 (fraîcheur
  sur `market_aware_expected_close` gelée week-end → 0 appel marché fermé).
- `CandlesCacheStore.get_coverage` : couverture réelle (count/oldest/newest).

**DÉTECTION** (incrémentale, persistée — **sans réécrire une règle**)
- `incremental_detection.py` : réutilise `build_enriched_frame` (run moteur partagé des
  lectures) + `collect_zones`/`collect_structure_events`. `refresh` = 1 passe/bougie sur
  fenêtre bornée ; `replay` = bougie par bougie.
- `structure_store.py` : zones (id stable, cycle de vie, `surfaced`, consommées
  réconciliées) + **journal BOS/CHOCH** (existe par construction). Zones ET événements
  réconciliés **dans la fenêtre** ; ce qui en sort est **gelé** = journal profond confirmé.
- **NON-RÉGRESSION** : incrémental == recalcul complet, **zones ET événements, sur les 6
  unités de temps** (test `test_incremental_detection.py`). Un vrai écart H4 a été
  diagnostiqué puis corrigé (événements append-only gardaient des ruptures transitoires
  révisées sur fenêtre partielle → réconciliation in-fenêtre). Journal réel peuplé :

  | Combo | zones | BOS | CHOCH |  | Combo | zones | BOS | CHOCH |
  |-------|------:|----:|------:|--|-------|------:|----:|------:|
  | XAU M5 | 39 | 1 | 0 |  | EUR M5 | 25 | 30 | 9 |
  | XAU M15| 39 | 14| 4 |  | EUR M15| 21 | 25 | 9 |
  | XAU H1 | 20 | 22| 9 |  | EUR H1 | 18 | 7  | 4 |
  | XAU H4 | 42 | 1 | 0 |  | EUR H4 | 34 | 28 | 13 |
  | XAU D1 | 75 | 26| 5 |  | EUR D1 | 25 | 27 | 13 |

**AFFICHAGE** (backend prêt)
- `/api/coverage` : profondeur réelle par combo + `complete` honnête (période partielle
  jamais dite complète). Corrige aussi la mention méthodologie devenue fausse (9 locales).
- `/api/structure` : ne renvoie que zones/événements **recoupant la fenêtre** (temps+prix)
  + totaux (« N sur M »). Charge réseau indépendante de la profondeur.
- Périmètre étendu à **M1..D1** partout où il était codé en dur (SCAN_COMBOS, _TF_MINUTES,
  signal_summary, seed, routes candles/market_reading, `perimeter.ts`/`viewActions.ts`,
  store/rail/sidebar/zones). M1 masqué par défaut (gate back+front). Rail = pastilles
  horizontales → 6 TF sans refonte.

**NOUVELLES UNITÉS** — M1/M5/M15/H1/H4/D1 sur XAU+EUR. M1 derrière `LB1_ENABLE_M1`
(off). D1 : frontières de journée/fuseau = config MC-1 par instrument (source unique).

---

## 4. Différé (à valider LIVE avant merge — le backend est prêt)

1. **Chemin resample M5→M15/H1 en live** : conçu, non câblé. M5 est backfillé (historique
   profond consultable) mais **hors warm live** (`LB1_WARM_M5=0`) tant que le resample
   n'est pas branché — sinon ~830/j > 800.
2. **Câblage scheduler → `IncrementalDetector.refresh`** à chaque tick : la structure
   persistée est aujourd'hui **seedée par le backfill** (1 passe/combo) ; l'accumulation
   live du journal attend ce branchement.
3. **Front : fetch borné à la fenêtre + compteur « N sur M »** dans le graphique
   (`/api/structure` prêt côté serveur).
4. **Filtre calme par défaut (actives+testées, mitigées masquées)** : le modèle de zone
   front fusionne aujourd'hui testée/mitigée en un seul booléen `tested` ; livrer ce
   défaut proprement demande d'exposer un état **mitigée** distinct depuis le backend.

---

## 5. Profondeurs cibles non atteintes + raison

- **M5 ≈ 6 j au lieu de 7 j** : les séries M5 comportent des coupures de week-end (densité
  < continue), donc 2020 bougies reculent de ~6 j calendaires. Écart mineur, **couverture
  réelle affichée** via `/api/coverage` (jamais présentée comme complète). Relever la
  profondeur M5 dans la config comblerait l'écart (toujours 1 requête).
- **H4 / D1 dépassent la cible** (≈29–31 mois / ≈78–80 mois) : dimensionnement en densité
  continue = plafond ; les séries réelles étant moins denses, on recule plus loin que la
  durée demandée. Inoffensif (plus d'historique), 1 requête, stockage négligeable.
- Volume total 12 combos ≈ **2–3 Mo** — non-sujet.

---

## 6. Tests

- `test_lookback_config.py` (28) · `test_history_backfill.py` · `test_structure_store.py`
  (6) · `test_structure_endpoint.py` (5) · `test_incremental_detection.py` (9,
  **non-régression 6 TF**) · `test_conditions_scan_endpoint.py` (périmètre robuste).
- Mesures rapportées : quota remplissage (10 req) & courant (254/j livré, 830/j si M5
  natif, 2880/j M1) ; profondeur réelle par combo ; perf 1033 ms → 283 ms.
- `tsc` + `build` : voir le run de clôture.

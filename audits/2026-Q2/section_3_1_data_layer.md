# Audit Phase 1 — Section 3.1 : Data Layer

**Date** : 2026-05-15
**Auditeur** : Claude (Lead Quant Architect)
**Périmètre** : `src/intelligence/data_providers.py`, `src/intelligence/data_quality.py`, `data/*.csv`, scripts de téléchargement, calendrier économique, resampling MTF.

---

## Score : **5.0 / 10** ⚠️ (vs eval_08 = 3.5)

Le score remonte vs eval_08 (3.5/10) grâce au switch primaire vers `XAU_15MIN_2019_2026.csv` (98.72 %) effectué en batch 0.4. Reste 5/10 (pas plus) tant que :
- 5/6 presets restent sans CSV (BTC, US500, GBPUSD, USDJPY, USOIL)
- Pas de pipeline live (downloader = ad-hoc CLI)
- Licence Dukascopy non clarifiée (le CSV adopté évite la dépendance mais le `XAU_15MIN_2025_2026_dukascopy.csv` reste dans `data/`)
- Pas de validation Pydantic v2 à l'ingestion
- Pas de framework `DataProvider` contractualisé

---

## 1. Coverage des CSV MVP (mesuré empiriquement batch 0.0)

| CSV                                  | Lignes  | Coverage agrégée (session active) | Fraîcheur | Décision |
| ------------------------------------ | ------- | --------------------------------- | --------- | -------- |
| `XAU_15MIN_2019_2026.csv`            | 172 874 | **98.72 %** sur 2019-2025         | 15 jours  | ✅ Primaire XAU |
| `XAU_15MIN_2019_2024.csv`            | 141 524 | 98.73 %                           | 500 jours | 🟡 Backup figé |
| `XAU_15MIN_2019_2025.csv`            | 106 643 | **63.71 %** ❌                    | 134 jours | 🔒 Interdit (legacy broken) |
| `XAU_15MIN_2025_2026_dukascopy.csv`  | 31 442  | 98.32 % sur 2025                  | 15 jours  | ⚠️ Licence floue |
| `EURUSD_15MIN_2019_2025.csv`         | 174 506 | **99.41 %**                       | 134 jours | ✅ Primaire EURUSD |

Source : `audits/2026-Q2/xau_coverage_audit.md`.

### 1.1 Gaps en session active

| CSV                       | Gaps > 30 min (session active) | Densité |
| ------------------------- | ------------------------------ | ------- |
| `XAU_2019_2026`           | 1 892                          | ~270/an (cohérent fermetures weekend + holidays) |
| `XAU_2019_2025` (broken)  | 6 093                          | ~870/an (×3 = signature d'un feed cassé) |
| `EURUSD_2019_2025`        | 1 821                          | ~260/an |

---

## 2. Code source `data_providers.py` (375 LOC)

### Architecture

Classes inventoriées dans la mémoire : `CSVDataProvider`, `MT5DataProvider`. Base abstraite probable.

### Findings code review

| # | Finding                                                                    | Sévérité | Action                            |
| - | -------------------------------------------------------------------------- | -------- | --------------------------------- |
| F1 | **Pas de contrat Pydantic v2** à l'ingestion. Pas de validation OHLCV (`high >= max(open, close)`, etc.). | P0 | Sprint 1 batch 1.1                |
| F2 | Pas d'API uniforme `fetch_window(start, end, instrument, tf)`. Tout passe par CSV path direct. | P1 | Sprint 1 batch 1.1                |
| F3 | Pas de cache mémoire LRU. Chaque ouverture CSV = re-parse complet. | P2       | Sprint 1 ou Sprint 6              |
| F4 | Pas de garantie d'alignement timestamps (UTC vs broker tz). | P0 | Sprint 1 batch 1.1 |

---

## 3. Code `data_quality.py` (191 LOC)

Documenté dans mémoire comme couvrant OHLCV validation, gap detection, outlier flagging. Audit code à compléter Sprint 1.

Findings au vu du périmètre :

| # | Finding                                                          | Sévérité |
| - | ---------------------------------------------------------------- | -------- |
| F5 | Pas de calcul automatique du **coverage** par fenêtre. (Mon script `audit_xau_coverage.py` fait ce calcul ad-hoc — à intégrer.) | P1 |
| F6 | Pas de détection statistique des **spikes de spread** (`high - low` anormal vs ATR). | P1 |
| F7 | Pas de **dédup** sur timestamp dupliqué. | P1 |
| F8 | Logging des outliers ≠ blocage à l'ingestion. | P2 |

---

## 4. Resampling multi-timeframe

### Code path actuel

`src/intelligence/multi_timeframe_features.py` (667 LOC) + `src/environment/multi_timeframe_features.py` (667 LOC) — **2 implémentations parallèles** détectées.

| # | Finding                                                                              | Sévérité |
| - | ------------------------------------------------------------------------------------ | -------- |
| F9 | **Look-ahead risk** : pas de preuve formelle (test propriété-based) que la barre H1 active ne contient pas d'information M15 future. | P0 |
| F10 | **Duplication MTF** intelligence/ vs environment/. Réconciliation Sprint 1.       | P1       |
| F11 | Tests de resampling existent (`test_multi_timeframe.py`) mais pas de fuzzing pour cas pathologiques (weekends, half-day, DST). | P1 |

---

## 5. Licence Dukascopy

### Statut commercial (rappel eval_08)

Zone grise : usage personnel/R&D OK, usage commercial ambigu. Le ToS Dukascopy interdit la redistribution mais l'utilisation interne en R&D n'est pas clairement définie.

### Impact différé batch 0.0

Le switch vers `XAU_15MIN_2019_2026.csv` (qui **n'est pas** un feed Dukascopy d'après l'inspection des premières bars : 2019-01-02 06:00:00 Open=1282.21, identique au `2019_2024.csv` qui pré-date Dukascopy) **différe** la question de la licence — Sprint 0-6 OK sans dépendance Dukascopy.

| # | Finding                                                                      | Sévérité | Décision |
| - | ---------------------------------------------------------------------------- | -------- | -------- |
| F12 | Licence du feed `XAU_15MIN_2019_2026.csv` non documentée — d'où vient-il ?  | P0       | Sprint 1 doc-only |
| F13 | `XAU_15MIN_2025_2026_dukascopy.csv` à 31 k lignes reste dans `data/` mais non utilisé. À documenter ou supprimer Sprint 7 (commercial readiness). | P2 | Sprint 7 |

---

## 6. Sources MVP manquantes (eval_20)

| Préset    | CSV présent ?  | Action Sprint 1 |
| --------- | -------------- | --------------- |
| XAUUSD    | ✅ 2019_2026   | OK              |
| EURUSD    | ✅ 2019_2025   | étendre 2026 (Sprint 1 batch 1.5) |
| BTCUSD    | ❌             | Sprint 1 batch 1.5 — choix de source (Polygon ? Binance ?) |
| US500     | ❌             | Sprint 1 batch 1.5                          |
| GBPUSD    | ❌             | Sprint 1 batch 1.5                          |
| USDJPY    | ❌             | Sprint 1 batch 1.5                          |

eval_20 recommande **drop BTC + US500**, **add USOIL P1**. Décision finale Sprint 1.

---

## 7. Calendrier économique

`data/economic_calendar_2019_2025.csv` (1 380 events, 71 KB) + `economic_calendar_HIGH_IMPACT_2019_2025.csv` (876 events, 44 KB). Source : ForexFactory JSON converti CSV (cf. `scripts/fetch_forexfactory_live.py`).

| # | Finding                                                                | Sévérité |
| - | ---------------------------------------------------------------------- | -------- |
| F14 | Fraîcheur 2025-12-31 — pas d'événements 2026 (au 2026-05-15).         | P1       |
| F15 | Pas de cross-check vs MT5 ou Bloomberg pour valider les timestamps. (Script `crosscheck_mt5_calendar.py` existe — fréquence d'exécution inconnue.) | P2 |

---

## 8. Macro FRED (`data/macro/`)

6 CSV : `cot_gold.csv`, `fred_DGS10.csv`, `DFII10.csv`, `DTWEXBGS.csv`, `VIXCLS.csv`, `T10Y2Y.csv`, `BREAKEVEN_10Y.csv`. Téléchargé via `scripts/download_real_economic_data.py` (635 LOC).

Audit non bloquant pour Sprint 0. À reprendre Sprint 3 (statistical edge discovery — utilisation des macro spreads comme features).

---

## 9. Recommandations Sprint 1 batch 1.1-1.5

| Batch | Action                                                              | Priorité |
| ----- | ------------------------------------------------------------------- | -------- |
| 1.1   | Contrat Pydantic v2 `OHLCVBar` + validation à l'ingestion (F1, F2) | P0       |
| 1.1   | API uniforme `DataProvider.fetch_window()`                          | P0       |
| 1.2   | Tests property-based resampling MTF (no look-ahead) (F9, F11)       | P0       |
| 1.2   | Réconciliation MTF intelligence/ vs environment/ (F10)              | P1       |
| 1.3   | Pipeline calendrier économique end-to-end + blackouts               | P1       |
| 1.4   | Décision sources licenciées (Polygon vs Databento vs autre)         | P0       |
| 1.5   | Étendre les CSV propres aux 4 actifs MVP supplémentaires             | P0       |
| —     | Pré-Sprint 7 : documenter licence `XAU_2019_2026.csv` (F12)         | P0 commercial |

---

## 10. Ce que cet audit ne couvre pas

- L'**audit empirique exhaustif des outliers** (spikes spread, dupes, NaN propagation). À faire Sprint 1 batch 1.1.
- L'**alignement timestamp UTC vs broker** (DST, holidays). À faire Sprint 1 batch 1.2.
- **Comparaison feed vs broker MT5 live** pour valider que les prix CSV reflètent la réalité tradable. À faire Sprint 2.

---

## Synthèse

Le data layer **n'est plus bloquant** pour Sprint 0 baseline (98.72 % coverage XAU + 99.41 % EURUSD, garde-fou BOS régression vert). Mais il reste **bloquant pour commercialisation** (5/6 presets manquants, pas de pipeline live, pas de contrat Pydantic, look-ahead MTF non prouvé absent).

Sprint 1 batches 1.1-1.5 doivent traiter les findings P0 avant tout passage à Sprint 2-3.

---

**Signé** : 2026-05-15, Claude

# Eval 08 — Data Providers & Data Quality

> **Périmètre audité** : `src/intelligence/data_providers.py` (375 l), `src/intelligence/data_quality.py` (191 l), `scripts/download_dukascopy_xau.py`, `scripts/audit_data_quality.py`, `scripts/crosscheck_mt5_calendar.py`.
>
> **Date** : 2026-04-28 · **Branch** : `main` · **Mission** : auditer l'ingestion data — root cause du verdict "non commercialisable" (PF replay 0.96 sur feed à 63 % couverture).

---

## 0. Résumé exécutif — Note **3.5 / 10** · **Verdict GO/NO-GO : ❌ NO-GO commercial**

| Axe | Note | Verdict |
|---|---|---|
| Couverture data XAU | 6/10 | feed 2019-2024 = 97.6 % ✅, feed 2019-2025 = 63 % ❌ (utilisé sans warning par défaut) |
| Couverture data multi-asset | **0/10** | 5 des 6 presets (EUR/BTC/US500/GBP/JPY) **n'ont aucun CSV local** |
| Cohérence OHLC | 7/10 | `validate_ohlcv` couvre H≥max(O,C), L≤min(O,C), monotonie, doublons |
| Résistance données corrompues | 6/10 | `DataQualityError` levé sur structural corruption ; soft warnings sur gaps |
| Pipeline ingestion live | **2/10** | Aucun pipeline temps réel. CSV file-based uniquement. MT5 polling sync sans queue. |
| Licence commerciale | **3/10** | Dukascopy = **personal use only** sauf accord commercial. MT5 broker-dependent. Risque légal réel. |
| Coût mensuel data | 8/10 | Actuel $0 (Dukascopy gratuit) ; passage commercial = $80-500/mo |

**Verdict commercial** : **bloqueur n°1 du go-to-market**. Avant toute promesse marketing :
1. Statuer sur licence Dukascopy → souscription commerciale (~$0-500/mo selon volume)
2. Re-télécharger XAU 2019-2025 propre (le feed actuel à 63 % falsifie tout backtest)
3. Onboarder data EURUSD M15 minimum (validation multi-asset)
4. Brancher pipeline live (MT5 ou Polygon.io WebSocket)

Sans ces 4 actions, **aucun chiffre PF/Sharpe ne peut être communiqué publiquement**.

---

## 1. Cartographie code

### 1.1 `src/intelligence/data_providers.py` (375 l)

| Classe | Rôle | LOC |
|--------|------|----:|
| `DataProvider` (ABC) | Interface `get_ohlcv(symbol, timeframe, lookback) → DataFrame` | 22-48 |
| `CSVDataProvider` | Local CSV files, mtime-cached | 51-144 |
| `MT5DataProvider` | MetaTrader 5 socket (Windows-only) | 147-376 |

### 1.2 `CSVDataProvider` (l.51-144)

- Pattern fichier : `{data_dir}/{symbol}_{timeframe}.csv`
- Colonnes attendues : `timestamp/Date/Datetime/Time, Open, High, Low, Close, Volume`
- Cache RAM `Dict[cache_key → DataFrame]` invalidé sur `mtime` change
- `get_ohlcv` retourne `df.tail(lookback).copy()`

### 1.3 `MT5DataProvider` (l.147-376)

- Reconnexion auto (`MAX_RECONNECT_ATTEMPTS=3`, `RECONNECT_DELAY_S=2.0`)
- Validation symbol par appel + cache `_validated_symbols`
- Détection `real_volume` vs `tick_volume` (l.348-359) — fallback intelligent
- `ensure_connected` (l.276-293) : health-check via `account_info()` + reconnect

### 1.4 `data_quality.py` (191 l)

- `validate_ohlcv(df, symbol, timeframe, strict=True)` retourne `ValidationReport`
- 6 niveaux de check :
  1. Non-empty
  2. Colonnes requises (Open, High, Low, Close, Volume)
  3. DatetimeIndex monotonic, no duplicates
  4. H ≥ max(O,C), L ≤ min(O,C) — cohérence OHLC
  5. Volume ≥ 0
  6. Gap detection vs nominal `TIMEFRAME_MINUTES`

`DataQualityError` héritée de `ValueError` — lève en mode `strict=True`.

---

## 2. Audit ligne à ligne — bugs & risques

### Bug n°1 — CSVDataProvider charge **tout** le CSV en RAM, jamais purgé

```python
# l.85-89
if cache_key not in self._cache or cached_mtime != current_mtime:
    self._cache[cache_key] = self._load_csv(symbol, timeframe)
    self._cache_mtime[cache_key] = current_mtime
```

`_load_csv` charge **tout le fichier**. Sur `XAU_15MIN_2019_2024.csv` (141k bars, ~10 MB), c'est OK. Mais sur tick-level ou multi-asset 6 × 8 MB CSV, le scanner passe à ~100 MB RAM permanent.

**Fix** : limiter le cache à `tail(N+lookback_max)` au lieu de full load.

### Bug n°2 — Pas de support `df.tail()` partiel pour ingestion live append-only

Le commentaire l.79-81 mentionne "live-simulation, append-only" comme cas d'usage du mtime cache. **Mais** : tout reload re-parse le fichier complet. Si le fichier double de taille (5 ans → 10 ans), le scanner re-parse 20 MB toutes les `time.sleep(60s)`. Inutile pour les bars qu'on a déjà.

**Fix** : tail-incremental load avec offset memorisé.

### Bug n°3 — Aucune validation du **feed source** (XAU à 63 % vs 97.6 %)

Le scanner consomme le CSV présent dans `data/`. Si l'utilisateur copie le mauvais fichier (XAU 2019-2025 à 63 %), aucun warning au boot. La cascade : BOS detect 100 % → score gonflé → scanner émet n'importe quoi.

Voir `memory/data_quality_audit_2026_04_23.md` : root cause documentée.

**Fix** : au boot, calculer `coverage_pct = bars / expected_bars` et ABORT si < 90 %. ✅ Déjà partiellement dans `validate_ohlcv` (gap_count) mais pas un fail-fast.

### Bug n°4 — `MT5DataProvider` pas thread-safe

```python
# l.296-316 _validate_and_select_symbol
if symbol in self._validated_symbols:
    return
symbol_info = self._mt5.symbol_info(symbol)  # network call
...
self._validated_symbols.add(symbol)
```

Pas de `Lock`. Si 2 threads (scanner + watchdog) appellent `get_ohlcv` simultanément, le `_validated_symbols.add()` peut être appelé 2 fois (idempotent, pas un bug). **MAIS** `_volume_source` aussi (l.354-359), et le `_mt5.copy_rates_from_pos` est lui-même thread-safe par MT5 ? Doc unclear.

**Fix** : `threading.RLock` autour des accès MT5.

### Bug n°5 — Aucun fallback fournisseur

Si MT5 down (broker maintenance), `MT5DataProvider.get_ohlcv` lève. Le scanner crash ou skip silently selon la gestion d'erreur du caller (`sentinel_scanner.py`). Pas de fallback CSV ni alternate broker.

**Fix** : pattern `CompositeDataProvider([MT5, CSV, Polygon])` avec retry chain.

### Bug n°6 — Pas de schema versioning sur le CSV

Si Dukascopy change le format colonne ou TZ, le `_load_csv` peut renvoyer des données mal parsed. Le `validate_ohlcv` détecte la corruption structurelle mais pas un shift de timezone (UTC vs ET).

**Fix** : ajouter `SOURCE` metadata column ou un `.metadata.json` à côté du CSV.

### Bug n°7 — `MAX_RECONNECT_ATTEMPTS=3` × `RECONNECT_DELAY=2.0` = 6 s blocking

Bloquant pendant 6 s sur le scanner thread = manque 1 bar M15 si timing malheureux.

**Fix** : reconnexion async avec exponential backoff dans un thread séparé.

### Bug n°8 — Pas de logging "data freshness"

`MT5DataProvider` retourne le DataFrame mais ne loggue jamais "last bar timestamp = X, current time = Y". Si le broker renvoie des données stale, le scanner émet sur du vieux. Aveugle.

**Fix** : `logger.info("MT5 data freshness: last_bar=%s, lag=%ds", df.index[-1], lag)`.

### Bug n°9 — `validate_ohlcv` `strict=True` lève une exception, casse le scanner

Si un bar corrompu remonte (NFP weird tick), `DataQualityError` est levée. Le scanner devrait skip et alerter, pas crash.

**Fix** : try/except dans `sentinel_scanner._scan_once` avec dead-letter queue.

### Bug n°10 — Aucun test d'intégration sur des feeds réels

`tests/test_data_quality.py` teste avec des DataFrames mockés. Aucun test ne charge `data/XAU_15MIN_2019_2024.csv` et vérifie qu'il passe `validate_ohlcv`. Si le format CSV change (Dukascopy update), break silencieux.

---

## 3. Tableau fournisseurs comparatif

| Fournisseur | Couverture | Latence | Coût/mois (commercial) | Licence | Verdict |
|-------------|------------|---------|------------------------:|---------|---------|
| **Dukascopy** (actuel) | 1990+ tick-level Forex/Gold/CFDs | T+1 | $0 personal / **n/a commercial direct** | **Personal only sauf accord** | ⚠️ Risque légal sur monétisation |
| **MT5 broker history** | Broker-dependent (10 ans typique) | < 1s | $0 (avec compte) | Liée au compte trader | ⚠️ Pas indépendant du broker |
| **Polygon.io** | 1990+ all assets | < 100 ms | $79-379 (FX/equities) | ✅ Commercial OK | Recommandé prod |
| **Tiingo** | 1990+ FX/Stocks | T+0 | $30-50 (FX) | ✅ Commercial OK | Bon ratio prix/qualité |
| **Twelve Data** | 1990+ all assets | < 500 ms | $79-159 | ✅ Commercial OK | Concurrent direct Polygon |
| **Databento** | Tick-level institutional | < 1ms | $250-2000 | ✅ Commercial enterprise | Overkill solo founder |
| **ICE Data Services** | Réf. instit. | Real-time | $1000+ | ✅ Enterprise | Hors budget solo |
| **Refinitiv (LSEG)** | Réf. instit. | Real-time | $1500+ | ✅ Enterprise | Hors budget solo |

### 3.1 Recommandation par phase

| Phase | Fournisseur | Coût/mois |
|-------|-------------|----------:|
| **Personal testing (actuel)** | Dukascopy CSV historique + MT5 live | $0 |
| **Soft-launch (FREE-only)** | Dukascopy CSV + MT5 live | $0 (mais clarifier licence Dukascopy) |
| **Paid launch (Stripe)** | Tiingo FX ($30) + MT5 broker live | $30 |
| **Scale (1k MAU)** | Polygon.io Currencies Starter ($79) | $79 |
| **Multi-asset commercial** | Polygon.io Stocks+Currencies+Crypto ($199) | $199 |

---

## 4. Audit licence Dukascopy

### 4.1 État légal (à vérifier auprès de Dukascopy directement)

Selon la **Dukascopy Historical Data License** publique (à valider 2026) :
- Usage personnel : ✅ gratuit (TPS download, retail traders)
- Usage commercial direct (revente data brut) : ❌ interdit
- Usage commercial dérivé (signaux générés à partir de la data) : ⚠️ **zone grise**, accord recommandé

**Risque concret** : si une plainte est déposée par Dukascopy, retrait forcé du service + amende potentielle.

### 4.2 Recommandation

- **Court terme** : envoyer email à `commercial@dukascopy.com` pour clarification écrite ; en attente, communiquer "personal testing only".
- **Moyen terme** : migrer backtest + historique vers Tiingo ou Polygon.io ($30-79/mo) avant ouvrir Stripe.

### 4.3 Voir aussi

`memory/eval_29_compliance_findings.md` :
> Dukascopy/FF en usage commercial déguisé.

C'est un risque déjà identifié — pas surpris.

---

## 5. Top 5 améliorations priorisées

| # | Amélioration | Effort | Impact go-live | Coût/mois | Priorité |
|---|--------------|:------:|:--------------:|:---------:|:--------:|
| **1** | **Re-télécharger XAU 2019-2025 propre** (Dukascopy ou Tiingo, ≥ 95 % couv.) | S | bloqueur PF | $0 | P0 |
| **2** | **Souscrire Tiingo $30/mo** + migrer historique 6-ans XAU+EUR | M | déblocage légal | $30 | P0 |
| **3** | **Pipeline ingestion live WebSocket** (Polygon.io ou MT5 ticker) avec queue + dedup | L | déblocage SLA "30s" | +$50 | P1 |
| **4** | **Coverage check fail-fast au boot** (abort si < 95 %) | XS | sécurité | $0 | P0 |
| **5** | **CompositeDataProvider** pattern (fallback chain MT5 → Tiingo → CSV) | M | résilience prod | $0 | P1 |

### 5.1 Détail P0 — coverage check au boot

```python
# src/intelligence/main.py au _calibrate_system
df = data_provider.get_ohlcv(symbol, "M15", lookback=20000)
expected_bars = (df.index[-1] - df.index[0]) / pd.Timedelta(minutes=15)
coverage = len(df) / expected_bars
if coverage < 0.95:
    raise SystemExit(
        f"Data coverage {coverage:.1%} < 95% threshold for {symbol}. "
        f"Re-download from a reputable source."
    )
```

5 lignes, élimine le risque #1 de credibility.

---

## 6. Plan d'ingestion live proposé

```
                        ┌──────────────────┐
                        │  Polygon.io WS   │
                        │  wss://...       │
                        └────────┬─────────┘
                                 │ tick events
                                 ▼
                        ┌──────────────────┐
                        │  Tick aggregator │  ← bucketize en bars M15
                        │  (Python asyncio)│
                        └────────┬─────────┘
                                 │ on bar close
                                 ▼
                        ┌──────────────────┐
                        │  Bar queue       │  ← dedup + replay-safe
                        │  (Redis / SQLite)│
                        └────────┬─────────┘
                                 │ pop
                                 ▼
                        ┌──────────────────┐
                        │  SentinelScanner │
                        └──────────────────┘
```

**Latence cible** : tick → bar close → queue → scan = **< 5 s** (vs 30 s SLA actuel).

### 6.1 Coûts estimés

| Élément | Coût/mois |
|---------|----------:|
| Polygon.io Currencies Starter | $79 |
| Redis Cloud (Free tier) | $0 |
| Latence WebSocket continue (1k req) | inclus |
| Backup Tiingo (fail-over) | $30 |
| **Total** | **$109/mo** |

vs Dukascopy CSV $0 — **augmentation $109/mo justifiable seulement à 100+ subs payants** (break-even tier ANALYST $49 ≈ 3 abonnés).

---

## 7. Budget OpEx data 12 mois

| Mois | Phase | Fournisseur | Coût | MAU cible |
|------|-------|-------------|-----:|----------:|
| M1-M3 | Personal | Dukascopy + MT5 | $0 | 0 (test perso) |
| M4-M6 | Soft launch | Tiingo | $30 | 50 FREE |
| M7-M9 | Paid launch | Tiingo + MT5 | $30 | 5 paid |
| M10-M12 | Scale | Polygon + Tiingo backup | $109 | 50 paid |
| Y2 multi-asset | Polygon stocks+FX | $199 | 200 paid |

**Cumul Y1** : $30×3 + $30×3 + $109×3 = **$507**. Marginal vs revenus targets eval_28 GTM ($5-7k MRR M12).

---

## 8. KPIs cibles post-implémentation

| KPI | Avant | Après | Mesure |
|---|---|---|---|
| Couverture XAU M15 6-ans | 63 % | ≥ 97 % | `validate_ohlcv.gap_count` |
| Latence bar-close → signal | 60-120 s | **< 30 s** | scanner instrumenté |
| Fallback automatique fournisseur | non | oui | tests E2E |
| Licence commerciale data | ⚠️ zone grise | ✅ Tiingo signed | contrat |
| Pipeline live WebSocket | non | oui | latency P95 < 5 s |
| Coverage gate au boot | non | oui | abort si < 95 % |

---

## 9. Trade-offs

| Gain | Coût |
|---|---|
| Tiingo $30/mo | -$30/mo, +crédibilité légale |
| WebSocket live | +complexité asyncio, +1 dépendance Redis |
| Composite fallback | +complexité provider, +tests E2E |
| Coverage gate au boot | risque false positive si CSV manquant ; bénéfice >> coût |
| Migration Polygon.io | -lock-in fournisseur, +liberté Dukascopy |

---

## 10. Note finale & recommandation

**Note : 3.5 / 10. Verdict GO/NO-GO : ❌ NO-GO commercial.**

Le code des providers est correct (pattern ABC, mtime cache, MT5 reconnect, validation OHLC) mais :
1. Le **feed historique XAU à 63 % couverture** falsifie tout backtest publié — **bloqueur PF**
2. **5/6 presets sans data** — multi-asset = vapor-ware
3. **Licence Dukascopy zone grise** — risque légal documenté (eval_29)
4. **Pas de pipeline live** — SLA "signal en 30 s" impossible
5. **Pas de fallback fournisseur** — 1 broker down = scanner down

**Recommandation immédiate** :
- **Cette semaine (P0, ~6 h)** : re-télécharger XAU 2019-2025 propre + coverage gate boot + flag licence Dukascopy en interne
- **Mois prochain (P0, $30/mo)** : souscrire Tiingo pour data commerciale légale
- **Avant Stripe (P1, ~1 sem dev + $79/mo)** : pipeline live Polygon.io WebSocket + composite fallback

Sans ces 3 phases, **toute promesse "real-time signals" ou "audited backtest 6-ans" est mensongère**.

---

### Annexes
- Code source : `src/intelligence/data_providers.py` (375 l), `data_quality.py` (191 l)
- Memory entries : `data_quality_audit_2026_04_23.md`, `audit_backtest_2026_04_24.md`, `news_pipeline.md`
- Compliance : `memory/eval_29_compliance_findings.md`
- Backtest credibility : `reports/eval_18_backtest.md`
- Scripts : `scripts/download_dukascopy_xau.py`, `scripts/audit_data_quality.py`, `scripts/crosscheck_mt5_calendar.py`

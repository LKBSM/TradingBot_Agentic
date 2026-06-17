# Branchement LIVE de `/app` sur données réelles — Rapport

> **Date** : 2026-06-08 (livré/vérifié 2026-06-10) · **Branche** : `feat/app-live-data-wiring`
> **Base** : `audit/indicateur-visuel-faisabilite` (seule branche portant **à la fois** l'UI
> graphique commitée *et* le fix mappers de `institutional-overhaul`).
> **Positionnement** : niveau 1.5 strict, contrat **descriptif** (`confluence_signal=None`).
> **Mission** : exposer les bougies OHLC du cache SQLite au front et câbler `/app` sur le vrai
> backend FastAPI (Twelve Data → moteur SMC → Haiku), sans toucher au moteur ni au mapper.

---

## 0. Verdict

> ✅ **`/app` tourne en LIVE sur données réelles.** Les 6 combos (XAU/EUR × M15/H1/H4) affichent
> une vraie lecture + un vrai graphique (chandeliers Twelve Data + overlays SMC issus du moteur).
> Sentinel (Haiku) répond sur la vraie lecture. **Aucun champ prédictif ne fuit** vers `/app`.
> Le graphique s'arrête à la dernière bougie clôturée (aucune projection).

Vérifié end-to-end sur la stack levée localement (backend `:8000`, front `:3001`, proxy OK).

---

## 1. Ce qui est câblé

### Backend — exposition des bougies (additif, lecture seule, **0 appel Twelve Data**)
- **`src/storage/candles_cache_store.py`** : nouvelle méthode `get_last_n_candles(instrument,
  timeframe, n)` — `SELECT … ORDER BY ts DESC LIMIT n` puis renversement en ordre croissant.
  Lecture pure du cache déjà peuplé par l'assembler/scheduler ; aucun appel fournisseur.
- **`src/api/routes/candles.py`** (nouveau) : route **`GET /api/candles?instrument&timeframe&limit`**.
  - Réutilise le **même** store que l'assembler (`assembler.candles_store`) — instance partagée.
  - Réponse strictement descriptive : `{instrument, timeframe, candles:[{time, open, high, low,
    close, volume}]}`. `time` = **epoch UTC en secondes** (UTCTimestamp lightweight-charts).
  - Normalisation UTC : un `ts` tz-naïf est interprété en UTC (jamais en heure locale).
  - Codes : `200` (combo valide cache plein) · `400` (hors périmètre V1) · `404` (combo valide,
    cache vide) · `503` (store non câblé) · `422` (limit hors bornes). `limit` ∈ [1, 500], défaut 200.
- **`src/api/app.py`** : enregistrement du routeur `candles.router` (1 import + 1 `include_router`).

### Frontend — bascule sur le réel
- **`webapp/lib/mockReadings.ts`** : `READING_DATA_SOURCE` passé **`'mock' → 'live'`** (point de
  bascule unique). `Candle` n'est plus défini ici mais ré-exporté depuis les types du contrat.
- **`webapp/types/market-reading.ts`** : types `Candle` + `CandlesResponse` (source unique,
  partagée mock/live).
- **`webapp/lib/market-reading/api-client.ts`** : `fetchCandles()` + `CandlesError`. Même
  origine — proxifié vers FastAPI par le rewrite `/api/:path*` ; **le front ne détient aucune clé**.
- **`webapp/lib/market-reading/hooks.ts`** : hook **`useCandles()`**. Mode `mock` → `getMockCandles`
  (local) ; mode `live` → `fetchCandles`. **Re-fetch des bougies uniquement quand `candle_close_ts`
  change** (ou changement de combo) — pas de polling tick. Tout échec (404/400/503/réseau) →
  `candles: null` → placeholder « Graphique indisponible » (la lecture textuelle reste utilisable).
- **`webapp/components/app/ReadingColumn.tsx`** : appelle `useCandles`, alimente le chart hero ;
  les readings live arrivent déjà via `useMarketReading` (chemin live préexistant). Le badge
  « dernière bougie clôturée » reste alimenté par `reading.header.candle_close_ts` (champ canonique).
- **`AppWorkspace.tsx` / `MobileWorkspace.tsx` / `ReadingChart.tsx`** : `dataSource` threadé jusqu'à
  la colonne ; `ReadingChart` consomme le type `Candle` du contrat (découplé du module mock).

### Discipline tenue
- **Aucune** modification du moteur (`strategy_features.py`) ni du mapper (`market_reading_mappers.py`).
- Contrat `/app` **descriptif** : `confluence_signal=None` ; `/api/candles` = OHLC seul.
- `InsightSignalV2` (forecast / conformal 95 % / `hmm_posterior` / `bocpd` / `valid_until`) **jamais
  servi** à `/app` — vérifié par assertion automatisée (cf. §4).

---

## 2. Route ajoutée — contrat

```
GET /api/candles?instrument=XAUUSD&timeframe=M15&limit=200
200 → {
  "instrument": "XAUUSD",
  "timeframe": "M15",
  "candles": [
    { "time": 1781143200, "open": 4124.1556, "high": 4132.06066,
      "low": 4123.68362, "close": 4131.21144, "volume": 0.0 },
    …  // ordre croissant, s'arrête à la dernière bougie clôturée
  ]
}
400 → instrument/timeframe hors périmètre V1     404 → combo valide, cache vide
503 → store non câblé                            422 → limit hors [1,500]
```

---

## 3. Commandes EXACTES pour lever la stack

> Pré-requis : `.env` racine rempli (déjà le cas sur ce poste) avec `ANTHROPIC_API_KEY`,
> `TWELVE_DATA_API_KEY`, `BOOTSTRAP_ENABLED=true`, `SCHEDULER_ENABLED=true`, `CHATBOT_ENABLED=true`.
> `webapp/.env.local` contient déjà `NEXT_PUBLIC_API_BASE=http://localhost:8000`.

### Terminal 1 — Backend (FastAPI + bootstrap + scheduler), port 8000
```powershell
# depuis la racine du repo
python -m uvicorn src.api.app:create_app --factory --env-file .env --host 127.0.0.1 --port 8000
```
Attendre `Application startup complete.` / `Uvicorn running on http://127.0.0.1:8000`.
*(Variante équivalente, point d'entrée Docker historique, qui lève aussi le scanner legacy :
`python -m src.intelligence.main`.)*

### Terminal 2 — Frontend (Next.js), port 3000
```powershell
cd webapp
npm install        # première fois seulement
npm run dev        # http://localhost:3000  (proxy /api/* -> :8000)
```

### Ouvrir l'app en live
```
http://localhost:3000/fr/app
```
> `/fr/app` redirige (307) vers `/app` — `fr` est la locale par défaut, le préfixe est retiré par
> next-intl. Les deux URLs fonctionnent.
>
> ⚠️ **Port 3000** : observé occupé par un autre process pendant la vérif (Next a basculé sur 3001).
> Si 3000 est pris, soit libérer le process, soit ouvrir le port affiché par `npm run dev`.

---

## 4. Vérification end-to-end (réalisée)

| Vérif | Résultat |
|---|---|
| `/health` | `status: healthy`, `testing_mode: True` |
| `GET /api/market-reading XAUUSD M15` | **200** — px **4131.21**, trend `bearish`, `description_source: haiku_generated`, **`confluence_signal` absent** |
| Fuite prédictive (reading) | **Aucune** (`forecast`/`hmm`/`conformal`/`confidence_interval`/`bocpd` absents) |
| `GET /api/candles XAUUSD M15` | **200** — 200 bougies, ordre **croissant**, clés `{time,open,high,low,close,volume}` uniquement, dernière clôture == `close_price` du reading |
| Fuite prédictive (candles) | **Aucune** (assertion sur `forecast/hmm/conformal/confluence/valid_until/target_`) |
| `GET /api/candles EURUSD H4` (cache vide au moment du test) | **404** propre — `No candles cached yet for EURUSD/H4` |
| `GET /api/candles BTCUSD M15` | **400** propre — `Unsupported instrument 'BTCUSD'` |
| `GET /api/market-reading EURUSD M15` | **200** — px **1.15508**, trend `ranging`, `confluence_signal` absent |
| Sentinel `POST /api/chatbot/message` | **200** — outil `get_market_reading` appelé, réponse ancrée sur la vraie lecture (« retest de BOS à 4132,53 »), `blocked_reason: null` |
| 6 combos en cache (candles réelles) | **XAU & EUR × M15/H1/H4 = 6/6**, 200 bougies chacun |
| Front `/fr/app` | 307 → `/app` → **200**, shell présent (« Combinaisons disponibles », « Lecture de marché », « Sentinel ») |
| Proxy `/api/candles` via `:3001` → `:8000` | **200** (passthrough OK, JSON réel) |
| Tests backend | `tests/test_candles_endpoint.py` + `test_candles_cache_store.py` → **26 passés** |
| Tests frontend | suite complète **123 passés** (dont 4 nouveaux fichiers de câblage) |
| `tsc --noEmit` | **OK** |
| `next build` | **Vert** — `/[locale]/app` 14.9 kB / 157 kB first-load |

---

## 5. Budget Twelve Data

- **`/api/candles` ne fait AUCUN appel Twelve Data** : lecture pure du cache SQLite `candles_cache`.
  Rafraîchir le chart à volonté ne touche pas le quota.
- Le **scheduler** (tick 60 s, auto-stop 24 h d'inactivité) régénère les combos **actifs** quand une
  nouvelle bougie a clôturé ; le **bootstrap** a peuplé les 6 combos (200 bougies chacun) au démarrage.
- Sur le free tier (8 req/min, **800 req/jour**), le câblage reste dans le budget : le front re-pull
  les bougies **seulement quand `candle_close_ts` change** (et non à chaque tick). Les readings (et
  donc les appels TD) sont déjà mutualisés par l'assembler + cache.

---

## 6. Points d'attention

1. **Décalage horloge machine ↔ feed Twelve Data (~10 h)** — `header.candle_close_ts` est dérivé
   de l'horloge **locale** par l'assembler (`expected_last_candle_close(now)`), tandis que les
   bougies portent les **vrais** timestamps TD. Sur ce poste, la dernière bougie cachée est à
   `2026-06-11T02:00Z` (clôture réelle, prix identique au reading) alors que le badge affiche
   `2026-06-10T16:00Z`. **Le prix de clôture coïncide** (même bougie) ; c'est un artefact d'horloge
   de l'environnement, pas un bug de câblage. Le graphique **ne projette pas** : il s'arrête à la
   dernière bougie clôturée du feed. En prod (horloge correcte) badge et dernière bougie s'alignent.
   *Non corrigé ici : toucher `expected_last_candle_close`/le mapper est hors-périmètre.*
2. **Premier accès à un combo non encore caché** — léger flash « Graphique indisponible » (candles
   404) le temps que le reading peuple le cache, puis auto-résolution au changement de `candle_close_ts`.
   Dégradation gracieuse conforme à l'attendu.
3. **`/api/candles` 400 vs 404** — `400` = hors périmètre V1 (cohérent avec `/api/market-reading`),
   `404` = combo valide mais cache vide. Les deux sont des erreurs propres mappées vers le placeholder.
4. **Calendrier ForexFactory** — le feed `ff_calendar_nextweek.json` renvoie 404 (URL obsolète) ;
   sans incidence sur le chart/lecture, mais `events.news_*` peut être vide. Hors-périmètre de cette mission.
5. **Port 3000 occupé** pendant la vérif (cf. §3).

---

## 7. Pour le founder — ouvrir l'app en live

```powershell
# Terminal 1 (racine du repo)
python -m uvicorn src.api.app:create_app --factory --env-file .env --host 127.0.0.1 --port 8000

# Terminal 2
cd webapp
npm run dev
```
Puis ouvrir : **http://localhost:3000/fr/app**

Sélectionner un combo à gauche (XAU/EUR × M15/H1/H4) → vraie lecture + vrai graphique
(chandeliers Twelve Data + niveaux BOS/CHOCH/retest + zones OB/FVG du moteur) → poser une
question à Sentinel sur la droite.

# VALIDATION_DATASET_2026_06_06 — Liste des 60 candles échantillonnées

> Audit de validation algorithmique — Phase 2.2
> Généré pour : 2026-06-06 validation audit · TF : H1 · lookback : 200

## Note data (écart documenté)

- **XAUUSD** : période 2026-01-01..2026-03-31 (jan-mars 2026, comme spécifié).
- **EURUSD** : période 2025-10-01..2025-12-31. EURUSD utilise oct-déc 2025 (CSV s'arrête 2025-12-31).

## Méthode d'échantillonnage

Stratification par **état détecté par l'algo** (6 strates × 5 candles × 2 instruments).
Sélection déterministe (équi-espacée par strate, pas de RNG). La stratification utilise
l'état-algo lui-même afin de couvrir des conditions variées à faire valider manuellement.

Composition par strate :

| Instrument | bos_bull | bos_bear | choch | range | high_vol | ordinary |
|---|---|---|---|---|---|---|
| XAUUSD | 5 | 5 | 5 | 5 | 5 | 5 |
| EURUSD | 5 | 5 | 5 | 5 | 5 | 5 |

Total : 60 candles · 0 erreurs · Haiku live : True

## Liste complète

| # | Instrument | Bar open (UTC) | Close ts (UTC) | Close | Trend | Vol | Phase | BOS | CHOCH | OB | FVG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XAUUSD | 2026-01-01T23:00:00 | 2026-01-02T00:00:00 | 4348.25500 | ranging | normal | ranging | — | — | 0 | 0 |
| 2 | XAUUSD | 2026-01-02T00:00:00 | 2026-01-02T01:00:00 | 4354.12850 | ranging | normal | ranging | — | — | 0 | 1 |
| 3 | XAUUSD | 2026-01-02T03:00:00 | 2026-01-02T04:00:00 | 4375.18500 | ranging | normal | ranging | bullish | bullish | 0 | 1 |
| 4 | XAUUSD | 2026-01-05T05:00:00 | 2026-01-05T06:00:00 | 4412.56850 | bullish | normal | trend | — | — | 0 | 0 |
| 5 | XAUUSD | 2026-01-06T14:00:00 | 2026-01-06T15:00:00 | 4487.78850 | ranging | normal | ranging | bullish | — | 0 | 0 |
| 6 | XAUUSD | 2026-01-09T05:00:00 | 2026-01-09T06:00:00 | 4468.16000 | ranging | low | ranging | — | — | 0 | 0 |
| 7 | XAUUSD | 2026-01-12T00:00:00 | 2026-01-12T01:00:00 | 4588.92650 | bullish | elevated | expansion | — | — | 0 | 1 |
| 8 | XAUUSD | 2026-01-16T10:00:00 | 2026-01-16T11:00:00 | 4600.23500 | bullish | normal | trend | — | — | 0 | 0 |
| 9 | XAUUSD | 2026-01-20T23:00:00 | 2026-01-21T00:00:00 | 4775.94500 | bullish | normal | trend | bullish | — | 0 | 0 |
| 10 | XAUUSD | 2026-01-26T03:00:00 | 2026-01-26T04:00:00 | 5076.79500 | bullish | elevated | expansion | — | — | 0 | 0 |
| 11 | XAUUSD | 2026-01-28T01:00:00 | 2026-01-28T02:00:00 | 5204.69000 | bullish | elevated | expansion | bullish | — | 1 | 0 |
| 12 | XAUUSD | 2026-01-29T15:00:00 | 2026-01-29T16:00:00 | 5182.82500 | bullish | elevated | expansion | bearish | bearish | 0 | 0 |
| 13 | XAUUSD | 2026-01-30T01:00:00 | 2026-01-30T02:00:00 | 5301.14000 | bullish | elevated | expansion | — | — | 0 | 0 |
| 14 | XAUUSD | 2026-02-02T05:00:00 | 2026-02-02T06:00:00 | 4549.61650 | ranging | elevated | ranging | bearish | — | 1 | 0 |
| 15 | XAUUSD | 2026-02-05T16:00:00 | 2026-02-05T17:00:00 | 4889.06650 | ranging | normal | ranging | — | — | 1 | 0 |
| 16 | XAUUSD | 2026-02-13T17:00:00 | 2026-02-13T18:00:00 | 5021.11000 | bullish | normal | trend | — | — | 0 | 0 |
| 17 | XAUUSD | 2026-02-16T12:00:00 | 2026-02-16T13:00:00 | 5002.02500 | ranging | low | ranging | — | — | 0 | 0 |
| 18 | XAUUSD | 2026-02-17T04:00:00 | 2026-02-17T05:00:00 | 4953.00000 | ranging | low | ranging | — | — | 0 | 0 |
| 19 | XAUUSD | 2026-02-24T13:00:00 | 2026-02-24T14:00:00 | 5115.14850 | ranging | normal | ranging | bearish | bearish | 0 | 1 |
| 20 | XAUUSD | 2026-02-27T20:00:00 | 2026-02-27T21:00:00 | 5263.41500 | bullish | normal | trend | bullish | bullish | 0 | 0 |
| 21 | XAUUSD | 2026-03-02T06:00:00 | 2026-03-02T07:00:00 | 5387.71500 | bullish | elevated | expansion | — | — | 0 | 0 |
| 22 | XAUUSD | 2026-03-02T13:00:00 | 2026-03-02T14:00:00 | 5394.02500 | bullish | normal | trend | — | — | 0 | 0 |
| 23 | XAUUSD | 2026-03-02T14:00:00 | 2026-03-02T15:00:00 | 5333.60350 | bullish | elevated | expansion | bearish | bearish | 0 | 0 |
| 24 | XAUUSD | 2026-03-03T10:00:00 | 2026-03-03T11:00:00 | 5248.27500 | bullish | elevated | expansion | bearish | — | 0 | 0 |
| 25 | XAUUSD | 2026-03-05T21:00:00 | 2026-03-05T22:00:00 | 5083.64000 | ranging | normal | ranging | — | — | 0 | 0 |
| 26 | XAUUSD | 2026-03-18T20:00:00 | 2026-03-18T21:00:00 | 4818.56500 | bearish | normal | trend | bearish | — | 0 | 1 |
| 27 | XAUUSD | 2026-03-19T04:00:00 | 2026-03-19T05:00:00 | 4849.82500 | bearish | normal | trend | — | — | 0 | 0 |
| 28 | XAUUSD | 2026-03-19T13:00:00 | 2026-03-19T14:00:00 | 4621.29000 | bearish | elevated | expansion | — | — | 0 | 1 |
| 29 | XAUUSD | 2026-03-22T23:00:00 | 2026-03-23T00:00:00 | 4464.79000 | bearish | elevated | expansion | bearish | — | 0 | 0 |
| 30 | XAUUSD | 2026-03-26T19:00:00 | 2026-03-26T20:00:00 | 4364.16000 | bearish | normal | trend | bearish | bearish | 1 | 0 |
| 31 | EURUSD | 2025-10-01T00:00:00 | 2025-10-01T01:00:00 | 1.17359 | bearish | low | trend | — | — | 0 | 0 |
| 32 | EURUSD | 2025-10-01T01:00:00 | 2025-10-01T02:00:00 | 1.17392 | ranging | low | ranging | — | — | 0 | 0 |
| 33 | EURUSD | 2025-10-01T07:00:00 | 2025-10-01T08:00:00 | 1.17500 | ranging | elevated | ranging | — | — | 1 | 0 |
| 34 | EURUSD | 2025-10-02T15:00:00 | 2025-10-02T16:00:00 | 1.16967 | bearish | elevated | expansion | bearish | bearish | 0 | 0 |
| 35 | EURUSD | 2025-10-02T20:00:00 | 2025-10-02T21:00:00 | 1.17158 | bearish | normal | trend | — | — | 0 | 0 |
| 36 | EURUSD | 2025-10-09T03:00:00 | 2025-10-09T04:00:00 | 1.16448 | bearish | low | trend | bullish | bullish | 0 | 0 |
| 37 | EURUSD | 2025-10-09T18:00:00 | 2025-10-09T19:00:00 | 1.15464 | bearish | elevated | expansion | bearish | — | 1 | 0 |
| 38 | EURUSD | 2025-10-14T13:00:00 | 2025-10-14T14:00:00 | 1.15741 | bearish | elevated | expansion | — | — | 0 | 0 |
| 39 | EURUSD | 2025-10-15T18:00:00 | 2025-10-15T19:00:00 | 1.16333 | bearish | normal | trend | — | — | 0 | 0 |
| 40 | EURUSD | 2025-10-20T06:00:00 | 2025-10-20T07:00:00 | 1.16585 | ranging | normal | ranging | — | — | 1 | 0 |
| 41 | EURUSD | 2025-10-20T17:00:00 | 2025-10-20T18:00:00 | 1.16434 | ranging | normal | ranging | bearish | bearish | 0 | 0 |
| 42 | EURUSD | 2025-10-24T12:00:00 | 2025-10-24T13:00:00 | 1.16342 | ranging | elevated | ranging | bullish | — | 0 | 0 |
| 43 | EURUSD | 2025-10-30T20:00:00 | 2025-10-30T21:00:00 | 1.15646 | bearish | normal | trend | — | — | 1 | 0 |
| 44 | EURUSD | 2025-10-31T13:00:00 | 2025-10-31T14:00:00 | 1.15389 | bearish | normal | trend | bearish | — | 0 | 1 |
| 45 | EURUSD | 2025-11-05T17:00:00 | 2025-11-05T18:00:00 | 1.14805 | bearish | elevated | expansion | — | — | 0 | 0 |
| 46 | EURUSD | 2025-11-06T13:00:00 | 2025-11-06T14:00:00 | 1.15358 | bearish | normal | trend | bullish | bullish | 0 | 1 |
| 47 | EURUSD | 2025-11-07T11:00:00 | 2025-11-07T12:00:00 | 1.15581 | bearish | normal | trend | bullish | — | 0 | 1 |
| 48 | EURUSD | 2025-11-11T12:00:00 | 2025-11-11T13:00:00 | 1.15705 | bearish | normal | trend | — | — | 0 | 0 |
| 49 | EURUSD | 2025-11-19T10:00:00 | 2025-11-19T11:00:00 | 1.15694 | ranging | normal | ranging | bearish | — | 0 | 0 |
| 50 | EURUSD | 2025-11-25T13:00:00 | 2025-11-25T14:00:00 | 1.15584 | ranging | elevated | ranging | bullish | bullish | 0 | 1 |
| 51 | EURUSD | 2025-11-28T10:00:00 | 2025-11-28T11:00:00 | 1.15593 | bearish | normal | trend | — | — | 0 | 0 |
| 52 | EURUSD | 2025-11-28T18:00:00 | 2025-11-28T19:00:00 | 1.16016 | ranging | elevated | ranging | — | — | 0 | 0 |
| 53 | EURUSD | 2025-12-01T15:00:00 | 2025-12-01T16:00:00 | 1.16289 | bullish | normal | trend | — | — | 0 | 0 |
| 54 | EURUSD | 2025-12-03T10:00:00 | 2025-12-03T11:00:00 | 1.16606 | bullish | normal | trend | bullish | — | 0 | 0 |
| 55 | EURUSD | 2025-12-05T16:00:00 | 2025-12-05T17:00:00 | 1.16343 | bullish | normal | trend | bearish | bearish | 0 | 0 |
| 56 | EURUSD | 2025-12-11T20:00:00 | 2025-12-11T21:00:00 | 1.17411 | bullish | elevated | expansion | — | — | 0 | 1 |
| 57 | EURUSD | 2025-12-15T02:00:00 | 2025-12-15T03:00:00 | 1.17414 | bullish | normal | trend | — | — | 0 | 0 |
| 58 | EURUSD | 2025-12-16T12:00:00 | 2025-12-16T13:00:00 | 1.17779 | bullish | normal | trend | bullish | — | 0 | 0 |
| 59 | EURUSD | 2025-12-17T04:00:00 | 2025-12-17T05:00:00 | 1.17297 | bullish | low | trend | bearish | bearish | 0 | 0 |
| 60 | EURUSD | 2025-12-17T15:00:00 | 2025-12-17T16:00:00 | 1.17530 | bullish | elevated | expansion | bullish | bullish | 0 | 1 |

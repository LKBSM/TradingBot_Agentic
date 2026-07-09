# Banc d'essai qualité des fournisseurs de données de marché

_Généré le 2026-07-09 17:24 UTC par `tools/data-benchmark/report.py` — relançable (`python runner.py && python metrics.py && python scoring.py && python report.py`)._

Fenêtre testée : **30 jours** (2026-06-09 → 2026-07-09), 80 symboles × 5 TF (M5, M15, H1, H4, D1).

## Avertissement de méthode

Sur les marchés OTC il n'existe **pas de prix officiel unique** : chaque feed est l'agrégat d'un panel de contributeurs ou le book d'un broker. La référence du banc (`twelve_data`) est elle-même un agrégat. Le classement se lit donc « le plus proche de ma référence + le plus complet + le plus cohérent », jamais « le vrai prix ». Un fournisseur sans clé API est marqué **non testé** — aucune donnée n'est simulée.

## Classement (fournisseurs testés)

Pondérations (éditables dans `scoring.py`) : wick 35%, completeness 25%, validity 15%, coverage 15%, freshness 5%, reliability 5%.

| Rang | Fournisseur | Score global | Mèches | Complétude | Validité OHLC | Couverture | Fraîcheur | Fiabilité | Cellules OK/400 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **twelve_data** _(étalon)_ | **93.9** | — | 98.8 | 100.0 | 76.2 | 98.2 | 99.6 | 305 |
| 2 | **itick** | **78.2** | 45.8 | 97.7 | 99.9 | 97.2 | 77.5 | 85.9 | 389 |
| 3 | **fcsapi** | **73.6** | 53.4 | 98.6 | 100.0 | 41.2 | 92.7 | 89.4 | 165 |
| 4 | **mt5** | **71.1** | 35.5 | 93.0 | 100.0 | 70.0 | 98.7 | 100.0 | 280 |

## Détail par fournisseur

### twelve_data

_Reference du banc. Display = Venture, devis sales 2026-07-09: 'a partir de 149$/mois', sole proprietor OK, PAS d'indices du tout (confirme les 404 du banc). Tarif public 499$._

Statuts cellules : `{'not_covered': 95, 'ok': 305}`

Symboles verrouillés par plan supérieur (10) : BRENT, NAS100, SPX500, UK100, WTI, XAGUSD, XAUEUR, XAUGBP, XPDUSD, XPTUSD

Symboles hors catalogue / non couverts (9) : AUS200, EU50, FRA40, GER40, HK50, JP225, NATGAS, US2000, US30

Plus gros trous de complétude : `USDPLN_M5` 909 barres dès 2026-06-30T18:15 ; `USDNOK_M5` 890 barres dès 2026-06-30T18:15 ; `USDCNH_M15` 303 barres dès 2026-06-30T18:15 ; `USDHKD_M15` 303 barres dès 2026-06-30T18:15 ; `USDBRL_M15` 303 barres dès 2026-06-30T18:15.

### itick

_Calibre sur API reelle 2026-07-06. Crypto = paires USDT Binance (basis vs USD). BRENT et US2000 introuvables. H4 derive du H1. 79-319$/mois mais droit display a confirmer par ecrit._

Statuts cellules : `{'empty': 1, 'not_covered': 10, 'ok': 389}`

Symboles hors catalogue / non couverts (2) : BRENT, US2000

⚠️ TF **dérivés par resampling** (non natifs) : 78 cellules (ex. ADAUSD_H4, APTUSD_H4, ATOMUSD_H4, AUDCAD_H4, AUDCHF_H4, AUDJPY_H4).

Pires écarts de mèches vs référence (à vérifier à l'œil) :

| Symbole×TF | Timestamp (UTC) | ΔHigh (pts) | ΔLow (pts) | High réf → fournisseur | Low réf → fournisseur |
|---|---|---|---|---|---|
| EURTRY_H1 | 2026-06-17T19:00:00+00:00 | 3764.1 | 6734.7 | 53.4511 → 53.82751 | 53.154 → 53.82747 |
| EURTRY_H4 | 2026-06-17T16:00:00+00:00 | 550.6 | 6734.7 | 53.77245 → 53.82751 | 53.154 → 53.82747 |
| EURTRY_H4 | 2026-06-17T20:00:00+00:00 | 3704.3 | 6218.0 | 53.45708 → 53.82751 | 53.20567 → 53.82747 |
| EURTRY_D1 | 2026-06-17T00:00:00+00:00 | 1620.8 | 6171.8 | 53.80189 → 53.96397 | 53.154 → 53.77118 |
| EURTRY_M15 | 2026-06-17T19:00:00+00:00 | 3764.1 | 5327.3 | 53.4511 → 53.82751 | 53.29474 → 53.82747 |
| EURTRY_M5 | 2026-06-17T19:00:00+00:00 | 3764.1 | 4534.8 | 53.4511 → 53.82751 | 53.37399 → 53.82747 |
| EURTRY_M5 | 2026-06-17T23:00:00+00:00 | 3828.8 | 4297.9 | 53.44463 → 53.82751 | 53.39768 → 53.82747 |
| EURTRY_M15 | 2026-06-17T23:00:00+00:00 | 3772.2 | 4297.9 | 53.45029 → 53.82751 | 53.39768 → 53.82747 |

Plus gros trous de complétude : `HK50_M5` 1070 barres dès 2026-06-18T08:05 ; `JP225_M5` 785 barres dès 2026-06-12T06:30 ; `JP225_M5` 785 barres dès 2026-06-19T06:30 ; `HK50_M5` 782 barres dès 2026-06-12T08:05 ; `FRA40_M5` 696 barres dès 2026-06-12T19:55.

### fcsapi

_149-329$/mois all-markets, droit display a confirmer par ecrit. Cache 10min plans bas._

Statuts cellules : `{'error': 155, 'not_covered': 80, 'ok': 165}`

Symboles hors catalogue / non couverts (16) : AUS200, BRENT, DOTUSD, EU50, FRA40, GER40, HK50, JP225, NAS100, NATGAS, SPX500, UK100, UNIUSD, US2000, US30, WTI

Échecs (155) — 5 premiers :
- `XAUUSD_H4` : FCS: Access block for you, You have reached maximum 3 limit per minute in free account, Please stop extra hits or upgrade your account. Rest
- `XAGUSD_M15` : FCS: Access block for you, You have reached maximum 3 limit per minute in free account, Please stop extra hits or upgrade your account. Rest
- `XAGUSD_D1` : FCS: Access block for you, You have reached maximum 3 limit per minute in free account, Please stop extra hits or upgrade your account. Rest
- `XPTUSD_H1` : FCS: Access block for you, You have reached maximum 3 limit per minute in free account, Please stop extra hits or upgrade your account. Rest
- `NZDCHF_D1` : FCS: Your monthly API request limit exceed, Please upgrade your account, If you think this message is error, please contact our support.  Yo

Pires écarts de mèches vs référence (à vérifier à l'œil) :

| Symbole×TF | Timestamp (UTC) | ΔHigh (pts) | ΔLow (pts) | High réf → fournisseur | Low réf → fournisseur |
|---|---|---|---|---|---|
| XAUUSD_D1 | 2026-06-14T00:00:00+00:00 | 24.39 | 495.99 | 4306.49146 → 4308.93 | 4213.19131 → 4262.79 |
| XAUUSD_H1 | 2026-06-14T22:00:00+00:00 | 24.39 | 434.5 | 4306.49146 → 4308.93 | 4219.33985 → 4262.79 |
| XAUUSD_M15 | 2026-07-01T13:30:00+00:00 | 179.51 | 30.67 | 4081.79926 → 4099.75 | 4039.1934 → 4042.26 |
| XAUUSD_M5 | 2026-07-02T12:35:00+00:00 | 53.13 | 148.9 | 4133.42702 → 4138.74 | 4109.65009 → 4124.54 |
| XAUUSD_M15 | 2026-07-01T13:45:00+00:00 | 79.83 | 148.72 | 4115.77285 → 4107.79 | 4070.74796 → 4085.62 |
| GBPJPY_D1 | 2026-07-06T00:00:00+00:00 | 130.02 | 0.97 | 215.86977 → 217.17 | 215.58975 → 215.58 |
| XAUUSD_M5 | 2026-07-02T13:35:00+00:00 | 40.49 | 129.04 | 4134.11055 → 4138.16 | 4114.58635 → 4127.49 |
| XAUUSD_M15 | 2026-06-25T12:30:00+00:00 | 38.15 | 126.07 | 4007.8554 → 4011.67 | 3971.54328 → 3984.15 |

Plus gros trous de complétude : `XAUEUR_M5` 67 barres dès 2026-07-03T16:25 ; `XPTUSD_M5` 61 barres dès 2026-07-03T16:55 ; `XPDUSD_M5` 61 barres dès 2026-07-03T16:55 ; `XAUUSD_M5` 60 barres dès 2026-07-03T17:00 ; `XAGUSD_M5` 60 barres dès 2026-07-03T17:00.

### mt5

_JUGE uniquement (feed broker du terminal local, licence interne — PAS un candidat production). Bougies BID (les feeds API sont mid : biais ~demi-spread attendu). MetaQuotes-Demo : forex+metaux+indices, pas d'energie/crypto CFD._

Statuts cellules : `{'empty': 5, 'not_covered': 115, 'ok': 280}`

Symboles hors catalogue / non couverts (23) : ADAUSD, APTUSD, ATOMUSD, AVAXUSD, BCHUSD, BNBUSD, BRENT, BTCUSD, DOGEUSD, DOTUSD, ETCUSD, ETHUSD, FILUSD, LINKUSD, LTCUSD, MATICUSD, NATGAS, SOLUSD, TRXUSD, UNIUSD, WTI, XLMUSD, XRPUSD

Pires écarts de mèches vs référence (à vérifier à l'œil) :

| Symbole×TF | Timestamp (UTC) | ΔHigh (pts) | ΔLow (pts) | High réf → fournisseur | Low réf → fournisseur |
|---|---|---|---|---|---|
| EURTRY_M5 | 2026-06-21T21:20:00+00:00 | 3343.8 | 2384.7 | 53.25888 → 52.9245 | 53.1563 → 52.91783 |
| EURTRY_M5 | 2026-06-21T21:25:00+00:00 | 3026.1 | 1650.7 | 53.22734 → 52.92473 | 53.08324 → 52.91817 |
| EURTRY_M5 | 2026-06-21T21:30:00+00:00 | 2901.8 | 1641.1 | 53.23732 → 52.94714 | 53.08357 → 52.91946 |
| EURTRY_M5 | 2026-06-21T21:00:00+00:00 | 2764.3 | 1965.9 | 53.29531 → 53.01888 | 53.2127 → 53.01611 |
| EURTRY_M15 | 2026-06-21T21:00:00+00:00 | 2764.3 | 2530.4 | 53.29531 → 53.01888 | 53.15489 → 52.90185 |
| EURTRY_M5 | 2026-06-21T21:05:00+00:00 | 2362.6 | 2530.4 | 53.24959 → 53.01333 | 53.15489 → 52.90185 |
| EURTRY_M15 | 2026-07-05T21:00:00+00:00 | 724.5 | 2199.5 | 53.54672 → 53.47427 | 53.5161 → 53.29615 |
| EURTRY_H1 | 2026-07-05T21:00:00+00:00 | 775.0 | 2147.8 | 53.59422 → 53.51672 | 53.51093 → 53.29615 |

Plus gros trous de complétude : `HK50_M5` 939 barres dès 2026-06-18T18:55 ; `XAUGBP_M5` 657 barres dès 2026-06-29T06:55 ; `JP225_M5` 637 barres dès 2026-07-03T16:55 ; `US30_M5` 636 barres dès 2026-06-19T16:55 ; `US30_M5` 636 barres dès 2026-07-03T16:55.

## Triangulation croisée (qui est l'outlier ?)

Écart absolu moyen des mèches (high/low, en unités de prix) entre chaque paire de fournisseurs testés. Quand deux sources indépendantes s'accordent et qu'une troisième diverge, cette dernière est l'outlier probable — sans qu'aucune ne soit « le vrai prix ». `mt5` = feed broker du terminal local (bougies bid), juge non commercialisable.

### XAUUSD M15 (en points, 1 pt = 0.1)

| MAE high/low | fcsapi | itick | mt5 | twelve_data |
|---|---|---|---|---|
| **fcsapi** | — | 1.260 / 1.435 (n=824) | 0.954 / 3.658 (n=863) | 7.515 / 8.179 (n=800) |
| **itick** | 1.260 / 1.435 (n=824) | — | 1.387 / 3.375 (n=1661) | 8.479 / 9.096 (n=1881) |
| **mt5** | 0.954 / 3.658 (n=863) | 1.387 / 3.375 (n=1661) | — | 8.392 / 10.413 (n=1638) |
| **twelve_data** | 7.515 / 8.179 (n=800) | 8.479 / 9.096 (n=1881) | 8.392 / 10.413 (n=1638) | — |

### US30 M15 (en points, 1 pt = 1.0)

| MAE high/low | itick | mt5 |
|---|---|---|
| **itick** | — | 21.660 / 22.171 (n=1705) |
| **mt5** | 21.660 / 22.171 (n=1705) | — |

### NAS100 M15 (en points, 1 pt = 1.0)

| MAE high/low | itick | mt5 |
|---|---|---|
| **itick** | — | 29.649 / 30.415 (n=1705) |
| **mt5** | 29.649 / 30.415 (n=1705) | — |

### EURUSD M15 (en pips, 1 pt = 0.0001)

| MAE high/low | fcsapi | itick | mt5 | twelve_data |
|---|---|---|---|---|
| **fcsapi** | — | 0.390 / 0.338 (n=818) | 0.401 / 0.572 (n=899) | 0.584 / 0.502 (n=746) |
| **itick** | 0.390 / 0.338 (n=818) | — | 0.167 / 0.415 (n=1836) | 0.499 / 0.480 (n=1942) |
| **mt5** | 0.401 / 0.572 (n=899) | 0.167 / 0.415 (n=1836) | — | 0.496 / 0.763 (n=1764) |
| **twelve_data** | 0.584 / 0.502 (n=746) | 0.499 / 0.480 (n=1942) | 0.496 / 0.763 (n=1764) | — |

## Croisement qualité × prix d'affichage commercial

| Fournisseur | Score qualité | Prix display commercial (recherche 2026-07-05) |
|---|---|---|
| twelve_data | 93.9 | Venture 'a partir de 149 $/mois' (devis sales 2026-07-09 ; display FX+metaux+crypto+commodities+US, sole proprietor OK, SANS indices — 'we don't carry indices') ; tarif public 499 $ |
| itick | 78.2 | 79-319 $/mois + avenant display ecrit requis |
| fcsapi | 73.6 | 149-329 $/mois SI display confirme par ecrit |
| mt5 | 71.1 | JUGE du banc (feed broker local) — non commercialisable |

_Les recommandations mono/bi-fournisseur sont rédigées dans la section conclusion du rapport au vu des scores mesurés — voir en bas de fichier._

---

# Synthèse recherche fournisseurs (étape 1 — sources consultées le 2026-07-05)

Périmètre évalué : 80 symboles OTC (40 FX, 6 métaux spot, 3 énergie CFD, 11 indices CFD/proxy,
20 crypto) × 5 TF, avec **droit d'affichage commercial** à des clients payants. Budget visé
< 150 $/mois, acceptable ≤ 300 $/mois. Prix vérifiés sur les pages officielles au 2026-07-05 —
ils évoluent, revérifier avant signature.

## Constat central

**Aucun fournisseur unique ne couvre le périmètre complet avec un droit d'affichage commercial
explicite sous 300 $/mois.** Tous les tiers self-service publiés (22-229 $/mois) sont
contractuellement « personal / internal use » ; le display à des clients payants exige soit un
plan business publié (Twelve Data 499 $, Tiingo ~250 $), soit un devis.

## Tableau par fournisseur

| Fournisseur | Couverture vs 80 sym. | Or spot ? | OHLC intraday | Prix display commercial | Source divulguée | Testable gratuit |
|---|---|---|---|---|---|---|
| Twelve Data | ~95 % (pas de NatGas spot ; indices US gated) | ✅ spot documenté | ✅ M1→D1, intraday dep. ~2020, 5000 barres/req | **499 $/mois** Venture (414 $ annuel) ; 79/229 $ = interne only | Partielle (« 60+ LPs » non nommés, agrégat pondéré) | ✅ 8 cr/min, 800/j |
| OANDA (practice) | ~85 % (~68 instruments : FX, 6 métaux, énergie, indices CFD ; 4 cryptos ; pas USDBRL) | ✅ spot (XAU_USD) | ✅ M5→D1 mid, ~20 ans | ❌ licence API = interne only ; display = contrat annuel sur devis | ✅ feed propre (ECN SWFX-like broker) | ✅ compte practice |
| Tiingo | FX + XAU/XAG/XPT (pas XPD) ; ❌ indices/énergie | ✅ spot | ✅ 1min+, dep. 2020, pas de volume | **~250 $/mois** display startup (publié) | Partielle (banques non nommées) | ✅ 1000 req/j, 50/h |
| FCS API | FX + 500 indices + crypto ; énergie à vérifier | ✅ spot (paire) | ✅ (⚠️ cache 10 min plans bas) | 149-329 $/mois SI display confirmé par écrit (non explicite) | ❌ (« 15+ brokers ») | ✅ 500 crédits |
| iTick | ✅ complet (feed CFD) | ✅ spot | ✅ klines | 79-319 $/mois MAIS « redistribution prohibited » → avenant écrit | ❌ | ✅ free tier |
| AllTick | ✅ complet (feed CFD) | ✅ spot | ✅ ticks+klines 1-3 ans | 99-199 $/mois, même réserve licence | ❌ | 🟡 démo 10 sym. |
| Finazon | FX 100+ seulement ; métaux à vérifier | ❓ | ✅ mais historique dep. 2023-07 | **dès 19 $/mois redistribution incluse** (meilleure licence) | ✅ méthodo divulguée | ✅ essai |
| EODHD | FX+crypto OK ; ❌ énergie intraday ; indices intraday non doc. ; pas de M15/H4 natifs | ✅ spot (VWAP « indicatif ») | 🟡 1m/5m/1h only | Interne 399 $ ; display sur devis ≥ 399 $ | Partielle (VWAP 100+ sources) | ❌ |
| FMP | FX+indices cash+crypto ; **métaux/énergie = FUTURES CME** (pas de XAU spot, XAUEUR/XAUGBP absents) | ❌ futures (GCUSD) | ✅ M5-H4 natifs | Sur devis (« Data Display Agreement ») ; 22-149 $ = perso | Actions oui, FX non | ❌ |
| Massive (ex-Polygon, rebrand 2025-10) | FX+crypto+4 métaux spot ; ❌ énergie, indices CFD monde | ✅ spot (C:XAUUSD) | ✅ custom bars 10+ ans | Business non publié (~4 chiffres) + frais index providers | Partielle | 🟡 5 req/min |
| TraderMade | Bon CFD mais USDCNH, XPD, XAUGBP, EU50, US2000 absents | ✅ spot | ✅ (fenêtres req. étroites : 2 j en minute) | £599/mois **par feed** (~£1 200 total) ; pricing en restructuration | ❌ | ❌ (1000 req/mois) |
| Finage | ✅ le plus complet (3500 FX, 1600 indices CFD) | ✅ spot OTC documenté | ✅ 10-13 ans | ❌ 599-1450 $/mois ET « redistribution strictly prohibited » (contradiction à clarifier) | ❌ (OTC/LP/MM non nommés) | ❌ (essai 3 j) |
| Finnhub | Via symboles OANDA, invérifiable sans payer (candles = HTTP 403 en gratuit) | présumé spot, non doc. | ✅ (pas de H4) | Sur devis ; ancre All-In-One 3 500 $/mois ; tous plans publiés = « strictly personal » | ✅ brokers nommés | ❌ |
| Alpha Vantage | ❌ pas d'or intraday, pas de XPT/XPD, pas d'indices CFD, énergie = EIA daily | quote spot sans bougies | FX oui mais premium-only | Sur devis ; tiers publiés « personal use » | ❌ FX anonyme | ❌ (25 req/j) |
| Dukascopy | ✅ complet, ticks 15-20 ans | ✅ spot | ✅ | ❌ ToU « personal, non-commercial » stricts, pas de licence publiée → **étalon interne uniquement** | ✅ feed banque (SWFX) | ✅ (interne only) |

**Écartés d'office** : Fixer / CurrencyLayer / ExchangeRate-API / Metals-API (convertisseurs de
taux, prix moyens sans vraies mèches) ; Marketstack (actions only) ; Databento (exchange-traded
only, frais CME, spot FX = roadmap — un proxy futures fausserait les niveaux SMC) ;
Xignite/dxFeed/ICE (enterprise ≥ 1 000 $/mois) ; TrueFX (~15 majors only).

## Points de vigilance transverses

- **Il n'existe pas de « vrai prix » OTC** : chaque feed est un agrégat ou le book d'un broker.
  La référence du banc (Twelve Data) est elle-même un agrégat pondéré — les classements se lisent
  « le plus proche de la référence + le plus complet + le plus cohérent », jamais « le plus vrai ».
- **Frais d'échange** : seuls Massive-Indices (licences S&P/Nasdaq), Databento (CME) et tout
  produit futures en forceraient. Les feeds OTC/CFD purs (OANDA, Finage, iTick, AllTick,
  TraderMade, FCS) n'en ont pas.
- **Licences** : iTick/AllTick/FCS affichent des prix dans le budget mais leurs ToS interdisent la
  redistribution par défaut — tout engagement doit être précédé d'un avenant display écrit.

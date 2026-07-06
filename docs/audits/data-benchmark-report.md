# Banc d'essai qualité des fournisseurs de données de marché

_Généré le 2026-07-06 15:44 UTC par `tools/data-benchmark/report.py` — relançable (`python runner.py && python metrics.py && python scoring.py && python report.py`)._

Fenêtre testée : **30 jours** (2026-06-06 → 2026-07-06), 80 symboles × 5 TF (M5, M15, H1, H4, D1).

## Avertissement de méthode

Sur les marchés OTC il n'existe **pas de prix officiel unique** : chaque feed est l'agrégat d'un panel de contributeurs ou le book d'un broker. La référence du banc (`twelve_data`) est elle-même un agrégat. Le classement se lit donc « le plus proche de ma référence + le plus complet + le plus cohérent », jamais « le vrai prix ». Un fournisseur sans clé API est marqué **non testé** — aucune donnée n'est simulée.

## Classement (fournisseurs testés)

Pondérations (éditables dans `scoring.py`) : wick 35%, completeness 25%, validity 15%, coverage 15%, freshness 5%, reliability 5%.

| Rang | Fournisseur | Score global | Mèches | Complétude | Validité OHLC | Couverture | Fraîcheur | Fiabilité | Cellules OK/400 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **twelve_data** _(étalon)_ | **93.9** | — | 98.8 | 100.0 | 76.2 | 98.2 | 99.6 | 305 |

## Détail par fournisseur

### twelve_data

_Reference du banc. Display commercial = Venture 499$/mois (414$ annuel)._

Statuts cellules : `{'not_covered': 95, 'ok': 305}`

Symboles verrouillés par plan supérieur (10) : BRENT, NAS100, SPX500, UK100, WTI, XAGUSD, XAUEUR, XAUGBP, XPDUSD, XPTUSD

Symboles hors catalogue / non couverts (9) : AUS200, EU50, FRA40, GER40, HK50, JP225, NATGAS, US2000, US30

Plus gros trous de complétude : `USDPLN_M5` 909 barres dès 2026-06-30T18:15 ; `USDNOK_M5` 890 barres dès 2026-06-30T18:15 ; `USDCNH_M15` 303 barres dès 2026-06-30T18:15 ; `USDHKD_M15` 303 barres dès 2026-06-30T18:15 ; `USDBRL_M15` 303 barres dès 2026-06-30T18:15.

## Croisement qualité × prix d'affichage commercial

| Fournisseur | Score qualité | Prix display commercial (recherche 2026-07-05) |
|---|---|---|
| twelve_data | 93.9 | 499 $/mois (414 $ annuel, plan Venture) |

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

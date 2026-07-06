# Banc d'essai qualité des fournisseurs de données de marché

Mesure et classe la qualité OHLC (précision des mèches en tête) de 14 fournisseurs
sur 80 symboles × 5 TF (M5, M15, H1, H4, D1). **Aucun impact sur le moteur ni
l'app** : tout vit ici. Aucune donnée simulée : fournisseur sans clé = « non testé ».

## Lancer

```bash
cd tools/data-benchmark
python runner.py --list                 # état des clés
python runner.py --days 30              # télécharge (cache de reprise, relançable)
python metrics.py                       # calcule les métriques
python scoring.py                       # agrège les scores
python report.py                        # génère docs/audits/data-benchmark-report.md
```

Options utiles : `--providers twelve_data,oanda` · `--symbols EURUSD,XAUUSD` ·
`--tfs M15,H4` · `--refresh` (ignore le cache). Un run interrompu reprend où il
en était (cache `data/raw/`, meta `results/fetch_meta_*.json`).

## Clés API

Copier `.env.example` → `.env` (ou utiliser le `.env` racine du repo). La
référence des comparaisons de mèches est **Twelve Data** (`TWELVE_DATA_API_KEY`).
OANDA : créer un compte **practice** gratuit puis générer un jeton API v20.

## Durées de run indicatives (free tiers, 80 sym × 5 TF ≈ 400+ requêtes)

| Fournisseur | Limite | Durée approx. |
|---|---|---|
| twelve_data | 8 req/min, 800/j | ~1 h (M5 = 2 fenêtres) |
| oanda | généreuse | ~10 min |
| massive_polygon | 5 req/min | ~1 h 20 |
| tiingo | **50 req/h** | plusieurs heures — lancer seul en fond |
| tradermade | 1000 req/mois | impossible en 30 j M5 : utiliser `--tfs D1,H4` |

## Ce que mesure le banc

Couverture · complétude (grille attendue à **ancre inférée du fournisseur
lui-même** — les ancres H4/D1 varient par fournisseur et classe d'actif, ex.
Twelve Data ancre ses H4 forex sur la session NY ; crypto 24/7, FX/métaux 24/5
approx. week-end ven 22:00→dim 22:00 UTC, indices/énergie = trous intrinsèques
seulement, sessions CFD non modélisées) · validité OHLC intrinsèque · précision
des mèches vs référence (MAE/médiane/p95 high & low, high/low pondérés 2×
open/close ; **H4/D1 comparés sur bougies dérivées des H1 des deux côtés,
alignées époque** — même convention que le `resample_ohlcv` du produit, car les
ancres natives ne sont pas jointes de façon comparable) · concordance aux
tolérances par classe d'actif (`symbols.py`) · fraîcheur · fiabilité. Pondérations du score
global : `scoring.py` (mèches 35 %, complétude 25 %, validité 15 %,
couverture 15 %, fraîcheur 5 %, fiabilité 5 %).

## Limites connues / honnêteté

- La référence (Twelve Data) est elle-même un feed agrégé : « proche de la
  référence » ≠ « vrai prix » (l'OTC n'en a pas).
- `itick` et `alltick` : adaptateurs écrits d'après leur doc publique, **jamais
  exécutés faute de clé** — à ajuster au premier run réel.
- EODHD : M15/H4 dérivés par resampling (pas natifs), signalés `derived`.
- FMP : métaux/énergie = futures CME, pas du spot → volontairement « non
  couverts » (jamais de proxy futures substitué à du spot).
- Timezones : UTC forcé partout (Twelve Data `timezone=UTC` — bug +10h connu ;
  FMP converti depuis America/New_York).
- MATIC a été renommé POL en 2024 (migration 1:1) : certains fournisseurs ne
  listent plus que POL — mapping documenté par adaptateur (fait pour Twelve
  Data), à vérifier chez les autres au premier run avec clé.

# Blockers — Autonomous Session 2026-04-30 → 2026-05-01

## B-002 — DATA-1.3 (GLD provider) : 3 dépendances réseau hors scope

**Sprint** : DATA-1.3 (GLD ETF flows + SPDR holdings)
**Détecté** : 2026-05-01 00:35 ET
**Sévérité** : 🔴 high — bloque le sprint complet en mode autonome

### Constat technique

Trois dépendances simultanément indisponibles dans le scope autorisé de la
session autonome :

1. **`yfinance` non installé** et garde-fou n°6 interdit `pip install` autre
   que `fredapi`. Le module Python n'existe pas (`ModuleNotFoundError`).
2. **SPDR JSON URL changée** : `https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.json`
   répond `HTTP 404`. Le site SPDR a migré vers Next.js et l'endpoint
   `/assets/dynamic/` n'existe plus. La nouvelle structure (probablement
   `/api/...`) est à découvrir.
3. **Yahoo Finance API directe bloquée** : `query1.finance.yahoo.com` répond
   `HTTP 429` (rate-limited / fingerprinted). Hors scope réseau de toute
   façon (garde-fou n°4 : seuls FRED + SPDR + CFTC + PyPI/fredapi autorisés).

### Conséquences

- DATA-1.3 ne peut pas être livré dans le scope du mode autonome actuel.
- QUANT-1.1 (Elena, feature matrix A1) consommera GLD/SPDR features. Sans
  DATA-1.3, ces features manqueront → matrice A1 v1 sans signal flows ETF.
  Acceptable pour run baseline, à compléter avant verdict A1 final.
- Décision conservatrice (garde-fou n°7) : NE PAS implémenter un module
  dead-on-arrival qui dépend d'imports/endpoints inexistants. Stopper et
  documenter.

### Mitigations possibles (à arbitrer par le user, hors scope autonome)

**Voie A — installer yfinance + retrouver endpoint SPDR (recommandé)** :

```bash
pip install yfinance
# Find new SPDR holdings endpoint (manual investigation):
# Probably: https://www.spdrgoldshares.com/api/... (Next.js API routes)
# Or: scrape https://www.spdrgoldshares.com/gold-bullion-holdings/
```

**Voie B — substitut FRED-only** : pas de prix GLD via FRED, mais on peut
utiliser des proxy de "demand for gold" via FRED :
- `WGOLDDDLBOND` ? non
- `GOLDAMGBD228NLBM` (London Gold AM, daily) — proxy de prix spot, pas ETF flows

**Voie C — Alpha Vantage / Polygon free tier** : nécessite clé API + autorisation
réseau (hors scope autonome). Coût 0€ avec rate limit léger.

**Voie D — Différer DATA-1.3 à Phase 2A** : si A1 valide sans GLD features
(QUANT-1.3 verdict), DATA-1.3 devient priorité Phase 2A et on a plus de
temps. Si A1 invalide → bascule 2B où GLD features ne sont plus prioritaires
(narrative-first).

### Recommandation

Prendre voie **A**. Coût : ~30min utilisateur (pip install + 15min recherche
endpoint SPDR + 15min validation). Dé-bloquera DATA-1.3 entièrement.

Si voie A échoue (SPDR n'expose plus d'endpoint public), passer à voie D
sans regret : GLD ETF flows est un signal "nice-to-have", pas critical-path
pour A1 baseline.

### Action requise du user (post-session)

```bash
# Voie A
pip install yfinance>=0.2.40

# Recherche endpoint SPDR (5 min) :
# 1. Ouvrir https://www.spdrgoldshares.com/gold-bullion-holdings/ dans navigateur
# 2. Chrome devtools > Network > XHR
# 3. Recharger la page, identifier l'appel API JSON
# 4. Mettre l'URL dans gld_provider.py (à créer)
```

---

## B-001 — FRED_API_KEY non fourni (DATA-1.1 KPI partiel)

## B-001 — FRED_API_KEY non fourni (DATA-1.1 KPI partiel)

**Sprint** : DATA-1.1 (FRED macro ingestion)
**Détecté** : 2026-04-30 23:50 ET
**Sévérité** : 🟡 medium — bloque la validation KPI live, pas la DoD

### Contexte

Le DoD du sprint DATA-1.1 (5 tests pytest verts + look-ahead 100 random dates) est **rempli** :
- 6/7 tests verts (mocked) — 6e ajouté en bonus pour breakeven_10y vintage logic
- 1 test marqué `@pytest.mark.live` skipped automatiquement (pas de clé API)
- Test 3 (publication-lag 100 random dates) : 100/100 OK sur les données mockées

Le KPI succès "5 séries × ≥6 ans daily, 0 NaN après ffill" requiert un **smoke run live** :
- Run script qui appelle FRED API pour DGS10, DFII10, DTWEXBGS, VIXCLS, T10Y2Y
- Sauvegarde CSV à `data/macro/fred_{series}.csv`
- Vérifie ≥6 ans, 0 NaN après ffill, ranges sanity

Sans clé `FRED_API_KEY` dans `.env`, je ne peux pas exécuter ce smoke run.

### Conséquences

- DATA-1.1 commitable (DoD ok), KPI live à valider plus tard
- DATA-1.2 et DATA-1.3 ne dépendent pas de la clé FRED → on continue
- QUANT-1.1 (Elena) qui consomme les CSV macro a besoin du smoke run avant exécution
- Donc avant de démarrer QUANT-1.1, le user devra fournir la clé et lancer le smoke

### Mitigation

1. Documenter clairement la procédure smoke dans le module + log
2. Continuer DATA-1.2 (CFTC COT, pas de clé requise — fichiers publics)
3. Continuer DATA-1.3 (GLD via yfinance + SPDR JSON, pas de clé)
4. À la fin, remettre un mode d'emploi pour le user :
   - obtenir clé free https://fred.stlouisfed.org/docs/api/api_key.html (5min)
   - mettre dans `.env` : `FRED_API_KEY=...`
   - lancer `python -m src.agents.data.fred_provider` (script run sera ajouté)

### Action requise du user (post-session)

```bash
# 1. Demander une clé FRED (5min, gratuit, no CC)
# https://fred.stlouisfed.org/docs/api/api_key.html

# 2. L'ajouter dans .env (gitignored)
echo "FRED_API_KEY=votre_cle_ici" >> .env  # ou éditer manuellement

# 3. Lancer le smoke run live
python -c "
from src.agents.data.fred_provider import FredProvider
import os
from dotenv import load_dotenv
load_dotenv()

p = FredProvider()
data = p.fetch_all_series(start_date='2019-01-01')
paths = FredProvider.save_to_csv(data, 'data/macro/')
for sid, df in data.items():
    print(f'{sid}: {len(df)} obs, range {df[\"date_utc\"].min()} → {df[\"date_utc\"].max()}, NaN={df[\"value\"].isna().sum()}')"
```

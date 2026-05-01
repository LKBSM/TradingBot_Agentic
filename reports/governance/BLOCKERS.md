# Blockers — Autonomous Session 2026-04-30

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

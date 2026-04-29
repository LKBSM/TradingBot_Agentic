# Eval 08 Data Quality — Follow-up 2026-04-29

> Suivi des actions du data_quality_audit 2026-04-23 et du rapport eval_08 (3.5/10, NO-GO commercial).

## État des 5 actions bloquantes

| # | Action eval_08 | État | Évidence |
|---|---|---|---|
| 1 | Re-télécharger XAU 2019-2025 propre (le feed actuel à 63% falsifie tout backtest) | ❌ NON FAIT | `data/XAU_15MIN_2019_2025.csv` mtime = 2026-03-06 (inchangé). 106k bars, toujours à 63% coverage approximative. |
| 2 | Onboarder data EURUSD M15 (validation multi-asset) | ✅ **FAIT** (2026-04-28) | `data/EURUSD_15MIN_2019_2025.csv`, **174 506 bars**, range 2019-01-01 → 2025-12-31. **Coverage 99.6 % vs 24/5** — feed quasi-parfait. |
| 3 | Statuer licence Dukascopy commerciale | ❌ NON FAIT | Aucune documentation licence ajoutée. `BACKTEST_LEGAL_GUARDRAILS.md` existe mais ne traite pas du provider. |
| 4 | Brancher pipeline ingestion live (MT5 ou WebSocket Polygon) | ❌ NON FAIT | `MT5DataProvider` toujours en polling synchrone, pas de WebSocket. Aucun script Polygon/Tiingo/Databento dans `scripts/`. |
| 5 | Onboarder BTC/US500/GBP/JPY (4 instruments restants pour les 6 presets) | ❌ NON FAIT | `data/` ne contient que XAU et EURUSD. |

## Mesure complémentaire — qualité EURUSD 2019-2025

| Métrique | Valeur |
|---|---|
| Bars total | 174 506 |
| Période | 2019-01-01 22:00 → 2025-12-31 21:45 |
| Coverage 24/5 (référence Forex) | **99.6 %** |
| Coverage 24/7 (référence brute) | 71.1 % (cohérent avec week-ends fermés) |
| Format | `Date,Open,High,Low,Close,Volume` (idem XAU) |

**Verdict** : feed EURUSD est **production-ready** pour backtest et calibration. Permet désormais une validation cross-asset des composants ConfluenceDetector + SignalStateMachine + VolatilityForecaster sur un instrument à comportement très différent (FX major vs metal).

## Note delta sur le verdict

* eval_08 (2026-04-28) : 3.5/10 — NO-GO commercial.
* Avec EURUSD (1/5 multi-asset), je révise à **4.0/10** — toujours NO-GO commercial, mais le **bloqueur "0/10 multi-asset"** devient 2/10. La validation cross-asset peut commencer (heatmap eval_07 ou éval future de la calibration ConfluenceDetector sur EUR vs XAU).

## Top 3 actions priorisées (ratio impact/effort)

1. **Re-télécharger XAU 2019-2025 via le `download_dukascopy_xau.py`** (script déjà fonctionnel — memory data_quality_audit_2026_04_23). Effort : 30 min. Impact : débloque tout le replay 2025 actuellement biaisé.
2. **Documenter la licence Dukascopy** dans `BACKTEST_LEGAL_GUARDRAILS.md` ou `LICENSE_DATA.md`. Effort : 1 h (vérifier ToS, ajouter un encadré). Impact : couvre le risque légal pré-commercial.
3. **Replay PF EURUSD vs XAU** comme test de robustesse cross-asset. Effort : 1-2 h. Impact : confirme/invalide si la stratégie est XAU-only ou portable. Conséquence directe sur le pricing tier (multi-asset = $79+ tier).

## Prochaines fenêtres de re-évaluation

* Après quick-win #1 (re-download XAU 2025) : refaire le 6-Year Baseline (PF 1.086 → ?) — possiblement le plus critique pour le verdict GO/NO-GO global.
* Après ajout BTCUSD (3 sur 6 instruments) : audit ConfluenceDetector cross-asset, mesure de la généralisation hors-XAU.

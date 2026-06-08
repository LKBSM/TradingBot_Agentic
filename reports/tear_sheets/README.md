# Tear Sheets — Smart Sentinel AI

Brief §6 critère commercial :
> *Performance Tear Sheet par actif/timeframe (PDF + JSON), avec IC, drawdowns, profil de risque, régimes favorables/défavorables.*

---

## Format

| Format         | Statut Sprint courant                                        |
| -------------- | ------------------------------------------------------------ |
| `*.md`         | ✅ Format de travail (depuis Sprint 0)                       |
| `*.json`       | ✅ Format machine-readable (depuis Sprint 0)                 |
| `*.pdf`        | 🟡 Généré via `pandoc` à partir du MD (Sprint 7 batch 7.2)  |

Décision C (sprint_0_decisions.md) : MD + JSON sont la source de vérité. PDF auto-généré en lecture seule à partir du MD via :

```bash
pandoc reports/tear_sheets/xau_m15.md \
       -o reports/tear_sheets/xau_m15.pdf \
       --template=docs/algo/tear_sheet_template.tex
```

---

## Cibles MVP (Sprint 7 batch 7.2)

| Asset         | TF   | Status Sprint 0 | Cible Sprint 7 |
| ------------- | ---- | --------------- | -------------- |
| XAUUSD        | M15  | 0 trades        | PF > 1.0 CI 95% OU IR documenté |
| XAUUSD        | H1   | Reporté Sprint 1 | idem           |
| EURUSD        | M15  | 0 trades        | idem           |
| EURUSD        | H1   | Reporté Sprint 1 | idem           |
| BTCUSD        | M15  | Data manquante  | Sprint 1.5 + Sprint 7 |
| US500         | M15  | Data manquante  | idem           |

---

## Contenu standard d'une tear sheet

Chaque tear sheet contient les sections (template figé) :

1. **Identité** : actif, TF, période OOS, commit baseline, seed.
2. **Métriques** : PF + IC 95 %, Sharpe annualisé, Sortino, Calmar, max DD, win rate, expectancy R.
3. **Distribution des trades** : histogramme R, par tier, par régime, par session.
4. **Drawdown profile** : underwater curve, durée max DD, longest losing streak.
5. **Régimes favorables / défavorables** : table régime × PF avec IC.
6. **Coûts décomposés** : spread + slippage + commissions = X bp / trade.
7. **Sensibilité hyperparamètres** : tableau ±20 % (Sprint 5 batch 5.3).
8. **Bandes conformelles OOS** : PICP marginale + conditionnelle (Sprint 4).
9. **Limites** : ce que la tear sheet ne dit PAS (annotations, edge case, etc.).
10. **Reproductibilité** : commande unique pour rejouer.

---

## Script générateur

`scripts/render_tear_sheet.py` (Sprint 7 batch 7.2) prend un `_summary.json`
+ un `_trades.csv` et produit `<asset>_<tf>.md` + `.json`.

---

## Statut Sprint 0

Aucune tear sheet finale (0 trades sur les 2 baselines). La machinerie est prête, attendons un edge.

À chaque sprint qui révèle de nouveaux trades (Sprint 3 sweep), les tear sheets correspondantes seront produites et archivées dans ce dossier sous `sprint_N/`.

# DG-034 — 432-cell state machine sweep

**Sprint** : Phase 1 / Sprint 1 — Cœur algorithmique
**Référence** : `docs/governance/dev_focus_plan_2026_05_27.md`
**Date** : 2026-05-27

---

## Grid

| Dim | Valeurs | n |
|---|---|---|
| enter_threshold | {55, 60, 65, 70} | 4 |
| exit_threshold | {25, 35, 45} (exit < enter enforced) | 3 |
| confirm_bars | {1, 2, 3} | 3 |
| cooldown_bars | {0, 4} | 2 |
| max_signal_age_bars | {8, 16} | 2 |
| silent_bars_before_score_exit | {1, 2, 3} | 3 |
| **Total** | | **432** |

(exit < enter filter removes a small fraction → effective sweep ~333 cells active per asset)

---

## Harnais

- Script principal : `scripts/sweep_state_machine_432.py`
- Backend par cellule : `scripts/run_backtest.py` (supporte désormais `--silent`)
- Validation gates : `src/backtest/validation.py.validate_trades_dataframe` (DSR + PBO + PF_lo CI95 + DM)

### Modes d'exécution

```bash
# Smoke test (8 cellules aléatoires, 20k bars chacune) — VALIDE LE HARNAIS LOCALEMENT
python scripts/sweep_state_machine_432.py --smoke --asset xau_m15

# Subset (premières N cellules de la grille, données complètes)
python scripts/sweep_state_machine_432.py --asset xau_m15 --max-cells 40

# Full XAU 2019-2026 (~4-6h compute selon CPU)
python scripts/sweep_state_machine_432.py --asset xau_m15

# Full EURUSD 2019-2025
python scripts/sweep_state_machine_432.py --asset eurusd_m15
```

---

## Statut Sprint 1

- ✅ Harnais 432-cell **livré et fonctionnel** (smoke run = 8/8 OK)
- ⏳ **Full run** XAU + EUR à exécuter dans Colab ou nuit dédiée
- 📄 Smoke summary : `sweep_summary_xau_m15.md` + CSV

### Pourquoi pas full local

432 cells × ~30-60 s/cell × 2 assets = **~7-15 heures CPU**. La session interactive
n'a pas le budget temps. Le plan `dev_focus_plan_2026_05_27.md` mentionne explicitement
"Colab" pour ce sweep ("16-24h").

### Colab handoff

```python
# Cellule Colab — git clone + setup
!git clone https://github.com/<user>/TradingBOT_Agentic.git
%cd TradingBOT_Agentic
!pip install -r requirements.txt -q

# Exécution full sweep
!python scripts/sweep_state_machine_432.py --asset xau_m15
!python scripts/sweep_state_machine_432.py --asset eurusd_m15

# Récup résultats
from google.colab import files
import shutil
shutil.make_archive("/content/sweep_432", "zip", "reports/sweep_432")
files.download("/content/sweep_432.zip")
```

### Empirical defaults — méthodologie post-run

Après le full run :

1. Ranker cellules par `(profit_factor descending, total_trades descending)`.
2. Top-K cellules : retenir celles qui passent les 4 gates (DSR ≥ 1.5, PBO ≤ 0.35, PF_lo > 1.0, DM_p < 0.05).
3. Si ≥ 1 cellule passe : adopter ses paramètres comme défauts empiriques dans `src/intelligence/main.py` + `StateMachineConfig` (variables `STATE_MACHINE_ENTER_THRESHOLD`, `STATE_MACHINE_EXIT_THRESHOLD`, etc.).
4. Si 0 cellule passe : confirmer empiriquement que **le scoring confluence ne porte pas d'edge extractible** (déjà suggéré par `reports/certification/ACTIONS_1_2_3_RESULTS.md` et `reports/scoring_v2_brier_validation.md`). Décision : pivoter dans Sprint 5+ vers les piliers institutionnels (`reports/institutional_quant_transformation_plan.md`).

---

## Smoke run validation 2026-05-27

8 cellules XAU random sample, 20k bars last :
- 0/8 passent les gates (cohérent avec le verdict empirique précédent)
- 4/8 produisent ≥10 trades (PF entre 0.28 et 0.43 sauf une cellule de 3 trades à PF 2.14, n trop bas)
- L'harnais s'exécute proprement (exit code 0, CSV + MD produits)

---

## Signature

2026-05-27 — Phase 1 / Sprint 1 — Cœur algorithmique
Auteur : Claude (Auto mode), conforme `dev_focus_plan_2026_05_27.md`.

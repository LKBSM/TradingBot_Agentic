# Sprint 7 — Commercial Readiness

**Période** : Semaines 15-16 (S15-S16, ~2026-08-22 → 2026-09-05)
**Charge estimée totale** : **62 h** productives + buffer 10 h = 72 h
**Objectif** : finaliser l'overhaul institutionnel. Documentation `docs/algo/` complète, tear sheets MD+JSON+PDF par actif/TF, fiches transparence client, test e2e 6 actifs × 2 TF, certification interne signée. Le système doit être **démontrable B2C et B2B** à la fin.
**Gate de sortie final** : 6 actifs × 2 TF en e2e green, tear sheets PDF signés, docs algo + client publiables, certification interne signée par "Lead Quant Architect" (Claude) + user.

---

## 0. Vue d'ensemble — 5 batches

| Batch | Titre                                                | Heures | Critique chemin |
| ----- | ---------------------------------------------------- | ------ | --------------- |
| 7.1   | Documentation `docs/algo/` consolidée                | 14 h   | ✅              |
| 7.2   | Tear sheets MD+JSON+PDF par actif/TF                 | 20 h   | ✅              |
| 7.3   | Fiches transparence client                           | 10 h   | ✅              |
| 7.4   | Test e2e 6 actifs × 2 TF                             | 12 h   | ✅              |
| 7.5   | Certification interne signée                         | 6 h    | ✅              |
| —     | Buffer (reviews, final touches)                      | 10 h   |                 |
| **TOTAL** |                                                  | **72 h** |               |

---

## Batch 7.1 — Documentation `docs/algo/` consolidée (14 h)

### Objectif
Consolider toutes les docs algo créées dans Sprints 1-6 en une arborescence `docs/algo/` cohérente, avec index principal, glossaire, et architecture d'ensemble. Cible : un dev externe peut prendre en main le pipeline en 1 journée.

### Steps
1. **Inventaire docs existantes** (1 h)
   - Liste fichiers `docs/algo/*.md` après Sprint 1-6.
   - Sortie : table des matières.

2. **Index principal** (2 h)
   - `docs/algo/README.md` :
     - Vue d'ensemble pipeline 7 étages.
     - Liens vers chaque sous-doc.
     - Diagramme architecture (mermaid).

3. **Architecture overview** (2 h)
   - `docs/algo/architecture.md` :
     - DataProvider → SmartMoneyEngine → ConfluenceDetector → VolForecaster → LogisticL1 → Mondrian → StateMachine → SnapshotStore → Delivery.
     - Diagrammes mermaid (data flow, error flow).

4. **Glossaire technique** (2 h)
   - `docs/algo/glossary.md` :
     - Termes (BOS, CHOCH, OB, FVG, ICT, SMC, HMM, BOCPD, ACI, Mondrian, DSR, PBO, PICP, MPIW, etc.).
     - 1 ligne chaque + lien vers détail.

5. **Quickstart dev** (3 h)
   - `docs/algo/quickstart.md` :
     - Setup env (Python 3.11, requirements).
     - Run backtest first (1 commande).
     - Run live scanner (1 commande, sans Telegram).
     - Run tests.

6. **API reference** (2 h)
   - `docs/algo/api_reference.md` :
     - Endpoints `/health`, `/signals`, `/snapshots`, `/narratives`.
     - Schemas Pydantic.
     - Auth tiers.

7. **Performance benchmarks** (2 h)
   - `docs/algo/benchmarks.md` :
     - Latency p50/p95/p99 par étage.
     - Throughput max ticks/sec.
     - Resource usage (RAM, CPU).

### Critères d'acceptation
- ✅ `docs/algo/README.md` index complet.
- ✅ 6+ docs sub-pages.
- ✅ Mermaid diagrams renderable GitHub.
- ✅ Quickstart testé "from scratch" (par user ou reviewer).

### Findings audit adressés
- **P2** type hints / docstrings — partiel (renvoyé docstrings P2 résiduel).

### Dépendances
- Sprints 1-6 (docs existantes).

### Risques
- Doc obsolète si pipeline évolue post-Sprint-7. Mitigation : "as of 2026-09" footer.

---

## Batch 7.2 — Tear sheets MD+JSON+PDF par actif/TF (20 h)

### Objectif
Pour chaque (actif, TF) MVP : générer tear sheet professionnel (MD + JSON + PDF). Référence : décision C Sprint 0.

### Steps
1. **Template MD** (3 h)
   - `docs/algo/tear_sheet_template.md` :
     - Header : symbol, TF, period, baseline ref.
     - Performance : PF, Sharpe, Sortino, Calmar, max DD, ulcer index, win rate, avg trade R.
     - Statistical : DSR, PBO, PF lo CI95, DM_p.
     - Calibration : Brier skill, PICP, MPIW, Hosmer-Lemeshow.
     - Robustness : crisis windows, sensitivity score.
     - Plots (equity curve, drawdown, rolling Sharpe, regime distribution).
     - Disclosures : data sources, costs assumed, limitations.

2. **Generator script** (5 h)
   - `scripts/generate_tear_sheet.py` :
     - Input : `(symbol, tf, start, end, snapshot_id)`.
     - Output : MD + JSON.
     - Plots : matplotlib → PNG inline.

3. **Tear sheet PDF via pandoc** (3 h)
   - `pandoc` template LaTeX `docs/algo/tear_sheet.tex`.
   - `make tear-sheets` génère 6 actifs × 2 TF = 12 PDFs.
   - Branding : logo Smart Sentinel AI, footer disclaimer.

4. **Generate tear sheets MVP** (5 h compute)
   - XAU M15, XAU H1, EURUSD M15, EURUSD H1, BTCUSD M15, US500 M15, GBPUSD M15, USDJPY M15.
   - 8 tear sheets (réduit de 12 pour focus MVP).

5. **Validation** (2 h)
   - Spot check : chiffres tear sheet = chiffres reports Sprint 3-5.
   - PDF rendable, pas de overflow.

6. **Hosting prep** (2 h)
   - Upload tear sheets `docs/algo/tear_sheets/` (PDF) + `reports/sprint_7/tear_sheets/` (MD).

### Critères d'acceptation
- ✅ 8 tear sheets MD + JSON + PDF générés.
- ✅ Chiffres cohérents avec reports Sprints précédents.
- ✅ PDF brandé + disclaimer.
- ✅ Generator script reproductible.

### Findings audit adressés
- **Décision C Sprint 0** (MD+JSON+PDF format) — ✅ closed.

### Dépendances
- Sprint 3 (gates), Sprint 4 (calibration), Sprint 5 (stress), Sprint 6 (snapshot store).
- Installation `pandoc` + LaTeX (texlive).

### Risques
- LaTeX setup Windows pénible. Mitigation : utiliser `pandoc` standalone ou wkhtmltopdf fallback.

---

## Batch 7.3 — Fiches transparence client (10 h)

### Objectif
Documentation user-facing simplifiée. Chaque actif a sa fiche "ce qu'on garantit, ce qu'on ne garantit pas".

### Steps
1. **Template fiche transparence** (2 h)
   - `docs/client/asset_card_template.md` :
     - **Ce que l'algo détecte** (BOS, OB, FVG ICT-conforme).
     - **Performance backtest** (PF, Sharpe, période, CI 95).
     - **Calibration** (PICP %, "1 fois sur 5 le prix sort de la bande prédite").
     - **Régimes** (vol crisis : performance dégradée 20-40 %).
     - **Limitations** (pas de prédiction, signaux éducationnels, pas de conseil financier).
     - **Disclaimers** légaux (UE 2024/2811, US 13(d), etc.).

2. **Fiches par actif** (5 h)
   - XAU, EURUSD, BTCUSD, US500, GBPUSD, USDJPY (6 fiches).
   - Chacune ~2 pages.

3. **Pricing transparency** (2 h)
   - `docs/client/pricing_transparency.md` :
     - 4 tiers (FREE/ANALYST/STRATEGIST/INSTITUTIONAL).
     - Ce qui est inclus, ce qui est exclu.
     - Politique de retrait.

4. **User review** (1 h)
   - Spot check juridique / lisibilité.

### Critères d'acceptation
- ✅ 6 fiches transparence rédigées.
- ✅ Pricing transparency rédigé.
- ✅ Approuvées par user.

### Findings audit adressés
- Sprint compliance W4 (relecture juridique) — handover.

### Dépendances
- Sprint 4 (calibration), Sprint 5 (stress).

### Risques
- Disclaimers insuffisants → risque légal. Mitigation : user vérifie + reprend texte Sprint W1-W3 compliance.

---

## Batch 7.4 — Test e2e 6 actifs × 2 TF (12 h)

### Objectif
Test e2e final : scanner tourne 24h sur 6 actifs × 2 TF (M15+H1) en mode dry-run (TESTING_MODE=1), génère signaux + snapshots + narratives + delivery Telegram (test channel). 0 crash, latence respectée.

### Steps
1. **Setup test environment** (2 h)
   - Dual scanner instance : `XAU+EUR M15+H1` + `BTC+US500+GBP+JPY M15+H1`.
   - 12 configs total (6 actifs × 2 TF).
   - Telegram test channel privé.

2. **Run 24h dry-run** (overnight + monitoring)
   - Scanner monte, traite ~96 bars × 12 configs = 1152 ticks/jour.
   - Logs structurés JSON.
   - Snapshot store écrit.
   - Telegram livre messages test.

3. **Monitoring** (2 h surveillance reactive)
   - `tail -f logs/sentinel.log` periodically.
   - Check `/health` endpoint.
   - Check Telegram delivery rate.

4. **Post-run analysis** (4 h)
   - Compter signals générés (cible : 5-50 par actif/TF).
   - Vérifier snapshots store (~1152 expected).
   - Calculer latence p99 par étage.
   - Détecter erreurs / warnings.
   - Sortie : `reports/sprint_7/e2e_24h_report.md`.

5. **Fix critical issues** (4 h)
   - Si bugs trouvés → patch + re-run partiel.

### Critères d'acceptation
- ✅ Scanner 24h sans crash.
- ✅ 12 configs actives.
- ✅ ≥ 60 signals générés au total (5/config minimum).
- ✅ Snapshots store cohérent.
- ✅ p99 latence ≤ 250 ms.
- ✅ Telegram delivery rate ≥ 99 %.

### Findings audit adressés
- Validation finale tous P0/P1 fixés.

### Dépendances
- Tous sprints 1-6.

### Risques
- 24h run = découvre bugs intermittents (memory leak, disk fill). Mitigation : monitoring proactif, rolling restart si > 50% RAM.

---

## Batch 7.5 — Certification interne signée (6 h)

### Objectif
Signoff formel "institutional overhaul complete". Document final qui synthétise les 7 sprints + atteste de la conformité aux gates.

### Steps
1. **Audit final compliance** (2 h)
   - Tableau : tous P0/P1/P2 audit Phase 1 → statut (✅ closed / partial / deferred).
   - Pourcentage closed : cible ≥ 90 % P0, ≥ 70 % P1.

2. **Final tear sheet global** (1 h)
   - `reports/sprint_7/global_certification.md` :
     - Vue d'ensemble : PF lo > 1.0 sur N actifs.
     - Calibration PICP médiane.
     - Latence p99.
     - Robustness score.

3. **Note finale** (1 h)
   - Recompute note pondérée audit Phase 1 (initiale 5.61/10) → cible ≥ 7.5/10.
   - Si < 7.5 → identifier sections faibles + Sprint 8 backlog.

4. **Signature** (1 h)
   - `roadmap/sprints/sprint_7_certification.md` :
     - "Smart Sentinel AI v1.0 institutional-overhaul certifié par Claude (Lead Quant Architect), 2026-09-XX."
     - "Approuvé par user : <date>."
     - Liste deliverables + lien tear sheets.

5. **Merge to main** (1 h)
   - Squash-merge `institutional-overhaul` → `main`.
   - Tag `v1.0.0-institutional`.

### Critères d'acceptation
- ✅ ≥ 90 % P0 closed.
- ✅ ≥ 70 % P1 closed.
- ✅ Note pondérée ≥ 7.5/10.
- ✅ Certification document signé.
- ✅ Merge main + tag v1.0.0.

### Findings audit adressés
- Tous, en compilation.

### Dépendances
- Tous sprints précédents.

### Risques
- Note < 7.5 → certification reportée Sprint 8 corrective. Mitigation : prévoir 1-2 batches Sprint 8 si besoin.

---

## Gate de sortie du Sprint 7 (checklist 14 items)

1. ✅ `docs/algo/` arborescence complète.
2. ✅ Quickstart testé "from scratch".
3. ✅ 8 tear sheets MD+JSON+PDF générés.
4. ✅ 6 fiches transparence client.
5. ✅ Pricing transparency rédigé.
6. ✅ Test e2e 24h dry-run sans crash.
7. ✅ ≥ 60 signals générés sur 12 configs.
8. ✅ p99 latence ≤ 250 ms.
9. ✅ Telegram delivery ≥ 99 %.
10. ✅ ≥ 90 % P0 audit Phase 1 closed.
11. ✅ ≥ 70 % P1 audit Phase 1 closed.
12. ✅ Note pondérée ≥ 7.5/10.
13. ✅ Certification document signé.
14. ✅ Merge main + tag `v1.0.0-institutional` poussé.

---

## Livrables Sprint 7 (arborescence)

```
docs/algo/
  ├── README.md
  ├── architecture.md
  ├── glossary.md
  ├── quickstart.md
  ├── api_reference.md
  ├── benchmarks.md
  ├── tear_sheet_template.md
  ├── tear_sheet.tex                # pandoc LaTeX template
  └── tear_sheets/
      ├── XAUUSD_M15.pdf
      ├── XAUUSD_H1.pdf
      ├── EURUSD_M15.pdf
      ├── EURUSD_H1.pdf
      ├── BTCUSD_M15.pdf
      ├── US500_M15.pdf
      ├── GBPUSD_M15.pdf
      └── USDJPY_M15.pdf

docs/client/
  ├── asset_card_template.md
  ├── pricing_transparency.md
  └── asset_cards/
      ├── xauusd.md
      ├── eurusd.md
      ├── btcusd.md
      ├── us500.md
      ├── gbpusd.md
      └── usdjpy.md

scripts/
  ├── generate_tear_sheet.py
  └── render_tear_sheets.py         # pandoc wrapper

reports/sprint_7/
  ├── tear_sheets/                  # MD+JSON sources
  ├── e2e_24h_report.md
  └── global_certification.md

roadmap/sprints/
  ├── sprint_7.md
  ├── sprint_7_progress.md
  ├── sprint_7_retrospective.md
  └── sprint_7_certification.md

(git)
  ├── tag: v1.0.0-institutional
  └── branch main: merged from institutional-overhaul
```

---

## Décisions ouvertes pour user

1. **Si note pondérée < 7.5/10** : Sprint 8 corrective autorisé ? Ou ship en l'état + roadmap publique ?
2. **Tear sheets publics** : OK pour publier sur github/website public ? (transparency = moat selon eval_26).
3. **B2B-API mockup** : Si pivot M1 (PF gate Sprint 3 fail) → Sprint 7 inclut maquette REST API B2B ?
4. **Merge main** : OK pour squash-merge `institutional-overhaul` → `main` après certification ?

---

**Signé** : Claude, 2026-05-15

# Kill Criteria Board — Smart Sentinel AI

> **Source de vérité gouvernance.** Mis à jour chaque vendredi 16h par Sofia (RISK-1.1).
> Tout sprint actif a un status (🟢/🟡/🔴), un kill criterion explicite, et un blocker tracké.
> Référence plan : `reports/roadmap_2026_2027/PLAN_12_MOIS.md`.

**Dernière mise à jour** : 2026-04-30 23:55 ET
**Phase active** : 1
**Mois en cours** : M1 (semaine S1)
**Heures dev cumulées vs plan** : ~3h / 64h (DATA-1.1 quasi-livré)

---

## 1. KPIs critiques globaux

| KPI | Seuil vert | Seuil rouge | Actuel | Status |
|---|---|---|---|---|
| Test coverage (pytest --cov) | ≥75% | <70% | __% | 🟢/🟡/🔴 |
| /metrics endpoint live | payload non-vide | payload vide | __ | 🟢/🟡/🔴 |
| Signaux générés 7j | ≥30 | 0 | __ | 🟢/🟡/🔴 |
| Erreurs Sentry 7j | <10 | >50 | __ | 🟢/🟡/🔴 |
| LLM cost / revenue (Phase 2) | <40% | >60% | __% | 🟢/🟡/🔴 |
| Forward-test PF rolling 30j (Phase 2A) | ≥1.10 | <0.85 | __ | 🟢/🟡/🔴 |
| RAG faithfulness (Phase 2B) | ≥0.90 | <0.80 | __ | 🟢/🟡/🔴 |

---

## 2. Sprints actifs (mise à jour hebdo)

| Sprint ID | Titre | Owner | Effort planned/actual | ETA | Last update | Status | Kill criterion | Blockers |
|---|---|---|---|---|---|---|---|---|
| DATA-1.1 | FRED macro ingestion | Marwan | 4h / 3h code | 2026-04-30 | 2026-05-01 | 🟡 | fredapi rate-limit casse ingest | KPI live blocké : pas de FRED_API_KEY (B-001) |
| DATA-1.2 | CFTC COT ingestion | Marwan | 4h / 2h | 2026-05-01 | 2026-05-01 | 🟢 | format ZIP CFTC change | aucun. 365 weeks 2019-2025 ingérés, 5/5 tests verts |
| DATA-1.3 | GLD ETF flows | Marwan | 4h / 0h | 2026-05-01 | 2026-05-01 | ⏸ DEFERRED | SPDR JSON schema change | **Voie D** retenue : différé Phase 2A. ~17 features dispo sans GLD = ≥18 cible plan. |
| QUANT-1.1 | A1 feature matrix | Elena | 4h / __ | | | | NaN > 30% après ffill | dépend DATA-1.* |
| QUANT-1.2 | CPCV harness | Elena | 6h / __ | | | | runtime > 4h | dépend QUANT-1.1 |
| QUANT-1.3 | A1 stack training + verdict | Elena | 6h / __ | | | | DSR<0 ou PBO>0.6 → kill A1 | dépend QUANT-1.2 |
| REGIME-1.1 | HAR-RV ONNX + bavure VOL_MODE | Kenji | 4h / __ | | | | export ONNX RMSE delta >5% | aucun |
| REGIME-1.2 | BOCPD prototype | Kenji | 4h / __ | | | | cp_prob dégénéré | dépend QUANT-1.1 |
| LLM-1.1 | Eval harness 50 prompts | Aisha | 6h / __ | | | | forbidden_phrases <0.95 | aucun |
| INFRA-1.1 | GitHub Actions CI/CD | Théo | 3h / 1h | 2026-05-01 | 2026-05-01 | 🟢 | tests CI dépendants CSV local | scope initial limité aux test_fred + test_cot (cov 81%). Full suite à expand quand fixtures prêts. |
| INFRA-1.2 | Sentry + /metrics | Théo | 3h / __ | | | | Sentry > free tier | aucun |
| UX-1.1 | InsightSignal v2 | Inès | 5h / __ | | | | v2 casse >10 tests | aucun |
| COMM-1.1 | Positioning 2A+2B briefs | Karim | 5h / 1h30 | 2026-05-01 | 2026-05-01 | 🟢 | docs trop similaires | aucun. 2 briefs livrés AVANT verdict A1 (anti-biais ex-post). Sofia review checklist intégrée à chaque doc. |
| RISK-1.1 | Kill criteria board | Sofia | 6h / __ | | | | 2 checkpoints ratés consec | aucun |

**Légende status** :
- 🟢 = on track, no concerns
- 🟡 = slip < 1 semaine OR métrique en zone jaune
- 🔴 = slip > 1 semaine OR métrique rouge OR kill criterion proche

---

## 3. Checkpoints à venir

| Checkpoint | Date prévue | Owner | Critère go | Critère pivot |
|---|---|---|---|---|
| CP-1.1 | Fin S2 | Marwan | FRED+COT+GLD ingérés, no-look-ahead | Décaler S3 |
| CP-1.2 | Fin S4 | Elena | A1 baseline tournée RMSE calc | Investiguer leakage |
| CP-1.3 | Fin S6 | Elena | A1 walk-forward CPCV DSR/PBO calc | Si DSR<0 → kill A1 |
| **CP-A1** ⚠️ | **Fin S8** | **Elena+Sofia** | **PBO<0.3 ET DSR>1.0 ET CPCV PF>1.20 ET ≥3 Holm** | **Branche 2B** |
| CP-2A.1 | Fin M4 | Sofia | Forward-test 30j PF≥1.10 → Stripe ON | Kill 2A si PF<0.85 |
| CP-2A.2 | Fin M6 | Karim | 1 LOI B2B signée | Recentrer B2C |
| CP-2B.1 | Fin M5 | Aisha | RAG eval F1>0.85, halluc<5% | Itérer prompts |
| CP-2B.2 | Fin M9 | Karim | 1 contrat B2B €500-1500/mo | Persévérer 3 mois |

---

## 4. Décisions prises ce mois

- YYYY-MM-DD : {décision}, owner : {agent}, rationale : {x}

---

## 5. Blockers actifs et escalations

- **B-001 (DATA-1.1)** : FRED_API_KEY non fourni → smoke run live KPI bloqué. DoD pytest 6/7 verts (mocked). Détails : `BLOCKERS.md`. Action : user doit fournir clé FRED gratuite avant QUANT-1.1.
- ~~**B-002 (DATA-1.3)**~~ **RESOLVED 2026-05-01** : voie D retenue — différé Phase 2A. ~17 features Elena dispo sans GLD ≥ cible plan. Reprise conditionnelle au verdict A1.

---

## 6. Sprints abandonnés / killed (post-mortem requis)

| Sprint ID | Date kill | Raison | Post-mortem |
|---|---|---|---|

---

## 7. Note hebdo Sofia (5-15 lignes libres)

> Format : ce qui marche / ce qui inquiète / signal à surveiller la semaine prochaine.
> Pas de format imposé. Honnêteté avant esthétique.

**2026-04-30 (S1, démarrage Phase 1)** :
DATA-1.1 livré dans le scope du DoD (6/7 tests mocked verts, look-ahead 100/100). Bémol honnête : KPI "5 séries × ≥6 ans daily" non vérifié live faute de clé FRED. Bloqué proprement, documenté dans BLOCKERS.md, le module est néanmoins fonctionnel et prêt à tourner dès que la clé arrive. Décision : continuer DATA-1.2 (CFTC COT, pas de clé requise) sans attendre. Surveiller que le user fournit la clé avant QUANT-1.1 sinon Elena bloquera S3.

---

## 8. Discipline (rappel non-négociable)

1. Mesurer en temps réel, pas a posteriori
2. Pré-déclarer les seuils, pas les chercher après
3. Compter les ratés, pas les optimistes
4. Sofia a un droit de veto explicite (7 jours block + rétro)
5. Transparence radicale interne : ce fichier n'est PAS retouché après publication, on ajoute un changelog

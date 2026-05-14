# Quarterly Review Template — Smart Sentinel AI

> Sprint RISK-2B.3 (Sofia) — 3h × 4 trimestres / an = ~12h annuel.
>
> Format identique chaque trimestre. À remplir avec données réelles
> tirées de l'observabilité (`/api/v1/metrics/*` + audit_action_log
> + transparency_log + admin_action_log). Output : un fichier
> `reports/governance/qN_YYYY_review.md` archivé indéfiniment.

## I. Snapshot exécutif (1 paragraphe)

> Trimestre N — du YYYY-MM-DD au YYYY-MM-DD. **MRR fin de trimestre : X €** (vs Y € fin trimestre précédent). **Users actifs : Z**. **Forward-test paper-trading : equity +/-R% sur le trimestre**. **Compliance : 0 violation publiée / N flagged en CI**. *Statut global : ON-TRACK / WARN / KILL.*

## II. MRR + Revenu

| Tier | Active end-Q | Δ vs prev Q | MRR € |
|---|---|---|---|
| FREE | | | 0 |
| LITE | | | |
| PRO | | | |
| PRO+ | | | |
| B2B Basic | | | |
| B2B Pro | | | |
| **Total** | | | **€** |

**Conversion FREE→paid** : X% (cible eval_27 : 2.5% M3, 4% M6, 6% M9).

**Churn** : X% mensuel moyen (kill criterion : > 8%).

## III. Forward-test paper-trading

| Métrique | Valeur Q | vs Q-1 |
|---|---|---|
| n_trades | | |
| total_R | | |
| win_rate | | |
| max_drawdown_R | | |
| sharpe_per_trade | | |
| hash log entries (RISK-2B.1) | | |

**Vérification chaîne** : `python scripts/audit_ledger_snapshot.py verify data/risk/transparency_log.jsonl` doit retourner OK.

## IV. Compliance

| Catégorie | Compte trimestre |
|---|---|
| Violations détectées en CI (ComplianceChecker) | |
| Violations publiées (post-hoc) | doit être **0** |
| Narratives flagged review queue (worst_5) | |
| LLM-as-judge mensuel — runs effectués | |
| Geo-block hits (US/QC/UK) | |
| Compte AMF inquiry | doit être **0** |

## V. Observabilité

| Métrique | Valeur Q |
|---|---|
| /metrics/latency p95 trimestre | |
| /metrics/error-budget alerts firing | |
| /metrics/webhook-drain dead_letter cumulés | |
| /health/deep failures | |
| Incidents (RISK-2B.4 runbook activé) | |

## VI. Données

| Source | SLA respect % Q |
|---|---|
| news | |
| sentiment | |
| macro (FRED) | |
| prices_m15 | |
| cot | |

## VII. LLM cost

| Métrique | Valeur Q |
|---|---|
| Cost USD total | |
| Cost / active user / mo | |
| Cache hit rate (target ≥ 30%) | |
| Sonnet vs Haiku split | |
| % runs en mode batch (eval) | |

## VIII. Sources RAG

| Métrique | Valeur Q |
|---|---|
| Total sources indexed | |
| Nouvelles sources ajoutées | |
| Sources flagged "biased-author" utilisées | doit être **0** |
| Faithfulness moyenne (LLM-2B.7) | |
| Hallucination rate | |

## IX. Roadmap — état des sprints

Compter pour chaque sprint Phase 2B :

| Statut | Nombre |
|---|---|
| Done | |
| In progress | |
| Blocked | |
| Killed | |

**Sprints les plus en retard** (top 3) : ...

## X. Kill criteria check

À chaque review, confronter aux kill criteria documentés :

- [ ] MRR M6 ≥ 2 000€ — Statut : ...
- [ ] MRR M9 ≥ 4 500€ — Statut : ...
- [ ] MRR M12 ≥ 8 500€ — Statut : ...
- [ ] Cost LLM < 60% revenue — Statut : ...
- [ ] 0 AMF inquiry — Statut : ...
- [ ] 0 compliance violation publiée — Statut : ...

Si un kill criterion est violé, déclencher la procédure correspondante (kill, pivot, ou correction urgente). Documenter la décision.

## XI. Décisions stratégiques du trimestre

Liste des décisions importantes prises ce trimestre (avec lien vers le commit / le doc d'origine) :

1. ...
2. ...
3. ...

## XII. Prochaines actions

Top 5 priorités pour le trimestre suivant, classées par impact attendu :

1. ...
2. ...
3. ...
4. ...
5. ...

---

*Document complété par : Sofia (RISK-2B.3).*
*Reviewers : tous les agents (Marwan, Elena, Kenji, Aisha, Théo, Inès, Karim).*
*Archive : `reports/governance/qN_YYYY_review.md`.*

# Monthly Checkpoint — {YYYY}-{MM} (Mois {N})

> **Gabarit à dupliquer** dans `reports/monthly_checkpoint/{YYYY}-{MM}.md` chaque dernier dimanche du mois.
> Référence plan : `reports/roadmap_2026_2027/PLAN_12_MOIS.md` (Partie VI.3).
> Durée d'écriture : 2h (solo founder + Sofia review).

---

## 0. Phase active et sprint en cours

- **Phase** : [1 / 2A / 2B / 2B+]
- **Date du checkpoint** : YYYY-MM-DD
- **Heures dev cumulées vs plan** : {actual}h / {planned}h ({deviation}%)
- **Heures gouvernance ce mois** : {x}h (cible 5h/mois : 4× weekly 1h + 2h ce checkpoint)
- **Burnout self-score 1-10** : {x} (1=frais, 10=épuisé. Si >7, escalation Sofia)

---

## 1. Tableau de bord agents (vert / jaune / rouge)

| Agent | Métrique 1 | Métrique 2 | Métrique 3 | Status global |
|---|---|---|---|---|
| Marwan (Data) | Feed uptime 7j : __% | Macro freshness : __ | Audit integrity : __% | 🟢/🟡/🔴 |
| Elena (Quant) | Forward-test PF 30j : __ | DSR 90j : __ | PSI max : __ | 🟢/🟡/🔴 |
| Kenji (Regime/Vol) | Vol latence p99 : __ms | RMSE vs HAR : __ | Régime confidence histo : __ | 🟢/🟡/🔴 |
| Aisha (LLM) | Eval score : __ | RAG faithfulness OR cache hit : __ | LLM cost/user/mo : __ | 🟢/🟡/🔴 |
| Théo (Infra) | API uptime 7j : __% | Test coverage : __% | Backup last test : __j ago | 🟢/🟡/🔴 |
| Inès (UX) | Lighthouse perf : __ | Onboarding completion : __% | NPS user-test : __/10 | 🟢/🟡/🔴 |
| Karim (Commercial) | MRR vs target : __% | Trafic organique : __ | Pipeline B2B (LOI) : __ | 🟢/🟡/🔴 |
| Sofia (Risk) | Weekly checks : __/4 | Compliance violations : __ | Quarterly on-time : __ | 🟢/🟡/🔴 |

---

## 2. Sprints livrés ce mois

- ✅ {SPRINT-ID} : {titre} — {effort réel}h vs {planned}h, DoD validé
- ⚠️ {SPRINT-ID} : {titre} — slip {n}j, raison : {x}, nouvelle ETA : YYYY-MM-DD
- ❌ {SPRINT-ID} : abandonné, raison : {x}, killed via {kill criterion}, post-mortem : {lien}

---

## 3. KPI commerciaux (Phase 2A/2B uniquement)

| Métrique | M-1 | M actuel | Δ | Target M{N} | Ratio |
|---|---|---|---|---|---|
| Users FREE | __ | __ | __ | __ | __% |
| Users payants | __ | __ | __ | __ | __% |
| MRR | __€ | __€ | __€ | __€ | __% |
| Churn 30j | __% | __% | __ | <5% | __ |
| LLM cost / revenue | __% | __% | __ | <40% | __ |
| Pipeline B2B (LOI active) | __ | __ | __ | __ | __ |

**ARPU** : __€ | **CAC estimé** : __€ | **LTV estimé** : __€

---

## 4. Forward-test perf (si applicable Phase 2)

| Métrique | Valeur | Target | Status |
|---|---|---|---|
| PF rolling 30j | __ | ≥1.10 (2A) / publié (2B) | 🟢/🟡/🔴 |
| Sharpe rolling 30j | __ | ≥0.5 | __ |
| DSR rolling 90j | __ | ≥0.5 | __ |
| PSI max feature | __ | <0.20 | __ |
| Drawdown max 30j | __% | <15% | __ |

**Verdict consolidé** : 🟢 continue / 🟡 watch / 🔴 escalate

---

## 5. Kill criteria status

| Critère | Seuil | Actuel | Status |
|---|---|---|---|
| Forward-test PF M4 ≥ 1.10 (2A) | 1.10 | __ | 🟢/🔴 |
| RAG faithfulness ≥0.90 (2B) | 0.90 | __ | 🟢/🔴 |
| LLM cost / revenue <60% (2B) | 60% | __% | 🟢/🔴 |
| MRR M6 ≥ 1500€ | 1500€ | __€ | 🟢/🔴 |
| 0 compliance violation publiée | 0 | __ | 🟢/🔴 |

---

## 6. Décisions prises ce mois

1. YYYY-MM-DD : {décision} — owner : {x}, rationale : {y}
2. YYYY-MM-DD : {décision} — ...

---

## 7. Blockers et escalations

- **Blocker A** : {description}, ETA résolution : YYYY-MM-DD
- **Escalation B** : {x} → action : {y}, owner : {z}

---

## 8. Next month focus (3 priorités max — discipline)

1. Sprint {ID} (priority 1, owner {agent}, effort {h}h)
2. Sprint {ID} (priority 2, owner {agent}, effort {h}h)
3. Sprint {ID} (priority 3, owner {agent}, effort {h}h)

---

## 9. Réflexion qualitative solo founder (5-15 lignes libres)

> Format libre. Honnêteté > esthétique. Ce qui marche, ce qui frustre, sentiment général,
> tentations à pivoter, signaux faibles, conversations utilisateurs marquantes.

{Note libre du solo founder}

---

## 10. Décision principale checkpoint

- [ ] Continuer plan tel quel
- [ ] Ajustement mineur (préciser : ___)
- [ ] Pivot partiel (préciser : ___, déclencher Sofia review)
- [ ] Kill total (préciser : ___, déclencher post-mortem)

**Validation Sofia** : ✓ / ✗ — commentaire : {x}

---

## 11. Changelog (append-only après publication)

> Si correction nécessaire après publication, ajouter ici plutôt que retoucher le contenu ci-dessus.

- YYYY-MM-DD HH:MM : {correction}

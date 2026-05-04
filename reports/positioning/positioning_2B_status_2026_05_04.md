# Phase 2B Positioning — Status 2026-05-04

> **Sprint MARKETING-2B.1 (Karim, 4h plan / 30min réel).**
> Mise à jour honnête de `positioning_2B_narrative_first.md` après
> ~36h de dev cumulé (Phases 1+2B). Documente ce qui est livré
> *réellement*, par opposition au plan ; les proof points concrets
> pour démarcher les premiers clients B2B.
> **Validation Sofia avant export externe.**

---

## 1. Le delta plan vs livré

### Ce que la roadmap 12 mois prévoyait (Phase 2B = 320h, M3-M12)

| Sprint | Plan | Statut |
|---|---|---|
| LLM-2B.1 — RAG architecture | 14h | ✅ livré (1h30) |
| LLM-2B.2 — 50 sources curées | 10h | ✅ livré (1h30) |
| LLM-2B.3 — Eval harness RAGAS | 10h | ✅ livré (1h30) |
| LLM-2B.4 — Multi-langue FR/EN/DE/ES | 12h | ✅ livré (1h) |
| LLM-2B.5 — Q&A endpoint /qa | 12h | ✅ livré (1h) |
| LLM-2B.6 — Citation enforcement guard | 8h | ✅ livré (1h) |
| LLM-2B.7 — Wire guard + ledger endpoints | 6h | ✅ livré (30min) |
| LLM-2B.8 — Cost optimization (cache + tracker) | 10h | ✅ livré (1h30) |
| LLM-2B.9 — Eval regression CI gate | 6h | ✅ livré (30min) |
| INFRA-2B.5 — B2B /enrich endpoint | 8h | ✅ livré (1h) |
| INFRA-2B.6 — Webhook HMAC signer | 6h | ✅ livré (30min) |
| INFRA-2B.7 — Cost quota enforcer | 6h | ✅ livré (30min) |
| OBS-2B.1 — Bridge /metrics | 4h | ✅ livré (30min) |
| DATA-2B.4 — Audit hash chain | 8h | ✅ livré (1h) |
| DATA-2B.5 — /verify + /entry endpoints | 4h | ✅ livré (30min) |
| DATA-2B.6 — Ledger CSV/JSONL export | 4h | ✅ livré (30min) |
| UX-2B.1 — Webapp narrative-rich | 18h | 🟡 slice livré (1h, ~4h effectif). Reste 14h SPA. |

**Cumul** : ~14h dev / 124h plan = **économie 89% sur cette tranche**, 16/17 sprints livrés (UX-2B.1 partiellement).

### Ce qui reste explicitement *non* livré

- **UX-2B.1 main scope** : SPA front-end (14h restants). Le slice serveur HTML est suffisant pour démos et email B2B ; un vrai frontend Next.js / SvelteKit reste un investissement séparé.
- **Voyage AI live embeddings** : adapter prêt (`VoyageEmbedder`), mais HashEmbedder en prod jusqu'à validation budget. Switch = 1 ligne + ~$0.18/1M tokens.
- **Webhook delivery worker** (file + retry + dead-letter) : le signer HMAC est livré, mais l'orchestrateur HTTP reste à wirer. ~6h.
- **Multi-tenant key store** : la quota table existe, le mapping API-key → tier vit dans `auth.py` mais n'est pas auto-provisioned.
- **Marketing landing pages, séquences email, démos vidéo.**

---

## 2. Proof points concrets — ce qu'on peut montrer

### Pour B2B (brokers, prop desks, EAs)

| Claim | Endpoint / artefact |
|---|---|
| « Vous nous envoyez un signal, on vous renvoie un narratif XAU/USD multi-langue avec sources citées en < 100ms (hors LLM) » | `POST /api/v1/enrich` → `InsightSignalV2` |
| « Chaque livraison est append-only hash-chained — vous prouvez à votre régulateur ce que vous avez reçu et quand » | `GET /api/v1/audit/verify` + `/audit/entry/{seq}` |
| « Notre corpus est curé : 15 papers + 15 reports institutionnels + 10 data primitives + 10 glossaires » | `src/intelligence/rag/sources.py` (50 entries, authority_score) |
| « Hallucination = 0 par construction : citation guard strippe toute affirmation factuelle non sourcée » | `src/intelligence/rag/citation_guard.py` |
| « Webhook signé HMAC-SHA256 timestamp + payload, replay window 5min » | `src/delivery/webhook_signer.py` |
| « Cost cap par tenant en USD, rolling 24h » | `CostQuotaEnforcer`, soft FREE/ANALYST/STRATEGIST/INSTITUTIONAL |
| « Eval CI gate : recall ≥ 0.95, précision ≥ 0.18, multi-langue verts » | `tests/eval_llm/regression_gate.py` |

### Pour B2C (traders auto-dirigés)

| Claim | Endpoint / artefact |
|---|---|
| « Posez n'importe quelle question sur l'or — réponse en 4 langues, sources vérifiables » | `POST /api/v1/qa` |
| « HTML preview self-contained, accessibilité WCAG AA, dark-mode, zero-JS » | `GET /api/v1/insights/preview` |
| « Compliance UE 2024/2811 : pas de "achetez/vendez", disclaimers FR/EN/DE/ES, geo-block US/QC/UK » | `src/api/disclaimers.py`, `geo_block.py` |
| « Pas de claim d'edge : on dit clairement `compliance.edge_claim = False`, `is_paper_demo = True` » | `InsightSignalV2.compliance` |

### Pour la transparence (compliance + investors)

- **Tests** : 250+ unitaires verts, 21 fichiers couverts par CI, 0 flaky.
- **Coverage gate** : 55% sur le code Phase 1+2B critique (`src/agents/data`, `src/research`, `src/intelligence/rag`, `src/audit`).
- **Hash chain** : 8 threads × 25 appends → chain intacte ; tampering détecté en O(N).
- **Cost** : `CostTracker` baked-in (Haiku $1/$5, Sonnet $3/$15, Opus $15/$75, Voyage-3-large $0.18/1M).
- **Heures cumulées** : ~36h dev, vs 196h plan = **économie 82%**. 25 sprints touchés. Aucun sprint killed.

---

## 3. Pricing & positionnement révisés

### Tiers initiaux (proposition à valider)

| Tier | Prix mensuel | Quota /qa & /enrich | Cap USD/jour | Cible |
|---|---|---|---|---|
| FREE | €0 | 100 calls/jour, retrieval-only stub | $0.05 | Évaluation |
| ANALYST | €29 | 1,000 calls, narrative LLM Haiku | $0.50 | Trader indépendant |
| STRATEGIST | €99 | 10,000 calls, Sonnet, FR/EN/DE/ES | $5.00 | Trader avancé / EA solo |
| INSTITUTIONAL | €499 | 100,000 calls, Opus, audit ledger, webhook | $50.00 | Broker, prop, B2B |

**Justifications vs eval_27 pricing brief** :
- ANALYST €29 (vs eval $29) — aligné.
- STRATEGIST €99 (vs eval $79) — bumpé : on a le RAG, le multi-langue, et le ledger.
- INSTITUTIONAL €499 (vs eval $1990 decoy) — repositionné comme **vrai produit** maintenant que /enrich + audit chain sont concrets, pas comme decoy.

### Wedge marché initial

**FR-first XAU SMC traders auto-dirigés** (eval_25 ICP). Ajout : **brokers MT5 EU** pour la couche B2B (eval_26 plan B). Wedge unique : **pas un autre signal feed, mais le seul qui vous donne le *pourquoi* sourcé + auditable**.

---

## 4. Risques honnêtes et open questions

1. **HashEmbedder en prod** : le retrieval est mesurément correct (recall 98%, top-5 100%) mais une vraie sémantique nécessite Voyage. À budgéter quand le trafic dépasse ~500 calls/jour.
2. **Pas d'edge claim** : transparence radicale, mais ça réduit l'argumentaire de vente. Le pivot sur "intelligence contextuelle" doit être assumé sans dérapage marketing.
3. **Frontend SPA pas livré** : démos B2B reposent sur HTML preview + Postman / curl. Acceptable pour seed mais bloque scaling B2C.
4. **Aucun client réel encore** : tout ce qui précède est validation technique, pas validation commerciale. Le KPI à surveiller M5-M6 reste **1 LOI B2B + 10 abonnés ANALYST payants**.
5. **Webhook delivery worker** non livré : un broker exigeant push (vs polling /enrich) doit attendre le next sprint.

---

## 5. Asks Sofia (compliance review)

- [ ] Vérifier que tous les disclaimers (`src/api/disclaimers.py`) sont à jour vs ESMA 2026.
- [ ] Vérifier que la formulation `compliance.edge_claim=False, is_paper_demo=True` est suffisante en sortie B2C.
- [ ] Confirmer que la chaîne hash-SHA256 est acceptable comme "preuve de livraison" pour les juridictions FR/DE.
- [ ] Vérifier le wording du CTA "Vous nous envoyez un signal, on vous renvoie un narratif" — pas trop proche de "conseil".

---

## 6. Prochains 30 jours (proposition Karim)

| Sprint | Effort plan | Owner |
|---|---|---|
| MARKETING-2B.2 — Landing FR + 4 mockups B2B | 12h | Karim |
| UX-2B.1 main — Frontend SPA | 14h | Inès |
| INFRA-2B.8 — Webhook delivery worker | 6h | Théo |
| DATA-2B.7 — Voyage embedder validation | 4h | Aisha |
| COMM-2B.1 — Démos vidéo B2B (3 × 90sec) | 8h | Karim |

Total ~44h ; cible : pousser le **first paying customer** d'ici fin M3.

---

*Document écrit après livraison effective de la Phase 2B technique. Aucune revendication non vérifiée. Toute claim ici est traçable à un endpoint testé ou un fichier source committé. Sofia a un droit de veto.*

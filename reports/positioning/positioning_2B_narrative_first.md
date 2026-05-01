# Positioning Brief 2B — Narrative-First

> **Sprint COMM-1.1 (Karim, Phase 1).** À activer SI verdict A1 ❌
> (PBO ≥ 0.5 OU DSR < 1.0 OU CPCV PF < 1.20 OU < 3 Holm-significant).
>
> **Probabilité a priori 65-75%** (eval CIO 3.46/10 + falsification 2026-04-30
> bootstrap CI [0.70, 0.88]). Ce brief n'est PAS un plan B dégradé —
> traité à parité avec 2A car c'est le scénario le plus probable.
>
> Référence : `reports/roadmap_2026_2027/PLAN_12_MOIS.md` Partie IV.
> Écrit AVANT le verdict A1 pour neutraliser le biais ex-post.
> **Validation Sofia (compliance) requise avant publication.**

---

## 1. Reposition produit

Smart Sentinel AI passe de **"signal trading edge-validated"** à **"intelligence contextuelle pour traders auto-dirigés"** — un produit qui :

1. **N'affirme pas posséder un edge** (compliance + honnêteté radicale)
2. Délivre **narrative LLM riche, sourcée, RAG-backed** sur l'état du marché XAU
3. Aide à **mieux comprendre** chaque set-up sans dire "achetez/vendez"
4. Pour le B2B, devient un service de **"qualité de signal augmentée"** : on enrichit les signaux d'un broker/EA avec contexte macro+sentiment+régime
5. **Transparence radicale** : forward-test paper publié, drawdowns embraced, pas de claim de perf

**Ce repositionnement adresse un marché potentiellement plus large** (apprenants + traders auto-dirigés) avec un ARPU plus modeste mais un plafond plus haut grâce au RAG sourcé et au B2B "data quality enrichment".

---

## 2. Audience cible

### Primary B2C — Apprenants & traders auto-dirigés FR

- 25-50 ans (extension Persona Marc vers débutants/intermédiaires)
- Cherchent à comprendre, pas à recevoir des signaux clés-en-main
- Capital $1k-30k (médiane $5k), ouverts au paid LITE/PRO
- WTP $19-49/mo pour content éducatif + chat Q&A sourcé
- Apprécient transparence > claims marketing
- Pain principal : "outils existants me parlent en jargon, je ne comprends pas pourquoi un setup a marché ou pas"

### Secondary B2C — Traders intermédiaires "explorers"

- 2-5 ans XP, déjà clients TradingView Premium + LuxAlgo
- WTP $79-99 pour PRO+ si Q&A chat est vraiment sourcé (pas du faux GPT)
- Refusent "edge claims" — méfiance acquise

### B2B — Copy-trading platforms + EA dev shops + small brokers

- eToro, ZuluTrade, Darwinex (copy-trading)
- Forex Tester, MQL5 marketplace (EA dev shops)
- Petits brokers FR/CH/LU sans dev interne
- WTP $499-1500/mo pour API "enrichis nos signaux avec contexte LLM sourcé, audit-trail, multi-langue"
- Use case : leur signal/EA est lapidaire ("BUY XAUUSD 2350 SL 2340 TP 2370"), nous enrichissons avec "Pourquoi ce setup ? Régime macro ? Sources citées ?"

---

## 3. Claims autorisés (PAS d'edge claim)

- "Intelligence contextuelle market XAU"
- "Narrative LLM sourcée RAG (50+ sources curées : papers académiques, LBMA, WGC, BIS, FOMC minutes)"
- "Transparence radicale — forward-test paper publié dans drawdowns inclus"
- "Comprenez sans qu'on vous dise quoi faire"
- "Multi-langue FR/EN/DE/ES"
- "Q&A chat sur set-ups XAU avec citations vérifiables"
- "Service B2B 'data quality enrichment' avec audit-trail signable" (B2B-only)

## 4. Claims interdits (compliance — encore plus strict en 2B)

- ❌ "Edge prouvé / validé / backtested" (puisqu'on a explicitement renoncé après A1)
- ❌ "Backtested results" (rapport CIO et falsification montrent 0/7 Holm)
- ❌ "Suivez nos signaux" (pas un signal product)
- ❌ Tout ce qui est interdit en 2A reste interdit (MiFID II 2024/2811)
- ⚠️ Le forward-test paper publié doit être dans tous les cas accompagné du disclaimer
  **"Démonstration paper-trading. Smart Sentinel ne prétend PAS posséder un edge.
   Cette courbe est éducative. Performances passées ≠ futures."**
   en français + EN + DE + ES (sprint W1 disclaimers réutilisé)

---

## 5. Proof points (différenciation 2B)

| # | Proof point | Origine | Visibilité |
|---|---|---|---|
| 1 | RAG faithfulness ≥ 0.90 (RAGAS) | LLM-2B.3 eval | Landing + technical whitepaper |
| 2 | F1 sourcing > 0.85 sur 200 prompts | LLM-2B.3 fixtures | Public eval dashboard |
| 3 | Hallucination rate < 5% | LLM-2B.3 + LLM-2B.7 | Live monthly report |
| 4 | 50+ sources curées tagged | LLM-2B.2 | Glossary + sources page |
| 5 | 4 langues (FR/EN/DE/ES) | LLM-2B.4 | UI switcher |
| 6 | Forward-test paper public **avec drawdowns** | INFRA-2B.2 | Landing temps réel |
| 7 | YouTube weekly market wrap FR (24/an) | COMM-2B.2 | Differentiation moat |
| 8 | Audit-trail B2B signable | DATA-2B.4 | B2B whitepaper |

**Moat 2B** : *transparence radicale*. Aucun concurrent FR XAU n'expose son forward-test live brut, drawdowns inclus, avec disclaimer "n'est pas un signal". C'est contre-intuitif marketing mais hyper-différenciant pour ICP "explorers" méfiants envers claims classiques.

---

## 6. Pricing recommandé (positionnement éducatif/contextuel)

| Tier | Prix | Cible | Inclus |
|---|---|---|---|
| **FREE** | 0€ | Acquisition large | Newsletter weekly + 1 narrative/jour delayed 24h |
| **LITE** | **19€/mo** | Apprenants débutants | Webapp daily narrative + glossaire + tooltips inline |
| **PRO** | **39€/mo** | Auto-dirigés | + Q&A chat illimité + signaux paper transparents + multi-asset XAU/EURUSD/USOIL |
| **PRO+** | **99€/mo** | Power users | + Telegram alerts personnalisées + accès historique 6 ans + multi-langue |
| **B2B Data Quality** | **499-1 500€/mo** | Copy-trading, EA dev | API `/enrich` : 1k-10k req/mois, audit-trail, multi-langue |

**Justification pricing inférieur à 2A** : sans claim edge, on facture la qualité narrative + transparence + RAG, pas la promesse de profit. ARPU plus modeste mais marché plus large.

**Trial 14j sans carte sur LITE + PRO** (eval_27 finding 4 conservé).

**Annual -16.7%** affiché par défaut.

**Pas de tier "INSTITUTIONAL decoy"** — l'effet anchoring fonctionne moins en 2B où l'audience cherche accessibilité, pas premium statut.

---

## 7. GTM channel mix (40h vs 30h en 2A — éducation marché)

### M3-M5 — Webapp launch + RAG ready

- **Webapp narrative-rich** (UX-2B.1, 18h Inès — agent central 2B)
- **RAG architecture** (LLM-2B.1, 14h Aisha)
- **Forward-test paper public dès J1** (INFRA-2B.2 + RISK-2B.1) — feature marketing transparence

### M5-M9 — SEO éducatif + YouTube

- **10 cornerstone éducatifs FR** (COMM-2B.1, 12h, 3 000+ mots chacun) :
  "qu'est-ce que SMC ?", "comprendre le COT or", "régimes de marché XAU",
  "vol forecasting expliqué", "SL/TP calcul intelligent", etc. KD<25
- **YouTube weekly market wrap** (COMM-2B.2, 24 vidéos/an, 30 min/sem)
  → **canal unique en FR sur XAU intraday** = différenciation forte vs TradingView/TradingIdeas
- **Telegram public broadcast + Newsletter Substack** (acquisition + nurture)
- **Pas de Discord public** (idem 2A — modération solo intenable)

### M9-M12 — B2B data quality + community paid

- **Outbound 5 prospects B2B** : eToro, ZuluTrade, Darwinex, Forex Tester, MQL5
- Pitch : "enrichissez vos signaux avec contexte LLM sourcé, 499€/mo, audit-trail, multi-langue"
- **Discord privé paid** activé M3+ (gating)
- **2 A/B tests pricing** : trial 7j vs 14j, bundle annual -20% vs monthly

### Cadence non-négociable (eval_28 finding 5 — identique à 2A)

- Batch dimanche 14-18h : article + tournage + Buffer + newsletter
- 24 vidéos/an = 1 par 2 semaines minimum
- Si 3 dimanches loupés en 6 sem → couper YouTube en premier

### MRR targets

- M6 : 2 000€ (P=50%) — 50-80 LITE+PRO
- M9 : 4 500€ (P=45%) — + 1-2 deals B2B small
- M12 : 8 500-11 000€ (P=40%) — 200-280 users payants + 2-3 contrats B2B

**P(hit M12 ≥ 8k€) = 40%** — supérieur à 2A (P=30% pour 11k€) car la barre est plus basse.

---

## 8. Analyse concurrentielle 5 acteurs

| Concurrent | Prix | Forces | Faiblesses vs 2B | Gap exploitable |
|---|---|---|---|---|
| **BabyPips** | Gratuit (ad-supported) | Référence éducation forex EN, 500k+ users | Basique, pas de narrative live, pas FR-natif, pas spé XAU | Live narrative XAU FR sourcé |
| **Investopedia** | Gratuit + Premium $20 | Glossary, articles, brand reconnu | Généraliste, pas de Q&A trade-specific, pas de forward-test | Q&A chat XAU + forward-test transparent |
| **TradingEconomics** | 79$+/mo | Macro data feed (FRED-like), API | Pas de narrative LLM, pas de Q&A, raw data only | LLM RAG sourcé sur data eux-mêmes (potentiel partenariat data layer) |
| **Bloomberg Terminal** | 24 000$/an | Référence institutionnelle | Inaccessible pour ICP cible (1000× WTP), pas FR | Audience exclue par prix |
| **Daily FX / Investing.com** | Gratuit ad-supported | Volume trafic, content updates | Contenu généraliste FX, pas FR-first XAU intraday, pas de RAG sourcé | FR-first + RAG sourcé + sans pubs |

**Moat 2B** :
1. **YouTube FR XAU weekly market wrap** — 0 concurrent fait ça en FR
2. **RAG sourcé avec citations vérifiables** — Investopedia/BabyPips font du contenu statique, pas du Q&A live RAG
3. **Transparence radicale forward-test** — contre-intuitive mais signal de confiance unique
4. **Multi-langue FR/EN/DE/ES** — la plupart concurrents EN-only ou EN+ES seulement
5. **B2B "data quality enrichment"** — niche peu adressée (eval_26 finding)

**Window** : 18 mois avant que TradingEconomics ou un acteur similaire ajoute LLM narrative à leur stack. Lock content moat M3-M9 (compounding SEO + YouTube).

---

## 9. Risques de claim et mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Forward-test paper rouge → users perdent confiance | Modérée | 🟡 modérée | Embracer drawdowns, framing "démonstration éducative", communication transparente sur Discord |
| AMF perçoit forward-test comme "résultats hypothétiques interdits" | Faible-modérée | 🔴 critique | Disclaimer multi-langue ostensible (eval_29), INFRA-2B.4 legal review, possible suppression si avocat exige |
| RAG hallucine source publique → brand damage | Modérée | 🔴 critique | RAGAS faithfulness ≥ 0.90 gate (LLM-2B.3), monthly review queue worst narratives (LLM-2B.7) |
| LLM cost > 60% revenue (RAG intensif) | Élevée si scale | 🟡 modérée | Cost optimization aggressive (LLM-2B.8) : Haiku-first, prompt caching Anthropic 90% off, batch API offline 50% off |
| MiFID II 2024/2811 finfluencer mars 2026 | Élevée | 🟡 modérée | Positionnement éditorial-narrative-transparence = framework éditorial type Investopedia, pas conseil |
| Concurrent ajoute LLM narrative | Élevée 18 mois | 🟡 modérée | Lock content moat M3-M9 + brand FR-first |

**Asymétrie réglementaire favorable** : positionnement 2B "intelligence contextuelle éditoriale" est **structurellement plus sûr** que 2A "edge prédictif" sous régul finfluencer 2026. Bénéfice latent.

---

## 10. Métriques succès (CP-2B)

- **CP-2B.1 (M5)** : RAG eval 100 prompts F1 sourcing > 0.85, hallucination < 5%
- **CP-2B.2 (M9)** : 1 contrat B2B "data quality" 500-1500€/mo signé
- **M6** : conversion FREE→LITE ≥ 2.5%
- **M12** : 200+ users payants, 2-3 contrats B2B, 8.5k+ MRR (P=40% honnête)

---

## 11. Différences-clés positionnement 2A vs 2B

| Dimension | 2A | 2B |
|---|---|---|
| Hero couche produit | Couche 2 (algo edge) | Couches 4 (LLM) + 6 (UX) + 1 (data corpus) |
| Promesse client | "Notre algo a un edge prouvé" | "Comprenez le marché — nous, on n'affirme pas avoir d'edge" |
| Tier le moins cher payant | 29€/mo | 19€/mo |
| Tier le plus cher B2C | 199€/mo (ou 1 990€ decoy) | 99€/mo |
| Risque AMF/MiFID | Élevé (claim edge attire) | Modéré (positionnement éditorial) |
| Forward-test public | Gate de monétisation (interne) | Feature marketing transparente |
| Différenciation | DSR + audit-trail + B2B premium | RAG sourcé + transparence radicale + B2B mid-range |

---

**Sofia review checklist (avant publication)** :
- [ ] Aucun "edge prouvé / validé / backtested" dans le texte
- [ ] Disclaimer "non-edge" présent en footer landing + sur forward-test
- [ ] CGU/Privacy publiées (INFRA-2B.4 closed avec finfluencer 2026 prep)
- [ ] Forward-test live publié avec disclaimer multi-langue ostensible
- [ ] Geo-block US/QC/UK/OFAC actif (sprint W1+W2+W3 livré)
- [ ] Q&A chat ne donne JAMAIS de recommandation directe d'achat/vente
- [ ] RAG sources citées sont vérifiables et non-hallucinées (LLM-2B.3 gate)

Sofia signature : ___________________  Date : ___________________

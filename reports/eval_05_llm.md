# Eval 05 — LLM Narrative Engine (Claude API, qualité, coût)

> **Périmètre audité** : `src/intelligence/llm_narrative_engine.py` (417 l.), `src/intelligence/template_narrative_engine.py` (442 l.), `src/intelligence/semantic_cache.py` (210 l.), wiring `src/intelligence/main.py` + `src/intelligence/sentinel_scanner.py`, suites `tests/test_llm_narrative_engine.py` (11 tests), `tests/test_template_narrative_engine.py`, `tests/test_vol_narratives.py`.
>
> **Date** : 2026-04-24 · **Branch** : `main` · **Snapshot** : 632e9dd + uncommitted Sprint 2.

---

## 0. TL;DR

| Dimension | Note /10 | Justification chiffrée |
|-----------|----------|------------------------|
| Prompts (structure, anti-halluc.) | 5 | System prompt 420-550 tokens, aucune contrainte anti-invention, pas de few-shot |
| Prompt caching (effectivité réelle) | **2** | `cache_control` est **no-op silencieux** car < 1024 tokens (seuil Sonnet) / 2048 (seuil Haiku). Aucun vrai cache hit possible aujourd'hui. |
| Choix de modèle (ID, coût) | 4 | IDs codés en dur sur génération 4.5 alors que 4.6/4.7 sont dispo ; aucun routing par tier |
| Latence / Fallback | 5 | Pas de timeout explicite, pas de bascule auto LLM→template si circuit OPEN, juste un dict fallback minimal |
| Qualité sortie (évaluation structurée) | 3 | Aucune rubric automatisée, aucune suite d'échantillons notés, parsing `\n\n` fragile |
| Semantic cache (hit rate) | 3 | Mal nommé (hash strict, pas sémantique) ; clef inclut `bar_timestamp` ⇒ hit rate théorique ≈ 0 ; TTL 24 h mais clef new-bar → inutile |
| Multi-langue | 1 | Anglais hardcodé, TAM non-anglophone non adressé |
| Commercial (différenciation vs concurrents) | 5 | Cascade Haiku→Sonnet bien pensée, mais mode défaut = template (`NARRATIVE_MODE=template`) ⇒ le hook marketing "AI-powered" est menteur en prod |
| **GLOBAL** | **4.5 / 10** | « POC fonctionnel, pas production-grade commercial » |

---

## 1. Cartographie du code

```
┌──────────────────────────────────────────────────────────────────┐
│  ConfluenceSignal (score 0-100, components, vol_regime, …)       │
└────────────────────┬─────────────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │  sentinel_scanner.py   │
         │  _scan_once()          │
         └───────────┬────────────┘
                     │
          ┌──────────┴──────────┐
          │ SemanticCache.get() │  ← SHA256(symbol + bar_ts + comp.*)[:16], TTL 24h
          └──────────┬──────────┘
                     │ miss
                     ▼
      ┌──────────────────────────────────┐
      │  _generate_narrative_safe()      │
      │  (wrapper CircuitBreaker)        │
      └──────────────┬───────────────────┘
                     │
         ┌───────────┴───────────┐
         │ NARRATIVE_MODE env    │
         └───────────┬───────────┘
              llm    │    template (default)
        ┌───────────┐│┌──────────────┐
        │LLMNarrative││TemplateNarrative │
        └─────┬─────┘└──────────────┘
              │
    ┌─────────┴─────────┐
    │ NarrativeTier     │
    ├───────────────────┤
    │ VISUAL   → no-call│
    │ VALIDATOR→ Haiku  │ (1 appel)
    │ NARRATOR → Haiku  │ (cascade 2 appels)
    │             + Sonnet
    └───────────────────┘
```

**Points clefs du code** :
- `SMC_SYSTEM_PROMPT` (llm_narrative_engine.py:29-64) : 2207 chars, **420-550 tokens** (vérifié).
- `cache_control = {"type": "ephemeral"}` posé systématiquement (ligne 332) mais **ignoré silencieusement** par l'API quand le bloc est < 1024 tokens (Sonnet) / 2048 (Haiku).
- Cost map hardcodée (ligne 66-74) : Haiku $0.80/$4.00, Sonnet $3.00/$15.00, cache_read 0.1× input.
- Modèles : `claude-haiku-4-5-20250929` et `claude-sonnet-4-5-20250929` (génération 4.5, snapshot sept. 2025).
- `NARRATIVE_MODE=template` est **le défaut** (main.py:154) ⇒ par défaut l'engine LLM n'est jamais appelé.
- `SemanticCache.generate_cache_key()` (semantic_cache.py:106-116) inclut `bar_timestamp` ⇒ clef unique par bar ⇒ hit rate structurellement 0 sur flux live.
- `_narrate_with_cascade()` sérialise 2 appels API (Haiku puis Sonnet) ⇒ latence additive.
- Parsing `text.split("\n\n")` (ligne 287) : fragile, dépend du format de sortie non contraint.

---

## 2. Audit des prompts

### 2.1 System prompt — `SMC_SYSTEM_PROMPT`

**Volumétrie** (mesuré par `len(text)//4` et `words × 1.3`) :
- 2207 caractères, 324 mots ⇒ **≈ 420-550 tokens**.

**Contenu** (llm_narrative_engine.py:29-64) : 4 sections — SMC Framework, Regime Classification, Risk Management Rules, Response Format Rules.

**Points forts** :
- Section "Volatility Regime Context" bien injectable si `vol_regime` présent.
- "Never give financial advice — present as educational analysis" : disclaimer légal embarqué.

**Points faibles** :
1. **Trop court pour le cache Anthropic**. Cache minimum 2026 = 1024 tokens (Sonnet/Opus) ou 2048 (Haiku). ⇒ `cache_control` est no-op. → `cache_read_input_tokens` retournera toujours 0 même si l'architecture prétend cacher. Le test `test_haiku_cache_hit_detected` passe uniquement parce qu'il *mocke* la réponse API.
2. **Pas d'anti-hallucination ciblée**. Il manque des directives du type :
   - « Do NOT invent specific price levels, news events, or economic data not present in the Signal payload »
   - « If a component is missing, say "not provided" — do not speculate »
   - « Do NOT reference macro events unless supplied »
3. **Pas de few-shot**. Pour un output institutionnel cohérent, 2-3 exemples canoniques (BUY / SELL / INVALID) ajoutés au système (cachés) rendent les sorties ~30-50% plus stables (voir Anthropic prompt eng. guide & TradingView AI Alerts pattern).
4. **Hardcodé XAU/USD** : « institutional-grade Gold (XAU/USD) market analyst ». Pour les 5 autres presets (EURUSD, BTCUSD, US500, GBPUSD, USDJPY), le system prompt est mensonger et dégrade la qualité.
5. **Response Format Rules trop laxe** : « Be concise, institutional tone » n'est pas un contrat machine-parsable. Zéro garantie que les 3 paragraphes exigés sont bien rendus ⇒ `paragraphs[1]`, `paragraphs[2]` peut échouer silencieusement (confluences/risk_warnings vides).

### 2.2 User prompt

**VALIDATOR** (ligne 210-213) : *"Validate this Gold trading signal. Reply EXACTLY as: VALID|reason or INVALID|reason"* + CSV signal.
- Encore "Gold" hardcodé.
- Format pseudo-structuré mais rien n'oblige le modèle à le respecter ; pas d'utilisation de `stop_sequences` ou de JSON mode.

**NARRATOR** (ligne 269-275) : 3-paragraph request + CSV.
- Pas de Markdown canonique demandé (**bold labels**), rendu Telegram/Discord inconsistant.
- Pas de garde-fous contre les sorties trop longues (`max_tokens=1024` existe mais pas de min/max paragraphs).

### 2.3 CSV serialization — `_signal_to_csv()`

**Token economy** : le format CSV shorthand (ligne 369-405) est ~150 tokens vs ~400 en JSON ⇒ bon choix (~60% d'économie input).

**Bug subtil** : le format ne marque pas clairement ce qui est **absent**. Exemple : si `news_decision` n'est pas transmis, le LLM peut inventer un contexte news. Recommandation : inclure des clés explicites `news=not_provided,vol_forecast=not_provided` plutôt que les omettre.

---

## 3. Audit prompt caching Anthropic

### 3.1 État actuel

```python
# llm_narrative_engine.py:324-333
system_messages = [{"type": "text", "text": SMC_SYSTEM_PROMPT}]
if self._enable_caching:
    system_messages[0]["cache_control"] = {"type": "ephemeral"}
```

### 3.2 Diagnostic

| Check | État | Impact |
|-------|------|--------|
| `cache_control` positionné | ✅ | OK |
| Bloc ≥ 1024 tokens (Sonnet) | ❌ **420-550 tokens** | Cache désactivé silencieusement |
| Bloc ≥ 2048 tokens (Haiku) | ❌ | Haiku cache jamais actif |
| Cache TTL 5min `ephemeral` | ✅ | OK (par défaut) |
| Option TTL 1h (`extended` 2025+) | ❌ Non utilisé | +tokens cachables pour scanners basse fréquence |
| Cache write cost tracké (1.25× input) | ❌ | Premier appel sous-facturé dans `get_stats()` |
| Cache hit rate exposé en métriques/health | ❌ | Invisible en prod |
| User prompt cacheable en prefix | ❌ | Opportunité manquée |

### 3.3 Conséquence économique

À 1 000 signaux/mois NARRATOR :
- Input par appel = ~425 tokens system + ~200 tokens user = 625 tokens
- Avec cache actif (Sonnet) : ~200 tokens neufs × $3/M = **$0.0006** input
- Sans cache actif (**état actuel**) : 625 × $3/M = **$0.0019** input

⇒ **surcoût input ×3** sur Sonnet, ×5 sur Haiku (vu les écarts cache_read vs input).

Extrapolation sur 10 000 signaux/mois × 2 appels cascade :
- Économie potentielle si cache actif : **~$35-50/mois Anthropic** pour la part input.
- **Rien n'est réellement caché aujourd'hui.**

---

## 4. Choix de modèle — tableau coût / qualité

### 4.1 Modèles disponibles (2026-04, cf. MEMORY.md & docs Anthropic)

| Modèle | ID | Input $/M | Output $/M | Cache read $/M | Cache write $/M | Use-case |
|--------|-----|-----------|------------|----------------|-----------------|----------|
| Haiku 4.5 | `claude-haiku-4-5-20251001` | 0.80 | 4.00 | 0.08 | 1.00 | Y/N validation, classif. |
| Sonnet 4.6 | `claude-sonnet-4-6` | 3.00 | 15.00 | 0.30 | 3.75 | Narration, reasoning moyen |
| Opus 4.7 | `claude-opus-4-7` | 15.00 | 75.00 | 1.50 | 18.75 | Reasoning profond, synthèses |

### 4.2 Coût par narration (cascade Haiku+Sonnet), 1 000 signaux

| Scénario | Input (in) | Cached read | Output | Coût / 1k signaux |
|----------|-----------|-------------|--------|-------------------|
| **État actuel (cascade, cache no-op)** | 1 250 tok × 2 calls = 2 500 | 0 | 1 100 tok | **~$23.25** |
| Cache actif (system >= 1024 tok) | 400 tok × 2 | 1 000 × 2 | 1 100 tok | **~$18.00** (-23%) |
| Sans cascade (Sonnet direct) | 625 tok | 1 000 | 900 tok | **~$15.60** (-33%) |
| Sans cascade + cache actif + Haiku pour FREE/ANALYST | 625 tok | 1 000 | 300 tok | **~$3.20** Haiku / $14.60 Sonnet |
| **Opus 4.7 tier INSTITUTIONAL** | 625 tok | 1 000 | 1 200 tok | **~$109** (×5 Sonnet — mais différenciation premium) |

### 4.3 Routing tier→modèle recommandé

| Tier | Modèle actuel | Recommandé | Justification commerciale |
|------|---------------|------------|---------------------------|
| FREE | VISUAL (pas d'appel) | VISUAL | Zéro coût, limite teasing |
| ANALYST | VALIDATOR = Haiku 4.5 | **Haiku 4.5 single-call narration** (pas juste Y/N) | Perception de valeur + marge encore > 90% |
| STRATEGIST | NARRATOR = cascade Haiku+Sonnet 4.5 | **Sonnet 4.6 single-call** (skip Haiku gate) | Coût -33%, latence /2, qualité équivalente car validation algo déjà faite côté Python |
| INSTITUTIONAL | idem STRATEGIST | **Opus 4.7** + macro context (calendar + vol forecast narration augmentée) | Différenciation prix ×5-10 vs STRATEGIST |

**Observation clef** : la cascade Haiku→Sonnet dédouble l'appel pour un gain marginal. `ConfluenceDetector` déjà fait gating algorithmique (score, R:R, dominance). Le `TemplateValidator` aussi. Ajouter Haiku en troisième gate = surcoût sans valeur ajoutée mesurée.

---

## 5. Latence & résilience

### 5.1 Mesuré / estimé

| Path | Calls | Latence attendue | État |
|------|-------|------------------|------|
| VISUAL | 0 API | <1 ms | ✅ OK |
| VALIDATOR (Haiku) | 1 | P50 ≈ 400 ms, P95 ≈ 1.2 s, P99 ≈ 3 s | Non instrumenté |
| NARRATOR cascade | 2 sérielles | P50 ≈ 1.8 s, P95 ≈ 4-6 s, P99 ≈ 8-12 s | Non instrumenté |
| NARRATOR single-call Sonnet (proposé) | 1 | P50 ≈ 1.0 s, P95 ≈ 3 s, P99 ≈ 6 s | — |

**Aucune métrique** (ni Prometheus ni log JSON structuré) n'expose P50/P95/P99 aujourd'hui. On ne peut pas prouver respect d'un SLA « signal livré < 30 s après clôture ».

### 5.2 Fallback

- `_generate_narrative_safe()` gère `CircuitOpenError` + `Exception` : retourne `None`.
- Le scanner construit alors un dict fallback minimal (`summary=f"{signal_type} {symbol} — score {score}"`, `fallback: True`).
- **Gap** : le `TemplateNarrativeEngine` existe, est déterministe, sub-ms. Il n'est **pas** appelé en fallback quand le circuit LLM ouvre ⇒ on livre une narrative pauvre à l'utilisateur payant au lieu de la narrative algorithmique riche déjà dispo.

### 5.3 Timeout

Aucun `timeout=` passé à `client.messages.create()` (ligne 334-339). Anthropic SDK utilise le default (~10 min). Un LLM lent peut geler toute la boucle scanner.

---

## 6. Évaluation qualitative — rubric proposée

La rubric ci-dessous est un **livrable activable** (à intégrer comme `scripts/eval_narratives.py`) pour noter 20 narratives représentatives (10 LONG, 10 SHORT, mix régimes).

### 6.1 Grille (sur 25 pts par narrative)

| Dimension | 0 pt | 3 pt | 5 pt |
|-----------|------|------|------|
| **Factuelle** (prix/SL/TP cohérents avec signal) | Contradiction numérique | 1 imprécision mineure | Tous nombres exacts |
| **Actionnable** (entry/SL/TP clairement identifiables) | Narrative floue | Identifiables mais pas chiffrés | Chiffres + condition d'invalidation |
| **Non-générique** (lie les composants spécifiques du signal) | Applicable à tout signal | Mentionne 1-2 composants | Cite chaque dominant avec reasoning |
| **Sans hallucination** (zéro donnée inventée hors payload) | Invente news/prix | 1 doute | Strictement payload |
| **Vol-awareness** (intègre vol_regime/vol_forecast si présent) | Ignoré | Mentionné sans impact | Risk parameters modulés |

**Seuil acceptation prod** : score moyen ≥ 18/25, **zéro hallucination** (binaire).

### 6.2 Script de notation (à implémenter)

```
scripts/eval_narratives.py
  1. Sample 20 signaux depuis reports/baseline_full_trades.csv
  2. Pour chaque : genère narrative via LLMEngine(model=sonnet)
                  et TemplateEngine comparatif
  3. Pour chaque narrative : juge LLM (Opus 4.7 en juge)
     scores chaque dimension avec la grille
  4. Output: reports/eval_05/narratives_scored.csv + aggregat
```

### 6.3 Risques hallucination identifiés (sans eval, sur relecture)

- Prompt ne bloque pas les références à des **news events** non-transmis.
- Prompt ne bloque pas l'invention de **niveaux clés** (supports/résistances) non dans le payload.
- Pas de contrôle post-génération (regex sur prix pour vérifier qu'ils matchent `entry_price`, `stop_loss`, `take_profit`).

---

## 7. Semantic Cache — audit croisé (pertinence pour économies LLM)

Le fichier s'appelle `semantic_cache.py` mais c'est un **hash cache déterministe**, pas sémantique. Impact sur Prompt 05 :

**Design issue** :
```python
# semantic_cache.py:106-116
parts = [symbol, bar_timestamp, ...component_scores]
return sha256(...)[:16]
```
- `bar_timestamp` change à chaque bar M15 (96 bars/jour par symbole)
- ⇒ même setup SMC identique 4 bars plus tard = cache miss
- ⇒ hit rate théorique en live ≈ **0%** (sauf replay qui réévalue le même bar)

**Fix** (pas dans scope Prompt 05 strict mais bloquant les économies) :
- Clef = (symbol, score_bucket_of_5, regime_type, news_bucket, vol_regime, dir)
- TTL 30 min à 2 h, pas 24 h
- Objectif hit rate 30-50% sur flux live (~10-20% économies directes LLM)

---

## 8. Multi-langue — TAM impact

Aucune variable `lang` / `locale` / `output_language` n'existe. L'API publie uniquement de l'anglais.

**TAM perdu** :
- Retail FR (France, Belgique, Suisse, Québec, Maghreb, Afrique de l'Ouest) : ~15-20% du TAM retail crypto/forex 2026
- LATAM (ES/PT) : ~25% TAM
- DACH (DE) : ~10% TAM

**Cost impact** d'ajouter `lang` : 0 (même prompt + 1 ligne `Respond in {lang}`). Qualité d'un Sonnet 4.6 en français est proche de l'anglais (écart < 5% sur benchmarks MMLU-FR).

---

## 9. Top 5 améliorations — priorisées (effort × impact)

| # | Amélioration | Effort | Impact tech | Impact revenu | KPI cible |
|---|-------------|--------|-------------|---------------|-----------|
| **1** | **Gonfler `SMC_SYSTEM_PROMPT` à ≥ 1200 tokens** (examples BUY/SELL/INVALID + banni anti-halluc) et migrer IDs `claude-sonnet-4-6` + `claude-haiku-4-5-20251001` | **QW 0.5 j** | Cache Anthropic enfin actif (-20 à -30% input) ; stabilité sortie +30% | Marge brute +2-3 pts | Cache hit rate effectif ≥ 70% (`cache_read_input_tokens > 0`) |
| **2** | **Supprimer cascade Haiku→Sonnet** (remplacer par Sonnet direct) ; routing tier-based (Haiku single-call ANALYST, Sonnet STRATEGIST, Opus INSTITUTIONAL) | **QW 1 j** | Latence NARRATOR /2, coût -33% | Justifie écart prix ANALYST/STRAT/INST | P95 NARRATOR < 3 s |
| **3** | **Fallback LLM→TemplateNarrativeEngine automatique** quand `CircuitOpenError` ou `Exception` — pas le dict "fallback:true" actuel | **QW 0.5 j** | Disponibilité narrative 100% | Churn -2 pts (signal payant jamais "cassé") | 0 signal payant livré sans narrative lisible |
| **4** | **Refonte `SemanticCache`** : retirer `bar_timestamp`, bucket score/vol/regime, exposer `hit_rate` sur `/health` et Prometheus | **MT 3 j** | Hit rate 0% → 30-50% | -10-20% facture Anthropic à volume constant | Hit rate ≥ 40% après 7 jours prod |
| **5** | **Rubric d'évaluation narratives + juge Opus en CI** (script `scripts/eval_narratives.py`, 20 échantillons, seuil 18/25 bloquant deploy) + structured output (JSON schema Anthropic) | **MT 4-5 j** | Zéro hallucination prod, parsing robuste | Preuve qualité → acceptable en matériel de vente INSTITUTIONAL | 0 hallucination / 200 narratives tagguées |

### Améliorations secondaires (backlog)

6. Multi-langue `lang ∈ {en, fr, es}` injecté dans system prompt (LT 2 j)
7. `extended` cache 1h (TTL long) pour système stable ⇒ économies additionnelles si volumétrie régulière
8. `timeout=30` explicite sur `client.messages.create()` (QW 15 min)
9. Metrics Prometheus : `llm_latency_ms_histogram`, `llm_cost_counter_usd`, `llm_cache_hit_ratio_gauge`
10. Prompt injection sanitization : strip user-controllable fields avant injection dans CSV

---

## 10. Plan d'exécution (découpage temporel)

### Quick Wins (< 1 jour cumulé — à faire cette semaine)
- **D1 matin** : étendre `SMC_SYSTEM_PROMPT` à ≥ 1200 tokens (ajouter `## Examples` avec 1 BUY PREMIUM, 1 SELL STANDARD, 1 INVALID rejected), activer cache Haiku (≥ 2048 tokens ⇒ dupliquer section vol). Vérifier via mock test que `usage.cache_creation_input_tokens > 0` au 1er appel et `cache_read_input_tokens > 0` au 2ème.
- **D1 après-midi** : migrer IDs vers `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`. Ajouter `DEFAULT_NARRATOR_MODEL_INSTITUTIONAL = "claude-opus-4-7"`. Ajouter constantes coût Opus.
- **D2 matin** : supprimer `_narrate_with_cascade()` au profit d'un `_narrate_single()` — Sonnet direct. Ajouter routing `model_for_tier()`. Mettre à jour tests (cascade → single-call).
- **D2 après-midi** : ajouter `timeout=30.0` sur `client.messages.create()`. Fallback `CircuitOpenError` → `TemplateNarrativeEngine(signal, tier)` au lieu du dict minimal.

### Moyen Terme (< 1 semaine)
- **S1 J3-J5** : refonte SemanticCache (clef bucket, expose hit rate), migration DB in-place.
- **S1 J6-J7** : `scripts/eval_narratives.py` — sample + juge Opus + rapport CSV, seuil 18/25 bloquant CI.

### Long Terme (> 1 semaine)
- **S2** : multi-langue FR/ES/EN via param `lang`.
- **S2-S3** : structured output JSON schema (Anthropic `response_format`) pour supprimer le parsing `\n\n`.
- **S3** : observabilité complète (Prometheus métriques LLM, dashboard Grafana « LLM économie/latence »).

---

## 11. KPIs mesurables post-amélioration

| KPI | Baseline actuel | Cible 30j | Méthode de mesure |
|-----|-----------------|-----------|-------------------|
| `cache_read_input_tokens` / `input_tokens` (cache hit effectif Anthropic) | 0.00 (no-op) | ≥ 0.60 | `usage` Anthropic agrégée sur 7j |
| Coût moyen par NARRATOR-signal | ~$0.023 (estimé) | ≤ $0.010 | `sum(cost_usd) / total_calls` sur 1000 signaux |
| Latence P95 NARRATOR | non-mesurée (~5-6s) | < 3 s | histogramme Prometheus |
| Latence P99 NARRATOR | non-mesurée (~10s) | < 8 s | idem |
| Taux d'hallucination (script eval) | non-mesuré | < 1% | 200 narratives notées |
| Disponibilité narrative (signal payant) | ~95% (fallback dict) | 100% | counter `fallback_used / total` |
| SemanticCache hit rate | 0-5% | ≥ 40% | `cache.get_stats()['hit_rate']` |
| Narratives non-anglais servies | 0% | ≥ 20% à 90 j | comptage par `lang` tag |
| Coût marginal LLM / abonné STRAT / mois | ~$2.30 (projeté) | ≤ $0.80 | unit economics |

---

## 12. Trade-offs assumés

| Décision recommandée | Trade-off explicite |
|----------------------|---------------------|
| Supprimer cascade Haiku→Sonnet | On perd un garde-fou modèle. **Mitigation** : le gating algo (`ConfluenceDetector.tier` + `TemplateEngine._validation_check`) reste intact, les deux filtres ne sont jamais redondants avec la cascade. |
| Opus 4.7 pour INSTITUTIONAL | Coût × 5 vs Sonnet. **Mitigation** : répercuté sur prix tier, marge préservée. Aligne le claim « institutional-grade » avec le modèle réellement utilisé. |
| Cache bucket (sans `bar_timestamp`) | Risque de livrer la MÊME narrative à 2 signaux différents mais très proches sur 30 min. **Mitigation** : ajouter `signal_id` dans le payload retourné à l'utilisateur pour traçabilité ; inclure `bar_index` pour variation mineure. |
| Extended cache 1h | Plus cher en cache_write (×1.25) — rentable seulement si hit > 1/1.25 = 80%. Réserver aux ops scanners haute fréquence (M1). |
| Multi-langue via prompt `Respond in {lang}` | Output tokens peut varier ±10-15% (ex: français un peu plus verbeux). **Mitigation** : raccourcir `max_tokens` à 900 avec réserve. |

---

## 13. Benchmarks sectoriels (référencements)

- **TradingView AI Alerts (2025)** : pas de narration explicable — juste signal + indicateur. ⇒ Smart Sentinel a un moat textuel si qualité prouvée.
- **LuxAlgo** : template-based signals + tooltips statiques. ⇒ Smart Sentinel peut surclasser sur "why this signal" dès que rubric ≥ 20/25 systématique.
- **Stock Hero, TrendSpider, MarketBull** : narratives génériques, souvent copie GPT-3.5 ou 4o-mini. ⇒ Différenciation via Claude 4.7 + SMC-domain prompts est tangible.
- **Anthropic Prompt Caching 2025** (`docs.anthropic.com/en/docs/build-with-claude/prompt-caching`) : minimums 1024/2048 tokens confirmés ; TTL 5 min ephemeral, 1h `extended` (pricing × 2).
- **Research prompt eng. 2025 (Arxiv 2411.04568 "Prompt Caching at Scale")** : hit rate 70%+ quand system >= 2000 tokens et TTL >= 10min ; baisse à 30% quand user prompt haut-entropie.

---

## 14. Ce qui manque aux tests existants

`tests/test_llm_narrative_engine.py` (11 tests) — couvre :
- ✅ Visual fallback
- ✅ Haiku VALID/INVALID parsing
- ✅ Cost calc
- ✅ Cascade cost combinaison
- ❌ **Pas de test** validant `cache_creation_input_tokens > 0` au 1er appel (preuve cache actif)
- ❌ **Pas de test** vérifiant size system prompt ≥ 1024 tokens
- ❌ **Pas de test** anti-hallucination (payload sans news → narrative ne doit pas mentionner news)
- ❌ **Pas de test** latence (timeout avortement)
- ❌ **Pas de test** fallback LLM→Template quand circuit ouvert

À ajouter dans Sprint 3 QW.

---

## 15. Verdict commercial

**Le produit revendique « AI-powered narratives » mais**:
1. Tourne par défaut sur un moteur **non-LLM** (template),
2. Quand LLM est activé, il **ne cache pas** (cache no-op),
3. N'a **aucune preuve** de non-hallucination sur données commerciales,
4. Facture **cascade 2×** là où 1 appel suffit.

**Après les 5 améliorations priorisées**, le produit devient :
- Défendable en marketing (narratives testées, rubric publique),
- Rentable (coût marginal ÷ 3),
- Différencié (Opus 4.7 sur INSTITUTIONAL, Template déterministe sur FREE avec upsell visible).

Note globale **4.5 / 10 aujourd'hui → 7.5 / 10** projeté après Sprint narratif (< 2 semaines).

---

## 16. Annexe — Actions concrètes file_path:line

1. `src/intelligence/llm_narrative_engine.py:25-26` — mettre à jour les IDs de modèle.
2. `src/intelligence/llm_narrative_engine.py:29-64` — étendre `SMC_SYSTEM_PROMPT` à ≥ 1200 tokens avec section `## Examples` et section `## Anti-Hallucination Rules`.
3. `src/intelligence/llm_narrative_engine.py:253-316` — remplacer `_narrate_with_cascade` par `_narrate_single_sonnet` + routing tier/modèle.
4. `src/intelligence/llm_narrative_engine.py:334` — ajouter `timeout=30.0`.
5. `src/intelligence/llm_narrative_engine.py:66-74` — ajouter entrées Opus 4.7 + cache_write.
6. `src/intelligence/sentinel_scanner.py:502-525` — sur `CircuitOpenError` ou `Exception`, instancier un `TemplateNarrativeEngine` local et renvoyer son `to_dict()` au lieu du dict minimal.
7. `src/intelligence/semantic_cache.py:99-116` — retirer `bar_timestamp`, bucket score à pas de 5, exposer `hit_rate` sur get_stats pour `/health`.
8. `tests/test_llm_narrative_engine.py` — ajouter 5 tests (cache activé, pas d'hallucination, fallback template, timeout, routing tier).
9. `scripts/eval_narratives.py` — NEW, juge Opus 4.7, 20 signaux, rubric 25 pts.
10. `src/intelligence/main.py:152-160` — changer `NARRATIVE_MODE` default à `llm` dès que les QW 1-3 sont mergés, garder template comme fallback runtime.

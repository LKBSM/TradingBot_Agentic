# PROMPT — Final Go/No-Go Audit, Smart Sentinel AI

> **Mode d'emploi** : copier-coller le bloc ci-dessous dans une nouvelle session Claude Code, dans le dossier `C:\MyPythonProjects\TradingBOT_Agentic`. Pas de pré-amorçage, pas de petites questions, pas de "what would you like" — Claude doit livrer le rapport directement.

---

# Rôle

Tu es un **senior staff engineer + tech-business advisor** mandaté pour rendre **un verdict Go/No-Go indépendant** sur le déploiement commercial de **Smart Sentinel AI**. Tu n'es pas là pour rassurer le founder. Tu es là pour le sauver d'un lancement raté.

Tu as l'autorité de dire "Non" et d'imposer un report. Si tu hésites, choisis "Non" — un mauvais lancement coûte 100× plus qu'un report d'un mois.

# Contexte produit

**Smart Sentinel AI** est un SaaS d'intelligence de marché propulsé par IA, destiné aux traders particuliers et semi-professionnels (focus initial XAU/USD M15).

**Pipeline** :
```
DataProvider → SmartMoneyEngine (BOS/CHOCH/OB/FVG)
            → ConfluenceDetector (scoring 0-100)
            → VolForecaster (HAR-RV / LightGBM / Hybrid)
            → LLMNarrativeEngine (Claude API)
            → SemanticCache → SignalStore (SQLite) → Telegram
```

**4 tiers** prévus :
- FREE ($0)
- ANALYST ($49/mo)
- STRATEGIST ($99/mo)
- INSTITUTIONAL ($149/mo)

**6 instruments** présets : XAUUSD, EURUSD, BTCUSD, US500, GBPUSD, USDJPY.
**Stack** : Python 3.11, FastAPI, SQLite, Anthropic Claude (Opus 4.7 / Sonnet 4.6 / Haiku 4.5), Telegram Bot, Railway.
**Founder** : solo, FR-natif, PhD-level expectations, code propre (1 366+ tests, 0 régression).

# Ce que tu dois savoir AVANT de commencer

L'équipe a déjà produit **20+ rapports d'évaluation** dans `reports/eval_*.md`. Lis-les en priorité avant tout autre fichier — ils contiennent le vrai état du produit.

**Synthèse des verdicts existants (notes /10)** :

| # | Domaine | Note | Findings clés |
|---|---------|------|---------------|
| 01 | Architecture orchestration | — | TESTING_MODE=1 par défaut = auth bypass risk |
| 02 | ConfluenceDetector | — | **Score à ZÉRO pouvoir prédictif** (Pearson −0.023, Brier > baseline) |
| 03 | Smart Money Engine | — | BOS/CHOCH OK post-fix, longs/shorts asymétriques |
| 04 | Volatility Forecasting | — | Claims "20-35 % RMSE improvement" non validés OOS |
| 05 | LLM Narrative Engine | — | `cache_control` no-op (system 500 < 1024 min), cascade Haiku→Sonnet redondante |
| 10-15 | API/Auth/Store/Tel/CB/Sec | — | 15 bugs non-évidents, 5 modules bloquants go-live |
| 21 | Performance & Scalabilité | **5.5** | Sync I/O dans async (`narratives.py:155`), RateLimiter en RAM divergent multi-worker |
| 22 | Deployment & Infra | **4.5** | Procfile pointait sur ancien RL bot (corrigé), nixpacks ignore Dockerfile (corrigé), TESTING_MODE fail-closed ajouté |
| 23 | MLOps Pipeline | **4.5** | **4 train/serve skews** identifiés (sessions, calendrier USD, blend, hybrid) — jusqu'à 15 % RMSE |
| 24 | Unit Economics | **5.5** | Institutional $149 sous-pricé (marge 48.5 % vs cible 80 %), Dukascopy non-commercial = blocker légal |
| 25 | PMF / ICP | **4.5** | "Pas prêt pour une niche payante", beachhead conditionnel = XAU SMC FR retail $20-49 |

**Moyenne 4.9/10**.

**Faits-réalité incompressibles** (issus du replay 7 ans 2019-2025) :
- Configuration production (seuil score=75) : **0 trade émis sur 7 ans de XAU M15**.
- Score max atteint en replay : **55.5/100** (plafond 70 car composants News+Vol=0 hors-temps-réel).
- Profit Factor max observé toutes configs confondues : **0.96** (donc perdante).
- Données XAU 2019-2025 : couverture **63 %** seulement (feed corrompu).
- Verdict mémoire : **"non commercialisable en l'état"** (2026-04-24).

**Fixes P0 récents (2026-04-26)** :
- `Procfile` + `railway.toml` corrigés (entry Sentinel, builder DOCKERFILE).
- `src/intelligence/main.py` : `assert_safe_production_config()` ajouté — exit 2 si `ENVIRONMENT=production` + `SENTINEL_TESTING_MODE=1`.
- `.dockerignore` créé (exclut `.env`, `data/*.csv`, `replay_*`, `models/`, etc.).
- `docker-compose.yml` : ports 8080→8000 alignés avec `main.py`.

# Mission

Produire **un rapport unique** : `reports/GO_NO_GO_DECISION.md` (~400-600 lignes), structuré exactement comme indiqué en section "Format de sortie" plus bas.

Tu dois rendre 4 verdicts indépendants, sur **7 axes critiques** :

1. **Performance produit** (le signal gagne-t-il de l'argent en backtest honnête, hors-sample ?)
2. **Performance technique** (l'app tient la charge, latence, uptime ?)
3. **Sécurité** (peut-on shipper sans risque de fuite/auth-bypass/RGPD ?)
4. **Économie unitaire** (marge brute > 70 % réaliste à chaque tier ?)
5. **Conformité légale** (license data, "conseil en investissement" risk, RGPD)
6. **Product/Market Fit** (un ICP réel paye le prix demandé pour la promesse actuelle ?)
7. **Maturité opérationnelle** (déploiement, observabilité, MLOps, support solo soutenable ?)

Pour chaque axe, livre :
- Un score /10
- Le top 1-3 blocker(s) avec **citation file:line ou chiffre du replay**
- Un verdict micro : **GO / GO-IF / NO-GO**

Puis un **verdict global** parmi 4 options :

- **🟢 GO** — déploiement immédiat sans réserve. Critère : 7/7 axes ≥ 7/10 ET 0 NO-GO axe.
- **🟡 GO-IF** — déploiement conditionnel après fix précis. Liste exhaustive des conditions, chacune chiffrée et datée. Critère : ≥ 5/7 axes ≥ 6/10 ET 0 axe < 4/10.
- **🟠 SOFT-LAUNCH** — beta gratuite fermée (max 50 users), pas de Stripe, collecte feedback. Critère : produit marche techniquement mais PMF non validé OU PnL < seuil.
- **🔴 NO-GO** — report ≥ 90 jours, retour à la planche. Critère : ≥ 1 axe < 4/10 OU PF live < 1.0 OU blocker légal/sécurité.

Une seule case à cocher. Pas de "ça dépend".

# Méthodologie OBLIGATOIRE

1. **Lire d'abord, juger ensuite.** Avant tout verdict, lis :
   - `MEMORY.md` (si présent à `C:\Users\bessa\.claude\projects\C--MyPythonProjects-TradingBOT-Agentic\memory\MEMORY.md`)
   - `reports/eval_21_performance.md`, `reports/eval_22_deployment.md`, `reports/eval_23_mlops.md`, `reports/eval_24_unit_economics.md`, `reports/eval_25_pmf_icp.md`
   - `reports/eval_02_confluence.md`, `reports/eval_05_llm.md`, `reports/eval_15_security.md`
   - `BUSINESS_PLAN_SMART_SENTINEL.md`, `COMMERCIALIZATION_REPORT.md`
   - `Procfile`, `railway.toml`, `infrastructure/Dockerfile`, `.dockerignore`, `infrastructure/docker-compose.yml`
   - `src/intelligence/main.py` (fonctions `main()` + `assert_safe_production_config()`)
   - `src/api/auth.py` (lignes 1-50, vérifier `TESTING_MODE` default)
   - Au moins 2 fichiers `replay_*.json` à la racine pour le PnL réel

2. **Vérifie 3 affirmations toi-même** (ne fais pas confiance aveuglément aux rapports) :
   - Lance `pytest tests/test_smoke_e2e.py -q` : doit passer 9/9.
   - Lance `python -c "import os; os.environ['ENVIRONMENT']='production'; os.environ['SENTINEL_TESTING_MODE']='1'; from src.intelligence.main import assert_safe_production_config; assert_safe_production_config()"` : doit `sys.exit(2)`.
   - Grep `Pearson` dans `reports/` pour voir si la confluence à pouvoir prédictif zéro est confirmée.

3. **Cite ou tais-toi.** Aucune affirmation sans `file:line` OU chiffre du replay. Exemples :
   - ✅ "Score 55.5/100 max sur replay 7 ans (`reports/audit_backtest_2026_04_24.md` ou réplica)" 
   - ✅ "Sync `requests.post` dans event loop (`src/api/routes/narratives.py:155`)"
   - ❌ "L'app pourrait avoir des problèmes de performance" (vague — rejeté)

4. **Anti-patterns à proscrire** :
   - "Looks production-ready overall" sans détail chiffré
   - Liste de "best practices" génériques (Docker, CI/CD…) sans rattachement au repo
   - Recommander de tout réécrire ("rewrite in Rust") au lieu de prioriser
   - Cacher des NO-GO derrière des "à surveiller" diplomatiques
   - Recommander d'ajouter Kubernetes / Kafka / microservices à un solo founder

# Format de sortie attendu

Écris **un seul fichier markdown** : `reports/GO_NO_GO_DECISION.md`.

Structure exacte (copie ce squelette) :

```markdown
# Smart Sentinel AI — Final Go/No-Go Decision

**Date**: <YYYY-MM-DD>
**Auditeur**: Independent staff engineer (Claude)
**Périmètre**: 7 axes, decision-bound

---

## 🎯 Verdict global

**Décision** : 🟢 GO / 🟡 GO-IF / 🟠 SOFT-LAUNCH / 🔴 NO-GO  
**Confiance** : Faible / Moyenne / Élevée  
**Réversibilité** : <que coûte un revirement à J+30 ?>

**TL;DR (3 phrases max)** : <résumé brutal de l'état + raison du verdict + prochaine étape>

---

## 📊 Scorecard

| # | Axe | Score /10 | Verdict | Top blocker |
|---|-----|-----------|---------|-------------|
| 1 | Performance produit (PnL) | X.Y | GO/IF/NO | <citation file:line ou chiffre> |
| 2 | Performance technique | X.Y | … | … |
| 3 | Sécurité | X.Y | … | … |
| 4 | Économie unitaire | X.Y | … | … |
| 5 | Conformité légale | X.Y | … | … |
| 6 | Product/Market Fit | X.Y | … | … |
| 7 | Maturité opérationnelle | X.Y | … | … |

**Moyenne pondérée** : X.Y/10 (poids : PnL ×3, autres ×1)

---

## Axe 1 — Performance produit

### Constat (chiffré)
- PF replay 7 ans : … (source : `reports/audit_backtest_2026_04_24.md` ou `replay_*.json`)
- Score max atteint : …
- Win rate, Sharpe, max drawdown, Calmar : …
- Asymétrie long/short : …

### Conséquence commerciale
- Avec PF X.YZ, un user perdant en moyenne $Y/mois avant churn.
- Refund risk : …

### Verdict micro
**GO / GO-IF / NO-GO** parce que <raison en 1 phrase>.

[Répéter le bloc pour les axes 2 à 7]

---

## 🚨 Blockers durs (NO-GO si non levés)

Liste ordonnée. Chaque blocker = 1 paragraphe :
- **B1** — <titre> — <fichier:ligne ou chiffre> — <effort fix> — <responsable>
- **B2** — …

---

## ⚠️ Conditions GO-IF (si verdict 🟡)

Pour passer du verdict actuel à 🟢, tu DOIS faire ceci dans cet ordre :

| # | Action | Effort | Owner | Deadline | Critère acceptation |
|---|--------|--------|-------|----------|---------------------|
| 1 | … | 0.5 j | founder | J+3 | <métrique mesurable> |
| 2 | … | … | … | … | … |

Pas de "améliorer X" — uniquement actions binaires (fait/pas fait).

---

## 🛑 Ce qu'il NE faut PAS faire (anti-recos)

3-5 items. Ce que le founder est tenté de faire mais qui nuit :
- Ne PAS lancer Stripe avant la résolution de B1.
- Ne PAS investir en paid acquisition (Google/Meta) tant que LP CVR non mesurée.
- …

---

## 📅 Plan d'exécution

### Si 🟡 GO-IF — 21 jours
- **J+1 à J+7** : (B1, B2, B3)
- **J+8 à J+14** : validation, soft-launch fermé 10 users
- **J+15 à J+21** : decision review

### Si 🟠 SOFT-LAUNCH — 60 jours
- **Mois 1** : beta gratuite 25 users, mesures (NPS, retention D7, signaux ouverts/cliqués)
- **Mois 2** : decision GO conditionnel sur seuils chiffrés ci-dessous

### Si 🔴 NO-GO — 90 jours minimum
- Plan de retour : <quoi attaquer en priorité, séquence>

---

## 📈 KPIs à instrumenter AVANT lancement (non négociables)

| KPI | Cible | Outil | Alerte si … |
|-----|-------|-------|-------------|
| PF live (rolling 30j) | > 1.20 | replay quotidien | < 1.0 → pause Telegram |
| P99 latence scanner tick | < 250 ms | Prometheus | > 500 ms 3 minutes |
| $/MAU | < $0.80 | spreadsheet hebdo | > $1.50 |
| Cache hit rate LLM | > 40 % | log Anthropic | < 20 % → rebuild prompts |
| Auth bypass attempts | 0 | logs auth.py | toute occurrence |
| NPS (signaux Telegram) | > 30 | bouton 👍/👎 | < 0 |

---

## 📝 Hypothèses fragiles & inconnues

5-10 items. Ce que tu n'as PAS pu vérifier mais qui pourrait tout changer :
- "Pricing Anthropic 2026 : non vérifié, peut faire varier marge ±40 %"
- "Niche XAU SMC FR retail : 0 interview faite — WTP estimée projetée"
- …

---

## 💬 Message au founder (10 lignes max)

Ton honnête. Si NO-GO, dis-le sans détour. Si GO, valide-le sans flatterie.
Exemple : "Tu as un produit techniquement propre mais commercialement creux. Le PF 0.96 n'est pas un bug à corriger en une semaine — c'est une preuve que la stratégie scoring n'a pas d'edge. Repousse de 90 jours, refonds le scoring (cf. eval_02), valide sur out-of-sample 2026 vrai, puis re-soumets cette décision."

---

**Fin du rapport. Cible : 400-600 lignes. Si tu débordes, c'est que tu vagues.**
```

# Décision : critères chiffrés stricts

Pour fixer ton verdict global, applique ces seuils :

| Verdict | Conditions ALL true |
|---------|---------------------|
| 🟢 **GO** | PF replay ≥ 1.30 ; 7/7 axes ≥ 7/10 ; 0 blocker légal ouvert ; cache hit rate mesuré ≥ 40 % ; ≥ 5 interviews ICP confirmant WTP ≥ tier price |
| 🟡 **GO-IF** | PF replay ≥ 1.10 ; ≥ 5/7 axes ≥ 6/10 ; 0 axe < 4 ; blockers légaux datés < 14j |
| 🟠 **SOFT-LAUNCH** | PF replay ≥ 1.0 ; technique fonctionne ; PMF non validé ; pas de Stripe |
| 🔴 **NO-GO** | PF replay < 1.0 OU ≥ 1 axe < 4 OU blocker légal sans plan OU 0 ICP interviewé |

**État actuel connu : PF max 0.96 → mathématiquement, le seuil 🟠 SOFT-LAUNCH n'est pas atteignable sans intervention sur le scoring.** Si après ta vérif, le PF est toujours < 1.0, ton verdict ne peut PAS être 🟢 ni 🟡. C'est mécanique.

# Hard rules

1. **Tu ne peux pas écrire de code.** Pas d'`Edit`, pas de `Write` sur les sources. Tu produis UN seul fichier : `reports/GO_NO_GO_DECISION.md`.
2. **Tu ne flattres pas.** Si la moyenne sectorielle est 4.9/10, dis-le.
3. **Tu lis les vrais fichiers.** Si un rapport dit "PF 0.96", confirme avec un `replay_*.json` ou rejette.
4. **Tu cites file:line.** Toute affirmation non sourcée = à supprimer.
5. **Tu rends la décision.** Pas "à voir avec l'équipe" — il n'y a pas d'équipe, c'est un solo founder qui doit décider lundi matin.
6. **Length cap : 600 lignes.** Au-delà, tu vagues.

# Examples (mini)

**❌ Mauvais output (flatteur, vague)** :
> "Smart Sentinel AI shows promising architecture and a thoughtful approach to AI-powered trading signals. With some additional polish around testing and monitoring, it could be ready for an initial launch. The team should consider implementing a robust CI/CD pipeline."

**✅ Bon output (factuel, daté, actionnable)** :
> "**Verdict : 🔴 NO-GO 90 jours.** Le scoring confluence a 0 pouvoir prédictif (Pearson −0.023, `reports/eval_02_confluence.md` §4) et le PF replay max est 0.96 sur 7 ans (`reports/audit_backtest_2026_04_24.md`). Lancer maintenant = vendre un signal aléatoire à $49/mo. Refonder scoring (B1, 4-6 sem) puis revalider OOS 2026 (B2, 2 sem) puis interviewer 5 ICP (B3, 2 sem). Re-décide 2026-07-26."

# Démarrage

Commence MAINTENANT par lire `MEMORY.md` puis les rapports cités. Ne demande aucune confirmation, ne propose pas de plan — exécute. Le rapport doit être prêt en une seule passe.

**Tu rends ta décision dans `reports/GO_NO_GO_DECISION.md`. C'est tout.**

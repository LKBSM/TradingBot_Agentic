# Plan de Commercialisation — Catégorie 20: Product UX & Frontend

> **Date** : 2026-05-23
> **Auteur** : Agent Catégorie 20 — Product UX & Frontend (sprint commercialisation)
> **Branche** : `institutional-overhaul`
> **Périmètre** : mockups B2C/B2B, multi-view modes (FOCUS / CO-PILOT / EXPERT), webapp frontend, design system, onboarding, démo, dashboard, mobile, B2B portal, status page.
> **Hors périmètre** : algo (cat. 2-4), data infra (cat. 5), backtest (cat. 6), risk (cat. 7), API backend (cat. 8), LLM (cat. 9), auth (cat. 10), delivery push (cat. 11), observabilité (cat. 12), tests infra (cat. 13), MTF (cat. 14), news (cat. 15), state machine (cat. 16), caching perf (cat. 17), compliance (cat. 18), MLOps (cat. 19), GTM (cat. 1).
> **Règle UX persistante** : *toujours 3 vues — FOCUS / CO-PILOT / EXPERT — sur la même donnée*. Source : `C:\Users\bessa\.claude\projects\C--MyPythonProjects-TradingBOT-Agentic\memory\feedback_multi_view_ux.md:1-22`. Validation utilisateur du 2026-05-18.
> **Verdict initial UX & Frontend** : 🔴 NO-GO. 4 mockups statiques (un seul comportant le toggle 3-vues : `mockups/v3/best_concept_demo.html`), **aucun frontend en production**, aucun design system codifié, aucune landing publique, aucune page onboarding, aucun B2B portal, aucun mobile.

---

## 0. Synthèse exécutive (TL;DR)

| Dimension UX | État actuel | Cible 12 sem | Source |
|---|---:|---:|---|
| Frontend webapp production | 🔴 0 (rien de servi) | 🟢 Next.js app router `/app` + `/dash` + `/auth` | `src/api/routes/webapp.py:1-100` rend du HTML inline SSR mais pas une SPA |
| Toggle 3-vues FOCUS/CO-PILOT/EXPERT | 🟡 mockup unique HTML | 🟢 implémenté en production, persistant `user_preferences` | `mockups/v3/best_concept_demo.html:1-2246` |
| Design system codifié | 🔴 inline CSS répété 5×, pas de tokens | 🟢 Figma + Tailwind tokens + shadcn/ui | Comparer `mockups/webapp_b2c.html:8-131` et `mockups/v3/best_concept_demo.html:15-42` : palette divergente |
| Landing page publique | 🔴 0 (domaine réservé, pas pointé) | 🟢 home + pricing + méthodologie + about | Cf. cat. 1, `reports/commercialization_sprint/01_commercialization_gtm.md:60-72` |
| Onboarding signup → first signal | 🔴 0 | 🟢 5 écrans, time-to-aha < 90s | Aucun fichier existant |
| B2B portal (clés API, usage, webhooks, docs) | 🔴 0 | 🟢 portal `/b2b` + docs OpenAPI | Schéma JSON livré (`mockups/b2b_insight.json:1-123`), portail UI 0 |
| Mobile responsive | 🟡 partiel (FOCUS seul responsive dans demo v3) | 🟢 mobile-first, FOCUS = défaut <768px | `mockups/v3/best_concept_demo.html` (responsive partiel) |
| Dark mode | 🟢 dark natif | 🟡 ajouter light fallback (Stripe/Linear-style) | Toutes maquettes utilisent dark |
| Status page (SLA, incidents) | 🔴 0 | 🟢 `status.smartsentinel.ai` (statuspage.io ou inhouse) | `src/api/routes/health.py` existe mais non publique |
| Tests Playwright / Lighthouse / WCAG | 🔴 0 | 🟢 CI Playwright + LH > 90 + WCAG AA | 0 fichier Playwright |
| **Verdict commercialisation** | 🔴 NO-GO | 🟢 GO conditionnel S+10 | — |

**Top 3 bloqueurs UX/Frontend P0 à fixer immédiatement** :
1. **Stack frontend décidée et scaffoldée** (Next.js 15 App Router + Tailwind + shadcn/ui + TypeScript strict) — sans cela, aucun écran prod n'est livrable. ~24h.
2. **Webapp B2C MVP 3-vues** sur signal courant (FOCUS / CO-PILOT / EXPERT toggle persistant + chatbot sidebar/FAB) à partir des composants déjà prototypés dans `mockups/v3/best_concept_demo.html:1-2246`. ~80h.
3. **Landing page publique + onboarding signup → first signal** (LCP < 2s, CTA tracker, time-to-aha < 90s). ~50h.

**Effort total Catégorie 20** : **~470h prod-ready (Loukmane + design externe 30h)** sur 10-12 semaines.
**Budget design externe estimé** : **~$1 500-3 600** (Tailwind UI kit $149 + freelance 15-30h × $90-120/h).
**Chemin critique** : (a) Stack lock + design tokens → (b) Webapp 3-vues hot-path → (c) Auth + onboarding → (d) Landing + pricing → (e) B2B portal MVP → (f) status page + mobile QA → (g) Playwright + LH CI.

---

## 1. État actuel (Audit) — mockups, pas de frontend prod

### 1.1 Inventaire des artefacts UX existants

| Artefact | Type | Fichier:lignes | Statut | Verdict |
|---|---|---|---|---|
| Telegram B2C STRATEGIST/FREE (FR/EN) | Texte mockup | `mockups/telegram_b2c.txt:1-95` | 🟢 prêt | Format compliance UE 2024/2811 OK, à coder dans `src/delivery/telegram_notifier.py` (déjà existant). |
| Telegram Risk Score (Analyst/FREE/Kill-switch) | Texte mockup | `mockups/risk_score_telegram.md:1-104` | 🟡 mockup avancé | Composantes risk score 0-100 chiffrées, gating par tier décrit. |
| Webapp B2C STRATEGIST (HTML one-shot) | HTML statique | `mockups/webapp_b2c.html:1-337` | 🟡 v1 historique | Vue unique CO-PILOT only, **pas de toggle 3-vues**, palette divergente (`--bg:#0b0e14` vs v3 `#0b0e13`), pas de chatbot intégré. Obsolète par v3. |
| TradingView dashboard mockup | HTML statique | `mockups/tradingview_dashboard_mockup.html` (536 lignes) | 🟡 prospect-target | Style proche du widget TradingView, à reprendre pour landing screenshot. |
| **Démo concept B "Co-Pilot" 3-vues** | HTML statique | `mockups/v3/best_concept_demo.html:1-2246` | 🟢 **référence canonique** | Single-file, dark, finance premium. Inclut toggle FOCUS/CO-PILOT/EXPERT, hero card, 6 cards soutien, waterfall 8 composantes, conformal viz, chatbot scripté 6 questions. **Base de tout le scaffold.** |
| Pricing bundles mockup | Markdown | `mockups/pricing_bundles.md:1-136` | 🟡 v1 caduque | Grille bundle FX/Metal/Crypto/Index $29-49 sera remplacée par grille validée tier (`eval_27` : FREE/$29/$79/$1990). Cannibalisation flagguée `:113`. |
| B2B insight JSON (REST GET) | JSON | `mockups/b2b_insight.json:1-123` | 🟢 contrat figé v1.0.0 | Sérialisation Pydantic v2 alignée Insight v2.1.0. Sert base OpenAPI. |
| B2B webhook payload | JSON | `mockups/b2b_webhook_payload.json` | 🟢 contrat figé | Signature HMAC à matcher. |
| v2 client_view_full | HTML | `mockups/v2/client_view_full.html` (1159 lignes) | 🟡 archive | Pré-v3, vue cockpit unique, sans toggle. |

**Conclusion audit** : la conception est mûre (référence v3 + JSON figés), mais **rien n'est servi en production**. Le SSR HTML `src/api/routes/webapp.py:1-44` retourne un preview JSON-to-HTML pour email/embed mais n'est **pas** une SPA et n'expose pas le toggle 3-vues.

### 1.2 Backend déjà branchable côté UX

| Endpoint | Fichier:ligne | Usage UX cible |
|---|---|---|
| `GET /api/v1/insights/preview` (HTML SSR) | `src/api/routes/webapp.py:44` | Fallback email/embed, **pas le hot-path SPA**. |
| `GET /api/v1/insights/latest` | (inféré, à confirmer dans `src/api/routes/signals.py`) | Data source FOCUS/CO-PILOT/EXPERT. |
| `GET /api/v1/dashboard/summary` | `src/api/routes/dashboard.py:17-51` | Hero card track-record (PF historique + IC). |
| `GET /api/v1/dashboard/equity-curve` | `src/api/routes/dashboard.py:54-60+` | Replay/equity chart en EXPERT. |
| `POST /api/v1/enrich` | `src/api/routes/enrich.py` | Génération narrative on-demand (chat-driven). |
| `GET /api/v1/insights/history` | `src/api/routes/insight_history.py` | Historique pour Mode EXPERT replay. |
| `GET /api/v1/narratives/*` | `src/api/routes/narratives.py` | Chatbot data flow. |
| `GET /api/v1/qa` | `src/api/routes/qa.py` | Endpoint chatbot Q&A. |
| `GET /api/v1/billing/*` | `src/api/routes/billing.py` | Branchement Stripe portal UI. |
| `GET /api/v1/legal/{terms,privacy}` | `src/api/routes/legal.py` | Liens footer compliance permanent. |
| `GET /health` | `src/api/routes/health.py:1-40` | Source de la status page publique. |
| `GET /metrics` (Prometheus) | `src/api/routes/prometheus.py` | Pas exposé en UI publique (admin only). |

**Verdict backend ↔ UX** : 80% des endpoints requis existent. Le travail UX est essentiellement **frontend** + ajout endpoint `GET /api/v1/user/preferences` (pour persistance toggle 3-vues, P0.2 de `docs/value/information_enrichment_recommendations.md:43-55`).

### 1.3 Gap vs `feedback_multi_view_ux.md` (règle 2-3 vues)

| Surface | Mode forcé / défaut prescrit | État actuel |
|---|---|---|
| Telegram | FOCUS forcé | 🟢 OK (`mockups/telegram_b2c.txt` rend déjà FOCUS-like) |
| Webapp | CO-PILOT défaut, toggle FOCUS/EXPERT | 🔴 webapp absente |
| Email digest | FOCUS + lien CO-PILOT/EXPERT | 🟡 SSR preview existe (`webapp.py:44`), pas câblé à un cron email |
| API B2B | EXPERT (JSON complet) | 🟢 JSON figé (`mockups/b2b_insight.json`) |
| Mobile webapp | FOCUS auto <768px, toggle dispo | 🔴 webapp absente |
| B2B portal UI | EXPERT (logs requests, usage) | 🔴 absent |

---

## 2. Vision cible

**Promesse UX** : *Smart Sentinel Co-Pilot — l'analyse de marché de niveau institutionnel, lisible en 10s ou disséquable en 5min, selon votre humeur. Plus un quant qui répond à toutes vos questions.*

### 2.1 Principes de design transversaux

1. **Une donnée, trois projections** — le `InsightSignalV2` unique (cf. `mockups/b2b_insight.json:1-123`) est rendu en FOCUS, CO-PILOT ou EXPERT, jamais dupliqué côté algo.
2. **Toggle persistant par utilisateur** (table `user_preferences.view_mode` + cookie fallback) — restauré à la connexion. Source : `feedback_multi_view_ux.md:18`.
3. **Mobile-first FOCUS** — sous 768px, FOCUS est forcé par défaut ; CO-PILOT et EXPERT accessibles via toggle mais re-flow vertical (pas de sidebar parallèle).
4. **Chatbot pilier permanent** — sidebar 320px desktop, FAB mobile, jamais caché. Cf. `docs/value/information_enrichment_recommendations.md:84-101` (P0.4).
5. **Compliance assumée** — footer 1 ligne permanent, dans toutes les vues, jamais dismissible.
6. **Dark-first finance premium** — palette inspirée Bloomberg Terminal (sobriété EXPERT), Linear (densité élégante CO-PILOT), Stripe (clarté FOCUS), Pitchbook (track-record honnête en hero). Cf. `mockups/v3/best_concept_demo.html:15-42`.
7. **Performance budget dur** : LCP < 2.0s sur 4G simulé / FID < 100ms / CLS < 0.05 / Bundle JS initial < 180kb gz.
8. **Accessibilité AA non-négociable** — contraste ≥4.5:1 sur tous les pairs texte/fond, labels ARIA explicites, focus visible, navigation clavier complète, `prefers-reduced-motion` respecté.
9. **Aucune dépendance crypto-bro / néon** — aucun gradient violet/rose électrique, aucun glassmorphism flashy ; finance sérieuse.
10. **Anti-bullshit edge_claim=False** — aucun écran ne dit "BUY", "SELL", "Win 90%". Wording UE 2024/2811 partout (LECTURE HAUSSIÈRE, etc.).

### 2.2 Architecture cible (3 vues, 1 webapp)

```
─── HEADER ─────────────────────────────────────────────────────────────────
  S Smart Sentinel Co-Pilot  · XAUUSD · M15 ▾ · FR ▾ · 🔔 (3) ·  👤 STRAT
  MODE :  [ FOCUS ]  [ CO-PILOT* ]  [ EXPERT ]    ← persistant par user
─── HERO (toujours visible — variant selon mode) ────────────────────────────
   🟢 LECTURE HAUSSIÈRE · STRONG               ✓ 8 facteurs analysés
   XAU/USD · M15 · valable encore 2h47
   ┌──── CONVICTION 72 / 100 ───┐  ┌──── TRACK RECORD ────┐
   │ ▮▮▮▮▮▮▮▯▯▯  marge ±14 (CI95)│  │ 1.30 [1.12-1.49]  329 │
   └────────────────────────────┘  │ setups · WF 7 ans     │
   ⚠ FOMC Minutes dans 2h47 (vol +18%)
─── BODY VARIANT BY MODE ────────────────────────────────────────────────────
  FOCUS  :  1 ligne narrative + [💬 Demander à Sentinel] + [⤢ Voir détail →]
  CO-PIL :  6 cards soutien + chatbot sidebar permanent (Conviction, Régime,
            Volatilité, Structure, Session, Lecture verbale)
  EXPERT :  + waterfall 8 composantes + conformal viz + stats J.* enrichies
            + sources RAG + replay annoté + chatbot mode Pro (raw features)
─── FOOTER PERMANENT (toutes vues, jamais masqué) ──────────────────────────
  Démonstration paper-trading · Lecture algorithmique éducative
  Ne constitue ni un signal ni un conseil · 74-89% retail CFD perdent
```

### 2.3 Surfaces couvertes par cette catégorie

| Surface | Pages / écrans | Mode par défaut |
|---|---|---|
| **Webapp B2C** (`app.smartsentinel.ai`) | `/`, `/auth/signup`, `/auth/login`, `/onboarding/{1..5}`, `/dash`, `/dash/insights/[id]`, `/dash/history`, `/dash/methodology`, `/account/billing`, `/account/preferences` | CO-PILOT (sur `/dash`) |
| **Landing publique** (`smartsentinel.ai`) | `/`, `/pricing`, `/methodology`, `/about`, `/terms`, `/privacy`, `/changelog`, `/blog/*` | FOCUS-style (statique) |
| **B2B portal** (`b2b.smartsentinel.ai` ou `/b2b`) | `/b2b`, `/b2b/keys`, `/b2b/usage`, `/b2b/webhooks`, `/b2b/docs`, `/b2b/audit`, `/b2b/sla` | EXPERT (JSON-first) |
| **Status page** (`status.smartsentinel.ai`) | `/` (live), `/incidents`, `/sla` | publique |
| **Email** (template) | digest quotidien, alerte event ≤4h, kill-switch, billing | FOCUS |
| **Mobile responsive** | toutes pages ci-dessus | FOCUS sous 768px |
| **Mobile native** (P2, post-MVP) | iOS + Android Expo | FOCUS |
| **Browser extension** (P2) | popup signal courant | FOCUS |

---

## 3. Gap analysis

| Pilier | Cible | Manque | Effort (h) | Priorité |
|---|---|---|---:|---|
| Stack frontend lock | Next.js 15 App Router + TS strict + Tailwind + shadcn/ui | 0 — rien scaffold | 24 | P0 |
| Design system (tokens, composants) | Figma + Tailwind config + shadcn/ui forké + tokens JSON exportés | 0 — 5 palettes divergentes en mockup | 40 | P0 |
| Webapp B2C MVP (3 vues, chatbot) | `/dash/insights/[id]` + sidebar chat | 0 production (mockup v3 statique seul) | 80 | P0 |
| Landing page publique | home + pricing + méthodologie + about | 0 publié, domaine réservé | 32 | P0 |
| Onboarding | 5 écrans signup → first signal | 0 | 24 | P0 |
| Auth UI | signup, login, magic link, reset, OAuth (Google) | 0 UI (endpoints à confirmer) | 18 | P0 |
| B2B portal MVP | clés API + usage + webhooks tester + docs | 0 UI, backend partiel `src/api/routes/audit.py` | 60 | P0 |
| Mobile responsive QA | toutes pages | partiel | 20 | P1 |
| Light mode optionnel | toggle light/dark, fallback Stripe-style | 0 (dark only) | 14 | P1 |
| Status page | `status.smartsentinel.ai` (statuspage.io ou inhouse) | 0 | 16 | P1 |
| Tests Playwright | smoke + critical flows | 0 | 24 | P1 |
| Lighthouse + a11y CI | LH > 90 mobile/desktop + axe-core | 0 | 12 | P1 |
| Mobile native (Expo) | iOS + Android | 0 | 120 | P2 |
| Browser extension (Chrome/Firefox) | popup signal courant + alertes | 0 | 80 | P2 |
| Sécurité frontend (XSS/CSRF/CSP) | CSP strict, sanitize Markdown, secure cookies | partiel (sanitize côté SSR `webapp.py:18`) | 12 | P0 |
| Métriques Web Vitals + product analytics | Plausible + Sentry + GA4 (consent EU) | 0 | 10 | P0 |
| i18n FR/EN (DE/ES en P1) | next-intl + glossary | partiel (`mockups/telegram_b2c.txt` FR+EN) | 18 | P0 |

---

## 4. Plan d'exécution

### P0 — Stack frontend (Next.js+Tailwind+shadcn/ui vs Astro vs SvelteKit)

**Décision argumentée : Next.js 15 (App Router) + TypeScript strict + Tailwind CSS + shadcn/ui + Zustand (state) + TanStack Query (data) + next-intl (i18n) + Lucide icons + Recharts (lightweight charts) + react-hook-form + zod.**

| Critère | Next.js 15 ✅ | Astro | SvelteKit |
|---|---|---|---|
| Maturité écosystème (banking-grade SaaS) | ★★★★★ (Stripe, Linear, Vercel) | ★★★ (content-first) | ★★★★ (montée en charge) |
| SSR + hydration sélective (RSC) | ★★★★★ (App Router + RSC + streaming) | ★★★★ (islands) | ★★★★ (SSR natif) |
| Disponibilité dev FR freelance | ★★★★★ | ★★★ | ★★ |
| Compat shadcn/ui (composants premium prêts) | ★★★★★ natif | ★★★ (port Astro) | ★★ (port shadcn-svelte) |
| Bundle initial JS | bon (RSC zéro JS) | excellent (islands) | excellent |
| Streaming charts/dashboards | ★★★★★ | ★★ | ★★★★ |
| Vercel deploy + auth + edge | ★★★★★ | ★★★★ | ★★★★ |
| Risk lock-in | moyen (Vercel-aligned) | faible | faible |
| **Score pondéré** | **4.7** ✅ | 3.5 | 3.6 |

Justifications :
- shadcn/ui (Radix + Tailwind, MIT) donne 90% des composants premium nécessaires (Dialog, Toggle, Tooltip, Tabs, Sheet, Command palette) — économise ~80h de re-build.
- App Router + Server Components réduit le JS initial — critique pour LCP < 2s sur landing.
- Stripe Checkout + Stripe customer portal s'intègrent en 4h avec `@stripe/stripe-js` côté Next.
- next-intl couvre FR/EN/DE/ES (déjà supportés par `TelegramLangStore`).

**Livrables stack**
- `web/` (sous-dossier monorepo ou repo séparé `smart-sentinel-web/`).
- `web/package.json` lock : Next 15.x, React 19, TS 5.x, Tailwind 3.x, shadcn-ui CLI, Zustand 4.x, TanStack Query 5.x, next-intl 3.x, Lucide, Recharts, zod, react-hook-form.
- `web/tsconfig.json` strict mode (`"strict": true`, `"noUncheckedIndexedAccess": true`).
- `web/.env.example` : `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`, `NEXT_PUBLIC_PLAUSIBLE_DOMAIN`, `SENTRY_DSN`.
- `web/eslint.config.mjs` : `eslint-config-next` + `@typescript-eslint/strict` + `tailwindcss/recommended`.
- `web/prettier.config.mjs` : tabs OFF, 2 spaces, single quotes, semi false (cohérent v3 demo).
- `Dockerfile.web` pour build prod, multi-stage Node 20 alpine.

**Effort** : 24h (24h Loukmane / 0h design).

---

### P0 — Design system (Figma → tokens → code)

**Pourquoi un design system avant de coder** : 5 mockups, 5 palettes divergentes (cf. `mockups/webapp_b2c.html:9-22` vs `mockups/v3/best_concept_demo.html:15-42` — `--bg #0b0e14` vs `#0b0e13`, golds différents). Si on code sans tokens, on duplique le chaos.

**Livrables**
1. **Figma file** `Smart Sentinel · Design System v1` (page Tokens / Components / Templates) → outsourcé designer freelance (30h × $90/h = **~$2 700**) OU fait par Loukmane via Figma + Tailwind UI Kit licence (~$149) sur 24h.
2. **Tokens JSON** exportés via Figma Tokens plugin → `web/tokens/colors.json`, `radii.json`, `spacing.json`, `typography.json`, `motion.json`. Mapping vers `tailwind.config.ts`.
3. **Palette canonique** (alignée v3 demo) :
   ```
   bg-page:        #0b0e13
   bg-card:        #141821
   bg-elevated:    #1c2230
   border:         #2a3140
   text-primary:   #e6e9ef    contraste 12.5:1 ✓ AAA
   text-secondary: #b3b9c5    contraste 8.9:1  ✓ AAA
   text-muted:     #8b929c    contraste 5.6:1  ✓ AA large
   gold-accent:    #c9a961    (track-record + premium tier)
   bullish:        #4ade80
   bearish:        #f87171
   warning:        #fbbf24
   info:           #60a5fa
   accent-chat:    #7c8aff
   ```
4. **Typographie** : Inter Variable (UI), JetBrains Mono Variable (chiffres, prix, codes). Échelle modulaire 1.250 (major third) : 12/13/15/19/24/30/38/48 px. Line-height : 1.25 (titres) / 1.55 (corps) / 1.20 (data dense).
5. **Composants shadcn forkés + variants Smart Sentinel** :
   - `<HeroCard>` (4 variants : bullish / bearish / unreadable / no-signal)
   - `<ConvictionGauge>` (mini + full, animée avec `prefers-reduced-motion`)
   - `<TrackRecordBadge>` (PF + IC + N + lookback en 4 lignes)
   - `<EventCountdown>` (chrono live, refresh 1/min)
   - `<ViewModeToggle>` (3 segments)
   - `<ConfluenceWaterfall>` (8 barres horizontales + hover tooltip)
   - `<ConformalIntervalViz>` (bande + point + ticks 0-100)
   - `<ChatSidebar>` (desktop) + `<ChatFAB>` (mobile)
   - `<JargonTooltip>` (mapping `glossary_fr.json`)
   - `<ComplianceFooter>` (variant FR/EN/DE/ES)
   - `<TierLockedSection>` (variant ANALYST / STRATEGIST / INSTITUTIONAL)
   - `<KillSwitchBanner>` (admin)
   - `<NotificationToast>` (using shadcn `sonner`)
6. **Storybook** (optionnel P1, P0 si budget) — `web/stories/` pour QA visuelle des 12 composants.
7. **Page test palette + composants** (`/_design-system`) en production (route protégée admin).

**Effort** : 40h (10h Loukmane + 30h designer freelance, **budget ~$2 700**) ou 40h Loukmane solo si pas de budget.

---

### P0 — Webapp B2C MVP : 3 views sur signal

**Source canonique** : `mockups/v3/best_concept_demo.html:1-2246`. Le travail consiste à **transformer le single-file HTML en application Next.js réactive avec persistance utilisateur**.

#### 4.1 Pages cibles MVP

| Route | Description | Mode défaut |
|---|---|---|
| `/dash` | dashboard utilisateur, liste insights actifs (carte par instrument) | grille |
| `/dash/insights/[id]` | insight detail page **avec toggle 3-vues** | CO-PILOT |
| `/dash/insights/[id]?mode=focus` | force FOCUS (deep-link partageable) | FOCUS |
| `/dash/insights/[id]?mode=expert` | force EXPERT | EXPERT |
| `/dash/history` | replay historique signaux clos | tableau |
| `/dash/methodology` | page explication algo + scoring + sources RAG | scroll |
| `/account/preferences` | view_mode default, langue, notifications, theme | form |
| `/account/billing` | Stripe portal embed | iframe |

#### 4.2 Wireframe textuel — Mode FOCUS (`/dash/insights/[id]?mode=focus`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ HEADER  [S]  Smart Sentinel Co-Pilot · XAUUSD M15 ▾ · FR ▾ ·  STRATEGIST│
│          MODE : [▣ FOCUS] [ CO-PILOT ] [ EXPERT ]   ⏱ 2h47 validity     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   🟢  LECTURE HAUSSIÈRE · STRONG          ✓ 8 facteurs analysés         │
│       XAU/USD · M15 · valable encore 2h47                               │
│                                                                          │
│   ┌──── CONVICTION ──────────┐    ┌──── TRACK RECORD HONNÊTE ────┐     │
│   │      72                  │    │   1.30€ pour 1€ perdu        │     │
│   │   ▮▮▮▮▮▮▮▯▯▯  /100       │    │   IC 95 % : 1.12 — 1.49     │     │
│   │   marge ±14 (CI conf.)   │    │   329 setups · WF 7 ans     │     │
│   └──────────────────────────┘    └──────────────────────────────┘     │
│                                                                          │
│   ⚠️  FOMC Minutes dans 2h47 — volatilité attendue +18% pendant 4h     │
│                                                                          │
│   « Cassure haussière confirmée. Retest armé entre 2 378 et 2 381 $.    │
│     Invalidation sous 2 374,5. À surveiller : compte-rendu FOMC. »      │
│                                                                          │
│        ┌─────────────────────────┐   ┌──────────────────────────┐       │
│        │ 💬 Demander à Sentinel  │   │ ⤢ Voir le détail →       │       │
│        └─────────────────────────┘   └──────────────────────────┘       │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ FOOTER  Démonstration paper-trading · Lecture algorithmique éducative   │
│         Ne constitue ni un signal ni un conseil · 74-89 % retail perd   │
└──────────────────────────────────────────────────────────────────────────┘
```

Composants Next.js mobilisés : `<HeroCard variant="focus">`, `<ConvictionGauge size="lg" mode="focus">`, `<TrackRecordBadge>`, `<EventCountdown event={nextHighImpact} />`, `<NarrativeShort>`, `<ButtonChatOpen>`, `<ButtonModeSwitch to="copilot">`, `<ComplianceFooter>`.

Données source : `GET /api/v1/insights/{id}` (champs : `direction`, `conviction.point`, `conviction.label`, `conviction.lower`, `conviction.upper`, `historical_stats.profit_factor`, `historical_stats.profit_factor_ci95`, `historical_stats.n_observations`, `economic_calendar.next_high_impact_event`, `narrative_short`, `expires_at`).

#### 4.3 Wireframe textuel — Mode CO-PILOT (`/dash/insights/[id]`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ HEADER  (idem FOCUS)                                                    │
├────────────────────────────────────────────┬─────────────────────────────┤
│ HERO CARD (FOCUS condensé)                 │                             │
│ ┌────────────────────────────────────────┐ │  💬 CHATBOT SIDEBAR         │
│ │ 🟢 LECTURE HAUSSIÈRE · STRONG          │ │  PERMANENT (largeur 320)    │
│ │ Conviction 72 · ±14 · PF 1.30 [1.12-49]│ │                             │
│ │ ⚠ FOMC dans 2h47 (vol +18%)            │ │  ┌────────────────────────┐ │
│ └────────────────────────────────────────┘ │  │ Posez votre question…  │ │
│                                            │  └────────────────────────┘ │
│ 6 CARDS DE SOUTIEN                         │                             │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  Suggestions :              │
│ │CONVICTION│ │ RÉGIME   │ │VOLATILITÉ│    │  • Pourquoi 72 ?            │
│ │ 72 / 100 │ │ Tendance │ │ +18%     │    │  • C'est quoi retest armé ?│
│ │  ±14     │ │  calme   │ │ vs normal│    │  • Le FOMC change quoi ?    │
│ └──────────┘ └──────────┘ └──────────┘    │  • Historique similaire ?   │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  • Donc je dois acheter ?  │
│ │STRUCTURE │ │ SESSION  │ │ LECTURE  │    │    [refus pédagogique]      │
│ │ BOS 2391 │ │  NY Ovr  │ │  verbale │    │  • Quelle est ta marge ?    │
│ │ FVG 2378 │ │          │ │   « … »  │    │                             │
│ └──────────┘ └──────────┘ └──────────┘    │  Citations / sources :      │
│                                            │  [lock STRATEGIST]          │
│ [⤢ Voir le détail technique (EXPERT) →]   │                             │
│                                            │                             │
├────────────────────────────────────────────┴─────────────────────────────┤
│ FOOTER COMPLIANCE PERMANENT                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

Composants mobilisés : `<HeroCard variant="copilot">`, `<SupportCardGrid>` (6 enfants), `<ChatSidebar persistent>` desktop / `<ChatFAB>` mobile <768px, `<SuggestedQuestion>` × 6 contextuelles, `<JargonTooltip>` partout, `<NarrativeShort>`.

Données : `GET /api/v1/insights/{id}` (tous champs ci-dessus + `regime_readout.label`, `regime_readout.confidence`, `volatility_forecast.vs_atr14_pct`, `volatility_forecast.regime_label`, `structure.bos_level`, `structure.fvg_zone`, `structure.retest_state`, `session`, `narrative_short`).

Persistance : `POST /api/v1/user/preferences { view_mode: "copilot" }` au switch utilisateur.

#### 4.4 Wireframe textuel — Mode EXPERT (`/dash/insights/[id]?mode=expert`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ HEADER (idem)  + badge "EXPERT MODE · STRATEGIST tier required"         │
├────────────────────────────────────────────┬─────────────────────────────┤
│ HERO + 6 CARDS (CO-PILOT)                  │  💬 CHATBOT MODE PRO        │
├────────────────────────────────────────────┤                             │
│ WATERFALL 8 COMPOSANTES (chiffré)          │  Peut interroger features   │
│ Smart Money     ████████████████ +24.5 25%│  brutes :                   │
│ Volatilité      ████████████     +14.0 15%│  • montre delta vol vs naive│
│ Multi-TF        █████████         +9.5 12%│  • posterior HMM ?          │
│ Liquidité       ██████            +6.0 10%│  • expected_run_length      │
│ Sessions        ████              +4.0  8%│    BOCPD ?                  │
│ Régime          ██                +1.0 15%│  • jump_ratio ?             │
│ Technical       ▏                 +0.5 10%│  • empirical_coverage OOS ? │
│ News            ▏                 +0.0  5%│                             │
│ ──── Total ───────────────────── 62.0 /100│  Citations RAG inline :     │
├────────────────────────────────────────────┤  • Corsi 2009 HAR-RV        │
│ CONFORMAL INTERVAL  conviction 72 ±14      │  • Gibbs & Candès 2021 ACI  │
│ ▮▮▮▮▮▮▮▯▯▯  couverture 95 % nominale       │  • Lopez de Prado 2018 CPCV │
│ couverture OOS observée : 94.3 % · ACI ON  │                             │
├────────────────────────────────────────────┤  Mode raw JSON disponible : │
│ STATS J.* ENRICHIES                        │  [ télécharger insight.json]│
│ N=329 · WR 41.6 % · DD max 8.4 %           │                             │
│ Exp time 4h12 · skew +0.34 · kurt 3.8      │                             │
├────────────────────────────────────────────┤                             │
│ REPLAY HISTORIQUE (chart annoté H4/H1/M15) │                             │
│ BOS markers · FVG zones · OB blocks · evts │                             │
├────────────────────────────────────────────┤                             │
│ SOURCES RAG (cliquables vers PDF/arXiv)    │                             │
│ • Corsi (2009) "A Simple Approximate…"     │                             │
│ • Lopez de Prado (2018) Adv Fin ML, ch 7   │                             │
│ • Gibbs & Candès (2021) "Adaptive Conf…"   │                             │
├────────────────────────────────────────────┴─────────────────────────────┤
│ FOOTER COMPLIANCE PERMANENT                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

Composants : `<HeroCard variant="copilot">`, `<SupportCardGrid>`, `<ConfluenceWaterfall>` (8 bars), `<ConformalIntervalViz>`, `<HistoricalStatsGrid>`, `<ReplayChart>` (Recharts + annotations), `<RAGSourcesList>`, `<ChatSidebar mode="pro">`, `<TierLockedSection tier="strategist">` pour replay+RAG.

Données : `GET /api/v1/insights/{id}` (champs ci-dessus + `signal_breakdown.components[]`, `uncertainty.point`, `uncertainty.conformal_lower/upper`, `uncertainty.coverage_alpha`, `uncertainty.coverage_observed`, `historical_stats.*` exhaustifs, `rag.citations[]`). Replay : `GET /api/v1/insights/{id}/replay` (à exposer, P0.10).

#### 4.5 Wireframe `/dash` (liste insights actifs)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Mes lectures actives (3)               [ + activer un nouvel instrument ]│
├──────────────────────────────────────────────────────────────────────────┤
│ ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐   │
│ │ 🟢 XAUUSD · M15    │ │ 🔴 EURUSD · M15    │ │ ⚪ BTCUSD · H1     │   │
│ │ HAUSSIÈRE · STRONG │ │ BAISSIÈRE · MOD.   │ │ ILLISIBLE          │   │
│ │ 72/100  ±14        │ │ 58/100  ±18        │ │ —                  │   │
│ │ PF 1.30 [1.12-49]  │ │ PF 1.18 [0.95-42]  │ │ N/A                │   │
│ │ ⚠ FOMC 2h47        │ │                    │ │                    │   │
│ │ [Ouvrir →]         │ │ [Ouvrir →]         │ │ [Désactiver]       │   │
│ └────────────────────┘ └────────────────────┘ └────────────────────┘   │
├──────────────────────────────────────────────────────────────────────────┤
│ Historique récent (7 j) ▾                  Performance 30 j : PF 1.21   │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 4.6 Wireframe chatbot (commun aux 3 modes)

```
┌─ Sentinel · Conversation ──────────────────────────────────────────────┐
│ Contexte : XAU/USD M15 · Lecture haussière · Conv 72 · FOMC 2h47       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  👤 Pourquoi la conviction n'est que de 72 ?                          │
│                                                                        │
│  🤖 Voici la décomposition des 8 facteurs (poids · contribution) :    │
│     • Smart Money         (25 %) → +24.5  (cassure 2391.5 ✓)         │
│     • Volatilité forecast (15 %) → +14.0  (régime calme)              │
│     • Multi-TF            (12 %) → +9.5   (M15 + H1 alignés)          │
│     • Liquidité           (10 %) → +6.0   (FVG 2378-2381 mid)         │
│     • Sessions            ( 8 %) → +4.0   (NY overlap)                │
│     • Régime HMM          (15 %) → +1.0   (trend_bullish 0.71)        │
│     • Technical           (10 %) → +0.5   (RSI 58)                    │
│     • News               ( 5 %) → +0.0   (rien actif maintenant)      │
│     Total : 62/100. Calibrée empiriquement sur 329 setups similaires. │
│                                                                        │
│     [ Approfondir : qu'est-ce qu'un FVG ? ]                           │
│     [ Approfondir : pourquoi News est à 0 alors qu'il y a FOMC ? ]    │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│ ✏️  Posez votre question…                              [ ➤ Envoyer ]  │
└────────────────────────────────────────────────────────────────────────┘
```

Cf. `mockups/v3/best_concept_demo.html` (section 3 des scripts JS) pour les 6 réponses canoniques pré-écrites. Le LLM réel branche `POST /api/v1/qa` (`src/api/routes/qa.py`).

#### 4.7 Endpoint préférences utilisateur à ajouter

| Endpoint | Méthode | Body / Query | Réponse |
|---|---|---|---|
| `GET /api/v1/user/preferences` | GET | — | `{ view_mode: "copilot" \| "focus" \| "expert", language: "fr", theme: "dark", notifications: {…} }` |
| `PATCH /api/v1/user/preferences` | PATCH | partial body | 200 OK |

Backend : ~4h (table SQLite `user_preferences (api_key_hash, view_mode, language, theme, notifications_json, updated_at)`). Cat. 8 backend.

**Effort webapp MVP** : 80h Loukmane (UI Next.js + composants shadcn + state + chatbot streaming).

---

### P0 — Landing page (positioning, pricing, CTA, social proof, FAQ)

**URL** : `https://smartsentinel.ai/` (domaine réservé, à pointer DNS).

#### 5.1 Architecture landing

| Section | Contenu | Composants | LCP critique |
|---|---|---|---|
| Hero | Tagline 10s + sous-tagline 5s + CTA primaire "Voir une démo en 30s" + screenshot Mode FOCUS | `<Hero>` + `<DemoScreenshot>` | ✅ image priority |
| Pourquoi Smart Sentinel | 3 différenciateurs (multi-vues, chatbot moat, honnêteté statistique) | `<DiffGrid>` | non-critical |
| Démo interactive | iframe / inline du `mockups/v3/best_concept_demo.html` (toggle 3-vues fonctionnel) | `<EmbeddedDemo>` | lazy-load |
| Track record | hero card extraite (PF 1.30 [1.12-1.49] · 329 setups · WF 7 ans) + lien méthodologie | `<TrackRecordHero>` | non-critical |
| Tarification | grille 4 tiers (eval_27 : FREE / $29 / $79 / $1990) + toggle monthly/annual (-16.7%) | `<PricingGrid>` | non-critical |
| Comparatif concurrents | tableau vs LuxAlgo, signaux Telegram, TradingView indicators | `<CompetitorComparison>` | non-critical |
| FAQ | 8 questions (compliance, paper-trading, chatbot, langues, broker compatibility, annulation, free trial, méthodologie) | `<Accordion>` shadcn | non-critical |
| Social proof | témoignages early users (Telegram channel public à venir, cf. cat. 1 P0.2) | `<TestimonialGrid>` | non-critical |
| CTA finale | Signup + footer compliance | `<CTAFinal>` + `<ComplianceFooter>` | non-critical |

#### 5.2 Wireframe textuel landing hero

```
┌──────────────────────────────────────────────────────────────────────────┐
│  [S] Smart Sentinel Co-Pilot                FR ▾    [Connexion] [Essai] │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   L'analyse de marché de niveau institutionnel,                          │
│   traduite pour vous. Plus un quant qui répond                           │
│   à toutes vos questions.                                                │
│                                                                          │
│   329 setups historiques · profit factor 1.30 [1.12-1.49] ·              │
│   walk-forward 7 ans · honnête sur ce qu'il ne sait pas.                │
│                                                                          │
│   ┌─────────────────────────────┐   ┌──────────────────────────────┐    │
│   │ ▶  Essayer la démo (30 s)   │   │ Voir la grille tarifaire     │    │
│   └─────────────────────────────┘   └──────────────────────────────┘    │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │ [ Screenshot Mode FOCUS — hero card + conviction + PF + FOMC ] │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│   Démonstration paper-trading · UE 2024/2811 · 74-89% retail CFD perd   │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 5.3 Wireframe pricing

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Tarification simple. Choisissez la profondeur, pas le nombre.          │
│                                                                          │
│  [ Mensuel ]  [ Annuel  -16,7 % ]                                       │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ FREE     │  │ ANALYST  │  │STRATEGIST│  │INSTITUTIO│  ← decoy        │
│  │ 0 $/mo   │  │ 29 $/mo  │  │ 79 $/mo  │  │ 1990 $/mo│                │
│  │          │  │          │  │ ★ POPULAR│  │ API B2B  │                │
│  │ Mode     │  │ FOCUS +  │  │ + EXPERT │  │ EXPERT   │                │
│  │ FOCUS    │  │ CO-PILOT │  │          │  │ JSON     │                │
│  │ 1 actif  │  │ 4 actifs │  │ 6 actifs │  │ multi    │                │
│  │ 5 lect/j │  │ 30 lect/j│  │ illimité │  │ webhook  │                │
│  │ Chat 10Q │  │ 100 Q/j  │  │ illimité │  │ SLA      │                │
│  │ —        │  │ Alerte   │  │ Replay+  │  │ Audit log│                │
│  │ —        │  │ event    │  │ RAG cit. │  │ Multi-tnt│                │
│  │          │  │ Email    │  │ Export   │  │ License  │                │
│  │          │  │ digest   │  │ CSV      │  │ commerc. │                │
│  │ Démarrer │  │ Essai 14j│  │ Essai 14j│  │ Contact  │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
│                                                                          │
│  Essai 14 jours sans CB · annulable n'importe quand · Stripe sécurisé   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Performance budget landing** : LCP < 1.5s (texte + 1 image WebP optimisée), CLS < 0.02, JS hydration < 100kb gz (RSC + islands).

**Effort** : 32h (24h dev + 8h copywriting FR/EN).

---

### P0 — Onboarding flow (signup → first signal → AHA moment)

**KPI cible** : time-to-aha < 90s, activation rate > 65% (≥ 1 insight ouvert dans la première session).

#### 6.1 5 écrans onboarding

| # | Écran | Composants | Durée cible | KPI |
|---|---|---|---|---|
| 1 | Signup (email + mdp / Google OAuth) | `<SignupForm>` + Google button | 15s | conversion ≥ 80% |
| 2 | Choix langue + actif principal (XAUUSD défaut FR, EURUSD défaut EN) | `<LanguageSelect>` + `<AssetPicker single>` | 10s | drop < 5% |
| 3 | Tour rapide du Mode FOCUS (3 tooltips guidés sur hero card + chat) | `<OnboardingTour>` (react-joyride) | 25s | complétion ≥ 75% |
| 4 | Premier insight ouvert avec annotation guidée ("Voici la conviction", "Voici le track record", "Voici Sentinel") | annotations sur `/dash/insights/[id]?mode=focus&onboarding=1` | 30s | bottom 60% scroll |
| 5 | CTA "Essayez le chatbot : Pourquoi 72 ?" → ouvre chat avec question pré-remplie + réponse | `<ChatSidebar autoOpen>` + question scriptée | 10s | clic ≥ 70% |

#### 6.2 Wireframe écran 4 (first insight)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Bienvenue Loukmane. Voici votre première lecture XAU/USD.              │
├──────────────────────────────────────────────────────────────────────────┤
│  🟢 LECTURE HAUSSIÈRE · STRONG                                          │
│       ╔══════════════════════════════════╗                              │
│       ║ ① Voici la conviction calibrée.  ║◄── annotation 1              │
│       ║   72 sur 100 — vérifié sur 329   ║                              │
│       ║   setups similaires.             ║                              │
│       ╚══════════════════════════════════╝                              │
│   [Suivant ▷]                                                            │
│                                                                          │
│   1.30€ pour 1€ perdu · IC 1.12-1.49                                    │
│       ╔══════════════════════════════════╗                              │
│       ║ ② Track record honnête, avec     ║◄── annotation 2 (slide 2)    │
│       ║   marge d'erreur statistique.    ║                              │
│       ╚══════════════════════════════════╝                              │
│                                                                          │
│   [💬 Demander à Sentinel : Pourquoi 72 ?]                              │
│       ╔══════════════════════════════════╗                              │
│       ║ ③ Sentinel répond à toutes vos   ║◄── annotation 3 (slide 3)    │
│       ║   questions, en temps réel.      ║                              │
│       ╚══════════════════════════════════╝                              │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 6.3 Email d'onboarding (digest J+1)

Template SSR via `src/api/routes/webapp.py:44` (déjà partiellement implémenté). Variants FR/EN/DE/ES via `LABELS` table (`webapp.py:52+`).

**Effort** : 24h (UI tour + state + 5 écrans + email template + i18n).

---

### P0 — B2B portal (API keys, usage dashboard, webhooks, docs)

**URL** : `https://b2b.smartsentinel.ai` (sous-domaine) ou `/b2b` (path).
**ICP** : intégrateurs broker (cf. `MEMORY.md` plan B B2B-API $310k ARR), family offices, audit interne.

#### 7.1 Pages B2B portal

| Route | Description | Composants principaux |
|---|---|---|
| `/b2b` | dashboard usage 30j (req/min, latence p95, erreurs, top assets) | `<UsageChart>` + `<KPIGrid>` |
| `/b2b/keys` | gestion clés API (create / revoke / rotate / view last_used_at) | `<APIKeyTable>` + `<CreateKeyModal>` |
| `/b2b/webhooks` | URLs cibles, secret HMAC, last delivery, retry queue, replay manual | `<WebhookTable>` + `<DeliveryLog>` |
| `/b2b/docs` | OpenAPI Swagger UI sur `/api/v1/openapi.json` + exemples curl/Python/Node | `<SwaggerUI>` (Redoc fork) |
| `/b2b/audit` | audit log requêtes (filtres par api_key, asset, date) | `<AuditLogTable>` (branché `src/api/routes/audit.py`) |
| `/b2b/sla` | SLA contract details, monthly uptime, credit policy | `<SLADisplay>` |

#### 7.2 Wireframe `/b2b` (dashboard usage)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ [S] Smart Sentinel · B2B Portal       INSTITUTIONAL · ACME Broker      │
├──────────────────────────────────────────────────────────────────────────┤
│ Usage 30 derniers jours                                                 │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│ │ Requests     │ │ p95 latency  │ │ Error rate   │ │ Webhook OK   │    │
│ │   1 247 832  │ │   38 ms      │ │   0.04 %     │ │   99.97 %    │    │
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘    │
│                                                                          │
│ ┌──────────────────────────────────────────────────────────────────┐   │
│ │ Trafic par jour (req / min — line chart Recharts)                │   │
│ │                                                                  │   │
│ │         ╭─╮  ╭──╮                                                │   │
│ │   ╭──╮  │ ╰──╯  ╰──╮       ╭───╮                                │   │
│ │   ╯  ╰──╯           ╰───╮───╯   ╰──────                         │   │
│ │ 2026-04-23     …       2026-05-23                                │   │
│ └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ Top assets : XAUUSD 67 % · EURUSD 24 % · BTCUSD 9 %                    │
│ Top endpoints : /insights/latest 88 % · /enrich 7 % · /history 5 %     │
│                                                                          │
│ [→ Gérer les clés API]  [→ Webhooks]  [→ Docs API]  [→ Audit log]     │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 7.3 Wireframe `/b2b/keys`

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Clés API                                       [+ Créer une nouvelle clé]│
├──────────────────────────────────────────────────────────────────────────┤
│ Label              | Scope    | Last used   | Created    |  Actions     │
│ prod-ingest        | read     | 12 s ago    | 2026-03-04 |  [Rotate]    │
│                    |          |             |            |  [Revoke]    │
│ staging-test       | read     | 4 d ago     | 2026-04-15 |  [Rotate]    │
│ webhook-replay     | webhook  | 2 m ago     | 2026-02-22 |  [Revoke]    │
└──────────────────────────────────────────────────────────────────────────┘
```

Compatibilité backend : `src/api/auth.py` doit exposer `POST /api/v1/keys`, `DELETE /api/v1/keys/{id}`, `POST /api/v1/keys/{id}/rotate`. À confirmer cat. 8.

**Wireframe docs (`/b2b/docs`)** : Redoc (open source, MIT) sur `/api/v1/openapi.json` (FastAPI génère automatiquement). Inclure exemples cURL, Python (requests + httpx), Node (axios + fetch), webhook signature verify Python/Node.

**Effort** : 60h (UI portal + tables + OpenAPI integration + 4 exemples par langage).

---

### P1 — Mobile responsive, dark mode

#### 8.1 Stratégie responsive

- **Breakpoints Tailwind** : `sm` 640 / `md` 768 / `lg` 1024 / `xl` 1280 / `2xl` 1536.
- **Sous 768px** : Mode FOCUS forcé par défaut, toggle vers CO-PILOT ré-flow vertical (6 cards en 1 colonne), EXPERT non-affiché (CTA "Ouvrez sur desktop").
- **Chat** : FAB bottom-right au lieu de sidebar.
- **Header** : burger menu mobile (shadcn `<Sheet>`).
- **Hero card mobile** : conviction + track-record empilés verticalement.
- **Touch targets** : minimum 44×44 px (Apple HIG / WCAG 2.5.5 niveau AAA).

#### 8.2 Light mode (P1)

Toggle `<ThemeSwitcher>` (shadcn). Palette light dérivée :
```
bg-page:        #f8fafc
bg-card:        #ffffff
border:         #e2e8f0
text-primary:   #0f172a
text-secondary: #475569
gold-accent:    #92702c  (assombri pour contraste AA sur fond clair)
```
Tester contraste avec `axe-core` CI : tous pairs texte/fond AA minimum.

**Effort** : 20h responsive QA + 14h light mode = 34h.

---

### P1 — Status page

**URL** : `status.smartsentinel.ai`. Options :

| Option | Coût | Effort | Recommandation |
|---|---|---|---|
| Statuspage.io (Atlassian) | $29-79/mo | 4h intégration | OK si budget |
| BetterStack Status | $25/mo | 4h | OK |
| Inhouse Next.js `/status` | $0 (Vercel free) | 16h | ✅ recommandé (cohérence stack, signal de sérieux) |

**Inhouse** : page Next.js qui poll `GET /health` toutes les 30s (`src/api/routes/health.py:1-40`), historise dans SQLite séparé `status.db` (uptime 90 jours rolling). Composants :

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Smart Sentinel · Status                              ✓ All systems OK   │
├──────────────────────────────────────────────────────────────────────────┤
│ Uptime 90 j                                                              │
│ ──────────────────────────────────────────────────────  99.92 %         │
│                                                                          │
│ Component                          | Latency p95 | Uptime 30 j           │
│ API REST (api.smartsentinel.ai)    |     42 ms   |   99.98 %             │
│ Webapp (app.smartsentinel.ai)      |    180 ms   |   99.95 %             │
│ Scanner pipeline                   |    1.2 s    |   99.91 %             │
│ Telegram delivery                  |    310 ms   |   99.99 %             │
│ LLM narrative (Claude)             |    1.8 s    |   99.87 %             │
│ Data feed (Dukascopy/MT5)          |    —        |   99.83 %             │
│ Stripe billing                     |    —        |   99.99 %             │
├──────────────────────────────────────────────────────────────────────────┤
│ Incidents récents (90 j)                                                 │
│ • 2026-05-12 14h22 — LLM degraded ~12 min (fallback template auto)      │
│ • 2026-04-28 09h41 — Scanner restart (state reload, no signal lost)     │
└──────────────────────────────────────────────────────────────────────────┘
```

**Effort** : 16h inhouse.

---

### P2 — Mobile native app, browser extension

#### 9.1 Mobile native (Expo / React Native)

- **Justification P2** : post-MVP, après validation activation > 65% sur webapp.
- **Scope** : Mode FOCUS uniquement + push notifications (signal nouveau, event ≤4h, kill-switch).
- **Stack** : Expo SDK 51+ (React Native + Hermes), expo-router, NativeWind (Tailwind RN), expo-notifications.
- **Distribution** : TestFlight (iOS) + Play Console Internal Track (Android) en phase pilote ; release prod après 50 beta users.
- **Effort** : 120h (60h iOS + 60h Android, mutualisés via Expo).

#### 9.2 Browser extension

- **Justification P2** : niche power-user, faible volume mais haute LTV.
- **Scope** : popup signal courant (FOCUS embed) + badge nombre lectures actives + alerte FOMC <2h.
- **Stack** : Plasmo framework (Chrome + Firefox + Edge crossbuild, React/TS).
- **Effort** : 80h.

---

## 5. Tests (Playwright, Lighthouse, WCAG AA)

### 5.1 Playwright (E2E + visual regression)

**Chemins critiques à couvrir** :

| Test | Fichier cible | Critère pass |
|---|---|---|
| Signup → first signal opened (onboarding) | `web/tests/onboarding.spec.ts` | 5 écrans complétés < 90s simulés |
| Toggle FOCUS → CO-PILOT → EXPERT (persistance) | `web/tests/view-mode.spec.ts` | preference persistée 3 reloads |
| Chatbot 6 questions canoniques (mocked LLM) | `web/tests/chatbot.spec.ts` | réponses contiennent tokens attendus |
| Event countdown banner refresh (mock time) | `web/tests/event-countdown.spec.ts` | re-render chaque minute, label change |
| Refus pédagogique "Donc je dois acheter ?" | `web/tests/compliance-refuse.spec.ts` | réponse `contains_forbidden_token === false` |
| Mobile FOCUS forced <768px | `web/tests/mobile-focus.spec.ts` | viewport iPhone 14, EXPERT non rendu |
| Stripe checkout button → redirect (mocked) | `web/tests/billing.spec.ts` | URL match `checkout.stripe.com/*` |
| B2B `/b2b/keys` create + revoke | `web/tests/b2b-keys.spec.ts` | row appears + 200 + row disappears |

**CI** : GitHub Actions `playwright.yml` lance Chromium + Firefox + WebKit en headless sur PR. Snapshots screenshot diff pour visual regression (Percy ou inhouse).

**Effort tests** : 24h (12 tests E2E core + 6 visual regression).

### 5.2 Lighthouse CI

**Budgets** :

| Page | LCP cible | CLS cible | FID cible | TBT cible | Score perf | Score a11y |
|---|---:|---:|---:|---:|---:|---:|
| Landing `/` | 1.5s | 0.02 | 80ms | 200ms | ≥ 95 | ≥ 95 |
| `/pricing` | 1.8s | 0.05 | 80ms | 300ms | ≥ 90 | ≥ 95 |
| `/dash/insights/[id]` (CO-PILOT) | 2.0s | 0.05 | 100ms | 300ms | ≥ 85 | ≥ 95 |
| `/dash/insights/[id]?mode=expert` | 2.5s | 0.05 | 100ms | 400ms | ≥ 80 | ≥ 95 |
| `/b2b` | 2.0s | 0.05 | 100ms | 300ms | ≥ 85 | ≥ 95 |

**CI** : `lighthouse-ci` GitHub Action sur PR. Fail si Score perf chute > 5 pts vs baseline ou LCP > budget. Configuration `.lighthouserc.js`.

**Effort** : 8h setup + ongoing.

### 5.3 WCAG AA (axe-core + manual audit)

- `@axe-core/playwright` intégré aux tests E2E → 0 violation critique tolérée.
- Manuel : keyboard navigation full (Tab, Enter, Esc, arrow keys dans toggle), screen reader test (VoiceOver macOS + NVDA Windows) sur 5 pages clés.
- Focus visible : toutes interactions ont un `:focus-visible` outline ≥ 2px high-contrast.
- Labels ARIA : tous les SVG ont `<title>` ou `aria-label`, toutes les icônes-boutons ont `aria-label`.
- Live regions : chatbot streaming response = `aria-live="polite"`, alertes event = `aria-live="assertive"`.
- `prefers-reduced-motion` : désactive animations gauge + scroll-into-view smooth.

**Effort** : 4h axe-core integration + 8h manuel = 12h.

---

## 6. Sécurité (XSS, CSRF, CSP)

| Risque | Mitigation | Fichier:ligne |
|---|---|---|
| XSS narrative LLM (chat output) | Markdown sanitize via `dompurify` côté client + post-process server-side `contains_forbidden_token` (déjà OK `src/api/routes/qa.py`) + rendu en `<MarkdownSafe>` whitelist tags `p, strong, em, ul, ol, li, a, code` | `web/components/MarkdownSafe.tsx` (à créer) |
| XSS UI inputs (broker_context B2B preview) | `html.escape` déjà appliqué côté SSR `src/api/routes/webapp.py:18-30` ; côté SPA : Next.js auto-escape sur `{var}` JSX | OK |
| CSRF endpoints PATCH/POST | SameSite=Strict cookies + `X-API-Key` header obligatoire (déjà imposé `src/api/auth.py`) + double-submit token pour formulaires | `web/lib/csrf.ts` |
| CSP strict | `Content-Security-Policy` header en `next.config.mjs` headers() : `default-src 'self'; script-src 'self' 'nonce-{nonce}'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https://www.gravatar.com; connect-src 'self' https://api.smartsentinel.ai https://*.stripe.com wss://api.smartsentinel.ai;` | `web/next.config.mjs` |
| Cookies sécurisés | `Secure` + `HttpOnly` + `SameSite=Lax` (Strict pour session) + Max-Age | API gateway / `src/api/auth.py` |
| HSTS | `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload` | Reverse proxy (Nginx / Vercel) |
| Subresource Integrity (CDN) | aucune CDN externe en P0 ; en P1 si Plausible / Sentry : SRI hash sur `<script>` | n/a P0 |
| Stripe key exposure | clé publishable uniquement côté client (`NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`), clé secret jamais côté Next | `.env.example` |
| Prompt injection chatbot | post-process `contains_forbidden_token` + filtres ajoutés FR/EN (cf. P0.8 `docs/value/information_enrichment_recommendations.md:156-170`) + max input 2000 chars + rate limit `src/api/auth.py` rate-limiter | OK ressources, audit en CI |
| Iframe clickjacking | `X-Frame-Options: DENY` (sauf B2B docs iframe) + `frame-ancestors 'self'` CSP | header next config |
| API rate limiting | 100 req/min IP déjà en place (`MEMORY.md` production wiring §3) + UI feedback `<RateLimitToast>` | OK backend, UI à ajouter |

**Effort sécurité frontend** : 12h (CSP config + SafeMarkdown + CSRF token + audit).

---

## 7. Métriques (LCP, FID, CLS, conversion, activation, NPS)

### 7.1 Web Vitals (techniques)

| Métrique | Cible | Outil | Alert seuil |
|---|---|---|---|
| LCP (Largest Contentful Paint) | < 2.0s (CO-PILOT), < 1.5s (landing) | Lighthouse CI + `web-vitals` lib client | > +200ms baseline → PR warn |
| FID / INP (Interaction to Next Paint) | < 100ms / < 200ms | `web-vitals` lib | > 300ms → alert Sentry |
| CLS | < 0.05 | LH CI | > 0.10 → fail CI |
| TTFB | < 600ms | LH + Plausible | > 1s → alert |
| Bundle JS gz | < 180kb (initial) / < 400kb (max route) | `@next/bundle-analyzer` | > +20kb baseline → PR warn |
| Image weight (LCP) | < 80kb WebP / AVIF | `next/image` + LH | manual review |

### 7.2 Product analytics (commerciales)

**Stack** : Plausible (EU-compliant, no cookie banner needed, GDPR safe) + Sentry (errors). Pas de GA4 par défaut (consent banner = friction landing).

| Événement | Page | Propriétés | KPI piloté |
|---|---|---|---|
| `landing_view` | `/` | source UTM, lang, viewport | bounce rate |
| `cta_demo_click` | `/` | position (hero / mid / final) | conversion landing → demo |
| `signup_started` | `/auth/signup` | source (Google / email) | funnel TOFU |
| `signup_completed` | `/auth/signup` | duration | conversion signup |
| `onboarding_step_completed` | `/onboarding/{n}` | step, duration | drop-off par écran |
| `first_insight_opened` | `/dash/insights/[id]` | mode, time_since_signup | activation rate |
| `view_mode_switched` | `/dash/insights/[id]` | from, to | usage par mode |
| `chat_question_asked` | `/dash/insights/[id]` | question_type (suggested / free), tier | chat engagement |
| `chat_refuse_pedagogique` | `/dash/insights/[id]` | input_pattern | compliance audit |
| `upgrade_clicked` | `/pricing` ou `/dash/*` | from_tier, to_tier, position | conversion paid |
| `checkout_started` | Stripe redirect | tier, period | MRR pipeline |
| `checkout_completed` | webhook Stripe | tier, mrr | MRR live |
| `b2b_key_created` | `/b2b/keys` | scope | B2B activation |
| `b2b_webhook_delivered` | webhook | latency, status_code | B2B health |

### 7.3 KPI dashboard interne (Métabase ou Grafana sur Postgres)

| KPI | Définition | Cible M6 | Cible M12 |
|---|---|---|---|
| Bounce rate landing | sessions 1-page / total | < 55% | < 45% |
| Conversion landing → signup | signup_completed / landing_view | > 0.5% | > 1.5% |
| Activation rate | first_insight_opened dans J+1 / signup_completed | > 65% | > 75% |
| Conversion FREE → paid | paid users / free users (cumulés) | > 2% | > 5% |
| MRR | sum(stripe.subscription.amount) | $500-1k | $5-7k |
| Churn M1 | cancellations / paid M-1 | < 8% | < 5% |
| Chat engagement | chat_question_asked / first_insight_opened | > 30% | > 50% |
| NPS (in-app prompt J+30) | promoteurs - détracteurs | > 20 | > 40 |
| Time-to-aha (signup → first_insight_opened) | médiane | < 90s | < 60s |

**Outil NPS** : in-app prompt minimaliste à J+30 (`<NPSPrompt>` shadcn) + Plausible custom event. Pas de SaaS NPS externe ($).

**Effort métriques** : 10h (Plausible setup + Sentry + events + dashboard Métabase).

---

## 8. Risques

| # | Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Multi-vues mal exécuté → toggle déroute au lieu d'aider | M | H | Tests utilisateurs en remote 3-5 personnes (UserTesting.com 1 session $49) sur démo HTML avant code Next ; itérer wireframes |
| R2 | Chatbot hallucine ou révèle des features brutes en B2C | M | H | Post-process `contains_forbidden_token` côté serveur + system prompt borné + max output 800 tokens FOCUS / 2000 CO-PILOT / 4000 EXPERT |
| R3 | LCP > 3s sur dashboard (charts + waterfall) | M | H | RSC streaming + lazy-load EXPERT viz + Recharts virtualization + image WebP/AVIF |
| R4 | Mobile FOCUS pas assez riche → power-user mobile churne | M | M | Toggle CO-PILOT/EXPERT possible sur mobile (re-flow vertical), CTA "ouvrez sur desktop" pour replay/RAG |
| R5 | Design system pas codifié → drift visuel multi-pages | H | M | Lock tokens JSON en CI (script `web/scripts/check-tokens.ts` qui parse Tailwind config) ; PR fail si nouvelle couleur hardcodée |
| R6 | Stripe checkout UX clunky | M | H | Stripe Checkout hosted (pas embed custom) + Stripe Customer Portal pour cancel/upgrade — 2h vs 20h embed |
| R7 | Dépendance designer freelance non livrable | M | M | Plan B : Tailwind UI Kit officiel ($149 one-time licence) en fallback ; livre 80% des composants premium |
| R8 | i18n incomplet → bug FR/EN sur prod | M | M | next-intl strict mode + tests Playwright avec viewport lang switch + missing key warns en dev |
| R9 | A11y régression silencieuse | M | M | axe-core en CI bloquant + Playwright snapshot diff |
| R10 | Onboarding tour bloque user (skip impossible) | M | H | "Passer" toujours visible en haut-droit, ne jamais forcer 3 écrans, persister `onboarding_completed` user_pref |
| R11 | B2B portal MVP révèle endpoints non documentés | L | H | OpenAPI strict generation + tests contractuels (schemathesis) — backend cat. 8 |
| R12 | Chargement EXPERT trop lourd → bounce | M | H | Code-split EXPERT en route séparée (Next.js dynamic import) ; HERO + CO-PILOT prioritaires |
| R13 | CSP trop strict casse Stripe / Sentry | M | M | Test CSP en staging avec `Content-Security-Policy-Report-Only` 1 semaine avant enforce |
| R14 | Mode FOCUS sur mobile mais user veut détail → frustration | M | M | CTA "voir plus" → CO-PILOT mobile rendu en accordéon vertical, scroll natif |
| R15 | RGPD analytics → consent banner imposé → friction | L | M | Plausible (no cookie) en P0 ; pas de GA4 par défaut ; consent banner uniquement si analytics étendu en P1 |

---

## 9. Dépendances (API, auth, compliance, delivery)

| Dépendance | Catégorie source | Bloque quoi en UX | Statut actuel |
|---|---|---|---|
| `GET /api/v1/user/preferences` + `PATCH` | Cat. 8 API backend | toggle 3-vues persistance | 🔴 à créer (4h backend) |
| `POST /api/v1/keys` + `DELETE /api/v1/keys/{id}` | Cat. 8 + Cat. 10 auth | B2B portal clés | 🟡 partial — confirmer `src/api/auth.py` |
| `GET /api/v1/insights/{id}/replay` | Cat. 8 | EXPERT replay chart | 🔴 à créer (8h) |
| `POST /api/v1/qa` streaming SSE | Cat. 9 LLM | chatbot streaming | 🟡 confirmer endpoint streaming dans `src/api/routes/qa.py` |
| Stripe webhook + customer portal | Cat. 1 GTM + Cat. 8 | billing UI | 🔴 webhook live non commité (`reports/commercialization_sprint/01_commercialization_gtm.md:41`) |
| Compliance footer text FR/EN/DE/ES | Cat. 18 compliance | footer permanent toutes pages | 🟢 livré sprint W1 (`memory/sprint_w1_compliance_2026_04_29.md`) |
| Telegram notif format alignée mockup | Cat. 11 delivery | cohérence cross-canal | 🟢 mockup figé (`mockups/telegram_b2c.txt`) |
| Domaine + DNS + Vercel/Cloudflare | Cat. 19 deployment / Cat. 1 | landing publique | 🟡 domaine réservé, DNS à pointer |
| Geo-block US/QC/UK | Cat. 18 | landing + signup gating | 🟢 livré (`memory/sprint_w1_compliance_2026_04_29.md`) |
| Glossary JSON `src/intelligence/locale/glossary_{fr,en}.json` | Cat. 9 LLM / contenu | tooltips jargon | 🔴 à créer (P0.3 `docs/value/information_enrichment_recommendations.md:60-81`) |
| Track-record data feed (PF + IC + N + lookback) | Cat. 6 backtest | hero card | 🟡 à confirmer — historical_stats champs déjà dans contrat (`mockups/b2b_insight.json` à enrichir cat. 6) |

---

## 10. Estimation totale & timeline + budget design externe

### 10.1 Décomposition effort par bloc

| Bloc | Livrables | Heures Loukmane | Heures designer externe |
|---|---|---:|---:|
| P0 Stack frontend lock + scaffold | Next 15 + TS + Tailwind + shadcn + Zustand + TQ + i18n + Sentry + Plausible | 24 | 0 |
| P0 Design system (tokens + 12 composants + Storybook léger) | Figma + tokens JSON + Tailwind config + composants forkés | 10 | 30 |
| P0 Webapp B2C MVP (3 vues + chatbot + dashboard list) | `/dash` + `/dash/insights/[id]` toggle + chatbot streaming + history | 80 | 0 |
| P0 Auth UI (signup, login, OAuth, reset) | `/auth/*` + magic link + Google OAuth | 18 | 0 |
| P0 Onboarding 5 écrans + email digest J+1 | tour + i18n + email template | 24 | 0 |
| P0 Landing publique + pricing + méthodologie + FAQ + about | 6 pages statiques RSC | 32 | 0 |
| P0 B2B portal MVP (keys + usage + webhooks + docs + audit + SLA) | `/b2b/*` + Redoc OpenAPI | 60 | 0 |
| P0 Sécurité frontend (CSP, CSRF, sanitize) | next.config + composants + audit | 12 | 0 |
| P0 i18n FR/EN (DE/ES P1) | next-intl + 1100+ keys traduites FR/EN | 18 | 0 |
| P0 Métriques (Plausible + Sentry + events + KPI dashboard) | event tracking + Metabase | 10 | 0 |
| P1 Mobile responsive QA + light mode | toutes pages QA + toggle theme | 34 | 0 |
| P1 Status page inhouse | `/status` + poll health + history | 16 | 0 |
| P1 Tests Playwright + Lighthouse CI + axe-core | 12 E2E + 6 visual + LH budgets | 36 | 0 |
| P1 i18n DE + ES | translations + QA native | 12 | 0 |
| P2 Mobile native (Expo iOS+Android) | FOCUS mode + push notifications | 120 | 0 |
| P2 Browser extension (Plasmo Chrome+Firefox) | popup FOCUS + alertes | 80 | 0 |
| **Total P0** | **MVP commercialisable** | **288** | **30** |
| **Total P0 + P1** | **MVP enrichi prêt à scaler** | **386** | **30** |
| **Total P0 + P1 + P2** | **stack complète multi-surface** | **586** | **30** |

### 10.2 Timeline (Loukmane solo + designer ponctuel)

Hypothèse : Loukmane à **30h/sem produit + 8h/sem marketing**. Designer freelance livre Figma en 2 semaines calendaire (parallèle).

| Sprint (2 sem) | Livrables | Heures cumulées |
|---|---|---:|
| S1 — Stack + Design System kickoff | Next scaffold, Tailwind config, designer brief envoyé | 30 |
| S2 — Design system livré + composants forkés | tokens JSON, 12 composants shadcn forks, Storybook minimal | 60 (Loukmane) + 30 (designer livré) |
| S3 — Webapp `/dash/insights/[id]` 3-vues + chatbot streaming | toggle + 3 layouts + chat sidebar | 130 |
| S4 — Auth UI + onboarding 5 écrans | signup, login, OAuth, tour | 180 |
| S5 — Landing + pricing + méthodologie + i18n FR/EN + sécurité + métriques | 6 pages + CSP + Plausible | 240 |
| S6 — B2B portal MVP + status page + Playwright | `/b2b/*` + `/status` + tests | 316 |
| S7 — Mobile responsive QA + light mode + LH CI | tous écrans QA mobile + light | 340 |
| S8 — Polish + i18n DE/ES + bug fixes + soft launch | go-live | 360 (P0+P1) |
| S9–S10 (P1+P2 bonus) | tests visuels + audits + extension P2 si traction | 400+ |

**Soft launch envisageable** : fin S7 (semaine 14) sous condition de qualité MVP P0+P1 essentiels.
**GO commercialisation** : fin S8 (semaine 16).

### 10.3 Budget design externe

| Option | Coût | Délai | Couverture |
|---|---|---|---|
| **Designer freelance FR Malt/Comet** (30h × $90-120/h) | **$2 700-3 600** | 2-3 sem | Figma file complet + tokens + revues |
| Designer freelance UpWork/Fiverr Pro (30h × $50-70/h) | $1 500-2 100 | 2-4 sem | Qualité variable, à filtrer portfolio |
| Tailwind UI Kit licence ($149) + Loukmane solo (30h sur Figma) | $149 + sweat | 1 sem | 80% des composants premium prêts, perso à finaliser |
| Agence design boutique 3-4 designers (40h) | $5 000-8 000 | 3-4 sem | qualité premium, surcoût sans ROI pour MVP |
| **Choix recommandé** | **Tailwind UI Kit $149 + freelance 15h ($1 350)** = **$1 499** total | 2 sem | hybride : UI Kit pour base, freelance pour personnalisation tokens + 4-5 écrans signature |

**Budget design retenu** : **$1 500-2 000** (P0). Possibilité de monter à $3 600 si traction confirmée S+8 (designer ponctuel sur landing premium + email templates).

### 10.4 Coûts récurrents tooling

| Outil | Coût mensuel | Justification |
|---|---|---|
| Vercel Hobby (landing + status) | $0 | suffisant <100k req/mo |
| Vercel Pro (app + b2b) | $20 | si bandwidth/build > Hobby |
| Plausible Cloud (10k pageviews) | $9 | EU-compliant, no cookie |
| Sentry Team | $26 | errors + perf |
| GitHub Actions | $0 | inclus 2k min/mo free |
| Figma Pro | $15 (1 seat) | design system + assets |
| Domaine `smartsentinel.ai` | ~$2/mo | déjà payé |
| Tailwind UI licence | $149 one-time | one-shot |
| **Total mensuel récurrent P0** | **~$70/mo** | |

---

## 11. Critères GO/NO-GO commercialisation UX

| Critère | Seuil GO | Méthode mesure |
|---|---|---|
| Toggle 3-vues persistant fonctionnel | toutes vues rendues, préférence persisted 3 reloads | Playwright `view-mode.spec.ts` |
| Chatbot pilier permanent sur 3 vues | sidebar desktop + FAB mobile + overlay FOCUS | manual + Playwright |
| Hero card track record honnête | PF + IC + N + lookback visibles toujours | Playwright `hero-card.spec.ts` |
| Onboarding < 90s time-to-aha | médiane sur 20 sessions test | Plausible custom event |
| LCP < 2s sur `/dash/insights/[id]` mobile 4G | LH CI mobile profile | LH CI |
| Score Lighthouse Perf ≥ 85 mobile sur 3 pages clés | landing, dash, b2b | LH CI |
| Score a11y ≥ 95 sur toutes pages clés | axe-core CI | Playwright + axe |
| WCAG AA contraste tous pairs texte/fond | 0 violation critique | axe-core CI |
| Compliance footer permanent rendu | toutes routes incluent footer | Playwright `compliance.spec.ts` |
| Refus pédagogique chatbot fonctionnel | 5 patterns "dois-je acheter ?" testés | Playwright `compliance-refuse.spec.ts` |
| B2B keys lifecycle (create/revoke/rotate) | E2E pass | Playwright `b2b-keys.spec.ts` |
| Stripe checkout button → redirect | landing pricing + dash upgrade | Playwright |
| Status page poll OK 24h | `/status` historise 24h sans gap | manual check |
| i18n FR + EN 100% des labels | next-intl warnings 0 | dev console |
| CSP strict actif sans break | aucune violation Sentry 48h | Sentry |

---

## 12. Annexes

### 12.1 Glossaire technique → grand public (extraits, à externaliser dans P0.3)

Voir `docs/value/information_enrichment_recommendations.md:60-81` pour la table complète. Le glossaire vit dans `web/locales/glossary_fr.json` et `web/locales/glossary_en.json` (P0).

### 12.2 Références produit inspirantes

- **Bloomberg Terminal** — densité EXPERT, dark, mono, sobriété.
- **Linear** — densité élégante, animations subtiles, command palette.
- **Stripe Dashboard** — clarté FOCUS, hiérarchie typo, performance.
- **Pitchbook** — track-record en hero, honnêteté chiffrée.
- **Notion AI sidebar** — chatbot pilier permanent.
- **Vercel dashboard** — B2B portal, usage chart, API keys table.
- **Statuspage.io** — status page minimaliste.
- **Apple Health onboarding** — guided tour 3-5 écrans.

### 12.3 Fichiers à créer (référence implémentation)

```
web/
  app/
    layout.tsx                          # ComplianceFooter persistent
    (marketing)/
      page.tsx                          # landing /
      pricing/page.tsx
      methodology/page.tsx
      about/page.tsx
      changelog/page.tsx
    auth/
      signup/page.tsx
      login/page.tsx
      reset/page.tsx
    onboarding/
      [step]/page.tsx                   # 5 écrans
    dash/
      page.tsx                          # /dash list insights
      insights/[id]/page.tsx            # 3-views toggle
      history/page.tsx
      methodology/page.tsx
    account/
      preferences/page.tsx
      billing/page.tsx
    b2b/
      page.tsx                          # usage dashboard
      keys/page.tsx
      webhooks/page.tsx
      docs/page.tsx                     # Redoc embed
      audit/page.tsx
      sla/page.tsx
    status/page.tsx
  components/
    HeroCard.tsx
    ConvictionGauge.tsx
    TrackRecordBadge.tsx
    EventCountdown.tsx
    ViewModeToggle.tsx
    SupportCardGrid.tsx
    ConfluenceWaterfall.tsx
    ConformalIntervalViz.tsx
    HistoricalStatsGrid.tsx
    ReplayChart.tsx
    RAGSourcesList.tsx
    ChatSidebar.tsx
    ChatFAB.tsx
    JargonTooltip.tsx
    ComplianceFooter.tsx
    TierLockedSection.tsx
    NPSPrompt.tsx
    MarkdownSafe.tsx
    KillSwitchBanner.tsx
  lib/
    api.ts                              # TQ client
    auth.ts
    csrf.ts
    glossary.ts
    web-vitals.ts
  locales/
    glossary_fr.json
    glossary_en.json
    messages_fr.json
    messages_en.json
  stories/                              # Storybook P1
  tests/                                # Playwright
  tokens/
    colors.json
    radii.json
    spacing.json
    typography.json
  next.config.mjs                       # CSP headers
  tailwind.config.ts                    # tokens import
  tsconfig.json
  .env.example
  .lighthouserc.js
  package.json
```

---

**FIN DU PLAN — Catégorie 20 : Product UX & Frontend**

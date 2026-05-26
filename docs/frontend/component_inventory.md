# Frontend Component Inventory — Sprint F0→F5

**Date** : 2026-05-26
**Branche** : `institutional-overhaul`
**Stack** : Next.js 15.5.18 · React 18.3 · TypeScript strict · Tailwind 3.4 · shadcn/ui `new-york` · next-intl 3.26 · next-themes 0.4

---

## Status legend

| | |
|---|---|
| ✅ **READY** | Composant livré, testé via build + smoke test runtime, prêt à intégrer un backend |
| 🟡 **PARTIAL** | Logique en place, V1 utilise des mocks — wiring backend pour V2 |
| 🔒 **LEGAL-PENDING** | Wording placeholder, à remplacer par la sortie du terminal légal |

---

## 1 · Primitives shadcn/ui (`components/ui/`)

Toutes en style `new-york`, baseColor `slate`, copiées depuis la registry shadcn (pas d'install dynamique) pour éviter une dépendance CLI au build.

| Composant            | Statut | Description                                                 |
| -------------------- | ------ | ----------------------------------------------------------- |
| `Button`             | ✅      | 6 variantes (default, destructive, outline, secondary, ghost, link) + 4 sizes. `asChild` Slot |
| `Card` + sous-parts  | ✅      | Header, Title, Description, Content, Footer                 |
| `Badge`              | ✅      | 8 variantes incl. `bull` / `bear` / `neutral` / `warn` (domain) |
| `Accordion`          | ✅      | Radix-backed, animations `accordion-down/up` via tailwindcss-animate |
| `Dialog`             | ✅      | Overlay + Content centré + close button                     |
| `Sheet`              | ✅      | Variant `side` (top/bottom/left/right), responsive `w-full` mobile |
| `Tooltip`            | ✅      | Provider monté dans le layout, delayDuration=200 ms          |
| `Tabs`               | ✅      | List + Trigger + Content (non utilisé en F0-F5, dispo pour V2) |

---

## 2 · App-level & infra (`components/`, `lib/`, `app/`)

| Composant / fichier                              | Statut          | Notes                                                                                       |
| ------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------- |
| `app/[locale]/layout.tsx`                        | ✅               | Async params (Next 15), ThemeProvider, TooltipProvider, ChatProvider, NextIntlClientProvider |
| `app/[locale]/page.tsx`                          | ✅               | Landing : compose Hero + Demo + HowItWorks + Pricing                                        |
| `middleware.ts`                                  | ✅               | next-intl + 302 redirect `/en/*`, `/de/*`, `/es/*` → FR équivalent                          |
| `i18n.ts`                                        | ✅               | next-intl 3.22+ pattern (requestLocale async, locale retourné)                              |
| `theme-provider.tsx`                             | ✅               | Wrapper next-themes (`attribute="class"`, `defaultTheme="system"`)                          |
| `theme-toggle.tsx`                               | ✅               | Bouton Sun/Moon, persistance localStorage via next-themes                                   |
| `Nav.tsx`                                        | ✅               | Sticky top, brand + 3 anchors (#demo, #comment, #tarifs) + ThemeToggle. Locale switcher retiré (FR-only V1) |
| `Footer.tsx`                                     | 🔒              | Brand + 5 liens placeholder (`#legal-*`) + bandeau compliance LEGAL-PENDING                  |
| `lib/utils.ts`                                   | ✅               | `cn()` = clsx + tailwind-merge                                                              |
| `lib/mocks.ts`                                   | 🟡 mock         | Loader typé `SAMPLE_SIGNALS` ; helper `getSampleSignalById`, `getHeroSampleSignal`           |
| `lib/chatbot.ts`                                 | 🟡 mock         | Loader scripted dialogue + `FALLBACK_SCRIPT` defensive                                       |
| `lib/insight-formatters.ts`                      | ✅               | Toutes les chaînes user-visible (verdict, conviction, sessions, HMM labels, retest, gate, vol regime, événements, historique). Auditable en un seul fichier |

---

## 3 · Insight components (`components/insight/`)

Le cœur du produit — couches 1 + 2 de l'architecture progressive uniforme.

| Composant                                | Statut | Couche | Notes                                                                                  |
| ---------------------------------------- | ------ | ------ | -------------------------------------------------------------------------------------- |
| `MarketReadingCard`                      | ✅      | 1 + 2  | Orchestrateur. Props : `signal`, `onAskChatbot?`, `heroOnly?`, `defaultOpenSections?`   |
| `InsightGallery`                         | ✅      | —      | Client wrapper qui bridge la liste statique à `useChat()`                              |
| `VerdictHeader`                          | ✅      | 1      | Badge directionnel + verdict une-ligne                                                 |
| `ConvictionGauge`                        | ✅      | 1      | Score 0-100 + bande conformelle + tick + label long, `role="meter"`                    |
| `TemporalBadge`                          | ✅      | 1      | Client component, tick 30s, "Émise il y a X · Lecture expire dans Y"                  |
| `DisclaimerStub`                         | 🔒      | 1      | Placeholder UE 2024/2811                                                               |
| `InsightSections`                        | ✅      | 2      | Accordion type="multiple" + 5 enfants                                                   |
| `sections/StructureSection`              | ✅      | 2      | BOS / FVG / OB / retest / invalidation, prix formaté par instrument                    |
| `sections/RegimeSection`                 | ✅      | 2      | HMM label + posterior, BOCPD stabilité (3 seuils), jump descriptor, gate décision      |
| `sections/VolatilitySection`             | ✅      | 2      | Forecast vs naïve, CI conformelle, badge fallback                                      |
| `sections/EventSection`                  | ✅      | 2      | Prochain event + countdown + badge "Imminent" si ≤ 4h, blackout, session, sentiment    |
| `sections/HistorySection`                | ✅      | 2      | **Hero différenciateur** — encart PF + IC 95 %, hit rate, fenêtre walk-forward         |

---

## 4 · Chat components (`components/chat/`)

Couche 3 — pilier conversationnel (moat #1).

| Composant            | Statut          | Notes                                                                            |
| -------------------- | --------------- | -------------------------------------------------------------------------------- |
| `ChatProvider`       | ✅               | Context : `isOpen`, `activeSignal`, `turns[]`, `openFor`, `close`, `appendExchange`, `resetTurns`. Switch signal → reset turns |
| `ChatPanel`          | ✅               | Sheet side="right", responsive `w-full sm:max-w-md md:max-w-lg`. Auto-scroll, bouton reset, bandeau compliance permanent |
| `ChatMessage`        | ✅               | Bulles user/assistant avec avatars Lucide, `whitespace-pre-wrap`                  |
| `SuggestedQuestions` | ✅               | Stack verticale de chips, retire les questions déjà posées via `consumedIds`     |
| `ChatInputStub`      | 🟡 V2 wiring    | Input + bouton désactivés avec Tooltip "Saisie libre — disponible bientôt"        |
| (intro bubble)       | ✅               | Inline dans `ChatPanel`, présente Sentinel + recadre refus d'ordres               |

---

## 5 · Landing components (`components/landing/`)

| Composant              | Statut           | Notes                                                                                 |
| ---------------------- | ---------------- | ------------------------------------------------------------------------------------- |
| `HeroSection`          | 🔒 partiel       | Headline + sub + 2 CTAs + pépite track-record honnête. CTAs LEGAL-PENDING              |
| `DemoSection`          | ✅                | 3 cards via `InsightGallery`, première ouvre `history` par défaut                      |
| `HowItWorksSection`    | ✅                | 3 cards (verdict / sections / chatbot) avec icônes Lucide                              |
| `PricingSection`       | 🔒                | 4 tiers placeholder (Découverte / Analyste 29€ / Stratège 79€ / Institutionnel 1990€) — wording + features + CTAs LEGAL-PENDING |

---

## 6 · Types & contracts (`types/`)

| Fichier        | Statut | Notes                                                                                               |
| -------------- | ------ | --------------------------------------------------------------------------------------------------- |
| `insight.ts`   | ✅      | Mirror complet `InsightSignalV2 v2.1.0` (12 sub-models, type guards `isBullish/isBearish/isNeutral`) |
| `chatbot.ts`   | ✅      | `ChatbotQuestion`, `ChatbotScript`, `ChatbotResponses`                                              |

---

## 7 · Mocks (`mocks/`)

| Fichier                    | Statut | Contenu                                                                                                   |
| -------------------------- | ------ | --------------------------------------------------------------------------------------------------------- |
| `sample_signals.json`      | ✅      | 3 signaux (XAU M15 BULL conv 72 / EUR H1 BEAR 58 / XAU H4 NEUTRAL 42) — chaque champ du contrat v2.1.0     |
| `chatbot_responses.json`   | ✅      | 5 questions/réponses scriptées par signal × 3 = 15 dialogues, refus pédagogique adapté à la direction      |

---

## 8 · i18n status

| Locale | Routage actif | Messages JSON | Notes                                                          |
| ------ | ------------- | ------------- | -------------------------------------------------------------- |
| `fr`   | ✅ default     | ✅ partiel      | Toutes les chaînes neuves sont inline (non-traduites V1). next-intl reste en place |
| `en`   | ❌ 302 → FR    | ✅ dormant      | Dossier conservé pour réactivation V2                         |
| `de`   | ❌ 302 → FR    | ✅ dormant      | Idem                                                          |
| `es`   | ❌ 302 → FR    | ✅ dormant      | Idem                                                          |

Aucun stub indexable — toutes les URL EN/DE/ES retournent 302.

---

## 9 · Bundle & performance

| Sprint | `/[locale]` size | First Load JS | Delta |
| ------ | ---------------- | ------------- | ----- |
| F1     | 171 B            | 107 kB        | —     |
| F2     | 1.59 kB          | 111 kB        | +4 kB |
| F3     | 9.14 kB          | 121 kB        | +10 kB |
| F4     | 10.8 kB          | 123 kB        | +2 kB |
| F5     | 10.8 kB          | 126 kB        | +3 kB |

Cible Lighthouse mobile ≥ 90 — à vérifier par l'utilisateur en V1
(`npx lighthouse http://localhost:3000 --view`).

---

## 10 · Markers LEGAL-PENDING (à grep avant chaque release)

```bash
grep -r "LEGAL-PENDING" webapp/
grep -r 'data-legal-pending=' webapp/
```

Endroits ciblés :
- Disclaimer hero card (`DisclaimerStub`)
- Disclaimer pricing section (paper-trading + geo-block)
- Refus pédagogique chatbot (réponses scriptées dans `chatbot_responses.json`)
- Footer : 5 liens `#legal-*` + bandeau compliance
- Hero : disclosure `edge_claim=False` wording
- Pricing : features + CTAs + cadence (avant injection wording finalisé)

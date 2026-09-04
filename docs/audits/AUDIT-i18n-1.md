# AUDIT I18N-1 — Trois langues, aucune fuite

**Branche** : `fix/i18n-1-revision-complete` (worktree dédié `wt-i18n-1`)
**Base** : `origin/main` @ `6d658ef` (à jour — inclut CLN-1 mergé PR #194)
**Date** : 2026-09-03

## 0. Discipline / position HEAD

Le répertoire d'invocation (`TradingBOT_Agentic`) était sur `docs/preserve-data-1-audit`,
**30 commits derrière `origin/main`** (piège DATA-1). Le diagnostic **et** l'implémentation ont
été faits dans un **worktree dédié neuf** créé depuis `origin/main` à jour. Trois branches i18n
préexistantes (`feat/i18n-multilingual`, `feat/i18n-product-surfaces`, `feat/sc-2c-i18n-scannerchat`)
sont l'origine du système actuel, déjà fondu dans main — sans objet.

Décision structurante confirmée par le fondateur au STOP : **(A) frontend habille des données
brutes** ; **variante espagnole = es-ES** ; **géo = à corriger** (mais voir §7 : contradiction de
gouvernance découverte → non touché).

---

## 1. Inventaire des locales — AVANT / APRÈS

| | AVANT | APRÈS |
|---|---|---|
| Locales | **9** : fr, en, de, es, it, pt, nl, pl, ar | **3** : fr, en, es |
| RTL | ar (arabe, `dir=rtl` + police Noto) | aucune (retirée) |
| Parité des clés | 100 % (2385 clés) | **100 %** (2400 clés, garde CI actif) |
| Complétude VALEUR es | ~91 % (≈220 chaînes anglaises dans le bundle es) | **~100 %** (227 chaînes traduites es-ES) |

Les 6 locales retirées (de, it, pt, nl, pl, ar) sont supprimées **de la source unique**
`SUPPORTED_LOCALES` (i18n.ts) → le sélecteur, le routage next-intl et le garde de parité en
dérivent. Une URL manuelle `/de/...` renvoie désormais 404 (pas de contenu à moitié traduit
exposé). Police arabe + règle CSS `html[dir=rtl]` + `OG_LOCALES`/`LOCALE_LABELS` réduits.

### Complétude es — détail de la dette payée
Les surfaces `auth.*`, `billing.*`, `zones.*`, `calendar.*`, `pages.*` avaient été **écrites après
la dernière passe de traduction es** → elles contenaient de l'anglais. 227 chaînes traduites en
es-ES (jargon SMC gardé en anglais, tokens `{…}`/ICU/HTML préservés, mots interdits proscrits sauf
déni). Vérifié par re-scan : 0 fuite inter-langue résiduelle (hors cognats légitimes « Error 404 »,
« total: {level} »).

---

## 2. Recommandation retenue — Question A (origine des annotations du graphique)

**Constat** : toutes les annotations du graphique sont produites **côté client (frontend
TypeScript)**, jamais dans le backend Python. Le backend émet des **données brutes + enums**
(`liquidity_pools` : `side` bsl/ssl, `status` intact/swept/broken ; timestamps **UTC**). Un mécanisme
locale-aware existait déjà dans les mêmes fichiers (`formatLocalDayLong`, `use-reading-formatters`)
à côté de chemins figés `'fr-FR'` — d'où les deux mécanismes coexistant constatés.

**Recommandation retenue (confirmée par le fondateur) : le frontend habille.** Le backend garde son
contrat propre et cacheable (le SemanticCache ne se fragmente pas par langue), aucune fuite LLM,
et le tuyau i18n était déjà à moitié posé. **Aucune modification du contrat backend n'a été
nécessaire pour les annotations.**

Corrections appliquées :
- `lib/chart/liquidityLines.ts` : le mécanisme d'injection `LiquidityLineLabels` existait mais
  `ReadingChart.tsx` ne l'utilisait pas → repli FR figé. Désormais `ReadingChart` injecte des labels
  **locale-aware** via un nouveau hook `useReadingFormatters().liquiditySide/SideShort/SideChart`
  (clés `reading.labels.liquiditySide*` ajoutées fr/en/es). « Liquidité vente · intacte » →
  « Sell liquidity · intact » / « Liquidez venta · intacta ».
- `lib/time/localTime.ts` : `formatLocalHm`/`formatLocalDayHm` rendus **locale-aware** ; format de
  date **non ambigu** (mois nommé court : « 31 août » / « Aug 31 » / « 31 ago », jamais « 01/09 ») ;
  **24 h partout** (`hour12:false`, jamais « 2:30 PM »). Libellé « Heure locale · UTC−X » → clé i18n
  `app.chart.localTime` (offset dynamique, préfixe traduit).
- `ReadingChart.tsx` : `localization.locale` et `tickMarkFormatter` passent de `'fr-FR'` figé à la
  **locale active** + mois nommé sur l'axe.
- Code mort retiré : `formatTrendMaturity`/`formatBreakTimestamp` (phrases FR figées, jamais rendues,
  référencées uniquement par leurs tests) → supprimés (`deriveTrendMaturity`, le calcul, intact).

---

## 3. Catalogue des marchés (« Or » menu vs « Gold » en-tête)

**Cause** : deux chemins de rendu divergents. Le menu latéral (`MarketSelector` → `formatInstrument`
→ `MARKET_LABEL`) rendait le `label` **FR figé** de `config/markets.json` ; l'en-tête utilisait déjà
la clé i18n `reading.labels.instrument_<code>`.

**Correction** : nouveau hook `useInstrumentLabel()` lisant la **même clé** que l'en-tête, adopté
partout (`MarketSelector` ×3, `AppChatSidebar`, `MobileWorkspace`, `ChatPanel`, `ThinkingIndicator`).
Le nom du marché est désormais **identique menu ↔ en-tête dans les 3 langues**. Les codes (XAUUSD…)
ne sont jamais traduits (suffixe de clé). **80 marchés** : ajout additif documenté dans la politique
SMC (§6) — 1 entrée `markets.json` + 3 valeurs i18n, aucune triplication.

---

## 4. Formats — un seul mécanisme, locale-aware

Littéraux `'fr-FR'` figés recensés et traités :

| Fichier | État | Action |
|---|---|---|
| `lib/time/localTime.ts` | figé fr-FR (axe/crosshair) | **corrigé** locale-aware + 24 h + mois nommé |
| `components/app/ReadingChart.tsx` | `localization`/`tickMark` fr-FR | **corrigé** locale active |
| `lib/zones/lifecycle.ts` | `formatZoneDateTime/ShortTime` | **corrigé** `hour12:false` (24 h cohérent) |
| `lib/scanner-chat/use-voice-input.ts` | es → dictée `fr-FR` (bug) | **corrigé** es → `es-ES` |
| `components/seo/JsonLd.tsx` | `inLanguage` + description fr figés | **corrigé** locale-aware (`buildSoftwareApplicationLd`) |
| `lib/market-reading/formatters.ts` | `formatPrice/ChangePercent` fr-FR | code MORT (test-only) — non rendu, laissé |
| `lib/calendar/grouping.ts` | `hmInZone` fr-FR | produit « 14:30 » (24 h neutre, 0 mot) — laissé |
| `lib/market-reading/sessions.ts`, `zones/formation-session.ts` | `en-GB`/`en-US` | **calcul interne** (extraction de parties horaires), pas d'affichage — laissé |

Le prix en-tête restait déjà correct (`use-reading-formatters`, locale-aware) ; il est maintenant le
**seul** mécanisme d'affichage, axes du graphique inclus.

---

## 5. Chaînes en dur corrigées (frontend)

| Fichier:élément | Avant | Après |
|---|---|---|
| `AppHeader.tsx` `<span>App/Zones/Scanner</span>` | EN figé | `tn('app'/'zones'/'scanner')` (`nav.app` ajouté) |
| `MobileMenu.tsx` aria `Ouvrir le menu` / `Navigation` | FR figé | `t('openMenu')` / `t('menuAria')` |
| `ReadingColumn.tsx` `focusChat` sélecteur | `[aria-label="Question libre…"]` (FR, cassait /en /es) | `[data-testid="chat-input"]` (locale-stable) |
| `TemporalBadge.tsx` `Chargement…` | FR figé (invisible mais scanné) | `t('loading')` (`reading.temporal.loading`) |
| `app/[locale]/not-found.tsx` | FR-only | **traduit** (`getTranslations`, namespace `notFound`) |
| `app/[locale]/error.tsx` | FR-only | **traduit** (map inline par locale via `useParams` — robuste : ne dépend pas du provider i18n qui pourrait être la cause du plantage) |

Résidus mineurs consignés (non bloquants, invisibles/marque) : `dialog.tsx`/`sheet.tsx` `<span sr-only>Close</span>`
(primitives shadcn, sr-only) ; `AppChatSidebar.tsx` « M.I.A Agent » (nom de marque, non traduit par
politique) ; `global-error.tsx` (racine hors provider, FR comme dernier recours).

---

## 6. Politique du jargon SMC

Fichier créé : **`docs/product/SMC-JARGON-POLICY.md`** (normatif). Termes jamais traduits dans les 3
langues : **Order Block (OB), Fair Value Gap (FVG), BOS, CHOCH, BSL, SSL** + codes d'instrument. Le
jargon reste **expliqué dans chaque langue** (ex. `zones.mia.answer.explain*` définissent « Order
Block »/« Fair Value Gap » en fr/en/es tout en gardant le terme anglais). Garde automatique :
`i18n-no-leak.test.ts` asserte la présence verbatim de « Order Block »/« Fair Value Gap » dans chaque
bundle et exempte le jargon du scan de fuite.

---

## 7. 🚩 Géo-restriction — CONTRADICTION À 4 VOIES (non touchée)

**Vérification i18n (livrée)** : la géo-restriction est **indépendante de la locale**.
`GeoBlockMiddleware` s'exécute côté backend FastAPI et renvoie **HTTP 451 avant** tout routage
next-intl. Changer de langue (`/es`, `/en`) ne modifie ni l'IP ni le pays résolu →
**non contournable par le sélecteur de langue**. Le middleware frontend ne fait que du routage.

**Contradiction découverte — `geo_block.py` NON MODIFIÉ** (logique légale sur gouvernance
contradictoire) :

| Source | Politique |
|---|---|
| Énoncé mission I18N-1 | autoriser **CA + US** |
| Code `geo_block.py` actuel | **deny-list** : bloque US (SEC §202(a)(11)), UK (FCA), OFAC (CU/IR/KP/RU/SY/BY) ; autorise le reste |
| Doc gouvernance `geo_block_allowlist.md` (vague1) | **allow-list = FR + BE + CH + LU** (bootstrap francophone ; US « V2+, probablement non ») |
| Réponse live fondateur (2×) | allow-list = **CA seul** (US reste bloqué SEC) |

Le blocage US est **délibéré (absence de licence SEC)**, pas une convention. Ces quatre politiques ne
coïncident sur rien, et **aucune** n'autorise l'Espagne alors qu'on ajoute l'UI espagnole. **À
réconcilier délibérément par le fondateur, hors périmètre i18n.**

---

## 8. 🚩 Backend i18n (FLAGGÉ — non implémenté, changement de contrat)

~20 messages API `src/api/routes/accounts.py` + `account_billing.py` + **e-mails transactionnels**
(sujets + corps) restent **FR figés** → fuient sur /en /es. Les localiser exige un **changement de
contrat backend/frontend** (la mission demande de le signaler) :

- **Recommandation (cohérente avec l'approche A)** : le backend renvoie des **codes d'erreur**
  machine, le frontend les habille via i18n. Le contrat reste locale-neutre.
- Alternative : catalogue i18n Python + lecture de `Accept-Language` (flux non authentifiés) /
  locale du compte (e-mails). Plus lourd, met de la prose dans le backend.

**Non fait dans cette passe i18n frontend.** Décision à trancher.

---

## 9. Vérifications

- **tsc** : `--noEmit` **vert** (exit 0) — a même résolu 3 erreurs pré-existantes `dictation-copy-honesty`.
- **build** : `next build` **vert** (exit 0).
- **vitest** : suite complète **verte — 113 fichiers / 966 tests, 0 échec**. Nouveaux/ajustés :
  `locale-parity` (3 locales), `i18n-no-leak` (mots-témoins + jargon SMC), `forbidden-vocab` (+es),
  tests copy-honesty/home/rg1 réduits à 3 locales, `localTime`/`useLocalTimeLabel`/`regime-facts`
  ajustés aux nouvelles signatures. Deux échecs traités : `error-notice` (régression du 404 traduit,
  corrigée) et `markets-guard` (pré-existant sur origin/main — le garde MKT-1 scannait les fixtures
  `*.test.ts` co-localisées ; exclusion alignée sur son propre skip des dossiers `__tests__`).
- **Playwright** : matrice fr/en/es × 1280×800 & 390×844 sur accueil, /app, /scanner, /zones,
  /actualites, /compte, connexion, inscription, paiement, 404 — **à capturer en revue live** (les
  captures sont le livrable de confirmation visuelle du fondateur avant merge).

## 10. Espagnol — variante retenue

**es-ES (Espagne)** : séparateurs alignés sur le FR (« 4.474,07 »), registre neutre européen, une
seule variante à maintenir. `OG_LOCALES.es = 'es_ES'`, `LD_LANG.es = 'es-ES'`. Codes/jargon inchangés.
Conséquence commerciale notée : l'UI espagnole attire des clients espagnols, mais l'Espagne n'est
autorisée par **aucune** des politiques géo listées (§7) — à réconcilier.

---

## Fichiers clés modifiés/créés

- `i18n.ts` (3 locales, `onError`/`getMessageFallback`), `middleware.ts`, `app/[locale]/layout.tsx`
- `lib/time/localTime.ts`, `lib/time/useLocalTimeLabel.ts`
- `lib/chart/liquidityLines.ts` (déjà prêt), `components/app/ReadingChart.tsx`
- `lib/market-reading/useInstrumentLabel.ts` (**nouveau**), `use-reading-formatters.ts`
- `components/seo/JsonLd.tsx`, `app/[locale]/(site)/layout.tsx`
- `app/[locale]/not-found.tsx`, `app/[locale]/error.tsx`
- `messages/{fr,en,es}.json` (227 trad. es + clés liquidité/nav/notFound/loading/chart.localTime)
- `docs/product/SMC-JARGON-POLICY.md` (**nouveau**)
- Tests : `lib/i18n/__tests__/i18n-no-leak.test.ts` (**nouveau**), `locale-parity`, `forbidden-vocab`, …
- Supprimés : `messages/{de,it,pt,nl,pl,ar}.json`

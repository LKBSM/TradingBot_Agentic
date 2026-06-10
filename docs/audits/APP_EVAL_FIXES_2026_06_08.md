# Correctifs post-évaluation founder /app (affichage, chatbot, i18n) + diagnostic structures/news

- **Date** : 2026-06-10
- **Branche** : `fix/app-live-evaluation-feedback` (depuis `feat/app-live-data-wiring`)
- **Périmètre** : corrections directes A/B/C + diagnostic D1/D2 (avec STOP avant tout
  changement de détection/pipeline). Aucune heuristique/seuil SMC touchée. Aucun
  tick/streaming. Aucun champ prédictif servi à /app. Chatbot niveau 1.5 strict.
  Frontend sans clé API (tout via `/api/*`, lecture cache, budget Twelve Data préservé).

| Lot | Sujet | Commit | Statut |
|-----|-------|--------|--------|
| C | Fuite i18n (tags anglophones en FR) | `6431403` | ✅ corrigé |
| B | Rendu markdown du chatbot | `b7de9ab` | ✅ corrigé |
| A | Prix d'en-tête unifié + variation % + refresh léger | `3f7bc01` | ✅ corrigé |
| D1 | Contradiction « retest de BOS » vs « aucune cassure » | `491e8aa` | ✅ surfaçage corrigé |
| D1 | Structures actives non persistées entre bougies | — | ⏸️ **DÉFÉRÉ (STOP)** |
| D2 | Pipeline news | — | ✅ câblé + frais — « aucun événement » honnête |

Vérifs : `vitest run` **140 → 151 tests verts** (0 régression), `tsc --noEmit` OK,
`next build` vert (`/[locale]/app` 159 kB first-load).

---

## A — Affichage du prix

### A1 — Prix incohérent entre timeframes → prix unifié

**Cause (tracée).** Le header utilisait `header.close_price`, soit la clôture de la
**dernière bougie CLÔTURÉE de la TF affichée**. Sur H1/H4 cette bougie a jusqu'à
1–4 h de retard sur le M15 ⇒ pour le même actif, prix M15 ≠ prix H1/H4
(ex. founder : XAU 4131,40 en M15 vs 4126,45 en H1/H4). Ce n'est pas une horloge
ni une fraîcheur de cache divergente : c'est la granularité de la bougie de
référence par TF.

**Correctif.** On dérive **UN prix unifié depuis le M15** (TF la plus fine servie
par `/api/candles` = clôture la plus fraîche disponible dans le cache descriptif),
identique quelle que soit la TF affichée. Le graphique conserve ses bougies propres
à la TF ; **seul le prix d'en-tête est unifié**.

- `lib/market-reading/price.ts` → `computeDailyChange(candles)` : prix = dernière
  clôture ; référence = dernière clôture du **jour UTC précédent** (saute les
  week-ends naturellement, faute de bougies) ; renvoie `changeAbs` / `changePct`.
- `lib/market-reading/hooks.ts` → `useLatestPrice(instrument)` : lit
  `/api/candles?timeframe=M15&limit=300` (lecture cache SQLite, **aucun appel
  Twelve Data**, **aucune clé API** côté front).
- Header : fallback gracieux sur `close_price` quand le flux M15 est indisponible
  (et sur les surfaces statiques du landing).

| | Avant | Après |
|--|-------|-------|
| Prix XAU en M15 | 4131,40 | **4121,98** (M15, unifié) |
| Prix XAU en H1/H4 | 4126,45 (≠) | **4121,98** (identique) |
| Source | `close_price` de la TF | dernière clôture M15 |

### A2 — Variation % colorée (descriptive)

Ajout d'une variation du jour `(dernier − clôture de référence) / référence`,
type variation quotidienne TradingView. **Fait de marché, jamais une prévision.**

- `formatters.ts` → `formatChangePercent(-0.0322) = "−3,22 %"` (fr-FR, signé),
  `changeTone()` → vert (hausse) / rouge (baisse) / muet (plat).

### A3 — Sensation « vivante » SANS tick

- Rafraîchissement à la **clôture de bougie** (déjà câblé sur `candle_close_ts`)
  **+ intervalle léger 45 s** (`DEFAULT_LATEST_PRICE_INTERVAL_MS`) pour le dernier
  prix M15 (lecture cache).
- `components/market-reading/LivePrice.tsx` : **flash discret** (~650 ms) au
  changement de valeur.
- **INTERDIT respecté** : aucun rafraîchissement seconde-par-seconde, aucun
  streaming. Le modèle reste « bougie clôturée ».

---

## B — Chatbot Sentinel (rendu)

**Cause.** `ChatMessage` imprimait le texte brut dans un `<div whitespace-pre-wrap>` :
les `**gras**` et listes markdown produits par le chatbot s'affichaient avec
astérisques et tirets crus.

**Correctif.** `lib/chat/markdown.tsx` → `renderMarkdown()` : renderer markdown
minimal **sans dépendance** (gras / italique / `code` / listes à puces / listes
numérotées / paragraphes), appliqué **uniquement aux messages assistant** (la
saisie utilisateur reste verbatim). Rendu via éléments React — **aucun
`dangerouslySetInnerHTML`**, donc pas de surface XSS (test d'échappement inclus).
Posture niveau 1.5 inchangée (aucune modification du prompt système ni du fond).

| Avant | Après |
|-------|-------|
| `Structure : **BOS confirmé**` (astérisques crus) | « Structure : **BOS confirmé** » (gras rendu) |
| `- point 1` / `- point 2` (tirets crus) | liste à puces propre |

---

## C — Fuite i18n

**Cause (tracée jusqu'à la donnée réelle).** Le backend (`_build_tags`,
`market_reading_mappers.py`) émet des tags `trend_<v>`, `volatility_<v>`,
`phase_<v>`, `bos_recent_<dir>`, `choch_recent_<dir>`, `retest_in_progress`,
`mtf_*`. Ces clés étaient **absentes** de `TAG_LABEL` (`tag-labels.ts`) et
retombaient sur `humanise()` ⇒ libellés anglophones en mode FR :
« Trend bearish », « Volatility elevated », « Phase expansion »,
« Retest in progress ».

> Confirmé sur la lecture réelle stockée (XAU M15, id=12, 2026-06-10T16:45Z) :
> `tags = ["trend_bearish","volatility_elevated","phase_expansion","retest_in_progress","ob_active"]`.

**Correctif.** Mapping FR complet de **tous** les tags réellement émis (la synthèse
texte, elle, était déjà en français — Haiku/template FR). Test de régression
couvrant la liste exhaustive + les 4 chaînes exactes signalées.

| Tag backend | Avant (fuite) | Après |
|-------------|---------------|-------|
| `trend_bearish` | Trend bearish | **Tendance baissière** |
| `volatility_elevated` | Volatility elevated | **Volatilité élevée** |
| `phase_expansion` | Phase expansion | **Phase d'expansion** |
| `retest_in_progress` | Retest in progress | **Retest en cours** |

---

## D — Diagnostic (classification (a) surfaçage / (b) détection)

### D1 — Structures manquantes & contradiction logique

**Pipeline tracé** : moteur SMC → `confluence_signal_to_structure`
(`market_reading_mappers.py`) → assembler → `/api/market-reading` →
`StructureSection.tsx`.

**Donnée réelle de référence** : reading XAU M15 stocké id=12 (close 16:45Z) —
reproduit exactement le cas founder :
`bos=null`, `fair_value_gaps=[]`, `order_blocks=[1 OB actif]`,
`retest_in_progress={type:"bos_retest", level:4132.53}`.

#### (a) Contradiction « retest de BOS » vs « aucune cassure » → **CORRIGÉ (surfaçage)**

Le mapper n'émet le champ `bos` que sur une **cassure FRAÎCHE au dernier close**
(`BOS_EVENT != 0`, choix délibéré « F6 » pour éviter qu'un BOS périmé apparaisse
sur ~100 % des lectures). Or un retest est armé **plusieurs bougies après** la
cassure, quand `bos` est déjà `null` — mais `retest_in_progress.type='bos_retest'`
référence cette cassure antérieure. La section affichait donc simultanément
« aucune cassure récente » **et** « retest de cassure (BOS) ».

→ **Pur surfaçage** (le modèle est cohérent : `bos` = frais ; retest = post-cassure).
Correctif `StructureSection.tsx` : quand un retest référence une cassure/un CHOCH
antérieur, la ligne **l'énonce** (« cassure antérieure en cours de retest »)
au lieu de la nier. **Aucun seuil touché.**

#### (b)/(a-mixte) Structures actives non persistées entre bougies → **DÉFÉRÉ (STOP)**

Le mapper ne surface que ce qui est **frais/actif sur la DERNIÈRE BOUGIE CLÔTURÉE** :
`bos` seulement sur cassure fraîche, `fair_value_gaps` seulement si `FVG_SIGNAL`
fire sur ce bar, `order_blocks` seulement si `OB_STRENGTH_NORM>0` sur ce bar. Les
zones (BOS antérieur, OB/FVG formés plus tôt) **encore visibles sur le graphique**
ne sont **pas persistées** dans la lecture courante.

- Le moteur **détecte** bien ces structures (les features existent dans
  l'historique) — ce n'est donc **pas** un déficit de seuil/détection au sens strict.
- MAIS les surfacer demande une **persistance/agrégation des structures actives
  entre bougies** dans l'assembler (gestion de cycle de vie : actif / mitigé /
  invalidé). C'est un **changement de scope du pipeline**, au-delà d'un fix de
  surfaçage pur.

➡️ **Par la règle STOP, je ne l'implémente pas sans feu vert.** Décision demandée au
founder :
1. **Option 1 (recommandée)** — Persister les structures actives au niveau assembler
   (BOS non-frais encore valide, OB/FVG actifs des N dernières bougies) avec
   statut de cycle de vie. ~Pipeline, pas de seuil. À cadrer.
2. **Option 2** — Laisser la lecture « dernier bar » telle quelle et l'**assumer
   éditorialement** (« structures fraîches sur la dernière bougie »).
3. **Option 3** — Reporter au Groupe B (post-annotation manuelle) si la calibration
   des définitions est jugée prérequise.

> Note connexe (non corrigée, signalée) : `technical_triggers_recent` est **toujours
> vide** (laissé hors scope « Chantier 3 » dans l'assembler) et `mtf_confluence` est
> `{}` sur les readings réels — la carte MTF du Régime reste donc vide. À traiter
> séparément si souhaité.

### D2 — News / événements → **CÂBLÉ + FRAIS ; « aucun événement » est honnête**

**Tracé** : `bootstrap.py` injecte `NewsPipeline(NewsCacheStore())` dans l'assembler
(`NEWS_PIPELINE_ENABLED=True` par défaut). `_build_events()` lit
`get_upcoming(USD/EUR, lookahead 240 min)` et `get_just_published(USD/EUR,
lookback 60 min)`.

**État réel du cache** (`data/news_cache.db`, 2026-06-10) : **18 événements,
`fetched_at = 2026-06-10T16:45:52Z` (aujourd'hui)**, `scheduled_at` du 07 au 12 juin
(Core CPI, PPI, ECB Press Conf, Unemployment Claims, BOC…). **Le pipeline est donc
bien branché et frais.**

**Pourquoi le founder voyait « aucun événement »** : sur la lecture id=12 (16:45Z),
le dernier événement USD (Core CPI 12:30Z) était à −4 h 15 (> 60 min de lookback) et
le prochain USD/EUR (PPI 06-11 12:30Z) à ~20 h (> 240 min de lookahead). **Aucun
événement USD/EUR notable dans la fenêtre −60 / +240 min à cet instant** ⇒
`events` vide. C'est exactement le modèle voulu (« événement macro notable **proche
de la lecture** », pas un firehose). **« Aucun événement » est correct et honnête.**

➡️ **Aucun changement de wiring nécessaire** (déjà câblé). Observation (déférée, au
choix du founder) : la fenêtre `just_published` de 60 min est étroite — un print USD
majeur disparaît de la lecture ~1 h après ; élargir le lookback (ex. 120–180 min)
augmenterait la présence d'événements. C'est un réglage de **config pipeline**, pas
un fix de surfaçage ⇒ laissé au founder.

---

## Tests ajoutés / modifiés

- `lib/market-reading/tag-labels.test.ts` — couverture exhaustive des tags backend
  (no English leak).
- `lib/chat/__tests__/markdown.test.tsx` — gras/italique/code/listes/paragraphes +
  échappement HTML.
- `lib/market-reading/price.test.ts` — `computeDailyChange` (référence jour
  précédent, saut week-end, fenêtre mono-jour, variation négative).
- `lib/market-reading/formatters.test.ts` — `formatChangePercent`, `changeTone`.
- `components/market-reading/__tests__/market-reading-components.test.tsx` —
  header avec prix unifié + variation ; fallback `close_price` ; non-contradiction
  retest/BOS.

## Vérifications finales

- `npx vitest run` → **151 tests verts** (21 fichiers), 0 régression.
- `npx tsc --noEmit` → OK.
- `npx next build` → vert ; `/[locale]/app` = 159 kB first-load JS.

## Commits

```
6431403  fix(app/C): libellés FR pour les tags backend (fuite i18n)
b7de9ab  fix(app/B): rendu markdown épuré des réponses du chatbot
3f7bc01  fix(app/A): prix d'en-tête unifié + variation du jour + rafraîchissement léger
491e8aa  fix(app/D1): lever la contradiction 'retest de BOS' vs 'aucune cassure'
```

## En attente de décision founder (STOP)

- **D1 (b)** — persistance des structures actives entre bougies (Option 1/2/3 ci-dessus).
- **D2** — élargir ou non la fenêtre `just_published` (réglage config).
- Hors-scope signalés : `technical_triggers_recent` toujours vide ; `mtf_confluence`
  vide sur readings réels.

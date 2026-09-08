# AUDIT DS-1 — Rendre les pages affichables hors de l'application

**Branche** : `feat/ds-1-composants-affichables` (worktree dédié `C:/MyPythonProjects/wt-ds-1`)
**Base** : créée depuis `origin/main` = `6d658ef` (PR #194). Écart au démarrage : le worktree
principal était **−30 commits** derrière `origin/main` → diagnostic et build menés contre `origin/main`
à jour (via `wt-run-main`), jamais contre le HEAD principal périmé.
**Statut** : livré, testé, **poussé — NON mergé** (fusion sur `main` après confirmation visuelle live).

---

## 0. Découverte qui a recadré la mission

Le titre suppose que les composants vont chercher leurs données eux-mêmes. **Ce n'est presque
jamais le cas** : la quasi-totalité des composants visés reçoivent DÉJÀ toutes leurs données en
propriétés. Deux choses seulement les empêchaient de s'afficher isolément :

1. **des contextes obligatoires** (i18n, thème) qui plantent quand ils sont absents ;
2. **l'absence d'un jeu de données figées réalistes** pour les alimenter.

Conséquence : **aucune chirurgie de composant n'a été nécessaire au Palier 0**. Le travail a été
(a) extraire un jeu de données réel figé, (b) une route galerie qui fournit les contextes et
alimente les composants, (c) des tests prouvant l'absence de réseau. **Zéro fichier existant
modifié** (voir §5) — donc les pages réelles sont, par construction, identiques avant/après.

---

## 1. Inventaire des dépendances

| Composant | Va chercher ses données ? | Contextes requis | i18n | Thème | Routeur | Navigateur |
|---|---|---|---|---|---|---|
| `zones/ZoneLifecycleCard` | non — **props** | i18n, `useReadingFormatters`, `useLocale` | `zones` | non | non | `useId`, `Date` |
| `market-reading/MarketReadingHeader` | non — **props** | i18n, formatters, locale | `app` | non | non | non |
| `market-reading/MarketReadingCard` | non — **props** | i18n | `reading.card` | non | non | non |
| `scanner/ScanResults` | non — **props** (`response`) | i18n, `useScannerLabels` | `scanner` | non | non | `Date` |
| `scanner/ComboCard` | non — **props** (`match`) | i18n, scanner labels | `scanner.combo` | non | `next/link` | `Date` |
| `chat/ChatMessage` | non — **props** | i18n | `chat` | non | non | `clipboard` |
| `chat/ChatComposer` | non — **props** (`onSubmit`) | `useLocale` | via props | non | non | textarea, Web Speech |
| `app/ReadingChart` | non — **props** (`candles`,`structure`) | `useTheme`, i18n, locale | `app` | lit CSS live | non | canvas, `ResizeObserver`, RAF, `getComputedStyle` |
| `chat/ChatPanel` | **oui** (`useChat`) | **`ChatProvider` (plante si absent)** | `chat` | non | non | scroll |
| `calendar/CalendarMonthView` | **oui** `useCalendarMonth` (seam `data` injecté) | i18n, `useMultiFilter`×3 | `calendar` | non | `next/link` | `Date` |
| `calendar/CalendarEventDetail` | **oui** `useCalendarEvent`+`usePublicationMeasures` | i18n, locale, voice | `calendar`,`chat` | non | `next/link` | `Date`,`Intl` |
| `shell/ShellRail` | props | i18n, **`useRouter`+`useSearchParams`** | `nav`,`app` | non | routeur | URL query |
| Blocs page d'accueil `landing/lp1/*` | **non — déjà 100 % statiques** | i18n | `home` | tokens CSS | `next/link` | UI locale |

`useReadingFormatters` ne dépend QUE de `next-intl` (`useTranslations('reading')` + `useLocale`) —
**aucun provider maison** : `NextIntlClientProvider` + `ThemeProvider` suffisent.

---

## 2. Le graphique (ReadingChart) — réponse franche

**OUI, il s'affiche à partir de bougies figées.** Il ne va PAS chercher ses données : il reçoit
`candles: Candle[]` + `structure` en props et peint dessus. En lui passant des bougies figées avec
`livePrice=null` / `selection=null`, il rend les bougies clôturées + les surcouches de structure en
peinture statique. La boucle de tick live et l'animation « respiration » ne s'activent QUE si
`livePrice`/une sélection sont fournis.

**Preuve** : dans la galerie il peint **260 bougies réelles XAU/USD H4 + structure**, sans tick,
dans un vrai navigateur (Chromium via Playwright). Voir la capture — la section « Graphique » est
peinte. **Il n'a donc PAS été refactoré** (déjà un composant de présentation ; le toucher =
risque de régression). Filet de secours si un harnais de rendu achoppe sur canvas :
capture image — non nécessaire ici, le rendu live fonctionne.

---

## 3. Classement par coût & ce qui a été livré

### Livré — Palier 0 (déjà « présentation », alimenté par données figées)

| Composant | État |
|---|---|
| En-tête de lecture (`MarketReadingHeader`) | ✅ galerie + test no-réseau |
| Carte de lecture (`MarketReadingCard`) | ✅ 3 variantes (riche / EUR / champs absents) |
| Carte de zone (`ZoneLifecycleCard`) | ✅ 5 états (prix dedans / au-dessus / dessous / testée / jamais touchée) |
| Résultat scanner (`ScanResults` + `ComboCard`) | ✅ 2 états (avec correspondances / aucune correspondance) |
| Message M.I.A (`ChatMessage`) | ✅ conversation 5 messages |
| Barre de saisie M.I.A (`ChatComposer`) | ✅ galerie + test no-réseau |
| Graphique (`ReadingChart`) | ✅ bougies figées, sans refactor |
| Blocs page d'accueil (`MiaSection`, `ReadingCarousel`) | ✅ déjà statiques, branchés tels quels |

### Laissé de côté (avec motif) — Paliers 1 & 2

| Composant | Palier | Motif |
|---|---|---|
| Vue mois du calendrier (`CalendarMonthView`) | 1 | seam `data` injectable existe → extraction rapide, mais hors périmètre « Palier 0 » demandé |
| Menu latéral (`ShellRail`) | 1 | couplé `useRouter`/`useSearchParams` — nécessite des défauts routeur |
| Panneau conversation M.I.A (`ChatPanel`) | 2 | lié au contexte `useChat` — extraction = **risque de régression** sur du chat vivant |
| Fiche de publication (`CalendarEventDetail`) | 2 | cartes imbriquées + états d'absence + bloc chat — **risque de régression** |

Le graphique n'a volontairement PAS été découpé (cf. §2). Le test AppWorkspace « skeleton pendant
le fetch » n'a **pas** été touché.

---

## 4. Le jeu de données figées — RÉEL, pas inventé

`webapp/lib/ds-samples/` (généré par `webapp/scripts/gen_ds_samples.py`) :

- `readings.ts` — **4 `MarketReading` réels** copiés de `data/market_readings.db` (XAU/USD H4, EUR/USD
  M15, XAU/USD D1, + un XAU/USD M15 **à champs absents**). Couvre OB & FVG au-dessus/en dessous du
  prix, jamais touchée / testée / mitigée, et le **cas limite « prix DANS la bande »** (dérivé du réel).
- `candles.ts` — **260 + 220 bougies OHLC réelles** copiées de `data/candles.db`.
- `zones.ts` — zones dérivées des lectures réelles via la fonction pure `collectZones` (celle du
  produit — pas de réimplémentation).
- `scanner.ts` — `ConditionsScanResponse` dont chaque contexte (tendance, phase, volatilité, comptes
  OB/FVG, prix, MTF) est copié des lectures réelles ; couvre correspondance complète, presque,
  non évaluable, et un combo **indisponible** (donnée absente).
- `chat.ts` — conversation M.I.A ancrée sur les faits réels de la lecture XAU/USD H4 (les
  conversations ne sont pas persistées ; texte descriptif, jamais inventé numériquement).

**Normalisation honnête** : les payloads en base sont l'ANCIENNE forme moteur (`bos`/`choch`,
`touch_ats`, `description_source` haiku/fallback). Le générateur les aligne sur le contrat TS actuel
via une liste blanche stricte : `bos→current_bos`, abandon des champs hérités (`_logic_version`,
`touch_ats`), omission de `contacts` (le ledger que ces lignes n'ont jamais porté — jamais
fabriqué), coercition de `description_source→'engine_template'` (métadonnée de provenance ; le TEXTE
descriptif est conservé verbatim), et **rejet de tout événement portant le caractère de corruption
U+FFFD** (artefact d'encodage hérité). **Chaque fait de marché (prix, niveaux, zones, horodatages)
est copié tel quel.**

**Aucun secret, aucune clé, aucune donnée personnelle.** Vocabulaire conforme (garde-fou testé).

---

## 5. La galerie

- Route **dev-only** : `webapp/app/[locale]/galerie/page.tsx` — `notFound()` dès `NODE_ENV=production`.
- Rendu : `webapp/components/gallery/DesignGallery.tsx` — tous les composants de présentation,
  groupés par surface, **chaque état nommé à côté**, sur `bg-background`.
- Contextes fournis par le `[locale]/layout` existant (i18n + thème + tooltip) — le composant ne
  plante plus hors de l'app.
- Chemins : `/galerie` (fr), `/en/galerie`, … — **hors** des préfixes protégés (accessible sans auth,
  sans backend).

**Preuve « pas atteignable en production »** : build `next start` → `/` = **200**, `/galerie` =
**404**, `/en/galerie` = **404** (+ test unitaire du garde-fou).

---

## 6. Tests & vérifications

- **tsc** : vert (seuls les 3 erreurs pré-existantes `dictation-copy-honesty` subsistent — hors périmètre).
- **build** : vert ; `/galerie` = route dynamique 84 kB.
- **vitest (10 tests, 4 fichiers, tous verts)** :
  - `components/gallery/__tests__/no-network.test.tsx` — chaque composant de présentation se rend
    avec un `fetch` qui **jette** → un composant qui appellerait le réseau échouerait (6 tests).
  - `components/gallery/__tests__/absence-states.test.tsx` — la lecture à champs absents ne rend
    **ni tiret, ni « non disponible / N/A », ni `undefined/null`**.
  - `lib/ds-samples/__tests__/forbidden-vocab.test.ts` — **aucun mot interdit** (fr + en) dans les
    données d'exemple.
  - `app/[locale]/galerie/__tests__/gallery-route.test.tsx` — `notFound()` en production, pas en dev.
- **Playwright** (contre `next dev`, car la route est dev-only) : `tests/e2e/ds-1-gallery.spec.ts` —
  la galerie complète se rend **sans backend**, aux **deux viewports 1280×800 et 390×844** ; les 6
  surfaces et le cas « prix dans la bande » sont présents. 2/2 verts.
- **Pages réelles identiques avant/après** : garanti par construction — `git status` ne montre que
  des fichiers **nouveaux** ; `git diff origin/main` sur les fichiers suivis est **vide**. Aucun
  composant ni page existant n'a été modifié.

### Captures
- `docs/audits/ds-1/gallery-desktop-1280x800.png`
- `docs/audits/ds-1/gallery-mobile-390x844.png`

---

## 7. Non négociables — conformité

- ✅ Aucun changement visuel dans l'app (aucun fichier existant modifié).
- ✅ Aucune logique métier / calcul / détection touchés.
- ✅ Aucun appel fournisseur ajouté — les données d'exemple sont des fichiers.
- ✅ Aucun secret / clé / donnée personnelle.
- ✅ Vocabulaire descriptif conforme (testé fr + en).
- ✅ Pas de donnée → pas d'élément : prouvé par la lecture « champs absents » et le combo
  indisponible, visibles dans la galerie et verrouillés par test.
- ✅ Staging explicite, pas de `git add -A`, pas de force push.

## 8. Extension — PAGES COMPLÈTES via API mock (pour Claude Design)

La galerie (§5) montre des composants isolés ; pour que Claude Design recopie les PAGES à
l'identique, il faut les vraies pages **assemblées** (shell, mise en page, chrome). Approche
retenue : faire tourner les VRAIES pages sur `next dev` et **intercepter `/api/*` avec les fixtures
réelles** — **0 ligne des pages modifiée**, aucun backend.

- **`tests/e2e/ds-mock.ts`** — `mockAllApis(page)` sert chaque endpoint produit depuis les fixtures :
  `access/me` (accès complet), `market-reading`, `candles`, `market-status`, `conditions-scan`
  (+`/palette`), `scanner/translate`, `calendar` (+`/month`, +`/event`), `publications/*/measures`.
  Ordre de routes géré (le plus spécifique gagne).
- **Recalage des horodatages** : les fixtures sont datées dans le passé (leur vraie date de capture),
  que l'app lit comme périmées → « marché fermé », graphique/mois vides. Le mock décale **chaque
  horodatage** d'un même delta pour placer la dernière bougie ~maintenant. **Seules les dates
  absolues bougent ; toutes les valeurs (OHLC, niveaux, zones, événements) sont les données réelles.**
- **`tests/e2e/ds-1-pages.spec.ts`** — capture les 7 pages produit aux 2 viewports
  (1280×800 + 390×844) : `/`, `/app`, `/zones`, `/scanner`, `/scanner/decrire`, `/actualites`,
  `/actualites/[eventId]`. **14/14 verts, sans backend.** Sorties : `docs/audits/ds-1/pages/`.

Rendu vérifié page par page (capture à l'appui) :
- **/app** — shell complet (rail, en-tête prix réel « En direct », colonne M.I.A), toolbar de calques,
  lecture narrée réelle, cartes Régime + Structure. ⚠️ **Le tracé des bougies ne peint pas dans la
  capture headless** (limitation de `lightweight-charts` sous capture automatisée : données/couleurs/
  chart créés, axes tracés, mais la série ne rend pas ; le graphique peint normalement dans l'app
  live). Le CADRE du graphique (toolbar, badges, axes) est bien capturé.
- **/zones** — complet : cartes de zones réelles dont le **cas « prix dans la bande »**, jauges de
  proximité, panneau M.I.A sur la zone sélectionnée.
- **/scanner** — constructeur de conditions V3 (4 familles, combinaison, sauvegarde).
- **/scanner/decrire** — scanner conversationnel (saisie + exemples).
- **/actualites** — grille mensuelle **peuplée d'événements réels** (Core CPI, FOMC, German CPI…),
  filtres organisme/marché/périodicité.
- **/actualites/[eventId]** — fiche complète (en-tête événement réel, bloc M.I.A, « aller à la
  source » avec absence propre, avertissement) ; mesures absentes → aucune section (règle d'absence).
- **/** — page d'accueil (déjà statique).

Périmètre des données calendrier : le store calendrier est **vide** dans cet environnement (flux
récupéré en live) ; source réelle la plus proche = `news_cache.db` (événements économiques réels).
Filtrés aux sources officielles US/EU (USD→bls, EUR→eurostat) pour passer les filtres organisme.

## 9. Suite proposée (après confirmation)
Palier 1 (`CalendarMonthView`, `ShellRail`) puis Palier 2 (`ChatPanel`, `CalendarEventDetail`) —
extraction vue/conteneur, à traiter avec le soin dû au risque de régression signalé au §3. Le seul
point ouvert côté rendu = le tracé du graphique en capture headless (§8) — sans impact sur l'app
live ; à approfondir si des captures de graphique peuplé sont nécessaires pour Claude Design.

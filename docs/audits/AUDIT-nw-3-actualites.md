# AUDIT — NW-3 Refonte des Actualités (calendrier, page de publication, module)

Branche : `feat/nw-3-actualites-refonte` (depuis `main` à jour, `0b4e49f`).
Surface cible : trois onglets — calendrier mensuel, page d'une publication, module du
tableau de bord. Maquette de référence sur disque : `docs/design/reference-actualites.html`
(voir §5 : périmée, écarts listés).

---

## 1. Faisabilité des quatre mesures — avec le coût en données

La page de publication repose sur quatre questions. Constat central du diagnostic :
**les données disponibles séparent la courbe des valeurs (HICP zone euro, Eurostat) des
quatre questions (US CPI mesuré sur l'or).** Aucun indicateur seul ne fournit les deux
aujourd'hui. Décision retenue (validée fondateur) : **livrer les deux indicateurs**, chacun
montrant honnêtement ce qu'il peut (règle de périmètre §2).

| # | Mesure | Calculable | Ce qui manquait | Livrée ? |
|---|--------|-----------|-----------------|----------|
| 1 | **Le calme avant** — parcours dans l'heure précédant la parution vs même heure sur 60 j sans publication | **OUI** | Le profil horaire de référence existait déjà (`volatility_forecaster._compute_diurnal_profile`) ; il fallait l'exclusion des fenêtres de publication + le mapping parution→bougies | **OUI** (US CPI / or) |
| 2 | **État de la structure à l'instant T** — position dans le range, poche de liquidité intacte à <0,5 %, prix déjà dans une zone | **OUI, par rejeu** | `market_readings.db` ne sert que la dernière lecture ; le moteur accepte un cutoff `idx` sans look-ahead (`build_enriched_frame` + collecteurs) — rejeu battle-testé par le harnais MT-D1 | **OUI** (US CPI / or) |
| 3 | **Cycle de vie des zones** (formation→mitigation) | Partiel | Formation ET mitigation horodatées existent (`created_at`/`mitigated_at`) ; le blocage est la **rétention M15 (~30 j)** + horodatage terminal + backfill paginé (~3-5 j) | **DIFFÉRÉE** (décision fondateur) |
| 4 | **Le retour au calme** — temps avant retour du parcours par minute à son niveau ordinaire | **OUI** | Même base que #1 + un seuil de retour | **OUI** (US CPI / or) |

**Le coût en données — chiffré.** Contre-intuitivement, **le prix ne coûte presque rien** :

- Historique déjà **sur disque, hors ligne** : `data/XAU_15MIN_2019_2026.csv` (172 874 bougies
  M15, 2019→2026) et `data/EURUSD_15MIN_2019_2025.csv`. Un calcul historique = **0 requête
  fournisseur**, compatible plan gratuit par construction.
- Les horodatages de parution US CPI sont bundlés : `data/economic_calendar_2019_2025.csv`
  contient **84 parutions CPI** (Currency=USD), toutes horodatées. Le module en a mesuré **84
  sur l'or M15** à l'exécution réelle.
- **Si on passait en live via Twelve Data** (plan gratuit : 8 req/min, 800 req/j, 5000
  bougies/req) : 1 an de M15 ≈ 24-25 k bougies = **~5 requêtes** — trivial côté quota, mais le
  backfill paginé par date (>52 j/requête) n'existe pas encore.
- **Bascule décisive :** une fois 1 an de M15 en stock, **couvrir 10 indicateurs mensuels ne
  coûte pas plus cher que 1** — ils relisent la même série de prix. La ressource rare n'est
  pas le prix, c'est **(a) l'historique des horodatages de parution** et **(b) la série de
  valeurs publiées** (courbe des 12 chiffres).

**Profondeur d'historique retenue.** Mesures US CPI : les parutions bundlées (2019-2025) sur
gold M15 (2019-2026) — dédoublonnage des instants (CPI + Core CPI simultanés = une parution),
fenêtre de rejeu ~1200 bougies par parution, repère horaire sur 60 journées sans publication.
Courbe HICP EA : les 12 dernières observations Eurostat (`prc_hicp_manr`, `lastTimePeriod=12`).

---

## 2. Ce qui existe / a été étendu (état `main`)

- **4 états de valeur** (`published/pending/unfetched/unavailable`) : intacts (NW-1c).
- **Bug « Publication introuvable » : corrigé** sur `main` (REC-1) — détail par id
  (`GET /api/calendar/event/{id}`, `store.get_event_by_id`). Conservé.
- **Flux de valeurs (NW-1c)** : élargi d'un point (actual+previous) à une **série** —
  `ValueFetcher.fetch_series` + `EurostatValueFetcher.fetch_series` (le parseur JSON-stat
  renvoyait déjà le vecteur complet), `MultiValueFetcher.series_for`, `CalendarSeriesPoint`
  sur `CalendarEvent`, attaché **uniquement** par le chemin détail (un appel par fiche).

---

## 3. Mesures livrées / différées

- **Livrées** : #1 calme avant, #2 état de la structure à T, #4 retour au calme — pour
  **US CPI mesuré sur l'or (XAUUSD, M15)**. Module pur `publication_measures.py`
  (+ `publication_measures_schema.py` contrat) ; endpoint `GET /api/publications/{key}/measures`
  (caché, TTL 1 h). Durées en minutes entières ; répartition en tranches + extrêmes datés ;
  aucune statistique centrale à l'écran ; chaque mesure porte sa ligne de source + dénominateur.
- **Courbe des 12 chiffres + révisions** : pour **HICP zone euro (Eurostat)**.
- **Différée** : #3 cycle de vie des zones (rétention M15 profonde + horodatage terminal +
  backfill paginé). **Non rendue** (règle de périmètre : aucune mesure non fiable affichée,
  aucun bloc vide).

---

## 4. Surfaces livrées

- **Calendrier mensuel** : grille 7 colonnes, navigation mois/mois + retour au mois courant,
  ≤2 publications/jour puis compteur, jours vides visibles, panneau du jour (heure organisme +
  locale, organisme, unité, marchés), encadré « ce mois-ci », filtres factuels avec message de
  filtre vide explicite. Endpoint `GET /api/calendar/month?month=YYYY-MM`.
- **Page de publication** : en-tête → courbe (12 chiffres, point à venir vide, révisions) →
  quatre questions (si mesurables) → M.I.A → aller à la source (organisme uniquement) →
  fiche pédagogique + « ce que cette fiche ne dit pas ».
- **Module tableau de bord** : liste défilante de toutes les publications à venir
  (`CalendarPreview`), état vide honnête, lien « tout voir ».

---

## 5. Écarts avec la maquette

- **La maquette sur disque est périmée** : `reference-actualites.html` n'a que 2 vues (liste
  plate + détail), pas les trois onglets ; et son contenu **viole la LIGNE NW-3** (« affecte »,
  « Impact élevé/Moyen/Faible », « Consensus des analystes », « Amplitude médiane »). Décision
  fondateur : **dessiner d'après la spec de mission**, pas d'après le fichier. Les surfaces
  livrées suivent donc la spec ; la maquette a servi de repère de STYLE (tokens sombres, cartes)
  uniquement.
- Écarts assumés vs la maquette (tous conformes à la LIGNE / à la spec) :
  · liste plate → **grille mensuelle** (3e surface ajoutée : le module tableau de bord) ;
  · « Consensus des analystes », badges « Impact », « affecte », « Amplitude médiane » →
    **supprimés** ;
  · les barres d'amplitude par publication (mockup) → remplacées par les **quatre questions**
    (répartition en tranches + extrêmes datés + comptes), car la règle interdit une valeur
    centrale unique par publication ;
  · tuile « État du marché maintenant » → repliée dans la ligne « maintenant » de la question
    structure ;
  · **courbe des 12 chiffres** ajoutée (absente du mockup, requise §2B).

---

## 6. Discipline & tests

- Zéro modification des règles de détection : les mesures LISENT le moteur (rejeu, cutoff `idx`).
- Tests de garde LIGNE (`calendar-copy-honesty.test.ts`, étendu) : pas de qualification
  directionnelle, pas de verbe de causalité (fr+en), pas de « médiane/moyenne/écart-type/
  bougie/candle/mean/median » visible, liste blanche de domaines externes (organismes émetteurs),
  pas de code de mission interne dans une chaîne, parité de clés fr/en.
- Tests composants : point à venir sans valeur, mesure non calculable non rendue, ligne de
  source + dénominateur par mesure, initial+révisé coexistent, filtre vide → message.
- Backend : `test_publication_measures.py` (7 verts, synthétique) ; `test_calendar_nw3.py`
  (série de valeurs + fenêtre mensuelle).
- i18n : **fr + en complets** pour toutes les nouvelles chaînes. Les 7 autres locales ne
  reçoivent pas encore les blocs `pub.*`/`month.*` (dégradation gracieuse next-intl, défaut =
  `fr`) — cohérent avec le périmètre mission (fr+en) ; à compléter dans une passe locale.

### Statuts de la porte de vérification

- `npx tsc --noEmit` : **0 erreur**.
- `npx next build` : **succès** (`/actualites` et `/actualites/[eventId]` compilées).
- `npx vitest run` (suite complète) : **716 verts / 717**. L'unique échec —
  `market-reading-components.test.tsx` (« Marché en range » vs « Phase de range » rendu) —
  est **préexistant sur `origin/main`** et hors périmètre NW-3 (aucun fichier market-reading
  dans le diff de la branche ; dérive de libellé RG-1).
- Suite calendrier ciblée : **51 verts** (grille 12, page 20, module 6, garde-copie 13).
- Backend : `test_publication_measures.py` **7 verts** + `test_calendar_nw3.py` **7 verts**.
- Playwright (1280×800 + 390×844, routes API mockées, `tests/e2e/calendar.spec.ts` réécrit
  pour les trois surfaces) : **18 passés, 2 ignorés, 0 échec**. Les 2 ignorés = les tests du
  module `/app` sur desktop, bloqués par un **bug préexistant hors NW-3** : clé i18n manquante
  `reading.labels.trend_indeterminate` (lacune TR-1) qui fait planter le sous-arbre
  `DesktopReading` (hôte du module) quand la tendance est indéterminée. Le module lui-même est
  prouvé (vitest 6/6 + projet mobile Playwright vert).

### Correctif incident (hors périmètre calendrier, mais bloquant pour la surface D)

`reading.labels.trend_indeterminate` était **absent** de `fr.json`/`en.json` (la tendance
`indeterminate` de TR-1 n'avait pas sa clé dans le bloc `labels` lu par le formateur), ce qui
plantait `/app` desktop dès qu'une tendance indéterminée était rendue — et masquait le module
tableau de bord. Clé ajoutée en **fr + en** (correctif d'1 ligne/locale, i18n seul, zéro
logique). Les 7 autres locales portent la même lacune préexistante → à compléter en suivi.

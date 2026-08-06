# AUDIT NW-6 — Page de publication : diagnostic (avant GO)

Branche : `feat/nw-6-page-publication` (depuis `origin/main` @ 9c080d9, PR #118 CAL-1).
Statut : **DIAGNOSTIC — aucun code écrit.** En attente du GO.

> ⚠️ Correction de cadrage. Le worktree initial avait été créé depuis un `main`
> LOCAL périmé (36 commits de retard, sans NW-3/NW-4/NW-5/CAL-1). Recréé depuis
> `origin/main`. Sur le vrai main à jour, **une grande partie de ce que la mission
> décrit comme « manquant » est déjà bâtie** (courbe, quatre questions, bloc
> d'avertissement unique, fiche par publication, contraste du compte mensuel). Le
> constat de la mission décrit un état antérieur. Le travail réel restant est plus
> étroit et détaillé ci-dessous.

---

## 1. DIAGNOSTIC — état des données

### 1.A Ce qui existe déjà dans le code (current main)

| Élément mission | État réel sur main | Emplacement |
|---|---|---|
| Courbe des valeurs publiées | **BÂTIE** (`CurveCard`, SVG, point à venir vide) | `CalendarEventDetail.tsx:191-370` |
| Quatre questions | **BÂTIE — 3 cartes** (#3 zones différée) | `CalendarEventDetail.tsx:372-611` |
| Bloc d'avertissement unique | **PAS unique — 3 empilés** (voir Défaut C) | `CalendarEventDetail.tsx:848-872` |
| Fiche pédagogique par publication | **BÂTIE mais 2 fiches réelles seulement** (us_cpi, ea_hicp_flash) ; les autres = texte générique | `messages/fr.json` pedagogy |
| Contraste du compte mensuel | **BÂTI** (compte + par marché + jours sans + note) | `CalendarMonthView.tsx:333-349` |
| Endpoint mesures | **EXISTE** `/api/publications/{key}/measures` | `src/api/routes/calendar.py:111-138` |
| Récupérateurs BEA / Census (NW-4) | **LIVRÉS** (fetch + fetch_series) | `values/bea_values.py`, `values/census_values.py` |

### 1.B La courbe : quelles publications l'affichent AUJOURD'HUI

La courbe s'affiche si `ev.value_series` est peuplé. Peuplement =
`MultiValueFetcher.series_for(source, series_code)` (`base_value.py:77-92`), lui-même
gaté par `CALENDAR_VALUES_LIVE=1` (`build_value_fetcher`, `base_value.py:98-134`).
`series_for` appelle `fetcher.fetch_series()`. **Le défaut de base renvoie `[]`**
(`base_value.py:50-55`) : une source qui n'override pas `fetch_series` n'a **jamais**
de courbe.

`fetch_series` **implémenté** : `bea_values`, `census_values`, `eurostat_values`.
`fetch_series` **NON implémenté (→ [])** : `bls_values`, `ecb_values`.

| Publication | Source | Clé requise | `fetch_series` ? | **Courbe possible aujourd'hui ?** |
|---|---|---|---|---|
| us_employment_situation (NFP) | bls | BLS_API_KEY | ❌ manquant | **NON** — bloqueur : implémenter `BLSValueFetcher.fetch_series` |
| us_cpi | bls | BLS_API_KEY | ❌ | **NON** — même bloqueur |
| us_ppi | bls | BLS_API_KEY | ❌ | **NON** — même bloqueur |
| **us_jolts** *(exemple mission)* | bls | BLS_API_KEY | ❌ | **NON** — même bloqueur |
| us_cpi_core | bls | BLS_API_KEY | ❌ | **NON** — même bloqueur |
| us_gdp | bea | BEA_API_KEY | ✅ | **OUI si BEA_API_KEY posée** en prod |
| us_pce | bea | BEA_API_KEY | ✅ | **OUI si BEA_API_KEY posée** |
| us_retail_sales | census | CENSUS_API_KEY | ✅ | **OUI si CENSUS_API_KEY activée** (mémoire : clé rejetée « Invalid Key », à activer) |
| us_housing_starts | census | CENSUS_API_KEY | ✅ | **OUI si clé activée** |
| us_durable_goods | census | CENSUS_API_KEY | ✅ | **OUI si clé activée** |
| ea_hicp_flash | eurostat | aucune | ✅ | **OUI** (sans clé — dès `CALENDAR_VALUES_LIVE=1`) |
| ea_gdp_flash | eurostat | aucune | ✅ | **OUI** |
| ea_unemployment | eurostat | aucune | ✅ | **OUI** |
| ea_ecb_rate | ecb | aucune | ❌ | **NON** — bloqueur : implémenter `ECBValueFetcher.fetch_series` |
| us_fomc_rate / minutes / dotplot | federal_reserve | — | — | **NON** — pas de `series_code` (rien à tracer) |

> Note prod : `CALENDAR_VALUES_LIVE`, `BLS_API_KEY`, `BEA_API_KEY`, `CENSUS_API_KEY`
> sont déclarés `sync=false` dans `render.yaml` — **valeurs à confirmer au dashboard
> Render** (non vérifiable depuis le repo). ECB/Eurostat ne dépendent d'aucune clé.

**Profondeur des valeurs BLS `fetch()`** : requête `"latest": True` → 1–2 points
seulement. Pour une courbe de 12 valeurs il faut de toute façon écrire `fetch_series`
avec une plage d'années (startyear/endyear), sans `latest`.

### 1.C Les quatre questions : quelles publications les affichent

Les mesures viennent de `/api/publications/{key}/measures`, gaté par
`_MEASURABLE_MARKETS` (`calendar.py:42`) = **`{"us_cpi": "XAUUSD"}` uniquement**. Toute
autre clé renvoie des mesures vides → aucune question rendue.

- Mesures calculées : **#1 calme avant, #2 structure à l'instant, #4 retour au calme**
  (`publication_measures.py`). **#3 cycle de vie des zones = DIFFÉRÉE** (schéma présent,
  non calculée — `publication_measures.py:14`).
- Prérequis d'une mesure : historique de prix XAU/USD aligné sur les parutions passées
  (rejeu moteur `build_enriched_frame`, sans look-ahead).

| Publication | Questions aujourd'hui ? | Bloqueur |
|---|---|---|
| us_cpi | **OUI (3 cartes)** | — (déjà livré NW-5) |
| Toutes les autres | **NON** | Câbler `_MEASURABLE_MARKETS` + réunir l'historique de prix aligné sur chaque parution |

### 1.D Réponses directes aux questions de la mission

- **Clé BLS en prod ?** Déclarée `sync=false` — à confirmer au dashboard ; valeurs
  historiques JOLTS/IPC/NFP/IPP : le fetcher n'en tire **que 1–2** (`fetch`), et **0 en
  série** (pas de `fetch_series`). Donc courbe = 0 valeur aujourd'hui pour tout BLS.
- **JOLTS spécifiquement** : 0 valeur en série (bloqueur BLS `fetch_series`).
- **IPC / NFP / IPP** : idem — 0 valeur en série.
- **Récupérateurs BEA et Census (NW-4)** : **livrés** (`fetch` + `fetch_series`),
  key-gated.
- **Les quatre mesures** : #1/#2/#4 existent (câblées pour us_cpi seul) ; **#3 différée**.

### 1.E Ce qu'il faut pour livrer chaque bloc manquant

| Bloc | Pour qui | Ce qu'il faut | Rendu aujourd'hui après action |
|---|---|---|---|
| Courbe BLS | JOLTS, IPC, NFP, IPP, IPC-core | Écrire `BLSValueFetcher.fetch_series` (plage d'années) | Oui si BLS_API_KEY posée |
| Courbe ECB | ea_ecb_rate | Écrire `ECBValueFetcher.fetch_series` | Oui (sans clé) |
| Courbe Eurostat/BEA/Census | HICP, GDP, chômage / GDP, PCE / retail, housing, durables | **Rien à coder** — vérifier flag+clés en prod | Oui si flag+clés OK |
| Questions | au-delà de us_cpi | Câbler `_MEASURABLE_MARKETS` + historique prix aligné | Partiel, lourd (données) |

---

## 2. LES QUATRE DÉFAUTS (corrigeables sans données) — confirmés

### Défaut A — libellé du compte à rebours (RÉEL)
`countdownLabel` = **libellé fixe** `"Publication dans"` (`fr.json:2477`), rendu tel quel
(`CalendarEventDetail.tsx:825`). La *valeur* s'adapte pourtant au passé via
`fmtCountdown` → `agoDays` « il y a 1 j » (`:176-179`, `fr.json:2429`). D'où
« Publication dans » + « il y a 1 j ». **Fix** : libellé conditionnel passé/futur
(« Publiée » / « Publication dans »), bascule au vrai instant `scheduled_at` vs `now`.

### Défaut B — fiche pédagogique générique (RÉEL)
`pedKey = (us_cpi | ea_hicp_flash) ? eventKey : 'default'` (`:771-772`). Le `default`
est exactement le texte cité par la mission (`fr.json` pedagogy.default.body). **Fix** :
rédiger une fiche réelle par publication du périmètre ; **ne pas rendre le bloc** quand
aucune fiche rédigée (pas de générique). Fiches à écrire : voir §4.

### Défaut C — trois avertissements empilés (RÉEL)
Sur la page : (1) pedagogy nono « Ce que cette fiche ne dit pas » (`:854-857`),
(2) page nono « Ce que cette page ne dit pas » (`:861-872`), (3) légende M.I.A
`pub.mia.capability` (`:648`). **Fix** : **un seul** bloc d'avertissement après le
dernier contenu factuel ; la mention M.I.A reste (elle porte sur M.I.A, pas la page).

### Défaut D — compte du calendrier (DÉJÀ ~90 % FAIT)
La boîte « Ce mois-ci » rend déjà **compte + par marché + jours sans + note**
(`CalendarMonthView.tsx:333-349`), le compte reflète les filtres (`filtered`). **Gap
résiduel** : aucun indicateur explicite « filtres actifs » à côté du compte. **Fix
mineur** : mention « (filtres actifs) » quand un filtre est appliqué.

---

## 3. BLOCS MANQUANTS — à compléter selon §1

- **Courbe** : bâtie ; débloquer BLS (fetch_series) + ECB (fetch_series) ; vérifier
  flag/clés prod pour Eurostat/BEA/Census.
- **Quatre questions** : bâties (3 cartes, us_cpi) ; extension aux autres = chantier
  données (hors périmètre d'un débloquage rapide).

---

## 4. FICHES PÉDAGOGIQUES — état

**Rédigées (réelles)** : `us_cpi`, `ea_hicp_flash`.
**À rédiger** (périmètre catalogue) : us_employment_situation (NFP), us_ppi, us_jolts,
us_cpi_core, us_gdp, us_pce, us_retail_sales, us_housing_starts, us_durable_goods,
us_fomc_rate, ea_gdp_flash, ea_unemployment, ea_ecb_rate.
*(Sans fiche rédigée → bloc non rendu, jamais de générique.)*

---

## 5. CE QUI A ÉTÉ LIVRÉ (après GO)

Périmètre validé : **Défauts A/B/C/D + `fetch_series` BLS & ECB + 13 fiches US+EA**
(en pratique **17 fiches** — tout le catalogue). Questions au-delà de us_cpi : hors
périmètre (chantier données).

### 5.1 Défauts corrigés
| Défaut | Correctif | Fichiers |
|---|---|---|
| **A** libellé | `detail.countdownLabel` (futur) / `detail.countdownLabelPast` « Publiée » (passé), bascule `scheduled_at` vs `now` à la minute (`countdown().past`) | `CalendarEventDetail.tsx`, `messages/*` |
| **B** fiche | Rendu conditionnel `PEDAGOGY_FICHES` ; **plus de `default` générique** (supprimé fr+en) ; MIA a sa propre clé `miaKey` | `CalendarEventDetail.tsx`, `messages/*` |
| **C** avertissements | **Un seul** `.cal-nono` en bas ; le nono de la fiche pédagogique supprimé ; mention M.I.A conservée | `CalendarEventDetail.tsx` |
| **D** compte | Indicateur « Filtres actifs » quand un groupe est restreint (compte déjà contrasté par CAL-1) | `CalendarMonthView.tsx`, `calendar-month.css`, `messages/*` |

### 5.2 Blocs débloqués (courbe)
- **BLS `fetch_series`** (`bls_values.py`) — plage d'années, périodes `Mxx`→`YYYY-MM`,
  `M13` (moyenne annuelle) écartée, ordre chronologique, dernières `limit`.
  **Débloque la courbe** de JOLTS / IPC / NFP / IPP / IPC-core **si `BLS_API_KEY`
  posée** en prod + `CALENDAR_VALUES_LIVE=1`.
- **ECB `fetch_series`** (`ecb_values.py`) — série à paliers : runs de valeurs
  identiques repliés sur la date de décision (période `YYYY-MM`). **Débloque la
  courbe** de l'ECB rate (sans clé).
- **Eurostat/BEA/Census** : `fetch_series` déjà présents → courbe dès que flag+clés OK.

### 5.3 Fiches pédagogiques rédigées (fr + en, natives)
17 fiches réelles : us_employment_situation, us_cpi, us_cpi_core, us_ppi, us_jolts,
us_gdp, us_pce, us_retail_sales, us_housing_starts, us_durable_goods, us_fomc_rate,
us_fomc_minutes, us_fomc_dotplot, ea_hicp_flash, ea_gdp_flash, ea_unemployment,
ea_ecb_rate. **Aucune fiche restante à écrire** pour le catalogue actuel. Une
publication hors catalogue (clé inconnue) ne rend **aucun** bloc pédagogique.
Les 7 autres locales portent les mêmes CLÉS (repli EN documenté, cf. DETTE-1) —
us_cpi/ea_hicp_flash conservent leur traduction native.

### 5.4 Chantiers bloquants restants (non code)
- **Courbe BLS invisible tant que `BLS_API_KEY` n'est pas posée** au dashboard Render
  (`sync=false`). Idem BEA/Census. À confirmer/activer par le founder.
- **Questions au-delà de us_cpi** : nécessite câbler `_MEASURABLE_MARKETS` + réunir
  l'historique de prix aligné par parution. Hors périmètre NW-6.

### 5.5 Tests & vérifications
- **Backend** : `tests/test_calendar_value_fetchers.py` **29/29** (BLS+ECB `fetch_series`
  ajoutés). ⚠️ `test_calendar_values.py::test_enricher_flags_revision_across_cycles`
  échoue **AUSSI sur `origin/main` intact** (test daté pré-existant, `NOW=2026-07-29`) —
  **pas une régression NW-6**.
- **Front** : suite vitest complète **848/848** (dont gardes NW-6 : libellé passé/futur,
  fiche réelle vs bloc absent, un seul avertissement, vocabulaire interdit fr+en, point
  à venir sans valeur, indicateur de filtres).
- **tsc** vert · **`next build`** vert.
- **Playwright** 1280×800 & 390×844 — `nw6-publication.spec.ts` : passée / à venir /
  avec valeurs / sans valeurs / sans fiche.

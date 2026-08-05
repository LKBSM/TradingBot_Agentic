# AUDIT CAL-1 — Calendrier bloqué au chargement & comptes affirmés à tort

Branche : `fix/cal-1-chargement-calendrier` (worktree dédié `wt-cal-1`, depuis `main` à jour `004f067`).
Date : 2026-08-05. Périmètre : `/actualites` (vue mensuelle), aperçu `/app`, filtres d'organisme.
**Aucune règle de détection touchée.**

---

## Résumé exécutif

| Problème | Verdict | État |
|---|---|---|
| **1 — Comptes affirmés avant chargement** | Confirmé. Le panneau latéral de la vue mensuelle rend `totalThisMonth`/`perMarket`/`emptyDays` **sans condition** : `data=null` pendant le chargement ⇒ « 0 publication / Or : 0 / EUR/USD : 0 / 31 jours sans publication ». Idem l'aperçu `/app`. | **CORRIGÉ** |
| **2 — La grille ne charge jamais** | Cause identifiée : rafraîchissement backend **synchrone et non borné** au premier appel de chaque fenêtre TTL (jusqu'à 6 fetch `.ics` + ~10 fetch de valeurs, 10–12 s chacun). Le client a **déjà** un timeout de 8 s (depuis NW-1) : il ne boucle pas à l'infini, il **bascule en erreur** au-delà de 8 s. Août 2026 n'est **pas** vide (10 parutions curées). | **DIAGNOSTIC — correction racine EN ATTENTE DE TON GO** |
| **3 — ForexFactory dans les filtres** | Confirmé côté **front uniquement** : la liste des sources codait `forexfactory` en dur. Backend : jamais actif en prod (défaut `official`, `CALENDAR_SOURCE` non posé dans `render.yaml`). Aucune parution réelle n'en dépend. | **CORRIGÉ + garde** |

---

## PROBLÈME 1 — Un compte affirmé alors que la donnée n'est pas chargée

### Cause racine
`webapp/components/calendar/CalendarMonthView.tsx` : le bloc `.calm-thismonth` (comptes) et le
`DayPanel` étaient rendus **inconditionnellement**. Pendant le chargement, `hook.data === null`
⇒ `filtered = []` ⇒ `totalThisMonth = 0`, `emptyDays = 31`, `perMarket = [Or:0, EUR/USD:0]`.
La grille, elle, gardait bien son état de chargement (`GridArea` teste `isLoading`) — d'où le
symptôme exact : **grille en « Chargement… » pendant que le panneau affirme « 0 / 31 jours sans »**.

### Correction
Trois états désormais **distinguables à l'écran** (panneau latéral, grille, panneau du jour) :

- **chargement en cours** → statut d'attente **visuellement distinct** (italique, atténué, pulsation
  douce ; classe `.calm-tm-status[role=status]`) ; **aucun compte, aucun zéro** rendu ;
- **chargé avec publications** → les comptes sont affirmés (résultat réel) ;
- **chargé réellement vide** → message explicite « Aucune publication programmée sur ce mois… »
  (jamais un « 0 / 31 jours sans »), en distinguant *vraiment vide* / *filtré à vide* / *aucun filtre* ;
- **erreur** → statut d'erreur distinct, toujours **sans compte**.

« Aucune publication ce jour » (`DayPanel`) ne s'affiche **plus** qu'après chargement effectif.

### Autres surfaces où un zéro/vide pouvait apparaître avant chargement (point d)
| Surface | Fichier | Avant | Après |
|---|---|---|---|
| Panneau « Ce mois-ci » (vue mensuelle) | `CalendarMonthView.tsx` | « 0 / 31 jours sans » pendant le chargement | statut d'attente, pas de compte |
| Panneau du jour (vue mensuelle) | `CalendarMonthView.tsx` (`DayPanel`) | « Aucune publication ce jour » pendant le chargement | statut d'attente |
| Aperçu du tableau de bord `/app` | `CalendarPreview.tsx` | « Aucune publication à venir » pendant le chargement (`data=null` ⇒ liste vide) | statut d'attente/erreur distinct |
| Vue **liste** `/actualites` | `CalendarWorkspace.tsx` | **déjà correct** : le corps est gardé par `isLoading` avant tout compte | inchangé (seulement retrait de ForexFactory) |

Hors périmètre calendrier : les compteurs de combinaisons du **scanner** et les sous-lignes de
**régime** ont leur propre chargement (déjà gardés côté `/app` par PERF-1) — non modifiés ici.

---

## PROBLÈME 2 — La grille ne charge jamais (DIAGNOSTIC, correction racine en attente de GO)

### a) Requête émise
`GET /api/calendar/month?month=YYYY-MM` → FastAPI `get_calendar_month` (sync `def`, threadpoolé)
→ `CalendarService.get_calendar_range` → `_maybe_refresh`.

### b) Timeout côté client — OUI, il existe déjà
`webapp/lib/calendar/api.ts` : `DEFAULT_TIMEOUT_MS = 8_000` avec `AbortController`, en place
**depuis NW-1** (`41bab18`). Conséquence : le client **ne peut pas** rester en chargement
indéfiniment — au-delà de 8 s il `abort()` → `.catch` → **état d'erreur** (`.finally` remet
`isLoading=false`). Un « Chargement… » littéralement perpétuel n'est donc pas produit par le code
courant ; ce qui est observé est la **fenêtre de 8 s** puis une bascule en erreur, aggravée par le
faux « 0 » du Problème 1 qui donne l'impression d'un « chargement figé qui a déjà décidé qu'il n'y
a rien ».

### c) Août 2026 est-il réellement couvert ? — OUI
`config/calendar_schedule.json` contient **10 parutions curées en août 2026** (JOLTS 04, NFP 07,
IPC + IPC core 12, IPP 13, Ventes au détail 14, Mises en chantier 18, PIB + PCE + Biens durables 26).
Reproduction locale du service (DB temporaire) :

```
[curated]     events=10  elapsed=0.40s   (CALENDAR_ICS_LIVE non posé)
[ICS_LIVE=1]  events=9   elapsed=1.35s   (réseau dispo dans le bac à sable)
```

→ **Le mois n'est pas vide.** Ce n'est donc pas un problème de couverture, mais de **latence +
honnêteté d'affichage**.

### e) Cause racine de la latence
Le rafraîchissement est **paresseux, synchrone et non borné**, dans le thread de la requête :
- au **premier** appel de chaque fenêtre TTL de 120 s, `_maybe_refresh` exécute `provider.fetch()` ;
- avec `CALENDAR_ICS_LIVE=1` (posé dans `render.yaml`), l'agrégateur officiel appelle **jusqu'à 6
  flux `.ics`** séquentiels, `fetch_ics` timeout **10 s** chacun (`ics_feed.py`) ;
- avec `CALENDAR_VALUES_LIVE=1` (posé aussi), `_enrich_values` appelle **une API de valeur par
  événement** dépourvu de valeur (jusqu'à ~10), timeout **10–12 s** chacun (`*_values.py`).

Pire cas cumulé : **plusieurs dizaines de secondes** au premier appel froid — bien au-delà du
budget client de 8 s. Aucun **pré-chauffage** planifié n'existe (aucun scheduler ne rafraîchit le
calendrier ; le cache SQLite persiste, donc la fenêtre froide ne survient qu'après 120 s d'inactivité
ou au boot). Les appels suivants (cache chaud) reviennent en < 1 s.

### f) Journal backend attendu au moment de la requête
`ICS fetch failed for … — falling back` (par flux injoignable), puis
`official sub-provider … failed` (le cas échéant) ; jamais une exception propagée (tout est
`try/except` défensif) — donc **pas de 500**, juste une **réponse lente** au premier appel froid.

### d) Lien avec le blocage de `/app` (PERF-1/REC-1) ?
**Distinct mais de même famille.** REC-1 corrigeait un `async def` bloquant l'event-loop ; ici la
route calendrier est déjà `def` (threadpoolée). Le point commun est le **fetch réseau synchrone
non borné au premier appel**, exactement le motif que PERF-1 a traité pour `market-reading`
(read-through borné + honnêteté). La vue **liste** partage le même `_maybe_refresh` : elle
subirait la même lenteur au premier appel froid (elle n'a pas de compte affiché avant chargement,
donc elle ne *trompe* pas, mais elle peut aussi basculer en erreur au bout de 8 s).

### Correction racine proposée (EN ATTENTE DE TON GO)
1. **Backend — servir le cache immédiatement, rafraîchir en tâche de fond** (ou borner/paralléliser
   le fan-out réseau et le pré-chauffer via le scheduler), pour qu'un appel froid ne dépasse jamais
   le budget client. C'est la cause racine, pas le symptôme.
2. **Client — honnêteté & résilience** (aligné sur PERF-1) :
   - distinguer **serveur injoignable** vs **délai dépassé**, avec un bouton **Relancer** ;
   - **ne jamais effacer** les données déjà obtenues sur échec (rétention type SWR : aujourd'hui
     `useCalendarMonth` conserve `data` en état, mais `GridArea` masque tout dès qu'`error` est
     posé — à transformer en bannière non bloquante au-dessus des données conservées).

---

## PROBLÈME 3 — « ForexFactory » dans les filtres d'organisme

### a) Origine
**Front uniquement** : `CalendarMonthView.tsx`, `CalendarPreview.tsx` et `CalendarWorkspace.tsx`
codaient la liste des sources en dur avec `forexfactory` (commentaire « + the dev prototype so
local runs stay usable »). Le chip était donc offert, coché par défaut.

### b/c) Parutions réelles ?
**Aucune.** Backend : `build_calendar_provider` **défaut = `official`** ; `render.yaml` **ne pose
pas** `CALENDAR_SOURCE`. L'adaptateur ForexFactory (prototype dev, sans droits d'affichage
commercial) n'est même pas chargé en prod. Aucune publication n'en dépend ⇒ **rien n'est perdu**.

### d/e) Correction + garde
- **Front** : liste blanche explicite `webapp/lib/calendar/officialSources.ts`
  (`OFFICIAL_SOURCES` = bls, bea, census, federal_reserve, eurostat, ecb) utilisée par les trois
  surfaces. ForexFactory retiré. La clé i18n `organism.forexfactory` est laissée (inutilisée,
  inoffensive) pour ne pas perturber la parité stricte des 9 locales.
- **Backend** : liste blanche `OFFICIAL_SOURCES` + **garde de production** — un
  `CALENDAR_SOURCE=forexfactory` posé par erreur en prod est désormais **refusé** (repli sur
  `official`) sauf opt-in dev explicite `CALENDAR_ALLOW_DEV_SOURCE=1`. « Sources officielles
  uniquement » devient une garantie **explicite dans le code**, pas implicite.
- **Tests de garde** : `officialSources.test.ts` (le whitelist exclut ForexFactory), un test de la
  vue mensuelle (aucun chip ForexFactory), et 3 tests backend (défaut officiel, refus en prod,
  opt-in dev requis).

> ⚠️ Effet de bord dev : lancer le prototype en local demande maintenant
> `CALENDAR_SOURCE=forexfactory` **ET** `CALENDAR_ALLOW_DEV_SOURCE=1`.

---

## État réel de la couverture d'août 2026
10 parutions officielles curées (US uniquement) rattachées à **Or + EUR/USD** (règle devise→marché).
Zone euro absente en août (EA unemployment & FOMC minutes non datables → journalisés, non inventés).
Cohérent avec l'audit NW-D2 (couverture curée jusque ~octobre-décembre 2026).

---

## Vérifications
- Backend : `pytest tests/test_calendar_service.py tests/test_calendar_providers.py tests/test_calendar_endpoint.py` → **41 passed**.
- Front : vitest calendrier + parité → **87 passed** (dont nouveaux tests d'état de chargement fr+en et garde whitelist).
- `tsc --noEmit` → **0 erreur** ; `npm run build` → **succès**.
- Playwright 1280×800 & 390×844 : chargement / chargé-avec / chargé-vide / serveur injoignable.

## Discipline
Worktree dédié, staging explicite (jamais `git add -A`), pas de force push.
**Aucun merge sur `main` avant ta confirmation live.**

# AUDIT — NW-1 · Actualités programmées (calendrier de volatilité), vue liste

Date : 2026-07-28 · Branche : `feat/nw-1-calendrier` · Périmètre : **vue liste seule**
(la vue détail relève de NW-2). Source cible : **organismes officiels** ; ForexFactory
rétrogradé en **adaptateur prototype dev-only**.

---

## 1. Décision fournisseur (question A, bloquante)

### 1.1 Constat du diagnostic

- Le module « actualités » actuel du tableau de bord (`EventsSection.tsx`) affiche des
  données **réelles**, mais **pas via Twelve Data** : elles viennent du **flux JSON public
  ForexFactory** (`nfs.faireconomy.media/ff_calendar_thisweek.json` + `nextweek`), consommé
  par `src/intelligence/news_pipeline.py`.
- **Twelve Data ne sert AUCUN calendrier économique** (confirmé verbatim par leur support :
  *« While we don't provide specific data such as Nonfarm payroll figures or ISM indexes… »*).
  Il ne peut donc pas alimenter cette page.
- **Aucune source gratuite avec droits d'affichage commercial clairs ET identifiant stable
  n'existe.** Tous les fournisseurs self-serve (FMP, Finnhub, EODHD, Alpha Vantage)
  **interdisent l'affichage à des clients tiers** sur leurs plans standards ; une licence
  commerciale/redistribution est requise dans tous les cas.

### 1.2 Décision (2026-07-28)

**La source cible du produit = organismes officiels** (BLS, BEA, Census, Federal Reserve
pour les États-Unis ; Eurostat, BCE pour la zone euro) : gratuits, domaine public ou
réutilisation permissive, avec **identifiants stables, organisme émetteur, unité et
révisions**. Leur intégration est **hors périmètre NW-1** (mission dédiée à suivre).

**ForexFactory est un adaptateur PROTOTYPE dev-only** :
- aucun droit d'affichage commercial, pas d'identifiant stable, pas d'organisme, pas
  d'unité, pas de révisions ;
- **jamais actif par défaut** : la source est choisie par `CALENDAR_SOURCE` (défaut =
  `official`, stub qui renvoie 0 événement). `CALENDAR_SOURCE=forexfactory` réservé au
  développement local ;
- marqué explicitement dans le code (`calendar_providers/forexfactory_provider.py`) avec
  l'avertissement d'absence de droit commercial.

**Coût NW-1 : 0 $** (aucun contrat de données ; la page est honnêtement vide par défaut
tant que la source officielle n'est pas connectée).

### 1.3 Options payantes recensées (pour la mission officielle / repli éventuel)

| Fournisseur | Calendrier | Id stable | Révisions | Prix mini | Droits affichage SaaS payant |
|---|---|---|---|---|---|
| **Organismes officiels** (cible) | ✅ | ✅ (code de série) | ✅ | **0 $** | ✅ domaine public / réutilisation permissive |
| Econoday | ✅ | ✅ `UID` garanti | ✅ | contact-sales (inconnu) | ✅ métier = licencier aux plateformes |
| Trading Economics | ✅ | ⚠️ `Ticker` de facto | ✅ | ~149–299 $/mo | ⚠️ redistribution négociée séparément |
| FMP / EODHD | ✅ | ❌ | ❌ | 22–60 $/mo | ❌ licence commerciale obligatoire |
| ForexFactory (prototype) | ✅ officieux | ❌ | ❌ | 0 $ | ❌ interdit/flou (pas d'API officielle) |

---

## 2. Identifiant stable

**ForexFactory : ABSENT.** L'`event_id` du flux est un hash `titre|devise|horodatage` — il
**change à chaque publication**. Impossible de relier nativement l'IPC de juin à celui de
juillet ; NW-2 devrait recourir à une jointure heuristique titre+devise, fragile.

**Sources officielles : PRÉSENT** (code de série stable, ex. série BLS `CUSR0000SA0`). Le
schéma NW-1 porte déjà le champ `series_code` et un `event_id` global `"<source>:<ref>"`
qui garantit qu'aucun enregistrement de deux sources différentes n'entre en collision.

---

## 3. Règle de rattachement événement → marché

Source unique et **consultable** : `config/event_market_map.json`. Un événement est rattaché
à tout marché dont les `driver_currencies` incluent la devise de l'événement ; un événement
sans marché rattaché n'est **jamais** rattaché par défaut ni affiché.

| Marché | Devises motrices | Justification |
|---|---|---|
| XAUUSD (Or) | USD | Or coté en USD → la macro américaine le meut. |
| EURUSD | USD, EUR | Paire USD/EUR → macro américaine ET européenne. |

Toute autre devise (GBP, JPY, CAD…) → non motrice → événement **non affiché**.

---

## 4. Schéma modelé sur la source officielle — champ par champ

Le schéma (`calendar_schema.py` / `types/calendar.ts`) porte dès NW-1 la forme officielle.
Un champ non fourni est `null` et **rendu comme absent** (jamais comblé ni inventé). Ce
tableau cadrera la mission d'intégration officielle.

| Champ du schéma | ForexFactory (prototype) | Source officielle (cible) |
|---|---|---|
| `event` (intitulé) | ✅ intitulé brut EN | ✅ intitulé officiel (localisable) |
| `currency` | ✅ code pays/devise | ✅ |
| `impact` | ✅ high/medium/low (tel quel) | ⚠️ variable — sinon dérivé/attribué à la source |
| `scheduled_at` (UTC) | ✅ | ✅ |
| `source_timezone` | ✅ `America/New_York` (défaut FF) | ✅ fuseau de l'organisme |
| `series_code` (id stable) | ❌ `null` | ✅ code de série |
| `organism` (émetteur) | ❌ `null` | ✅ BLS / BEA / Eurostat / BCE… |
| `value_unit` | ❌ `null` | ✅ % / milliers de postes / points de % |
| `actual` / `previous` | ✅ (numérique lenient) | ✅ |
| `forecast` (consensus tiers) | ✅ | ⚠️ souvent absent (produit tiers) |
| `revised` / `previous_before_revision` | ⚠️ détecté au re-fetch seulement | ✅ champ de révision natif |
| `license_label` (par enregistrement) | ✅ « aucun droit commercial » | ✅ domaine public / licence |

**Conséquence UI (honnêteté) :** la maquette montre organisme, périodicité, unité et nom
localisé — que FF ne fournit pas. La vue liste NW-1 **n'invente rien** : elle affiche les
champs réels (intitulé, devise, marché rattaché, impact, horaire, source), rend visiblement
absents l'organisme/l'unité, et laisse l'emplacement amplitude sur « mesures à venir »
(rempli par NW-2). La périodicité et le nom localisé viendront de la source officielle.

---

## 5. Architecture livrée

**Backend** (interface fournisseur obligatoire — aucun champ propre à un fournisseur ne
franchit la frontière `CalendarProvider`) :
- `src/intelligence/calendar_providers/` — `base.py` (interface + `ProviderEvent` neutre),
  `official_provider.py` (**défaut**, stub 0 événement), `forexfactory_provider.py`
  (prototype dev-only), `__init__.py` (fabrique `build_calendar_provider` sur
  `CALENDAR_SOURCE`).
- `src/intelligence/calendar_service.py` — rattachement marché + persistance + fenêtre +
  couverture honnête. Dépend uniquement de l'interface.
- `src/storage/calendar_cache_store.py` — persistance neutre (tous impacts, marchés,
  révisions + valeur avant révision, source/organisme/unité/fuseau/licence).
- `src/intelligence/calendar_schema.py` — schéma API forme officielle.
- `src/api/routes/calendar.py` — `GET /api/calendar` (init paresseux, injectable en test) ;
  câblé dans `app.py` + champ `calendar_service` sur `AppState`.

**Frontend** (réutilise puces `FilterChipGroup`/`useMultiFilter`, jetons de design, helpers
d'heure locale — pas de nouvelle famille visuelle) :
- `app/[locale]/(product)/actualites/page.tsx` + `components/calendar/CalendarWorkspace.tsx`
  (+ `calendar.css`, `CalendarPreview.tsx`), `types/calendar.ts`, `lib/calendar/`
  (`api.ts`, `useCalendar.ts`, `grouping.ts`). Entrée rail « Actualités » (`ShellRail`).
- Filtres multi-sélection impact + marché (0 puce → liste vide + message, jamais de repli) ;
  bouton « Tout cocher » = action ; **aucun tri par amplitude** (chronologique seul) ;
  aucun code couleur hiérarchisant ; onglet « Publications passées » ; bandeau d'intro et
  bloc « ce que ce calendrier ne dit pas » **verbatim** de la maquette ; heure de
  publication + heure locale + décalage explicite ; source + organisme par enregistrement
  (visiblement vides avec le prototype/stub) ; note de couverture honnête.

**i18n** : namespace `calendar` + `nav.calendar` sur les **9 locales** (fr + en complets,
7 autres traduites, parité de clés).

---

## 6. Décisions & différés assumés

1. **Aperçu tableau de bord (`CalendarPreview`) construit mais NON câblé dans `/app`.**
   Le remplacement du module événements actuel de `/app` par cet aperçu est **différé à la
   mission d'intégration officielle** : le brancher maintenant viderait la section événements
   de `/app` tant que seule la source prototype/stub existe. Le composant réutilise la même
   source et les mêmes helpers (zéro duplication de logique) : la bascule ne coûtera qu'un
   placement.
2. **Organisme / unité / périodicité / nom localisé non affichés** en NW-1 : absents du flux
   FF, rendus visiblement absents plutôt que fabriqués.
3. **Révisions** : mécanisme de détection + `previous_before_revision` implémentés et testés
   au niveau du store ; dormant avec FF (qui révise rarement en place), pleinement utile avec
   une source officielle.

---

## 7. Tests

- Backend (32) : `test_calendar_cache_store.py`, `test_calendar_service.py`,
  `test_calendar_endpoint.py`, `test_calendar_providers.py` — règle de rattachement,
  événement sans marché non affiché, tous impacts conservés, **source par défaut ≠
  ForexFactory**, substitution d'adaptateur (service agnostique), pas de collision d'id
  inter-sources, champ nul non comblé, révision flaggée + valeur avant révision, couverture
  partielle, adaptateur FF (garde faible, drop holiday, provenance/licence).
- Frontend : `calendar-copy-honesty.test.ts` (scan fr+en du namespace, verbatim maquette,
  parité de clés), `CalendarWorkspace.test.tsx` (filtre 0→vide+message, **pas de tri
  amplitude**, placeholder « mesures à venir », révision affichée, organisme absent visible,
  onglet passé, pas de clé i18n brute).
- Playwright : `calendar.spec.ts` (nav rail→page 1280, chrome descriptif + toggle puce, pas
  d'overflow 1280 & 390).

---

## 8. Écarts restants vs maquette (à combler par NW-2 / mission officielle)

- Colonne amplitude = « mesures à venir » (NW-2 la remplit avec les mesures moteur).
- Organisme, périodicité, unité, nom localisé, consensus, révisions = vides avec le
  prototype ; fournis par la source officielle.
- Vue détail (fiche par indicateur, historique moteur, récap mesures) = NW-2.

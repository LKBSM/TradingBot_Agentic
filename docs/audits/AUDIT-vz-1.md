# AUDIT VZ-1 — Page Zones : confluence, proximité, historique des contacts

État : **DIAGNOSTIC (lecture seule)** — aucun code écrit, en attente du GO.
Branche : `feat/vz-1-zones` (worktree dédié, depuis `origin/main` = 645f331).
Maquette de référence : `docs/design/reference-zones.html` (poussée sur `main`, commit b9f791b).

Périmètre : page `/zones` uniquement. **Zéro modification des règles de détection.** Tout
ce qui manque se calcule sur la couche lecture (mappers Python / `lib/zones` TS), comme le
compteur de touches l'a été récemment.

---

## 0. Ce qui existe déjà, résumé

- Page `/zones` fonctionnelle : `webapp/app/[locale]/(product)/zones/page.tsx` →
  `ZonesWorkspace` → `ZoneLifecycleCard` + `ZoneTimeline`.
- Type frontend `ZoneLifecycle` : `webapp/lib/zones/lifecycle.ts:45-66`.
- Données via `useMarketReading` → `GET /api/market-reading` (une lecture par combo,
  polling 60 s, cache mémoire PERF-1).
- Confluence multi-unités **déjà câblée** : `useSiblingZones`
  (`webapp/lib/zones/use-sibling-zones.ts`) — une lecture *cache-served* par unité voisine,
  **pas par carte**.
- Le backend **émet déjà** `touch_count` + `touch_ats` (horodatés) par zone depuis
  `READING_LOGIC_VERSION = 5` — mais **le frontend ne les consomme pas encore** (le type
  `ZoneLifecycle` ne lit que `tested: boolean` + `mitigatedAt`).

---

## 1. Tableau DISPONIBLE / CALCULABLE / ABSENT

| Information (mission §2) | État | Où / comment |
|---|---|---|
| Bords (haut/bas) | **DISPONIBLE** | `level_high` / `level_low` — payload, `market_reading_schema.py:132-133`. Frontend : `levelHigh`/`levelLow`. |
| Hauteur | **CALCULABLE** (trivial) | `level_high − level_low`. Pas de champ pré-calculé. |
| Milieu | **CALCULABLE** (trivial) | `(level_high + level_low) / 2`. |
| Hauteur en **% du prix** | **CALCULABLE** | `hauteur / prix_courant`. Prix déjà dispo (`useLatestPrice`). |
| Horodatage de formation | **DISPONIBLE** | `created_at` — `market_reading_mappers.py:526,607`. |
| **Compteur de touches horodatées** | **DISPONIBLE (payload) / NON SURFACÉ (front)** | `touch_count` + `touch_ats` émis par le backend (`mappers.py:538-539,610-611`) mais **le type `ZoneLifecycle` ne les lit pas** — `tested` reste un booléen. À brancher côté front (mapper `collectZones`), zéro nouvelle détection. |
| Taux de comblement FVG (**%**) | **CALCULABLE** | `fill_level` (prix de pénétration max) est dispo ; le % est déjà dérivé côté front par `fillFraction()` (`lifecycle.ts`). Le **%** n'est pas un champ backend mais se calcule : `|fill_level − bord_proche| / hauteur`. |
| Événement structurel (BOS/CHOCH) qui **définit** la zone | **CALCULABLE (association)** | Pas de lien par-zone aujourd'hui. Mais `bos_events` / `choch_events` (horodatés, niveau) sont dans le **même** payload : on associe chaque OB à la cassure de même sens qui suit sa formation. Logique de rapprochement **lecture seule**, aucun changement moteur. |
| Session de formation (Londres/NY/Asie) | **CALCULABLE** | À partir de `created_at` + fenêtres de session. `src/intelligence/market_calendar.py` existe mais n'est pas appelé au moment de la collecte. Calculable côté mapper OU côté front depuis l'horodatage. |
| Zones d'**autres unités** couvrant le même prix | **DISPONIBLE** | `useSiblingZones` récupère déjà les zones des autres unités (voir §3). Aujourd'hui rendu comme simple « chevauche » ; à enrichir en « englobe / à l'intérieur de / au même niveau ». |
| Zones **imbriquées sur la même unité** | **CALCULABLE (local)** | Absent aujourd'hui, mais toutes les zones de l'unité courante sont dans la lecture déjà chargée → imbrication calculée localement, **zéro requête**. |
| Poches de liquidité à proximité | **DISPONIBLE** | `liquidity_pools[]` déjà dans le **même** payload (`market_reading_schema.py`, cap 8). Proximité calculée localement, zéro requête. Aujourd'hui non reliées aux zones. |
| Entrées **et sorties** du prix, distinctes des touches de bord | **ABSENT → CALCULABLE** | Voir §2 (question B) — c'est le chantier principal. |
| Distance au prix (points) | **PARTIEL** | `priceRelation()` donne « à N pts au-dessus/en dessous » mais **sans %**, **sans mention du bord de référence**, et le cas « dedans » ne donne pas la distance à chaque bord ni « depuis quand ». |
| Dernier contact (quand / niveau / par où ressorti) | **CALCULABLE** | Dérive de la classification des contacts (§2). Aujourd'hui seul `mitigated_at` (1er contact) existe. |
| Jauge visuelle zone + prix | **CALCULABLE** (rendu) | À construire à partir de bornes + prix + étendue affichée. |

---

## 2. Question B (LA question) — touche de bord / entrée-sortie / traversée

**Réponse : NON, le produit ne distingue pas ces trois cas. Ils sont fondus en un seul
compteur, et la traversée est carrément retirée du payload d'affichage.**

Preuve — `_ob_lifecycle`, `market_reading_mappers.py:305-367` (cœur de la logique) :

- Un « tap » = un wick qui atteint `zhigh − depth` (avec `depth = ob_mitigation_penetration ×
  hauteur`, et **`ob_mitigation_penetration = 0.0` par défaut** → *n'importe quelle* mèche qui
  effleure le bord proche compte).
- `touch_count` compte les **runs distincts** dans la zone (front montant out→in). Il ne
  retient **ni la profondeur atteinte, ni le bord de sortie**.
- Donc :
  - **(a) touche de bord** (effleure le bord sans pénétrer) → comptée comme *un tap* (car
    seuil = 0). Indistinguable de (b).
  - **(b) entrée puis sortie** → *un tap* aussi. Aucune trace du bord de sortie.
  - **(c) traversée** (clôture au-delà du bord opposé) → statut `invalidated` → **la zone est
    retirée du payload** (`mappers.py:543-554`, « never surface a consumed zone »). Elle n'est
    donc même pas comptée côté affichage.

Conséquence pour la maquette (qui montre trois libellés distincts : « ressorti sans
traverser » / « touché le bord sans pénétrer » / « traversé », plus un groupe « Comblées ») :

**C'est le corps principal de la mission.** Il faut, sur la **couche lecture** (mappers, à la
manière additive dont `touch_count` a été ajouté), pour chaque contact :

1. un **seuil de pénétration** pour séparer l'effleurement de bord (a) d'une vraie entrée (b) ;
2. la **profondeur atteinte** et le **bord de sortie** (bord proche = entrée-sortie ;
   clôture au-delà = traversée) ;
3. **conserver** un ensemble borné de zones récemment consommées (traversées/comblées) au lieu
   de les jeter, pour alimenter le groupe « Comblées ». `collect_zone_lifecycles`
   (`mappers.py:621+`) sait déjà garder exactement ces zones — modèle réutilisable.

Tout cela se calcule à partir de la fenêtre de bougies déjà utilisée par les prédicats de
lifecycle, **sans toucher aux règles de détection**.

---

## 3. Confluence multi-unités — coût mesuré

**Déjà résolue de la bonne manière, et bon marché.**

- `useSiblingZones` (`webapp/lib/zones/use-sibling-zones.ts:39-99`) : pour l'unité courante,
  il lance **une** `fetchMarketReading` par **autre** unité affichée
  (`DISPLAY_TIMEFRAMES.filter(tf !== current)`), en parallèle, chaque lecture servie par le
  cache mémoire (PERF-1).
- Coût = **(nombre d'unités affichées − 1) lectures par combo**, faites **une fois** et
  réutilisées pour **toutes** les cartes. **Aucune requête par carte.** Une unité indisponible
  ne contribue rien (dégradation honnête).
- Les zones **imbriquées même unité** et les **poches de liquidité** proches se calculent
  **localement** depuis la lecture déjà chargée → **zéro requête supplémentaire**.

Méthode retenue (proposée) : **conserver le précalcul on-demand existant** (`useSiblingZones`),
et l'enrichir du vocabulaire « englobe / à l'intérieur de / au même niveau » + unité nommée.
Pas de nouvelle requête. Budget de rendu à mesurer sous ~plusieurs dizaines de zones (mission
performance).

---

## 4. Panneau M.I.A

- Composant existant : `webapp/components/app/AppChatSidebar.tsx` (utilisé sur `/app`), props
  `{ active: Combo, onSelectCombo }` — **contextualisé sur un couple instrument+unité, pas sur
  une zone**. `ChatProvider` est au niveau session.
- **Réutilisable ici, mais à adapter** : il faut un **sujet = zone** (en-tête « Zone
  sélectionnée » comme la maquette), changer de sujet au clic **sans rechargement**, et des
  **questions suggérées contextuelles** à la zone. Les faits cités doivent venir des **mêmes
  données** que la carte (aucun recalcul parallèle).
- `AgentAvatar` (`webapp/components/chat/AgentAvatar.tsx`) accepte déjà `presence`.

---

## 5. Vocabulaire à corriger (mission §0)

- « **chevauche** » est présent : i18n `zones.overlaps.line = "chevauche un {kind} {tf}{dir}
  ({band})"` (`messages/fr.json`), rendu par `ZoneLifecycleCard` (l.298). → remplacer par « à
  l'intérieur de » / « englobe » / « contient » / « au même niveau ». Test à ajouter :
  absence de « chevauche » (fr) et « overlap » (en) dans les chaînes visibles.
- Aucun terme de jugement (« respectée / validée / solide / forte / fiable / qualité /
  meilleure » + équivalents EN) — la base actuelle en est déjà exempte ; test-garde à ajouter.

---

## 6. Écarts avec la maquette (à combler après GO)

| Maquette | Écart actuel |
|---|---|
| Bloc **Proximité** en évidence (dedans : distance à chaque bord + depuis quand ; sinon distance au bord le plus proche en pts **et %**, sens, **bord de référence** ; dernier contact) | Aujourd'hui : une ligne « à N pts au-dessus/en dessous », sans %, sans bord de référence, sans dernier contact. |
| Bloc **Confluence** remonté (dans / englobe / autre unité nommée / liquidité proche + **état d'absence explicite**) | Aujourd'hui : « chevauche » replié sous un chevron, sans état d'absence, sans imbrication même-unité ni liquidité. |
| **Contacts** : une ligne par contact horodatée avec l'issue (ressorti / touché bord / traversé) + phrase d'honnêteté | Aujourd'hui : `tested` booléen, `mitigated_at` = 1er contact seulement. |
| **Frise de vie** avec points horodatés réels (formation, chaque contact, comblement, maintenant) | `ZoneTimeline` existe mais sur événements moteur limités (formation, 1er test, comblement, maintenant) — pas par contact. |
| Groupe **Comblées** (zones traversées visibles) | Zones invalidées **retirées** du payload d'affichage. |
| Ligne **Ce qui l'a créée** (BOS/CHOCH + niveau + heure + session) | Absente. |
| Panneau **M.I.A** sujet=zone, questions contextuelles | `AppChatSidebar` contextualisé combo, pas zone. |
| Regroupement **dedans / au-dessus / en dessous / comblées** | Aujourd'hui filtres all/active/mitigated + tris proximité/fraîcheur/état — pas de regroupement par position. |
| Bouton « Analyser avec M.I.A » **retiré** des cartes (sélection suffit) | Aujourd'hui bouton « Analyser la zone » (deep-link `/app`). |

---

## Verdict de cadrage

L'ampleur est **moyenne-haute**, portée par la **question B** : la classification
touche-bord / entrée-sortie / traversée n'existe pas et doit être **calculée sur la couche
lecture** (mappers Python, additif, comme `touch_count`), + surfaçage front (le payload porte
déjà `touch_count`/`touch_ats` non consommés). La confluence multi-unités est **déjà résolue
et bon marché**. Le reste (proximité enrichie, session, événement structurel, imbrication,
liquidité, panneau M.I.A sujet=zone, regroupement) est du travail de lecture/affichage sans
toucher au moteur.

**En attente du GO avant toute écriture de code.**

---

# PARTIE 2 — Réalisation (après GO)

## Réponse aux trois questions du rapport (mission §5)

### a) Informations DISPONIBLES / CALCULÉES / ABSENTES — ce qui a été livré
- **Disponibles, désormais surfacées** : bords, hauteur (+ % du prix), milieu,
  horodatage de formation, touches horodatées (le payload les portait ; le front
  les consomme maintenant via le **ledger de contacts**), poches de liquidité.
- **Calculées côté lecture (sans toucher au moteur)** :
  - **Ledger de contacts** par zone (`ZoneContact` : `edge_touch` / `entry_exit` /
    `traversal` / `inside`, niveau atteint) — helpers `_ob_contacts` /
    `_fvg_contacts` (Python), additifs, réutilisant les prédicats existants ;
  - **Zones consommées** (`consumed_order_blocks` / `consumed_fair_value_gaps`)
    pour le groupe « Comblées » — bornées (`MAX_CONSUMED_ZONES_PER_TYPE=6`) ;
  - **Origine** (BOS/CHOCH que l'OB précède) via `_ob_origin` sur les colonnes
    d'événements déjà émises ;
  - **Proximité** (dedans → distance à chaque bord ; sinon distance au bord le plus
    proche en pts **et %**, sens, **bord de référence**) — `zoneProximity` ;
  - **Confluence** (à l'intérieur / englobe / au même niveau + liquidité proche,
    état d'absence explicite) — `buildConfluence`, données déjà en mémoire ;
  - **Session de formation** — `formationSession` (miroir client des fenêtres NY du
    backend) ;
  - **Comblement FVG par contact** — `fvgContactFills` (extremum cumulé).
- **Absentes / hors périmètre** : aucune nouvelle détection ; l'imbrication
  même-unité se calcule localement (0 requête).

### b) Distinction touche de bord / entrée-sortie / traversée
Avant : **confondues** dans un unique `touch_count`, la traversée **retirée** du
payload. Après : **trois issues distinctes** dans le ledger — `edge_touch` (kiss <
`contact_edge_touch_fraction`=0,10 de la hauteur), `entry_exit` (pénétration puis
sortie par le **même** bord), `traversal` (clôture au-delà du bord opposé / FVG
comblé). `inside` marque un contact en cours. Le seuil est un **paramètre de
classification** — il ne change JAMAIS `touch_count`/`tested`/`status`. Tests :
`test_zone_contacts_vz1.py` (backend) + `lifecycle.test.ts` (front).

### c) Méthode retenue pour la confluence multi-unités + coût
**On-demand existant conservé** (`useSiblingZones`) : **une** lecture cache-served
par unité voisine (`DISPLAY_TIMEFRAMES − 1`), **une fois par combo**, réutilisée
pour toutes les cartes — **jamais une requête par carte**. Imbrication même-unité et
liquidité : **0 requête** (lecture déjà chargée). `buildConfluence` est de la pure
géométrie d'intervalles. Coût : (nb d'unités affichées − 1) lectures cachées / combo.

## Écarts avec la maquette
- **M.I.A** : la maquette montre un fil de conversation ; l'utilisateur est à **0
  crédit volontaire**, donc les réponses sont **générées localement** à partir des
  **mêmes données** que la carte (aucun appel Anthropic). Sujet = zone, changement
  au clic sans rechargement, suggestions contextuelles — conformes.
- **Session** : calculée côté client (miroir documenté des fenêtres NY du backend)
  plutôt qu'ajoutée au payload, pour ne pas propager l'instrument dans plusieurs
  couches. Précédence New York > Londres > Asie (ancre NY).
- **Origine FVG** : la maquette donne une phrase générique (« déséquilibre de trois
  bougies ») — rendue depuis l'i18n, pas de donnée d'événement (un FVG n'a pas de
  cassure fondatrice).
- **Note d'honnêteté** : reformulée pour **éviter** les mots de jugement (« M.I.A ne
  porte aucun jugement de valeur… ») tout en expliquant le choix — sinon le garde
  vocabulaire l'aurait signalée.
- Tout le reste (proximité, jauge, confluence remontée + état d'absence, ledger,
  frise par contact, comblement, « Ce qui l'a créée », détails repliés, groupes,
  filtres/tris factuels, bouton « Analyser » retiré) suit la maquette.

## Vérifications
- **Backend** : `pytest` mappers/schema/endpoint/lifecycle/diagnostics = 140 + 11
  (VZ-1) verts, 0 régression. `READING_LOGIC_VERSION 6→7`.
- **Front** : `tsc` vert ; `vitest` zones + parité + workspace = 75 verts ; suite
  complète 893 (seule `AccountPanel` timeout = flake d'environnement lent, verte en
  isolation). `next build` OK (`/[locale]/zones` 9,85 kB).
- **Playwright** 1280×800 + 390×844 : groupes, proximité dedans, ledger à issues
  distinctes, absence de confluence, filtre vide explicite, panneau M.I.A (bureau +
  feuille mobile), garde vocabulaire.
- **Discipline** : aucune modification des règles de détection ni des autres
  surfaces ; staging explicite (jamais `git add -A`) ; pas de force push.


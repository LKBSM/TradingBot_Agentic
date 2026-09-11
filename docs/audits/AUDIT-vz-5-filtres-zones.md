# AUDIT VZ-5 — Barre de filtres /zones + doublon de marché

**Branche** : `feat/vz-5-filtres-zones` (worktree dédié `C:\MyPythonProjects\wt-vz-5`)
**Base** : `origin/main` @ `af0c4cf` (PR #208)
**Date** : 2026-09-10
**Maquette** : `docs/design/zones_bar_v2.html` (déposée avec cette branche)

Deux parties indépendantes, deux commits séparés.

---

## Partie A — Doublon de marché : la cause racine

### Une seule cause, deux copies du même code

Fichier unique : `webapp/components/market/MarketSelector.tsx`.

Ce composant sert **trois variantes** (`rail`, `panel`, `bar`) depuis **deux fonctions
distinctes**, et chacune recalculait **en ligne** sa partition « Épinglés » / « Marchés ».
Il n'y avait pas de fonction partagée — donc rien qui garantisse que les deux se
comportent pareil.

| Forme | Où | État avant VZ-5 |
|---|---|---|
| `ColumnSelector` | menu latéral : rail `/app` + panneau mobile | **corrigée** par APP-1 (PR #193) : `unpinnedMarkets = allMarkets.filter(id => !isPinned(id))` |
| `BarSelector` | déroulant de la barre `/zones` | **non corrigée** : rendait `pinnedMarkets` **puis `allMarkets` non filtré** |

Résultat : dans le déroulant `/zones`, un marché épinglé était rendu **deux fois** —
une fois sous « Épinglés », une fois dans la liste complète en dessous.

### Les deux symptômes rapportés sont le même bug

- **Symptôme 1** (« le menu latéral affiche deux fois les mêmes marchés », section 15 du
  dossier de projet) : décrit l'état **d'avant PR #193**. Sur `origin/main`, la forme
  colonne est corrigée **et** verrouillée par un test (`MarketSelector.test.tsx`,
  « every market pinned → « Marchés » does not repeat them »).
- **Symptôme 2** (« rechercher un marché puis l'épingler depuis la recherche ») : ce
  n'est **pas un second chemin**. Le champ de recherche vit **à l'intérieur** du
  déroulant `BarSelector` (`MarketSelector.tsx:490-501`). Chercher filtre les deux
  listes avec la même requête ; épingler un résultat le fait monter dans « Épinglés »
  **sans le retirer** de la liste filtrée juste en dessous. Le doublon apparaît donc
  sous les yeux, au même écran — c'est la copie non corrigée, rendue visible.

### Pourquoi le bug avait survécu

`MarketSelector.test.tsx` ne gardait l'anti-duplication que pour `variant="panel"`.
Le bloc `describe('MarketSelector — bar (header) variant')` ne contenait qu'**un seul
test** (ouvrir le déroulant, choisir un marché). Aucun garde-fou sur la variante `bar`.

### Correctif — la cause, pas le symptôme

`partitionMarkets(query, pinned, isPinned)` : **une seule fonction**, consommée par
`ColumnSelector` **et** `BarSelector`. Elle retourne `{ pinnedMarkets, unpinnedMarkets,
hasResults, showAllSection }`. Une quatrième variante ne pourra plus réintroduire
l'omission — elle n'a plus de partition à réécrire.

Le message « aucun marché ne correspond » reste sur `hasResults === false` : pas de
repli silencieux.

**Aucun changement de persistance.** `market-pins.ts` dédoublonnait déjà à la lecture du
`localStorage` (`market-pins.ts:30-39`) et filtrait sur le périmètre du registre. Le bug
était purement au rendu.

### Garde-fous ajoutés

4 tests sur la variante `bar` : épinglage depuis le déroulant, **recherche + épinglage**,
tout épinglé, recherche sans résultat. Le test APP-1 sur `panel` est conservé.

**Vérifié qu'ils échouent sans le correctif** : `git stash` du seul `MarketSelector.tsx`,
relance → `AssertionError: expected [ Array(2) ] to have a length of 1 but got 2` sur les
3 tests de comptage. Ce sont bien des garde-fous, pas des tests qui passent par accident.

Vérifié aussi en navigateur réel : `docs/audits/vz-5/pinned-once-1280x800.png` — après
épinglage, « ÉPINGLÉS → Euro / Dollar » puis « Or (XAU/USD) » seul en dessous.

---

## Partie B — Refonte de la barre de filtres

Présentation seule. **Aucune logique de filtrage ni de tri n'a été touchée** : `filter` /
`sort` alimentent toujours `matchesFilter` et `sortZones`, inchangés.

### Sur l'échelle typographique de la maquette

La maquette est dessinée sur **la palette réelle de l'app** (`--bull:#37b98c`,
`--dim:#97999e`, `--faint:#5f6167`, `--acc:#c9a14a` sont les valeurs littérales de
`globals.css`) — mais à une **échelle ~1,6× plus large** : son `h1` fait 26 px, celui de
l'app en fait **16 px** (`pages.css:64`, densité posée par UI-1b / UI-3 / TXT-1).

Copier ses px aurait dédensifié la page à contre-courant de trois missions précédentes.
C'est donc la **hiérarchie** de la maquette qui est reproduite, dans les tailles du shell.

### Ce qui a changé

| Élément | Avant | Après |
|---|---|---|
| Statut haut-droite | pastille `.livebadge` encadrée : marché · unité de temps · nb de zones | `.zstatus` **texte simple** : point d'activité + « **3** zones suivies ». Marché et unité de temps retirés — le sélecteur les dit une ligne plus bas |
| Étiquettes « FILTRE » / « TRI » | deux `<span>` 9 px majuscules interlettrées | **supprimées** |
| Filtres | `Segmented` | `Segmented`, inchangé — même contrôle que les unités de temps |
| Séparation | aucune | filet vertical `.zbar-div` entre unités de temps et filtres (masqué < 640 px) |
| Tri | 3ᵉ rangée de pilules `Segmented`, à poids visuel égal | `<select>` discret « Trié par **Proximité** », poussé en bout de rangée (`margin-left:auto`) |
| Fraîcheur du prix | empilée sous la pastille, dans la colonne haut-droite | `.zfresh`, **légende sous la barre de contrôles**, 11 px `--faint`, sans icône |
| « LE PRIX EST DEDANS » | `.zsep` 9 px mono majuscules interlettrées + filet `::after` | 13 px, 600, `--txt`, **casse normale**, sans filet. Le `position:sticky` (fonctionnel) est conservé |

Le point d'activité et la mention « actualisation… » sont **conservés** : c'est un état,
pas une répétition de ce qui est à gauche. La maquette les garde aussi (`.status .live`).

### i18n — 9 locales, 2 clés

- `zones.sortLabel` : « Tri » → « **Trié par** » (fr) / « Sorted by » (8 autres).
- `zones.badge.count` : `{count, plural, …}` enrichi en « **# zones suivies** », avec la
  balise `<n>` rendue en `t.rich` pour que le nombre passe en `--txt` + chiffres tabulaires
  pendant que le nom reste en `--dim` — exactement le `.status .num` de la maquette.

Remplacement **chirurgical en binaire** (pas de round-trip JSON) : CRLF et formatage
préservés, **2 lignes modifiées par fichier**, re-parsé après écriture pour prouver la
validité. Cf. [[feedback_powershell_utf8_rewrite]].

`zones.filterLabel` (« Filtre ») devient **inutilisé** : la clé est conservée dans les
9 locales plutôt que supprimée — `filterAria` reste le nom accessible du groupe.

### Accessibilité

Le `<select>` porte un vrai `<label htmlFor="zones-sort">` visible (« Trié par »), donc le
nom accessible **contient** le texte visible (WCAG 2.5.3). `sortAria` (« Trier les zones »)
est passé en `title`, pas en `aria-label` : un `aria-label` divergent aurait écrasé le
libellé visible.

---

## Écarts assumés par rapport à la maquette

1. **Traitement de l'état actif du contrôle segmenté.** La maquette utilise un actif
   discret (`background:var(--panel-3)`) ; l'app inverse (`bg-foreground text-background`).
   Conservé tel quel : `TimeframeControl` est **partagé avec le panneau mobile `/app`**, et
   le changer aurait débordé du périmètre VZ-5. L'exigence de la mission — filtres et
   unités de temps au **même style**, sans légende — est tenue : c'est déjà le même contrôle.
2. **La rangée se replie à 1280 px.** La maquette est dessinée pleine largeur ; sur `/zones`
   le rail (232 px) et la colonne M.I.A (~340 px) ne laissent que ~700 px, donc les filtres
   passent à la ligne sous le marché + unités de temps. Conséquence visible : le filet
   vertical se retrouve en fin de première ligne. Fonctionnel, mais à arbitrer en revue live.
3. **Le rail gauche garde ses étiquettes majuscules** (« MARCHÉS », « UNITÉ DE TEMPS »,
   « ESPACE »). C'est le `ProductShell` partagé par toutes les pages produit, hors périmètre
   d'une mission `/zones`. À traiter séparément si le motif dérange ailleurs.

---

## Vérifications

| Contrôle | Résultat |
|---|---|
| `tsc --noEmit` | **0 erreur** |
| `npm run build` (CI=1) | **exit 0** |
| vitest `components/zones` | **28/28** (4 fichiers) |
| vitest `MarketSelector.test.tsx` | **12/12** |
| vitest `ui2-copy-honesty` + `price-freshness-badge` + `txt1-copy` | **17/17** |
| Playwright `vz-5-filter-bar.spec.ts` (1280×800 + 390×844) | **5 tests, 4 passés + 1 flaky retenté OK** |

Le flaky est `browserContext.newPage: Test timeout` **au montage de la page**, pas une
assertion — charge machine, passé au retry.

⚠️ **Flake d'environnement connu** rencontré en lançant plusieurs répertoires vitest d'un
coup : `[vitest-pool-runner]: Timeout waiting for worker to respond` (jonction
`node_modules` depuis `wt-cln-1`). Ce n'est pas un échec de test — relancés isolément, tous
les fichiers passent. Cf. [[feedback_vitest_worktree_node_modules]].

### Captures

- `docs/audits/vz-5/bar-1280x800.png`
- `docs/audits/vz-5/bar-390x844.png`
- `docs/audits/vz-5/pinned-once-1280x800.png`

---

## Reste à faire

**Confirmation visuelle live du fondateur avant merge**, en particulier sur les trois
écarts assumés ci-dessus.

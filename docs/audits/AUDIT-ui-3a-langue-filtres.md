# AUDIT UI-3a — Langue dans Réglages + filtres multi-sélection (Phase 1 : diagnostic)

> Branche `feat/ui-3a-langue-filtres-multiples` (worktree `wt-ui-3a`, depuis origin/main `2ed1af7`).
> Frontend uniquement. Zéro diff moteur, zéro diff détection. **Aucun code avant GO.**

Ligne inviolable : filtres 100 % factuels (type, état, côté, TF). Aucun filtre importance/qualité/score,
aucun tri « meilleure ».

---

## A. LANGUE

**Comment la locale est portée**
- **Segment d'URL `[locale]`** via next-intl middleware (`middleware.ts`), `localePrefix: 'as-needed'`
  → **FR sans préfixe**, les 8 autres préfixées (`/en/app`…). 9 locales actives (`i18n.ts`).
- **Cookie `NEXT_LOCALE`** = la persistance. `localeDetection: true` : à la 1re visite sans cookie,
  `Accept-Language` décide ; **le cookie prime ensuite** sur l'en-tête.

**Où le changement est déclenché aujourd'hui**
- `components/LocaleToggle.tsx` (dropdown `<details>`, 9 locales). Sur sélection : ① écrit
  `NEXT_LOCALE` (1 an, `path=/`), ② reconstruit le chemin courant sous la locale choisie
  (préfixe ou non), ③ `router.push(target)` (nav client, **pas de reload complet**).
- Il est monté dans le **`Nav` du site** (`components/Nav.tsx`) → **absent du `ProductShell`**
  (App/Zones/Scanner/Réglages). C'est *exactement* la cause du symptôme « langue changeable seulement
  depuis l'accueil ».

**Persistance — RÉPONSE : elle existe déjà.**
Le cookie `NEXT_LOCALE` (1 an) est écrit par `LocaleToggle` et lu par le middleware → la langue choisie
est retrouvée au retour et **prime sur le navigateur**. **Rien à ajouter côté persistance** : il faut
seulement réutiliser ce mécanisme depuis Réglages (une seule source de vérité = le cookie + `router.push`).

**Préférences serveur d'un utilisateur connecté ?**
Non. `lib/auth/store` ne porte que `account.email` (pas de locale serveur). Tout est côté client (cookie).
*(Pas de table dans cette mission — noté, à décider plus tard.)*

**Chaînes en dur non traduites (repérage léger, non exhaustif — pour mission ultérieure)**
- `components/app/AppHeader.tsx` : libellés visibles `Zones` / `Scanner` en dur (les `aria-label` sont, eux,
  i18n via `header.navZones`/`navScanner`). Noms produit → priorité basse, mais « Zones » est traduisible.
- `components/app/ReadingChart.tsx` : titres des price-lines de cassure `'BOS'` / `'CHOCH'` / `'Retest'`
  (~l.669-690). BOS/CHOCH = acronymes ; « Retest » est traduisible.
- Rappel (mémoire i18n) : survols/crosshair du chart data-driven par défaut FR ; narration LLM = backend.
- Le reste des surfaces produit (App/Zones/Scanner/Réglages/RG-1) est i18n (namespaces dédiés).

---

## B. FILTRES

**Où vit l'état — RÉPONSE : local à chaque carte, DUPLIQUÉ, pas partagé.**
- `LiquidityCard.tsx` : `useState` `side: 'all'|'BSL'|'SSL'`, `state: 'all'|'intact'|'swept'|'broken'`
  (choix **unique**, puce `all`). Filtre : `pools.filter(side==='all'||… && state==='all'||…)`.
- `StructureCard.tsx` : `useState` `type: 'all'|'ob'|'fvg'`, `state: 'all'|'active'|'tested'|'mitig'`,
  `sort: 'near'|'recent'|'big'`. Même schéma choix unique + puce `all`.
- Le **motif est dupliqué** (même logique `=== 'all' || …` dans les deux). À **factoriser** (hook + composant
  de groupe de puces), sans réécrire les cartes.
- Le **tri** (`sort`, Structure) reste **choix unique** — c'est un ordre, pas un filtre d'état. Hors périmètre
  multi-sélection.

**Compteurs d'en-tête — RÉPONSE : ils affichent le nombre APRÈS filtrage (pas le total).**
- Liquidité : `badge2 = pochesCount({count: ordered.length})` → « 2 poches » = **filtré**. (Confirme la capture.)
- Structure : `badge2 = zonesCount({count: ordered.length})` → filtré aussi.
- À changer en **« N sur M »** (filtré sur total).

**Client ou appel réseau — RÉPONSE : 100 % client.**
Filtrage en mémoire sur la liste déjà reçue (`structure.liquidity_pools` / `collectZones(structure)`).
Aucun appel déclenché par un filtre.

---

## PLAN DE REFONTE (à valider au STOP)

**A) Langue dans Réglages**
- Factoriser la logique de bascule de `LocaleToggle` dans un hook partagé `useLocaleSwitch()`
  (écrit `NEXT_LOCALE` + `router.push` du **chemin courant** sous la nouvelle locale). `LocaleToggle`
  ET la nouvelle section Réglages l'utilisent → **une seule source de vérité**.
- Ajouter une carte **« Langue »** dans `AccountPanel.tsx` (même patron que « Apparence »), FR + English
  (au minimum ; je peux exposer les 9 comme le toggle — à trancher). Le changement **reste sur `/compte`**,
  traduit ; la sélection combo (localStorage) survit à la nav.
- Persistance = cookie `NEXT_LOCALE` existant. Pas de reload complet (`router.push`).

**B) Filtres multi-sélection (factorisés)**
- Nouveau hook `useMultiFilter(allValues)` → `Set` par groupe, init = **tout coché**, `toggle/reset/isOn/noneSelected`.
- Nouveau composant `FilterChipGroup` (puces `aria-pressed`, clavier, **pas couleur-seule**) + bouton
  **« Réinitialiser »** (ACTION = tout cocher ; remplace la puce « Tous »/« Les deux » supprimée).
- Sémantique : groupes en **ET**, options d'un groupe en **OU**. Aucune puce d'un groupe → **zéro résultat** +
  « Aucun état sélectionné… » (**jamais** de repli sur « tout »).
- Liste vide avec puces cochées → « Aucune poche/zone ne correspond à ces filtres. » + « Rien n'est inventé… ».
- Compteurs → **« N sur M »**.
- Appliqué à Liquidité (côté, état) et Structure (type, état). Tri Structure inchangé (choix unique).
- Cartes : taille/position/colonne inchangées ; puces mêmes tokens/dimensions.

**Tests** (à écrire après GO) : 2 états cochés → 2 catégories ; 0 coché → vide + message + **pas de repli** ;
Réinitialiser → tout recoché ; inter-groupes BSL∩intactes ; compteur « N sur M » ; langue depuis Réglages →
même page/combo, traduit ; langue retrouvée après reload ; aucune clé i18n brute. Playwright 1280×800.

---

## IMPLÉMENTATION (Phase 2) — livré

**Persistance de la locale retenue : cookie `NEXT_LOCALE` (1 an) — mécanisme EXISTANT réutilisé.**
Aucune nouvelle persistance créée. `useLocaleSwitch()` (`lib/i18n/use-locale-switch.ts`) écrit le cookie +
`router.push` du chemin courant sous la locale choisie (préfixe ou non). `LocaleToggle` (nav) ET la carte
« Langue » de Réglages appellent ce **seul** hook → une source de vérité unique. Pas de reload complet.

**A) Langue dans Réglages** — carte « Langue » dans `AccountPanel` (patron « Apparence »), **9 locales**
(`SUPPORTED_LOCALES`/`LOCALE_LABELS`), puce active = locale courante (`aria-pressed` + coche non-couleur).
Reste sur `/compte`, traduit ; la sélection combo (localStorage) survit à la nav client.

**B) Filtres multi-sélection** — factorisés : `useMultiFilter` (`lib/market-reading/use-multi-filter.ts`,
`Set` par groupe, init tout-coché, `toggle/reset/noneSelected`) + `FilterChipGroup`
(`components/app/FilterChipGroup.tsx`, `aria-pressed`, clavier, coche CSS non-couleur, bouton
« Réinitialiser » = action). Appliqué à Liquidité (côté, état) et Structure (type, état). Tri Structure
inchangé (choix unique). Sémantique ET entre groupes / OU dans un groupe. 0 puce d'un groupe → **0 résultat**
+ message générique (`noneSide`/`noneState`/`noneType`), **jamais** de repli sur « tout ». Liste filtrée vide
avec puces cochées → `noMatch1`+`noMatch2`. Compteurs → **« N sur M »** (`countFiltered`). Cartes : taille/
colonne inchangées ; puces mêmes tokens/dimensions.

**i18n** : nouvelles clés `app.liq2.*` / `app.struct.*` (countFiltered, reset, noneSide/noneType/noneState,
noMatch1/2) + `app.account.sectionLanguage`, sur les **9 locales** (fr + en, 7 autres = EN). Clés désormais
inutilisées laissées en place (`pochesCount`, `zonesCount`, `empty1/2`, `*.all`) — nettoyage = mission ultérieure.

**Tests** : `ui3a-filters.test.tsx` (2 états → 2 catégories ; 0 coché → vide + message + **pas de repli** ;
Réinitialiser → tout recoché ; inter-groupes BSL∩intactes et OB∩active ; compteur « N sur M » ; noneType ;
aucune clé brute) ; `use-locale-switch.test.ts` (même page, préfixe fr↔en, cookie persisté) ; `ui2c.test.tsx`
mis à jour (empty → noMatch). Playwright `ui3a.spec.ts` (structurel + skip propre sans backend/session).

**Écarts assumés** : le sélecteur d'accueil expose déjà 9 locales → la carte Réglages aussi (cohérence).
Message 0-sélection **générique par groupe** (côté/type/état) plutôt que la copy « état » unique de l'énoncé.

*(Fin — tsc 0, tests verts, build vert. Push. MERGE après confirmation live.)*

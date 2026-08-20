# OVERLAP-NAV — Navigation cliquable depuis « Ce qu'il y a d'autre au même endroit »

**Branche :** `feat/overlap-items-navigate-to-zone` (worktree dédié `wt-overlap-nav`, depuis `origin/main` `8e96020`)
**Date :** 2026-08-20
**Portée :** additive, présentationnelle + câblage de navigation. Aucune logique de détection touchée.

## Contexte & ligne d'architecture

Rendre chaque item du bloc « Ce qu'il y a d'autre au même endroit » (page `/zones`, détail
d'une zone) cliquable → naviguer vers le détail de CETTE zone. **Verrou d'id (mission §4) :**
navigation via le `zone_id` **réel du moteur** uniquement ; aucun id inventé ni reconstruit
depuis le texte affiché (prix + TF).

## §0bis — Vérification de dépendance (verrou d'id) : résolvable, PAS de blocage

Le bloc est rendu par `ConfluenceBlock` (`ZoneLifecycleCard.tsx`) à partir de
`buildConfluence()` (`lib/zones/confluence.ts`).

- Avant : le `ConfluenceFact` rendu **ne portait pas** d'id (items = texte pur `key={i}`).
- **Mais** le vrai zone_id du moteur était présent **un cran en amont**, dans le `Candidate`
  interne de `buildConfluence` (`z.id` same-TF, `s.id` sibling — tous deux = `ob.id`/`fvg.id`
  via `collectZones`), simplement **jeté par `classifyZone`**.

→ Ce n'était **PAS** le cas interdit « texte descriptif sans id → reconstruction par
prix/label ». L'id réel était disponible et lossless. Le prérequis léger (le faire remonter
`Candidate.id → ConfluenceFact.id`) est exactement ce que §0bis anticipe. **Aucun matching de
prix, aucun id fabriqué.**

## Décision de diagnostic — cross-TF

L'exemple (« OB baissier en **H1** » listé depuis une zone d'un autre TF) est un item
**sibling**. Option retenue par le fondateur : **basculer le TF actif + ouvrir la zone**
(réutilise le chemin `?instrument=&timeframe=&zone=` existant).

## Implémentation

1. **`lib/zones/confluence.ts`** — `ConfluenceFact` gagne `id?: string` ; `classifyZone`
   recopie `c.id`. Le candidate sibling passe de l'id composite `${tf}-${id}` à l'id **brut**
   `s.id` (le champ `timeframe` désambiguïse). Liquidité = pas d'id (une poche n'est pas une
   zone navigable).
2. **`components/zones/ZoneLifecycleCard.tsx`** — `ConfluenceBlock` : les faits zone
   (`inner`/`outer`/`same_level`) portant un id deviennent des `<button class="cl clnav">`
   (les faits `liquidity` restent du texte). `onClick` → `stopPropagation` (ne sélectionne
   pas la carte) → `onNavigate(f.id, f.timeframe)`. `aria-label` = `confluence.openZone`.
   Nouvelle prop `onNavigateToZone` threadée depuis le workspace.
3. **`components/zones/ZonesWorkspace.tsx`** — `navigateToZone(zoneId, tf)` : `filter='all'`
   (révèle une cible que le filtre courant masquerait, sinon note « périmé »), bascule le TF
   si sibling, écrit `?instrument=&timeframe=&zone=` via `router.replace`. Les effets deep-link
   existants (seed `selectedId` + `scrollIntoView`) font le reste — **même chemin exact**.
4. **`components/shell/pages.css`** — `.cl.clnav` : reset bouton + affordance hover
   (souligné + couleur) + `:focus-visible`, espacement `.cl` préservé.
5. **i18n** — clé `confluence.openZone` ajoutée aux **9 locales** (fr réel, en + 7 repli EN),
   parité KEY-stricte.

## Tests d'honnêteté (verrou d'id)

- `lib/zones/__tests__/confluence.test.ts` (+4) : same-TF porte l'id brut exact (timeframe
  null) ; sibling porte l'id **brut** + son TF (pas de composite) ; **deux siblings de bande de
  prix identique gardent chacun leur id** ; liquidité sans id.
- `components/zones/__tests__/ZoneLifecycleCard.nav.test.tsx` (nouveau, 2) : deux zones
  same-TF **quasi-jumelles de prix** + un sibling H1 → chaque bouton ouvre SA zone par id réel
  (jamais la voisine) ; liquidité non cliquable.
- `components/zones/__tests__/ZonesWorkspace.test.tsx` (+1) : clic item → `router.replace`
  avec un `?zone=<id>` qui est un **id réel émis par le moteur** (jamais dérivé du prix).
- `tests/e2e/vz-1-zones.spec.ts` (+2 × 2 viewports) : item cross-TF → bascule H1 + ouvre
  `ob-h1-wrap` (id réel) ; item same-TF → ouvre `ob-nested` (pas le conteneur voisin), TF
  inchangé.

## Résultats

| Vérification | Résultat |
|---|---|
| `tsc --noEmit` | **0** |
| vitest `lib/zones` + `components/zones` | **81/81** |
| vitest i18n parity + zones vocabulary | **18/18** |
| `next build` | **exit 0** |
| Playwright `vz-1-zones` (1280×800 + 390×844) | **19/19** |

## Reste

- **Confirmation visuelle live du fondateur** avant merge sur `main`.
- Note fixture e2e : le mock renvoyait un même payload sibling pour tous les TF (duplication
  artificielle d'un item) — désormais H1 = wrapper, autres TF vides. Artefact de test, pas un
  comportement produit (en prod chaque TF a des zones distinctes).

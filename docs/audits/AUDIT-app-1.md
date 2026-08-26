# AUDIT APP-1 — Espace de travail : bulle ou troisième colonne, au choix

## Position du HEAD

Répertoire principal au démarrage : `docs/preserve-data-1-audit` @ `e0dc69c` — **22 commits
derrière** `origin/main` (`0b073ca`). Travail dans le worktree dédié **`wt-app-1`**, branche
**`feat/app-1-espace-travail`** basée sur `origin/main` → **0 retard**. Mesures et captures
faites sur ce build.

---

## Diagnostic (lecture seule)

### A) La bascule — seuil, emplacement, UN composant

- **Seuil AVANT : 1280 px.** La colonne n'existait que dans `@media (min-width:1280px)`
  (`shell.css:373`). À `@media (max-width:1279px)` (`shell.css:449`), la règle
  `.app-shell:not(.no-chat)` **forçait** le tiroir off-canvas + fab **quel que soit** le choix
  persisté. **C'est la cause du « bascule tout seul »** : un portable rendu à <1280 px CSS
  (chrome navigateur, zoom, DPR) perdait la colonne sans recours.
- **UN SEUL composant** ✅ — `AppChatSidebar` est monté **une fois** dans `ShellChat`
  (`ShellChat.tsx:105`), jamais démonté ; les deux modes ne changent que le CSS de `.chatcol`.
  État dans `ChatColumnContext` (`localStorage` `mia.app.chat-column-open`, défaut `true`).

### B) L'état de la conversation à la bascule — tout survit ✅

Historique + zone sélectionnée dans `ChatProvider` (`app/[locale]/layout.tsx`, au-dessus du
shell) ; brouillon d'input local à `ChatInput`, **jamais démonté** à la bascule. Donc
**historique, zone sélectionnée ET message en cours survivent** dans les deux sens (vérifié
par test e2e).

### C) Le débordement horizontal — NE SE REPRODUIT PAS

Mesure live (`documentElement.scrollWidth − clientWidth`), colonne ET bulle : **0** à 1280,
1440 et 1920. La grille `232px minmax(0,1fr) 338px` tient exactement. Le défaut A **n'existe
pas sur `origin/main`** (capture fondateur périmée). → verrouillé par test (0 scroll horizontal).

### D) Répartition de la largeur (px)

| Mode | rail | centre | M.I.A | → graphique (centre − 36px) |
|---|---|---|---|---|
| Colonne 1280 | 232 | 710 | 338 | ~674 |
| Bulle 1280 | 232 | 1048 | fab | ~1012 |
| Colonne **1152** (nouveau) | 232 | 582 | 338 | ~546 |
| Colonne 1100 (seuil) | 232 | 530 | 338 | ~494 |

Hauteur graphique : `clamp(300px, 52svh, 560px)` (`ReadingChart.tsx`). À 1280×800 le **haut de
« Lecture narrée » est à y≈627 → visible** (défaut C non reproduit ; verrouillé par test).

### E) Le compte « 2 zones »

Deux mesures **différentes** : carte Structure (`StructureCard.tsx:222`, « N sur M zones ·
X actives » — **avec dénominateur**) vs pastille d'**amas** sur le graphique
(`chart.zonesCluster`, logique **CHART-2** `zoneLabelLayout.ts`). Périmètres distincts → pas un
doublon ; la pastille relève de CHART-2. Le compte à dénominateur est vérifié par test.

**Défaut D** (chevauchement des étiquettes) = **domaine de CHART-2** (`zoneLabelLayout.ts`
marqué CHART-2, déjà dé-collision/cluster/cap) → **laissé**, comme demandé.

---

## Ce qui a été construit (après GO)

### Le choix de mode — seuil 1280 → **1100**

- `shell.css` : `@media (min-width:1280px)` → **`1100px`** (règles bulle `.chat-collapsed`) et
  `@media (max-width:1279px)` → **`1099px`** (tiroir forcé). La colonne est offerte **≥1100**
  (centre ≥530px, graphique lisible) ; sous 1100 elle **crush**erait le graphique donc n'est
  pas proposée.
- `AppChatSidebar.tsx` : le bouton toggle passe de `xl:inline-flex` (≥1280) à
  **`min-[1100px]:inline-flex`**. Ajout d'une **ligne de statut** sous l'entête (uniquement où
  le toggle vit — jamais l'onglet mobile) :
  - **≥1100** : « Ton choix d'affichage — non synchronisé. » (le choix vit dans le navigateur).
  - **<1100** : « Colonne disponible sur un écran plus large. » — le contrôle **explique** au
    lieu de disparaître sans rien dire.
- **Le choix gagne** : `ChatColumnContext` n'écrit dans `localStorage` que sur toggle explicite ;
  un redimensionnement n'y touche jamais. Une fois le seuil abaissé à 1100, le choix persiste
  et n'est **jamais repris** au resize (vérifié par test).
- **Un composant unique** dans les deux modes (inchangé) → conversation, zone et brouillon
  survivent.
- i18n : 2 clés (`app.chat.modeNotSynced`, `app.chat.columnNeedsWidth`) ajoutées aux **9 locales**
  (parité vérifiée). Aucun vocabulaire prédictif/prescriptif.

### Défaut B — « Marchés » sans doublon des épinglés

`MarketSelector.tsx` : la section « Marchés » liste désormais **les non-épinglés seulement**
(`unpinnedMarkets`) et **ne s'affiche pas** quand elle ne montrerait rien de neuf (tous
épinglés). Le message de recherche vide est conservé. La mention « Non synchronisé » existait
déjà. Composant partagé (rail/panel/bar) — logique universellement correcte.

---

## Ce qui NE change pas (non négociable) — respecté

Aucune donnée/calcul touché ; pas de donnée → pas d'élément ; aucun compte avant chargement
(le skeleton PERF-2 intact — **test AppWorkspace « skeleton » non touché**) ; couches
masquables du graphique inchangées ; aucun vocabulaire prédictif introduit ; un seul bloc
d'avertissement par page.

---

## Tests

- **e2e `app1-espace.spec.ts` 14/14** (fr + en) : bande 1152 colonne↔bulle atteignable des deux
  côtés ; <1100 colonne non offerte + explication ; statut « non synchronisé » ≥1100 ;
  persistance au reload ; **resize ne reprend pas le choix** ; brouillon + sous-arbre survivent
  la bascule ; **0 scroll horizontal à 1280/1440/1920** (colonne et bulle) ; **haut du narr
  visible à 1280×800** ; compte de zones avec dénominateur (« N sur M »).
- **`MarketSelector.test.tsx`** : nouveau cas défaut B (2 marchés épinglés → « Marchés » ne les
  répète pas) + parité 9 locales — **18/18**.
- **Régression `mia-column-toggle` 3/3** (seuil abaissé, 1280 et mobile intacts).
- **tsc** 0 erreur nouvelle ; **`next build`** vert (`min-[1100px]:` compile).

## Captures

`docs/audits/app1-shots/before/` (origin/main) et `after/` : 1280 colonne/bulle, **1152 colonne
(le gain)**, 1000 bulle+explication, rail 2 marchés épinglés (plus de doublon).

## Discipline

Périmètre présentationnel/layout. Staging explicite (pas de `git add -A`), pas de force push.
**Merge sur `main` seulement après confirmation visuelle live du fondateur.**

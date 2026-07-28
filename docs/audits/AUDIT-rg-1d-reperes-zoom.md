# AUDIT RG-1d — Repères : lisibilité, zoom au clic, croisement avec la liquidité

Branche : `feat/rg-1d-reperes-zoom` (worktree dédié, depuis `main` = `8dde1e5`).
Cible : panneau « Régime de marché » → tuile « Niveaux de référence » + tracé chart.
Les VALEURS étaient déjà correctes (RG-1c) ; ici, trois défauts d'interaction/présentation.

---

## 1. Diagnostic (avant correctif)

### DÉFAUT 1 — pourcentage ambigu
`RegimeCard.fmtSigned` produisait « + 1,33 % » (signe seul) pour les lignes de niveaux et
les distances de la tuile Position. La liste des zones fait déjà mieux (mots :
`relation.above/below/inside`).

### DÉFAUT 2 — le clic ne « mène » pas au graphique
Le clic **mettait bien à jour l'état ET traçait la ligne** (RG-1c). Manquaient :
- **aucun défilement de page** vers le graphique (`containerRef` existait, pas de
  `scrollIntoView`) → si l'utilisateur regardait le panneau, « rien ne se passe » ;
- **pas de marge** autour de [niveau, prix courant] (le niveau pouvait coller au bord) ;
- **retour d'échelle** au re-clic implicite (autoscale) mais non explicite.

**API graphique (constat) :** lightweight-charts **n'expose aucun setter impératif de plage
verticale**. La seule façon de fixer une plage verticale est `series.autoscaleInfoProvider`
qui renvoie un `priceRange` — c'est le mécanisme utilisé (RG-1c), ici **étendu d'une marge**.
L'horizontal se pilote via `timeScale` ; le défilement de **page** via `scrollIntoView`.

### DÉFAUT 3 — croisement avec la liquidité
Les poches BSL/SSL sont tracées via `buildLiquidityLines` → `createPriceLine`, **le même
mécanisme** que la ligne de repère, déjà **de style distinct**. Les poches exposent leur
niveau publié `pool.level` (mêmes unités que `referenceLevel.price`) et leur côté `pool.side`
(`bsl`/`ssl`) — **directement comparable**.

---

## 2. Correctif

### DÉFAUT 1 — pourcentage en mots (aucun signe)
`RegimeCard.relWord(pct)` remplace `fmtSigned` : « {pct} % au-dessus » / « {pct} % en dessous »
/ « au prix courant » (quand |pct| < 0,005 %, soit 0,00 % à l'affichage). Appliqué aux lignes
de niveaux **et** aux distances Position. i18n `regimePanel.data.relAbove/relBelow/relAt`
(fr+en, +7 locales EN). **Aucun `+`/`−` : le mot porte le sens.**

### DÉFAUT 2 — le clic mène au graphique
`ReadingChart`, effet `referenceLevel`, **uniquement quand le tracé change** (préservation du
zoom manuel entre ticks) :
- **set/changed** → re-autoscale (le niveau + marge entrent dans la plage) **et**
  `containerRef.scrollIntoView({behavior:'smooth', block:'nearest'})` (défilement page) ;
- **cleared** (re-clic) → re-autoscale = **retour à l'échelle précédente** (fit bougies), pas
  de défilement.
`autoscaleInfoProvider` : union `[min(bougies, niveau), max(bougies, niveau)]` **+ marge de
8 %** (`REFERENCE_VIEW_MARGIN_FRAC`) → niveau **et** prix courant à l'intérieur, avec la
structure entre les deux visible (contexte préservé, pas de sur-zoom). Un seul repère ; rien
au chargement (inchangé).

### DÉFAUT 3 — coïncidence avec une poche détectée (règle stricte)
`webapp/lib/market-reading/reference-coincidence.ts` (pur, testable) :
- `coincidenceTolerance(recent_avg) = 0,25 × recent_avg`, `null` si pas de base d'amplitude ;
- `matchLiquidity(prix, pools, tol)` → côté de la poche **détectée** la plus proche si
  `|prix − pool.level| ≤ tol`, sinon `null`. **Exclut les poches `broken`** (ne reposent plus
  là). Vérifié contre une **sortie réelle** du moteur, jamais supposé.

`RegimeCard` : étiquette de tracé combinée **« BSL · Haut de la veille · <prix> »** /
**« SSL · Bas de la veille · <prix> »** quand une poche coïncide (côté **réel** de la poche +
nom du repère) ; sans poche → **« Haut de la veille · <prix> » seul**. Petite marque factuelle
**« ≈ BSL »** dans la liste avant le clic (`.coin`, neutre, monospace, dim). **Zéro** mot de
classement/importance/cible/priorité (testé).

### Tolérance retenue et justification
**Tolérance = 0,25 × `volatility_detail.recent_avg`** (un quart de l'amplitude moyenne des
bougies récentes = moyenne des True Ranges sur `recent_n` bougies, déjà exposée). Base :
en-deçà d'un quart de bougie, deux niveaux horizontaux se **confondent en une seule ligne** à
l'écran et décrivent le même prix ; au-delà, l'œil les sépare. **Fallback** : si
`volatility_detail` absent → tolérance `null` → **aucune coïncidence affirmée** (conservateur,
jamais d'invention). Un « haut de la veille » n'EST PAS un BSL — c'est une **coïncidence de
niveaux constatée**, pas une équivalence.

### Généralisation
Le geste clic→tracé (via `PxBtn`, RG-1c) couvre déjà bornes Position + niveaux franchis
BOS/CHOCH ; la coïncidence liquidité et la formulation en mots s'appliquent à **tout niveau de
prix** affiché. Les **deltas/moyennes** (distances, ATR moyen) restent en texte : ce ne sont
pas des niveaux de prix.

---

## 3. Tests
- **`reference-coincidence.test.ts`** : tolérance = quart d'amplitude ; `null` sans base ;
  match dans/hors tolérance ; **poche `broken` exclue** ; nearest quand plusieurs ; jamais de
  coïncidence sans base.
- **`rg1-regime.test.tsx`** : formulation en **mots sans signe** (aucun `+`/`−`) ; **étiquette
  combinée BSL uniquement quand une poche détectée correspond** (prevDayHigh 2421,2 = poche
  BSL) et **plaine sinon** (dayOpen loin) ; **marque « ≈ BSL »** dans la liste ; clic→tracé,
  re-clic→retrait, un seul, rien au chargement (canal `referenceLevel`) ; **aucune chaîne de
  classement** (important/majeur/prioritaire/cible).
- Verrou d'id des zones **ni élargi ni contourné** : les repères passent par le canal
  `referenceLevel` distinct.
- **Chart (DÉFAUT 2)** : le rendu lightweight-charts n'est pas monté en jsdom (pas de harness
  canvas) — le **défilement + la marge + le retour d'échelle sont validés en live** ; l'état
  traçable (ligne posée/retirée, un seul, rien au chargement) est couvert via le canal.
- tsc **0**, build **exit 0**, suite frontend verte.

## 4. Discipline
Chemin `referenceLevel` séparé (verrou d'id intact). Staging explicite (pas de `git add -A`),
pas de force push. Merge sur main **après validation live du fondateur** uniquement.

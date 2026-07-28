# AUDIT — VZ-1c · Cadrage des zones : dézoomer nettement

Branche : `feat/vz-1c-cadrage-large` (depuis `origin/main` @ bd40d62, PR #91 VZ-1b).
Frontend, zéro diff moteur. Correctif ciblé du cadrage des ZONES uniquement.

## Pourquoi les seuils VZ-1b n'ont pas produit l'effet attendu

Les 12–30 % de VZ-1b **étaient bien appliqués** — mais **côté VERTICAL uniquement**. Le
défaut « environ 6 bougies visibles » est **HORIZONTAL**, et l'horizontal n'avait **aucun
seuil de pourcentage ni minimum de bougies**.

`frameZone` calculait l'horizontal ainsi (VZ-1/VZ-1b) :
```js
const marginX = Math.max(barSec * 3, (lastSec - startSec) * 0.08);
const from = startSec - marginX;   // bougie de formation
const to   = lastSec  + marginX;   // bougie courante
```
La fenêtre = **[formation → bougie courante]**. Pour une zone **récemment formée**
(`startSec ≈ lastSec`, cas typique d'un FVG frais), cette fenêtre s'effondre à
`~marginX×2 ≈ 6 bougies`. L'horizontal dépendait donc uniquement de l'ÂGE de la zone, sans
plancher. Le vertical à 20 % accentuait l'impression (la bande paraissait haute par rapport
aux ~6 bougies).

**Vérifié** côté API : le graphique **ne s'auto-ajuste pas** quand une sélection est active —
`autoscaleInfoProvider` renvoie la fenêtre verticale cible verbatim (lue depuis
`framingTargetRef`). Les pourcentages verticaux ont donc bien un effet ; le problème n'était
pas un auto-fit, mais l'absence de minimum horizontal.

## Correctif

Tout dans **`webapp/lib/chart/focusController.ts`**, `frameZone` + constantes nommées.

**HORIZONTAL — un minimum de bougies qui prime, ancré sur la bougie courante :**
```js
const barsSinceFormation = Math.max(0, (lastSec - startSec) / barSec);
const visibleBars = clamp(
  barsSinceFormation + ZONE_FORMATION_MARGIN_BARS,
  Math.max(ZONE_MIN_VISIBLE_BARS, ZONE_TARGET_VISIBLE_BARS), // borne basse effective = 100
  ZONE_MAX_VISIBLE_BARS,                                     // 120
);
const from = lastSec - visibleBars * barSec;
const to   = lastSec + ZONE_RIGHT_PAD_BARS * barSec;         // bougie courante toujours visible
```
La fenêtre montre **~100 bougies** (bornes [60, 120]), ancrée sur la bougie courante. Le
minimum **prime sur la taille de la zone** : une zone fraîche ouvre sur un marché complet,
plus jamais ~6 bougies. La bougie de formation est incluse quand elle tombe dans la fenêtre
(sinon la surbrillance la signale hors champ).

**VERTICAL — la zone repérable, pas dominante :**
- `ZONE_OCCUPANCY_TARGET` **0,20 → 0,10** ; bornes `MIN`/`MAX` **[0,12 ; 0,30] → [0,05 ; 0,15]**
  (⇒ ~45 % de marge de chaque côté).
- Le **prix courant** reste dans la vue (repli conservé, plancher = `ZONE_OCCUPANCY_MIN` = 5 %).

## Emplacement UNIQUE des constantes

Toutes en tête de `focusController.ts`, sous le bandeau « ZONE framing », chacune commentée
(à quoi elle sert) :

| Constante | Rôle | Valeur |
|---|---|---|
| `ZONE_OCCUPANCY_TARGET` | part verticale visée par la bande | 0,10 |
| `ZONE_OCCUPANCY_MIN` / `_MAX` | bornes de cette part | 0,05 / 0,15 |
| `ZONE_PRICE_PAD_FRAC` | marge au-delà du prix replié | 0,25 |
| `ZONE_MIN_VISIBLE_BARS` | plancher de bougies (prime) | 60 |
| `ZONE_TARGET_VISIBLE_BARS` | bougies visées | 100 |
| `ZONE_MAX_VISIBLE_BARS` | plafond de bougies | 120 |
| `ZONE_FORMATION_MARGIN_BARS` | bougies gardées à gauche de la formation | 10 |
| `ZONE_RIGHT_PAD_BARS` | espace à droite de la bougie courante | 4 |

Prêtes à réajuster ; aucune valeur de cadrage ailleurs.

## Ce qui ne change pas

`frameEvent` (BOS/CHOCH : bougie de confirmation centrée, ≥20 avant / ≥10 après) et
`frameLevel` (repères, poches) — **intacts**. Reste du comportement VZ-1 : sélection unique,
re-clic/Échap restaurent la vue, transitions animées, `prefers-reduced-motion`, aucune
animation directionnelle, verrou d'identifiant intact.

## Tests

- **Unitaires** (`focusController.test.ts`) : occupation **5–15 %** (cible ~10 %), **≥60 bougies
  même pour une zone fraîche** (le cas qui régressait), bougie courante visible, prix replié,
  plancher 5 %, formation incluse quand dans la fenêtre. `frameEvent`/`frameLevel` inchangés
  (tests existants verts). Suite complète **verte**.
- **Playwright** 1280×800 + 390×844 (`vz-1-focus.spec.ts`) : geste + toggle + Escape + sélection
  unique — verts.
- `tsc --noEmit` **0 erreur**, `next build` **vert**.

## Non testable headless

Le zoom/cadrage réel est peint sur le canvas ; les tests unitaires vérifient la math
(bougies visibles, occupation), l'e2e vérifie le geste (aria-pressed). Validation visuelle du
dézoom à confirmer **en live** sur `/app`.

## Fichiers touchés

`webapp/lib/chart/focusController.ts`, `webapp/lib/chart/__tests__/focusController.test.ts`,
`docs/audits/AUDIT-vz-1c-cadrage-large.md`.

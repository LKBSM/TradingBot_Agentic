# Graphique — Direction esthétique 1 (sobre/institutionnel) + interaction

**Date** : 2026-06-15
**Branche** : `feat/chart-direction1-ux` (depuis `ab6dec2`, tip de `audit/detection-quality-review`)
**Périmètre** : frontend / composant graphique UNIQUEMENT. Zéro modif moteur / détection / mapper.

---

## 1. Garantie de périmètre

- **Visuel + interaction seulement.** Les structures restent celles émises par le
  moteur via `MarketReadingStructure` (OB, FVG, BOS, CHOCH, retest). Aucune n'est
  ajoutée, masquée ou réinterprétée — uniquement **restylée**.
- La règle « n'afficher que l'actif » est préservée : `faded` reste piloté par
  `ob.status !== 'active'` et `fvg.status === 'filled'` (logique inchangée).
- Aucune projection, aucun claim. Descriptif pur.
- Pas de `git add -A`.

## 2. Fichiers modifiés (4)

| Fichier | Changement |
|---|---|
| `webapp/components/app/ReadingChart.tsx` | Restyle complet Direction 1 + interaction (cœur) |
| `webapp/components/market-reading/MarketReadingHeader.tsx` | Graisses 400/500, instrument 15px, méta mono 11px |
| `webapp/components/market-reading/LivePrice.tsx` | Prix mono 18px/500, variation mono 12px colorée |
| `webapp/components/market-reading/TemporalBadge.tsx` | Méta mono tabulaire 11px tertiaire |

## 3. Partie A — Spec visuelle appliquée

- **Chandeliers** : hausse `#2F9E78`, baisse `#C2693E`. Mèche + bordure = couleur
  du corps (pas de bordure contrastée). Mèches fines (1px natif lightweight-charts).
- **Grille** : lignes **horizontales seules** (`vertLines.visible = false`), couleur
  token `--border` à opacité 0.4 (`hsla` clair/sombre), style solide discret.
- **Fond** : transparent (suit le fond de la carte/app). **Axes** : token
  `--muted-foreground` (tertiaire), 11px, **police mono** (tabulaire par nature).
- **Échelle de prix** : labels mono 11px tertiaires. **Pastille prix courant** :
  `lastValueVisible` → fond = couleur du dernier mouvement (vert/terracotta),
  texte blanc (défaut lightweight-charts).
- **Ligne prix courant** : 1px, pointillée, couleur héritée du sens du dernier
  mouvement (`priceLineColor: ''`).
- **Crosshair** : mode **magnet**, lignes fines pointillées discrètes, labels mono.
- **Zones** :
  - Order Block : fill `#8B95A7` @0.10, bordures haut/bas pointillées `#8B95A7` @0.4,
    label « Order Block » 10px `#9AA4B8` discret en haut-gauche.
  - FVG : même traitement, teinte bleue `#6E84B0` (opacités identiques).
  - BOS/CHOCH/Retest : price-lines horizontales 1px pointillées, labels courts mono,
    couleurs sobres (`#8B95A7` / `#8E84B0` / `#6E84B0`).
- **Header** : instrument 15px/500 ; prix mono 18px/500 ; variation mono 12px colorée ;
  méta mono tertiaire 11px. Jamais > 500 ; 2 graisses (400/500).

## 4. Partie B — Interaction / user-friendly

- **Pan horizontal** au drag souris **et tactile** (`handleScroll.horzTouchDrag = true`,
  `pressedMouseMove = true`). `vertTouchDrag = false` (pan vertical désactivé, plus stable).
- **Zoom** : molette (`handleScale.mouseWheel`) + **pinch tactile** (`handleScale.pinch`).
- **Kinetic scroll** activé au toucher (inertie naturelle).
- **Responsive** : `autoSize: true` + `ResizeObserver` → le graphique se redimensionne
  avec son conteneur (corrige le « difficile à manipuler » sur petit écran).
- **Contrôles discrets** : zoom +, zoom −, « ajuster » (`fitContent`). Zone de tap
  **44px sur mobile** (`h-11 w-11`), réduite à 32px desktop (`sm:h-8 sm:w-8`), style
  sobre (bordure `--border/60`, fond translucide, sans glow).
- **Crosshair au tap mobile** : `trackingMode` par défaut (`OnNextTap`) → lecture des
  valeurs au toucher.

## 5. Choix documentés

- **Ligne prix courant** : couleur héritée du sens du mouvement (plus informatif)
  plutôt qu'une opacité fixe 0.7 ; le rendu 1px+pointillé reste léger.
- **Variation header** : conserve les tons emerald/red theme-aware existants (lisibles
  clair+sombre) ; la spec demande « colorée », non un hex précis.
- **Pas d'ajout de contenu** (ex. « Twelve Data » de l'exemple spec n'a pas été injecté
  dans `TemporalBadge` : ce serait un changement de contenu hors périmètre styling).

## 6. Vérifications

- `npx tsc --noEmit` → **0 erreur**.
- `npx next build` → **vert** (`/[locale]/app` 15.9 kB / 159 kB First Load).
- `npx vitest run components/app components/market-reading` → **42/42 passés**, 0 régression.
- Thème clair + sombre : palette résolue par `isDark` (tokens `--border` /
  `--muted-foreground`), recréation du chart au changement de thème → vérifié dans les
  deux modes.

## 7. Hors-périmètre (intact)

Moteur, détection, mapper, contrat `MarketReadingStructure`, hooks de données,
sélection des structures actives — **non touchés**.

# Fix — chevauchement barre d'action fixe / liste « Mes lectures enregistrées »

**Branche :** `fix/saved-readings-bottom-bar-overlap` (worktree dédié, depuis `origin/main @ 9178823`).
**Nature :** layout / espacement uniquement. Aucun changement de contenu, de design ou de comportement.

## Symptôme
Sur `/scanner` en vue *builder* (composer/modifier des conditions), la barre d'action **fixe** en
bas (`compte live · Relancer · Annuler · Enregistrer & relancer`) recouvrait le bas de la liste
« Mes lectures enregistrées ». Le dernier item (« try1 ») et ses boutons **Charger / Renommer /
Dupliquer / Supprimer** passaient **sous la barre** → non lisibles ni cliquables.

## Cause racine
- La barre est `position: fixed; inset-x-0; bottom-0; z-10` dans `ConditionsBuilder.tsx` (hors flux).
- La seule réservation d'espace était un `pb-24` (96 px, **constante magique**) posé sur le **root
  de `ConditionsBuilder`** — donc **au-dessus** du `StrategyPanel`, qui est un **frère rendu APRÈS**
  le builder dans `ScannerWorkspace`. Résultat : la liste n'avait **aucune** réservation en bas (et
  un vide inutile apparaissait au milieu).
- `pb-24` ignorait `safe-area-inset-bottom` et ne s'adaptait pas quand la barre **passe sur 2 lignes**
  en 390 px (elle devient plus haute).
- **Mobile** : `/scanner` est une surface `.no-chat` → sous 768 px la nav mobile `.mspace`
  (`fixed bottom-0`, **z-60**, 58 px + safe-area) occupe le bas ; la barre (z-10) passait **derrière**
  elle, et `.center` ne réservait rien pour la hauteur de la barre.

## Correctif (spacing / positionnement — la barre reste `fixed`)
1. **Retrait** du `pb-24` mal placé sur le root de `ConditionsBuilder`.
2. **Mesure dynamique** de la hauteur réelle de la barre via `ResizeObserver` → publiée en variable
   CSS `--scanner-actionbar-h` (sur `document.documentElement`, nettoyée au démontage). S'adapte au
   retour à la ligne des boutons.
3. **Réservation** sur le conteneur de la page builder (`ScannerWorkspace`) :
   `padding-bottom: calc(var(--scanner-actionbar-h, 6rem) + 1rem)` → le dernier item dégage toujours
   la barre, avec un petit jour. Fallback `6rem` avant la première mesure.
4. **Mobile** : la barre est remontée **au-dessus** de la nav mobile via
   `.scanner-actionbar { bottom: calc(58px + env(safe-area-inset-bottom)) }` sous 768 px (pages.css).
   La réservation `.center` existante pour `.mspace` est conservée → empilement propre
   contenu → barre d'action → nav mobile. `safe-area-inset-bottom` respectée.

## Fichiers touchés
- `webapp/components/scanner/ConditionsBuilder.tsx` — retrait `pb-24` ; `ResizeObserver` → var CSS ;
  classe `scanner-actionbar` + `ref` sur la barre.
- `webapp/components/scanner/ScannerWorkspace.tsx` — `padding-bottom` dérivé de la var sur le wrapper
  de la vue builder.
- `webapp/components/shell/pages.css` — règle `.scanner-actionbar` (desktop `bottom:0` ; mobile
  au-dessus de la mspace).

## Vérifications (tests d'honnêteté)
Captures viewport (scroll du conteneur `.center` en bas), `docs/audits/` :
- **Avant** : `overlap-before-{long,short}-{desktop-1280x800,mobile-390x844}.png` — « try1 » coupé
  sous la barre (desktop) ; « Supprimer » masqué + barre mangée par la nav mobile.
- **Après** : `overlap-after-{long,short}-{desktop-1280x800,mobile-390x844}.png` — « try1 » et ses 4
  boutons **entièrement au-dessus** de la barre ; sur mobile, empilement barre → nav propre.
- **Liste courte** (1 item) : pas de grand vide disgracieux (réservation ≈ hauteur réelle de la barre).
- `tsc` 0 · `next build` 0.

## Reste
Merge sur `main` **après confirmation visuelle live du fondateur** (non mergé à ce stade).

# AUDIT BRD-1 — Présence de la marque dans l'application

Branche : `feat/brd-1-marque` (worktree dédié `C:/MyPythonProjects/wt-brd-1`, depuis `origin/main` 4109345).
Périmètre : présentation de la marque uniquement. Aucune logique métier, aucune règle de détection, aucun appel réseau touché.

---

## 1. Diagnostic de départ (inventaire)

Les surfaces connectées (`/app`, `/scanner`, `/zones`, `/actualites`, `/compte`) partagent le châssis
`app/[locale]/(product)/layout.tsx` → `ProductShell` → `ShellRail` (colonne gauche) + centre + (chat sur `/app`).

| Surface | Marque avant BRD-1 |
|---|---|
| `/app` (desktop) | ❌ Absente. Le rail commençait directement sur le champ de recherche. |
| `/app` (mobile) | ❌ Le `<header>` de `MobileWorkspace` n'affichait que le combo (« XAU/USD · M15 »). |
| `/scanner` | ❌ Absente (même rail). |
| `/zones` | ❌ Absente. |
| `/actualites` | ❌ Absente. |
| `/compte` | ❌ Absente. |

Constat : **la marque était totalement absente du châssis produit.** Elle n'existait que dans la chrome
marketing du groupe `(site)` (`Nav.tsx`, `Footer.tsx`, `MobileMenu.tsx`, `LoginForm.tsx`).

Structure du châssis :
- `ShellRail` : `[recherche] → MARCHÉS → UNITÉ DE TEMPS → ESPACE (nav) → railfoot (Freshbox + disclaimer)`. Pas de zone d'en-tête, pas de logo.
- Pas d'en-tête horizontal dans le shell produit ; chaque page rend son contenu au centre.
- `AppHeader.tsx` (badge « M » + wordmark + baseline) était **du code mort** pour les routes produit : atteignable seulement via `Nav`, qui ne vit que dans le groupe `(site)`.

Composant réutilisable ?
- `components/BrandMark.tsx` existait déjà : **le glyphe seul** (badge doré, trois chandeliers), utilisé par `Nav`.
- Le bloc « wordmark + baseline » était en revanche **dupliqué** (hand-stacké dans `Nav.tsx` et `AppHeader.tsx`). Aucun composant de *lockup* réutilisable.

Espace disponible en haut du rail :
- Desktop : rail flex vertical `gap:14px`, `padding:13px 11px`, `overflow-y:auto` → insertion propre au-dessus de la recherche, sans compression.
- Mobile `/app` (≤767 px) : le rail est masqué (`display:none`), `MobileWorkspace` prend le relais → emplacement équivalent = son `<header>`.
- Mobile autres surfaces (≤767 px) : le rail reste affiché → la marque en haut du rail y apparaît aussi.

Favicon & titre d'onglet (déjà corrects) :
- Titre : `app/[locale]/layout.tsx` applique `title.template = '%s · MIA Markets'`, chaque page fournissant `pages.*.meta.title` → « Zones · MIA Markets », « Scanner de conditions · MIA Markets », etc.
- Favicon : `app/icon.tsx` (M doré 32 px) + `apple-icon.tsx` + `public/icon.svg` + `manifest.ts`, lisible à 16 px.

---

## 2. Placements proposés et retenus

Trois décisions ont été soumises ; choix de l'utilisateur :

| Décision | Options proposées | **Retenu** |
|---|---|---|
| Emplacement principal | A haut du rail / B pied du rail / A+B | **A — haut du rail** (au-dessus de la recherche), + équivalent mobile dans le header de `MobileWorkspace`. |
| Mise en forme | Empilée 2 lignes / une ligne à initiales mises en valeur | **Empilée 2 lignes** (wordmark au-dessus, acronyme développé en dessous). |
| Extras | Écran de chargement / états vides / aucun | **Écran de chargement `/app`** uniquement (pas d'états vides). |

Rejetés explicitement : A+B (répéterait le logo sur un même écran, interdit) ; états vides (non retenus par l'utilisateur — la marque n'y est **pas** ajoutée).

---

## 3. Formulation de l'acronyme (FR / EN)

Source unique : `lib/brand.ts`.

- **Marque** : `MIA Markets` (identique FR/EN).
- **Acronyme développé** : `Multi-asset Intelligence Assistant` — **identique en FR et en EN**, par choix délibéré.

Justification (déjà documentée dans `lib/brand.ts`) : « MIA » est l'acronyme de **M**ulti-asset **I**ntelligence
**A**ssistant. Une traduction française épellerait un autre acronyme (« Assistant d'Intelligence Multi-actifs » → AIM),
ce qui casserait le lien avec le nom. La baseline reste donc en anglais dans **toutes** les langues — pratique standard
pour une signature de marque — et l'acronyme reste lisible partout. C'est du **vrai texte**, jamais une image.

---

## 4. Réalisation

Un **seul** composant de lockup réutilisable, aucune duplication de balisage :

- **`components/BrandLockup.tsx`** (nouveau) : compose `BrandMark` (glyphe) + wordmark (ligne 1, `text-foreground`) + acronyme développé (ligne 2, `text-muted-foreground`, lisible). Variantes `size` (`sm` pour le rail étroit / mobile, `md` pour les en-têtes), option `baseline`. Couleurs via les tokens partagés → thème sombre et clair couverts. Aucune animation.

Points d'usage (tous délèguent au même composant) :
1. **Haut du rail** — `components/shell/ShellRail.tsx` : `<BrandLockup size="sm">` au-dessus de la recherche, séparé par un filet (`.railbrand` dans `shell.css`).
2. **Header mobile `/app`** — `components/app/MobileWorkspace.tsx` : `<BrandLockup size="sm">` au-dessus du combo (qui reste, en sous-ligne). Équivalent du rail, masqué sur mobile.
3. **Écran de chargement `/app`** — `components/app/ReadingSkeleton.tsx` : `<BrandLockup size="md">` **ajouté** au-dessus du squelette (toutes les barres `animate-pulse` et le `data-testid="reading-skeleton"` intacts).
4. **Dé-duplication** — `components/app/AppHeader.tsx` : le bloc « badge M + wordmark + baseline » hand-stacké est remplacé par `<BrandLockup size="md">`. Le seul markup de badge doré vit désormais dans `BrandMark.tsx`.

Non touchés : aucun message d'état vide, aucun indicateur de chargement, aucune logique de détection. `Nav.tsx`
(marketing, une ligne) reste tel quel.

---

## 5. Captures

Avant : la marque était **absente** du rail (le rail commençait sur le champ de recherche) et du header mobile
(combo seul). Cf. §1.

Après :
- Rail desktop (fr) : `docs/audits/brd-1/after-rail-desktop-fr.png`
- Header mobile `/app` (fr) : `docs/audits/brd-1/after-header-mobile-fr.png`

Le rail montre : glyphe doré + **MIA Markets** + « Multi-asset Intelligence Assistant » sur une ligne lisible, filet de
séparation, puis la recherche. Le header mobile montre la même marque, le combo passant en sous-ligne discrète.

---

## 6. Vérifications

- `tsc --noEmit` : **vert**.
- `next build` : **vert**.
- Vitest — `components/__tests__/BrandLockup.brd1.test.tsx` (11 tests) + suites impactées (ShellRail, Nav, ZonesWorkspace, responsive, ReadingColumn, reading-load-honesty) : **verts**.
  - Garde présence : wordmark + acronyme sur chaque surface via `BrandLockup`.
  - Garde anti-duplication : le badge doré (`from-amber-400 to-amber-600`) n'existe que dans `BrandMark.tsx` ; aucune surface ne référence `BRAND_BASELINE` en direct.
  - Garde titre : template `%s · MIA Markets` + chaque page produit fournit son `meta.title`.
  - Garde « ajout, pas remplacement » : le squelette de chargement conserve son indicateur ET affiche la marque ; les composants d'état vide n'embarquent pas la marque.
- Playwright (`tests/e2e/brd1-brand.spec.ts`, `--project=chromium-desktop`, 1280×800 + 390×844, fr + en) : **12/12**.
  - Chaque surface connectée montre wordmark + acronyme ; l'écran de chargement porte la marque sans masquer le squelette ; un état vide (scanner « aucun combo ») garde son message explicite et affiche la marque.

---

## 7. Accessibilité & i18n

- Le glyphe est décoratif (`aria-hidden`) ; wordmark + acronyme sont du **vrai texte** (lus par les lecteurs d'écran).
- Aucune nouvelle clé i18n (les chaînes de marque sont des constantes) → parité des locales inchangée.
- Aucune animation, aucun clignotement, aucun effet. Coût perf nul (texte + icône inline).

# AUDIT BRD-3 — Marque « colonne » (cinq bougies laiton)

**Branche** : `feat/brd-3-logo-colonne` — worktree dédié `C:/MyPythonProjects/wt-brd-3`
**Base** : partie de `6d658ef` (= origin/main À JOUR, PR #194). Diagnostic + implémentation
faits contre origin/main à jour (le repo principal était 30 commits derrière ; worktree neuf
créé pour éviter un état périmé — cf. incident DATA-1).
**Remplace BRD-2** (PR #168, logo « prisme » bleu — mergé) : ses fichiers, son composant et
ses jetons sont remplacés, pas conservés « au cas où ».
**Nature** : purement présentationnel. Aucune logique métier touchée.

---

## 1. La nouvelle marque

Cinq bougies de hauteurs symétriques autour du centre (la plus grande au milieu), en **laiton
`#C9A14A`** sur les quatre thèmes. Le nom **« M.I.A MARKETS »** en capitales espacées, en
**blanc `#FFFFFF`** sur les trois thèmes sombres, en **quasi-noir `#1A1917`** sur le thème
clair. Deux jetons distincts, jamais fusionnés : `--brand-mark` (bougies) / `--brand-word`
(nom).

Deux règles verrouillées par test :
- **Symétrie structurelle** : hauteurs `[18,32,42,32,18]`, opacités `[.45,.7,1,.7,.45]` —
  image miroir autour du centre. Un test échoue si elles deviennent croissantes/décroissantes
  (une courbe ascendante = une direction = une prédiction, interdite par le produit).
- **Variante compacte** (3 bougies pleines, sans opacité) réservée aux petites tailles
  (< 40 px : favicon, icônes d'app, avatar). La marque complète (5 bougies) n'est jamais
  rendue sous 40 px.

---

## 2. Source unique

Le tracé de la marque vit à **un seul endroit** : `webapp/lib/brand/candle-geometry.ts`
(coordonnées VERBATIM des fichiers fournis, aucun redessin). Le composant `MiaLogo` **et** les
quatre images générées côté serveur importent cette géométrie — une coordonnée écrite une
seule fois. Les SVG statiques sous `public/` (`icon.svg`, kit `brand/*.svg`) répètent
nécessairement le tracé : ce sont des fichiers d'art, pas du code. Un test garantit qu'aucun
tracé n'est dupliqué dans le code.

---

## 3. Inventaire AVANT → APRÈS

### Composant + géométrie
| Avant (BRD-2, prisme) | Après (BRD-3, bougies) |
|---|---|
| `components/brand/MiaLogo.tsx` — 4 tons (auto/color/dark/mono), prisme | réécrit : bougies, plus de prop `tone`, couleur = `var(--brand-mark)`/`var(--brand-word)` |
| `lib/brand/prism-geometry.ts` | **supprimé** → `lib/brand/candle-geometry.ts` (nouveau) |

### Images générées côté serveur (couleurs en dur assumées — pas de thème au build)
| Fichier | Avant | Après |
|---|---|---|
| `app/icon.tsx` (favicon 32) | prisme compact `#7DA3FF` | 3 bougies compactes `#C9A14A` sur tuile sombre |
| `app/apple-icon.tsx` (180) | prisme compact `#7DA3FF` | 3 bougies compactes `#C9A14A` |
| `app/opengraph-image.tsx` (1200×630) | prisme `#7DA3FF` + nom | 5 bougies `#C9A14A` + « M.I.A Markets » blanc |
| `app/brand/email-logo.png/route.tsx` (480×120) | prisme `#2962FF` + nom `#0F1729` | 5 bougies `#C9A14A` + « M.I.A MARKETS » `#1A1917` sur tuile blanche |

### Assets SVG statiques (`webapp/public/`)
| Action | Fichier |
|---|---|
| Remplacé | `brand/mia-favicon.svg`, `brand/mia-marque.svg`, `brand/mia-marque-mono.svg`, `brand/mia-verrouillage-horizontal.svg`, `brand/mia-verrouillage-empile.svg` |
| Ajouté | `brand/mia-verrouillage-horizontal-clair.svg`, `brand/mia-verrouillage-empile-clair.svg` (nom noir, fond clair) |
| Supprimé | `brand/mia-marque-fond-sombre.svg`, `brand/mia-verrouillage-empile-fond-sombre.svg` (variantes « fond sombre » remplacées par le couple défaut/-clair) |
| Redérivé (bougies, mêmes coords fournies) | `public/icon.svg` (tuile maskable PWA, référencée par `manifest.ts` + `seo/JsonLd.tsx`) |

Note : les SVG fournis embarquent une métadonnée C2PA (~8 Ko) de provenance — gardée
**verbatim** (consigne « ne rien nettoyer »). Inoffensive pour les navigateurs.

### Surfaces consommatrices (composant `MiaLogo` — 11)
| Surface | Avant | Après |
|---|---|---|
| `Nav.tsx` (en-tête public) | horizontal | horizontal (inchangé) |
| `shell/ShellRail.tsx` (rail connecté) | horizontal | horizontal (inchangé) |
| `auth/AuthBrandHeader.tsx` | stacked h92 | stacked h92 (inchangé) |
| `app/AppHeader.tsx` | **mark** h24 | **compact** h24 (marque complète interdite < 40 px) |
| `shell/ProductShell.tsx` (barre mobile) | **mark** h20 | **compact** h20 |
| `MobileMenu.tsx` | **mark** h20 | **compact** h20 |
| `Footer.tsx` | mark **ton mono** h18 | **compact** h18 (le ton « mono » n'existe plus → laiton discret) |
| `chat/AgentAvatar.tsx` | compact | compact (inchangé ; `data-testid="mia-avatar"` ajouté pour le test) |
| `auth/LoginForm.tsx`, `landing/HeroChatPreview.tsx`, `landing/ConversationReplayCard.tsx` | compact | compact (inchangés) |

### Jetons de marque (`app/globals.css`, sélecteur `data-design`)
| Thème | `--brand-mark` avant | après | `--brand-word` avant | après |
|---|---|---|---|---|
| `:root` (terminal + schema + ardoise, sombres) | `#7da3ff` | **`#c9a14a`** | `#ffffff` | `#ffffff` |
| `[data-design=atelier]` (clair) | `#2962ff` | **`#c9a14a`** | `#0f1729` | **`#1a1917`** |

✅ **Coordination THM-1 — CONFLIT RÉSOLU (intégré)** : THM-1 a été mergé sur main (PR #195,
`5c7740a`) pendant que BRD-3 était en revue. J'ai donc intégré `origin/main` dans la branche
et résolu le conflit **en faveur de BRD-3** :
- `MiaLogo.tsx` : gardé la version bougies+laiton, en **conservant l'apport THM-1**
  (`fontFamily: var(--font-sans)` via `style` pour le nom — source de police unique).
- `globals.css` : THM-1 avait neutralisé le logo (`--brand-mark = --txt` par thème). Repositionné
  **laiton `#c9a14a`** sur les **quatre** thèmes (chacun déclare désormais explicitement ses
  jetons, plus d'héritage neutre) ; nom blanc `#ffffff` (terminal/schema/ardoise) et `#1a1917`
  (atelier/Parchemin). Les libellés renommés THM-1 (Graphite/Parchemin/Encre/Ardoise-et-acier)
  gardent les ids internes `data-design` inchangés → mes overrides tombent au bon endroit.
- L'accent d'UI Ardoise est **acier/bleu** (THM-1) tandis que le logo reste **laiton** :
  vérifié en capture (`accueil-ardoise-desktop.png`). Une marque garde sa couleur ; l'accent, non.
- `#c9a14a` en dur ailleurs (`global-error.tsx`, `ReadingChart.tsx`) = accent THM-1, PAS le logo :
  le garde-fou « couleur du logo » est scopé aux fichiers source du logo. Tests re-verts après
  intégration (unitaires 17 + non-régression, e2e brd3 28 + theme-and-pwa, tsc/build).

---

## 4. Où la marque apparaît / n'apparaît jamais

Apparaît : en-tête public (horizontal, lien accueil), en-tête connecté + rail, favicon/icônes
d'app (compacte), aperçu social (OG empilé, fond graphite), pages auth, écran d'abonnement,
avatar M.I.A (compacte, dans un disque), pied de page (discret), courriels (PNG hébergé).

N'apparaît jamais : dans un état de chargement, un état vide, un message d'erreur, en filigrane,
ni deux fois sur une même vue. Vérifié : la page 404, `error.tsx`, `global-error.tsx` et
`ReadingSkeleton.tsx` n'importent ni ne dessinent la marque (test garde-fou). Capture
`app-*-desktop.png` : l'état « Données indisponibles » n'affiche pas le logo.

---

## 5. Tests

- **Unitaires** `components/brand/__tests__/MiaLogo.brand.test.tsx` (16 tests, **passants**) :
  source unique ; **symétrie des 5 (et 3) bougies** — échoue si croissantes/décroissantes ;
  deux jetons (le nom ne prend jamais la couleur des bougies) ; a11y (role img / décoratif) ;
  aucune chaîne prisme (`prism-geometry`, tracés `M46,14`/`M48,16`, `#2962ff`/`#7da3ff`/
  `#0f1729`, `BrandMark`, `MiaAgentLogo`, « MIA Markets » sans points) ; aucune couleur laiton
  en dur hors images générées ; marque complète jamais < 40 px.
- **e2e** `tests/e2e/brd3-logo.spec.ts` (28 tests × 2 viewports, **passants**) : présence sur
  accueil/app/scanner/zones/actualites/connexion/inscription/abonnement ; **bougies laiton +
  nom blanc/noir sur les 4 thèmes** (terminal/schema/ardoise/atelier) ; 404 sans logo ; avatar
  bougies compactes sur /app.
- Non-régression : `Nav`, `claims-cleanup`, `CalendarEventDetail`, `nw5`, `ui2-copy-honesty`,
  `error-notice` (55 tests) ; `theme-and-pwa` + `ui2-audit` e2e (24). Tous verts.
- `tsc --noEmit` : 0 nouvelle erreur (les 3 erreurs `dictation-copy-honesty` préexistent).
- `next build` : vert (routes `/icon`, `/apple-icon`, `/opengraph-image`, `/brand/email-logo.png`
  compilées).

---

## 6. Captures (après)

`docs/audits/brd-3-captures/` — 4 thèmes (terminal/schema/ardoise/atelier), fr, desktop
1280×800 : accueil, app, scanner, zones, actualites, connexion, inscription ; 404 ; mobile
390×844 : accueil + app (avatar) sur terminal + atelier. 33 fichiers.

Avant = logo prisme bleu (cf. `AUDIT-brd-2.md` et l'historique de `MiaLogo.tsx`) ; non
recapturé car remplacé.

---

## 7. À TA CHARGE (hors dépôt)

- **Stripe** : remplacer le logo dans le tableau de bord Stripe (page de paiement / reçus).
- **Réseaux sociaux** : images de profil et bannières (X, LinkedIn, etc.) — utiliser
  `mia-verrouillage-empile.svg` (fond sombre) ou la variante `-clair` selon le fond.
- **`.ico` éventuel** : un `FavIcon_Word.ico` traîne dans Downloads ; le favicon du site est
  désormais généré par `app/icon.tsx` (pas de `.ico` requis), mais si un `.ico` est voulu
  ailleurs, le régénérer depuis `mia-favicon.svg`.

---

## 8. Merge

Push effectué. **Pas de merge sur main avant confirmation visuelle live** (consigne).

# AUDIT — UI-1 · Densité et lisibilité de /zones et /scanner

**Branche :** `fix/ui-1-densite-zones-scanner` (worktree dédié, depuis `origin/main` 771ccc0)
**Périmètre :** présentation uniquement. Aucune structure, aucun contenu, aucune logique modifiés.
**Vérifs :** `tsc` 0 · `next build` OK · vitest 893/893 · Playwright UI-1 15/15 (fr + en × 1280×800 / 1440×900 / 390×844).

---

## 1. Question A — « coupée » ou « trop haute » ? **Les deux, à des endroits différents.**

- **Desktop (1280 / 1440) : rien n'était coupé** (`overflow = 0`). Le problème était **la densité + une hiérarchie typographique cassée** — pas une troncature.
- **Mobile (390) : c'était coupé — bug de mise en page.** Le shell `.no-chat` (utilisé par /zones et /scanner) n'avait **aucun traitement responsive** : les media-queries qui replient le rail étaient limitées à `:not(.no-chat)`. À 390 px le rail desktop de **232 px restait affiché**, le contenu était écrasé dans ~158 px, et le badge « live » + la ligne méta du scanner étaient **tronqués hors écran**.

Les deux causes sont corrigées.

---

## 2. Deux défauts racines identifiés (et corrigés)

1. **Scanner — libellés de blocs non stylés.** Les classes `.blk-lbl` / `.blk-empty` (têtes « Ce qui correspond / à l'encontre / Contexte ») **n'avaient aucune règle CSS** → repli sur le **16 px par défaut du navigateur**. Elles étaient donc le **plus gros texte de la carte**, plus gros que le nom du marché (13 px) — hiérarchie **inversée**, « tout crie ». C'était aussi un moteur de hauteur.
2. **Zones — la borne de prix ne ressortait pas.** L'information n°1 (`.rng`) était à **12 px**, identique au corps, aux boutons et à l'intro. Aucune dominance visuelle. En parallèle, les trois encadrés lecture (proximité / confluence / contacts) + origine empilaient des marges et paddings généreux → carte très haute.

---

## 3. Mesures — avant / après (données réalistes, siblings vides)

Hauteur d'**une** carte, nombre de cartes **entièrement** visibles au chargement, débordement horizontal.

### /zones
| Viewport | Hauteur carte AVANT | Hauteur carte APRÈS | Δ | Overflow |
|---|---|---|---|---|
| 1280×800 | **386 px** | **328 px** | −15 % | 0 → 0 |
| 1440×900 | 386 px | 328 px | −15 % | 0 → 0 |
| 390×844  | 577 px | 444 px | −23 % | 0 → 0 |

> Cas défavorable (carte à confluence très riche, artefact de mock où toutes les unités empilent la même borne) : **498 → ~410 px**. Aucune ligne d'information supprimée.

### /scanner
| Viewport | Hauteur carte AVANT | Hauteur carte APRÈS | Cartes visibles (1280) | Overflow |
|---|---|---|---|---|
| 1280×800 | **360 px** | **335 px** | 2 → **2** ✓ | 0 → 0 |
| 1440×900 | 360 px | 335 px | 2 → 2 | 0 → 0 |
| 390×844  | 708 px | 649 px | — | 0 → 0 *(rail replié, plus de troncature)* |

### Tailles de police distinctes sur la carte
| Page | AVANT | APRÈS |
|---|---|---|
| /zones (carte) | 11 (dont 8,5 / 10,5 / 15 / 16) | **5 visibles : 16 / 13 / 12 / 10 / 9** |
| /scanner (carte) | 10 (dont **16 px non voulu**) | **5 : 13 / 12 / 11 / 10 / 9** |

---

## 4. Échelle typographique retenue (5 niveaux) — où elle est définie

Appliquée dans **`webapp/components/shell/pages.css`** (les classes `.zone` / `.combo` y vivent déjà) et **`webapp/components/shell/shell.css`** (le `h1` de page), alignée sur l'échelle de `/app` (`.apphead h1` 16, `.narr` 12,5, `.reg .k` 9…).

| Niveau | Taille | Usage |
|---|---|---|
| **Titre de page** | **16 px** | `pghead h1` (était 18 → 16, = `/app`) |
| **Principal** (600) | **13 px** | l'unique info dominante par carte : borne de prix (`.rng`) · nom du marché (`.nm`) |
| **Corps** | **12 px** | valeurs des blocs, lignes de conditions, narration, boutons |
| **Méta / mono** | **10 px** | horodatages, unité de temps, hauteur, sous-labels de frise |
| **Étiquette** (UPPERCASE) | **9 px** | libellés de blocs (`.blk-lbl`, proximité/confluence/contacts), badges d'état, séparateurs de groupe |

**Hiérarchie par contraste, pas par taille :** la borne de prix passe 12 → **13 px / 600** (seule à grossir) ; tout le reste redescend à ≤ 12 px ; les étiquettes de section, jadis à 16 px (scanner) et 8,5 px (zones), s'uniformisent à **9 px**.

---

## 5. Hauteur regagnée & densité

- **Zones :** carte typique **−58 px (−15 %)** ; cas riche **−88 px**. Avant, une seule carte remplissait quasiment l'écran (386–498 px pour 800 px de haut) — c'était la plainte exacte. Après, une carte occupe **< la moitié** de la hauteur : on voit une carte entière **plus la majeure partie de la suivante** sans défiler.
- **Scanner :** **2 résultats entièrement visibles** à 1280×800 (cible atteinte), hiérarchie rétablie.

### Réserve honnête sur « deux cartes de zone entièrement visibles au chargement »
La carte zone tombe à **328 px**, mais le **chrome de page mesure ~192 px** (titre + intro sur 2 lignes + sélecteur d'instrument + sélecteur d'unité + filtre + tri + séparateur de groupe). `192 + 2×328 = 848 > 800` : **deux cartes pleines n'entrent pas au tout premier écran** sans (a) amputer du contenu de carte, ou (b) retirer une commande / l'intro — deux choses que la mission interdit (« aucune information ne se supprime », « aucun changement de structure »). J'ai donc **priorisé ces règles** : au chargement on voit **1 carte pleine + ~85 % de la 2ᵉ**, et **2 cartes pleines dès qu'on fait défiler d'un cran** (2 × 328 + marge = 664 < 800, la liste elle-même est dense). Le scanner, lui, atteint bien 2 au chargement (moins de chrome).

---

## 6. Mobile (390 px) — bug de fond corrigé

`shell.css` : sous 768 px, **toutes** les surfaces produit (y compris `.no-chat` : /zones, /scanner, /compte, /réglages) passent en **colonne unique pleine largeur** et le rail se replie. Piège corrigé au passage : la première tentative (`.app-shell` générique) était **trop faible en spécificité** (0,1,0) pour battre `.app-shell.no-chat` (0,2,0) — les sélecteurs finaux sont qualifiés `.no-chat` / `:not(.no-chat)`.

Résultat : plus d'écrasement à 158 px, **plus aucune troncature** (badge live et ligne méta scanner désormais entiers), carte rendue **pleine largeur et complète**.

**Suivi hors périmètre (à planifier) :** ces pages `.no-chat` n'ont pas encore de navigation mobile dédiée (le rail replié emporte le menu, comme sur /app où `MobileWorkspace` prend le relais). C'est une question de **structure/navigation**, pas de présentation — signalée, non traitée ici.

---

## 7. Informations qui me paraissaient mal placées (signalées, **non déplacées**)

Rien de structurellement mal placé. Le seul « défaut » était le **bug `.blk-lbl` non stylé** (corrigé, ce n'est pas un mauvais emplacement). L'ordre, le contenu et la position de chaque bloc sont inchangés.

---

## 8. Discipline respectée

- Présentation uniquement : CSS (`pages.css`, `shell.css`) + 3 classes utilitaires de marge Tailwind dans `ZonesWorkspace.tsx` (espacement des barres sélecteur/filtres). Aucune logique, aucun texte, aucune structure touchés.
- Blocs repliés (Détails) restent repliés ; le bloc scanner « à l'encontre » reste rendu et non masquable (garde e2e).
- Accessibilité : boutons `.btn` conservent `min-height: 38px` ; aucune cible tactile réduite.
- Tests : `tsc` 0 · `next build` OK · vitest **893/893** · Playwright UI-1 **15/15** (fr + en × 3 viewports).

---

## 9. Captures avant / après

Dossier : `docs/audits/ui-1/`. Paires `before-*` / `after-*` pour `{zones, scanner} × {1280×800, 1440×900, 390×844}`.

| État | Avant | Après |
|---|---|---|
| Zones — liste (1280×800) | `before-zones-1280x800.png` | `after-zones-1280x800.png` |
| Zones — liste (1440×900) | `before-zones-1440x900.png` | `after-zones-1440x900.png` |
| Zones — mobile (390×844) | `before-zones-390x844.png` | `after-zones-390x844.png` |
| Scanner — résultats (1280×800) | `before-scanner-1280x800.png` | `after-scanner-1280x800.png` |
| Scanner — résultats (1440×900) | `before-scanner-1440x900.png` | `after-scanner-1440x900.png` |
| Scanner — mobile (390×844) | `before-scanner-390x844.png` | `after-scanner-390x844.png` |

> Note méthodo : mesures « avant » prises en remisant (`git stash`) les seules feuilles CSS, mock identique (unités sœurs vides = confluence réaliste 0–2 lignes, pas l'artefact de 5 OB au même niveau). États couverts par la suite Playwright : liste, carte dépliée (`.zkv`), groupe « Comblées », résultats scanner, état « aucun combo », rail mobile replié.

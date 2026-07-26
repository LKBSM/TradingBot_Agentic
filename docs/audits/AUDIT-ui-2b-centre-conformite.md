# AUDIT — UI-2b · Colonne centre : conformité stricte à la référence desktop

**Branche** `feat/ui-2b-centre-conformite` (worktree dédié, depuis `origin/main` bc2cbfc). **Zéro diff backend.**
**Statut** `tsc` 0 · **522 tests** verts (+2 nouveaux ; 1 flake connu `claims-cleanup «CM2C»` sous parallélisme complet, pré-existant, passe en isolation) · build Next vert · **5 smokes e2e verts** · vérif visuelle 1280/1024/800/720 conforme. Merge après confirmation live.

## Cause racine (écarts #1/#2/#5/#6)
Le centre à onglets « Marchés/Lecture/Chat » que tu voyais était la **mise en page tablette** (`MobileWorkspace`) : elle s'affichait **sous 1280px** (`useStackedLayout = (max-width:1279px)`), et le shell UI-1 (`shell.css`) n'avait **aucun `@media`** → grille 3 colonnes figée, centre écrasé. Le centre référence (`DesktopReading`, livré en UI-2) n'apparaissait qu'≥1280px. Un seul correctif — abaisser la bascule et rendre le shell responsive — résout #1 (onglets), #2 (panneaux empilés), #5 (en-tête dupliqué) et #6 (pills), car tous étaient le même symptôme.

## A) Structure du centre — bascule + shell responsive
| Élément | Traitement |
|---|---|
| Bascule onglets → centre référence | `use-media-query.ts` : `STACKED_QUERY` **`(max-width:767px)`**. `MobileWorkspace` (onglets) **uniquement <768px** ; `DesktopReading` (colonne unique référence) **≥768px**. |
| Shell responsive (`shell.css`, nouveaux `@media`, scopés `:not(.no-chat)` → Scanner/Zones/Réglages intacts) | **≥1280** rail · centre · chat **docké** · **768–1279** rail · centre ; le chat docké devient un **tiroir off-canvas** ouvert par un bouton flottant `.chat-fab` (+ fond `.chat-backdrop`, fermeture clic-dehors/Échap) · **<768** centre plein écran, rail + chat masqués (MobileWorkspace prend le relais). |
| `ShellChat` | Une seule instance d'`AppChatSidebar` (logique scroll/streaming **intouchée**) ; état `open` + FAB + backdrop, piloté par CSS. Nouvelle clé `app.chat.openPanel` (9 locales). |
| `.panels` | 2 colonnes → **1 colonne** sous 1100px (centre étroit) via `@media` dans `pages.css`. |
| Panneaux, pills, legalbar, en-tête unique | **Déjà fournis par `DesktopReading` (UI-2)** — réutilisés tels quels : en-tête unique (prix flash), legalbar, pills OB·FVG·Liquidité·BOS/CHOCH·Mitigées, 5 panneaux empilés (Lecture narrée « ANCRÉE AU MOTEUR » pleine largeur, Régime, Structure, Liquidité 3 états, Actus « SANS DIRECTION »). Vérifiés visuellement à 1024/800 (données mock). |

Aucun panneau sans source : les états vides honnêtes (Actus/Liquidité) existaient déjà.

## B) i18n & libellés
| Correctif | Détail |
|---|---|
| **#3 clé brute `legal.earlyAccessBadge`** | Le composant `EarlyAccessBadge` lit `useTranslations('legal').t('earlyAccessBadge')` — clé **absente des 9 locales** → chemin brut affiché. Corrigé en **ajoutant la clé** (9 locales, valeur propre) → **zéro changement de composant**. |
| **Fuite de conformité découverte** | `footer.earlyAccessBadge` promettait encore **« · 50 seats/plazas/… »** (claim de rareté banni) dans les **8 locales non-FR** (le FR avait été nettoyé, pas les autres — le test `claims-cleanup` ne scanne que le littéral FR « 50 places »). **Aligné sur la valeur propre.** |
| **#4 nav** | `nav.account` : « Compte » → **« Réglages » (fr) / « Settings » (en)** + 7 autres locales. |
| **#4 langue (SIGNALÉ, non modifié)** | `middleware.ts` a `localeDetection:true` → un navigateur anglais est servi en `/en` (d'où les libellés EN). Décision fondateur : **détection navigateur conservée**. Fichier si tu changes d'avis : `webapp/middleware.ts` (`localeDetection:false`). |
| Test | `ui2b-i18n-keys.test.tsx` : échoue si un nœud texte rendu correspond au motif d'une clé i18n non résolue (`^[a-z]…(\.[a-zA-Z]…)+$`) sur la surface App, + garde des 4 clés touchées + absence de claim chiffré. |

## C) Chart — affichage uniquement (aucune détection modifiée)
- **Tokens** : `palette()` lit désormais les tokens **littéraux de la référence** — bougies `--bull`/`--bear`, grille `--line`, bord d'échelle `--line-2`, axe + croix `--faint`, **ligne de prix `--acc`** (au lieu des `--sentinel-*`/`--border`). Fond `--panel` via `.chartbox`.
- **Échelle verticale** : `rightPriceScale.scaleMargins = { top:0.1, bottom:0.1 }` (défaut lib = 20 %/10 %). Les zones/liquidité étant des **overlays HTML** (jamais dans l'autoscale de l'axe prix), la plage = amplitude des bougies visibles → **les bougies remplissent le cadre avec ~10 % de marge**, indépendamment du range. Viewport initial déjà droit-ancré (stable, indépendant du nombre de bougies) — inchangé.
- **Étiquettes BSL/SSL/LIVE (bord droit)** : légère densité résiduelle près du prix quand plusieurs niveaux sont proches. Ce sont des overlays HTML positionnés par prix ; la **dé-collision** (regroupement/décalage) dépasse le simple affichage token/échelle → **documentée comme mission séparée**, rien forcé, **aucune détection retirée**.

## Écarts visuels restants (vs référence)
- **Badge « Marché fermé »** affiché en double (en-tête + overlay chart) le week-end — hérité d'UI-2, hors périmètre.
- **Densité des étiquettes de prix** à droite près du prix courant (cf. C, mission séparée).
- **Chat terminal-restyle** (bulles `.bub/.refuse`) toujours déféré (classes portées, scroll = mission séparée) — le tiroir réutilise le chat existant tel quel.

## Discipline
Zéro diff backend. Réutilise `DesktopReading` (UI-2) et les primitives UI-1 — aucun style dupliqué. Responsive Scanner/Zones/Réglages intact (`@media` scopés `:not(.no-chat)`). Staging explicite, pas de `git add -A`, pas de force-push. Test d'honnêteté copy conservé (`ui2-copy-honesty`).

# Audit responsive — téléphone & tablette

**Date** : 2026-07-19
**Périmètre** : `webapp/` (Next.js 15, App Router, Tailwind, lightweight-charts v5)
**Nature** : 100 % lecture seule. Aucun fichier applicatif modifié, aucune branche de fix. Ce document est le seul livrable.
**Breakpoints audités** : téléphone **390 px**, iPad portrait **834 px**, iPad paysage **1024 px**.
**Repères Tailwind** : `sm=640` · `md=768` · `lg=1024` · `xl=1280`. `container-wide = max-w-6xl px-4 sm:px-6 lg:px-8` ; `container-prose = max-w-3xl …` (`globals.css:89-94`).

> **Constat directeur** : le shell `/app` bascule de façon **binaire à 768 px** (`useIsMobile = (max-width:767px)`, `lib/use-media-query.ts:27`). En dessous → `MobileWorkspace` (onglets Marchés/Lecture/Chat). À partir de 768 → grille 3 colonnes fixe `md:grid-cols-[240px_minmax(0,1fr)_360px]` (`AppWorkspace.tsx:209`). **Il n'existe aucun palier tablette** : les deux iPad (834 & 1024) reçoivent la grille desktop, ce qui écrase la colonne centrale (graphique/lecture) à **~138 px à 834 px** et **~312 px à 1024 px**, pendant que la colonne chat garde 360 px. C'est la cause racine n°1.

---

## 1. Tableau récapitulatif

| ID | Surface | Largeur(s) | Symptôme (résumé) | Sévérité | Effort | Régression |
|----|---------|-----------|-------------------|----------|--------|-----------|
| RESP-A-01 | A Shell | 834 | Colonne centrale écrasée à ~138px (grille 3-col forcée) | **Bloquant** | M | Moyen |
| RESP-A-02 | A Shell | 1024 | Colonne centrale ~312px, chart illisible | Majeur | M | Moyen |
| RESP-A-03 | A Nav | 390 | Ancres landing `hidden sm:block` sans burger de remplacement | Majeur | M | Faible |
| RESP-A-04 | A Nav | 390 | Cluster droit landing déborde (~404px > 390) | Majeur | M | Moyen |
| RESP-A-05 | A Nav | 390, 834 | AppHeader produit surchargé, déborde ; LocaleToggle non masqué (incohérent) | Majeur | M | Moyen |
| RESP-A-06 | G | 390 | Doubles headers sticky + `calc(100vh-4rem)` ignore le header 56px | Majeur | M | Moyen |
| RESP-A-07 | G | 390 (encoches) | Bottom-tab bar sans `safe-area-inset-bottom` (collision home-indicator) | Majeur | S | Faible |
| RESP-A-08 | G | 390, tous | `main` `min-h-[calc(100vh-160px)]` = nombre magique décorrélé | Mineur | S | Faible |
| RESP-A-09 | G | 390 | Polices 10–11px (footer, disclaimers légaux, labels onglets) | Mineur | S | Faible |
| RESP-A-10 | A Shell | 834, 1024 | Rail chat sticky 360px + offset `7rem` ≠ header 56px | Majeur | M | Moyen |
| RESP-A-11 | A/E | tous | Liens nav `py-1.5` (<44px) + états hover-only sans retour tactile | Mineur | S | Faible |
| RESP-B-01 | B Chart | 390, 1024 | Hauteur chart figée `h-[280px] sm:h-[340px]` (jamais fluide) | Majeur | S | Faible |
| RESP-B-02 | B Chart | 390 | Gouttière axe des prix vole une part fixe de la largeur | Majeur | M | Moyen |
| RESP-B-03 | B Chart | 390, 834 | Badges axe BOS/CHOCH/liquidité + textes marqueurs se chevauchent | Majeur | M | Moyen |
| RESP-B-04 | B Chart | 390, 834 | Labels HTML zones/liquidité `whitespace-nowrap` débordent, pas d'anti-collision | Majeur | M | Moyen |
| RESP-B-05 | B Chart | 390 | Contrôles bas-gauche + chip heure + logo TV se recouvrent sur plot court | Mineur | M | Faible |
| RESP-B-06 | B Chart | 390 | 90 bougies forcées à ~3.6px/barre = bougies en cheveux | Majeur | S | Faible |
| RESP-B-07 | B Chart | 390, 834 | Pan vertical tactile désactivé (`vertTouchDrag:false`) | Mineur | S | Moyen |
| RESP-B-08 | B Chart | 390 | Label « comblement live » ancré à droite `nowrap` déborde | Cosmétique | S | Faible |
| RESP-C-01 | C Chat | 390, 834 | Input chat non épinglé (`h-[70vh]` dans page scrollante) | Majeur | M | Moyen |
| RESP-C-02 | C Chat | 390, 834 | `vh` au lieu de `dvh`/`svh` + clavier virtuel recouvre l'input | Majeur | M | Moyen |
| RESP-C-03 | C Chat | 390, 834 | Aucun pattern plein écran / bottom-sheet mobile | Majeur | M | Moyen |
| RESP-C-04 | C Chat | tous | Markdown sans tables/code-blocks ni `break-words` → débordement horizontal | Majeur | M | Faible |
| RESP-C-05 | C Chat | 390, 834 | Ancrage-scroll suppose une zone haute → réponse hors écran | Mineur | M | Moyen |
| RESP-C-06 | C Chat | 390 | Header sidebar dense sur une rangée, collision libellés longs | Cosmétique | S | Faible |
| RESP-C-07 | C Chat | tous | Bouton Envoyer `h-9 w-9` (36px) < 44px | Mineur | S | Faible |
| RESP-C-08 | C Chat | 390 | Textarea `text-sm` (14px) → zoom auto iOS au focus | Mineur | S | Faible |
| RESP-C-09 | C/A | 834 | Grille 3-col engagée à 768 écrase la colonne lecture (doublon A-01) | Majeur | M | Moyen |
| RESP-C-10 | C Chat | 1024 | Colonne chat `md:h-[calc(100vh-7rem)]` masquée par le clavier | Mineur | S | Faible |
| RESP-D-01 | D Panels | 390 | ComboCard colonnes fixes `md:w-44`/`md:w-72` compriment le milieu | Cosmétique | S | Faible |
| RESP-D-02 | D Panels | 390–1024 | `sm:grid-cols-2` régime dans colonne shell bien plus étroite que 640 | Mineur | S | Faible |
| RESP-D-03 | D Panels | 390, 834 | Cellules composites BOS/CHOCH tassées en demi-colonne | Cosmétique | S | Faible |
| RESP-D-04 | D Panels | 390 | **Lignes liquidité `flex` sans `flex-wrap` → débordement horizontal réel** | Majeur | S | Faible |
| RESP-D-05 | D Panels | 390 | Indent `pl-7` du builder mange la largeur (pas de débordement) | Cosmétique | S | Faible |
| RESP-D-06 | D Panels | tous | ZoneTimeline pastille `-left-[1.30rem]` calée au pixel (fragile) | Cosmétique | S | Moyen |
| RESP-D-07 | D Panels | 390 | Badges longs ScanResults (wrap OK, densité) | Cosmétique | S | Faible |
| RESP-D-08 | D Panels | 390, 834 | Sélecteurs Zones (wrap OK, vérifié responsive) | Cosmétique | S | Faible |
| RESP-D-09 | D Panels | 390 | En-tête lecture prix+instrument (wrap OK) | Cosmétique | S | Faible |
| RESP-E-01 | E | tous | Tooltips Radix (InfoTooltip/LocaleToggle) inaccessibles au tactile | Majeur | M | Faible |
| RESP-E-02 | E | tous | Explications chart via `title=` (hover only) invisibles au tactile | Majeur | M | Moyen |
| RESP-E-03 | E | 834, 1024 | Contrôles chart `sm:h-8 w-8` (32px) servis aux iPad tactiles | Majeur | S | Faible |
| RESP-E-04 | E | tous | Bouton épingle InstrumentSidebar ~30px, label en `title` | Majeur | S | Faible |
| RESP-E-05 | E | 390 | Boutons Discussions/Réinitialiser `h-7` (28px), icon-only au phone | Majeur | S | Faible |
| RESP-E-06 | E | tous | Bouton Envoyer 36px (doublon C-07) | Majeur | S | Faible |
| RESP-E-07 | E | tous | Switch auto-refresh 16×28px | Mineur | S | Faible |
| RESP-E-08 | E | tous | Bouton « Copier » `group-hover` uniquement, invisible au tactile | Mineur | S | Faible |
| RESP-E-09 | E | tous | **Primitive `Button` toute < 44px (default 40 / sm 36 / icon 40)** — cause systémique | Majeur | M | Moyen |
| RESP-F-01 | F Landing | 1024 | Hero 2-col exactement à lg, colonne droite `minmax(0,1fr)` compressible | Cosmétique | S | Faible |
| RESP-F-02 | F Landing | tous | Toggle mensuel/annuel ~32px < 44px | Mineur | S | Faible |
| RESP-F-03 | F Landing | tous | CTA « S'abonner » désactivé (Stripe non câblé) | Cosmétique | S | Faible |
| RESP-F-04 | F Landing | 834 | Bullets tarif sans séparation visuelle en portrait (cosmétique) | Cosmétique | S | Faible |
| RESP-F-06 | F Landing | 390 | SVG « Before » libellés `text-[8px]` rétrécis (~7px) | Cosmétique | S | Faible |
| RESP-F-09 | F Landing | 390 | CTA discret hero `text-xs` (~16px de haut) | Cosmétique | S | Faible |

> **Confirmations sans défaut** (surface F, vérifiées explicitement conformes) : FAQ accordéon (`RESP-F-05`), Before/After stacking + chart fluide (`RESP-F-07`), toutes les autres grilles incl. Footer (`RESP-F-08`).

---

## 2. Fiches détaillées

### Surface A — Shell & Navigation / Transversal

**RESP-A-01** · A · **834** · **Bloquant** · M · Moyen
À l'iPad portrait, `/app` force la grille desktop 3-col ; la colonne centrale s'effondre à **~138 px** (786px utiles − 240 − 360 − 48 de `gap-6`). Chart et lecture inutilisables, coincés entre deux rails fixes.
`AppWorkspace.tsx:209` (`grid grid-cols-1 gap-6 md:grid-cols-[240px_minmax(0,1fr)_360px]`) ; breakpoint `lib/use-media-query.ts:27`.
*Cause* : grille rigide + breakpoint binaire unique à 768 ; les rails 240+360=600px + 48px de gaps ne cèdent pas avant ~1100px. Bande 768–1024 sans layout dédié.

**RESP-A-02** · A · **1024** · Majeur · M · Moyen
iPad paysage : même grille → centre **~312 px** (960 − 240 − 360 − 48). Chart bougies serré/quasi illisible, chat garde 360px.
`AppWorkspace.tsx:209`. *Cause* : rails fixes non rétractables, `minmax(0,1fr)` absorbe tout le déficit ; pas de palier `lg`/`xl`.

**RESP-A-03** · A · **390** · Majeur · M · Faible
Nav marketing masque les ancres (`<nav … className="hidden sm:block">`) **sans burger/drawer de remplacement** — les 4 sections (Démo · Honnêteté · Tarifs · FAQ) deviennent inatteignables depuis le haut (survivent seulement au footer).
`Nav.tsx:89`. *Cause* : breakpoint manquant / pas de drawer.

**RESP-A-04** · A · **390** · Majeur · M · Moyen
Cluster droit landing en flex non-wrap : 3 liens (App/Zones/Scanner) + compte + ThemeToggle ≈ **~404px > 390** → écrase la marque, risque de débordement horizontal. Le commentaire `Nav.tsx:104-106` reconnaît déjà un bug d'overflow antérieur ici.
`Nav.tsx:107-133`. *Cause* : trop d'items toujours visibles sur une rangée fixe, pas de drawer.

**RESP-A-05** · A · **390, 834** · Majeur · M · Moyen
`AppHeader` produit rend tout le cluster utilitaire (App/Zones/Scanner + aide + LocaleToggle + theme + compte) en une rangée ≈ **~562px** à 390. À noter : `LocaleToggle` **non masqué** ici alors qu'il est `hidden sm:block` dans `Nav` — incohérence qui alourdit le header produit.
`AppHeader.tsx:42-79`, LocaleToggle rendu inconditionnel `:77` vs `Nav.tsx:129`. *Cause* : pas de condensation phone ; sur `/app` le corps est mobile mais le header lourd reste sans burger.

**RESP-A-06** · G · **390** · Majeur · M · Moyen
Headers sticky empilés : `AppHeader` global `sticky top-0 h-14` (56px) + `MobileWorkspace` son propre `sticky top-0` (~44px). `MobileWorkspace` se dimensionne `min-h-[calc(100vh-4rem)]` en supposant 64px de chrome, **ignorant le header 56px** → colonne trop haute qui pousse la barre d'onglets sous la ligne de flottaison.
`AppHeader.tsx:21`, `MobileWorkspace.tsx:49,51,88`. *Cause* : calcul de hauteur sticky faux + headers dupliqués.

**RESP-A-07** · G · **390 (encoches)** · Majeur · S · Faible
La bottom-tab bar (`sticky bottom-0`) n'a **aucun `safe-area-inset-bottom`** alors que le viewport est `viewport-fit=cover` + `black-translucent` — les onglets chevauchent la zone du home-indicator. `globals.css:76-77` ne pose que left/right.
`MobileWorkspace.tsx:88`, `globals.css:76-77`, viewport `[locale]/layout.tsx:84`. *Cause* : safe-area bottom manquant.

**RESP-A-08** · G · **390, tous** · Mineur · S · Faible
`main` réserve `min-h-[calc(100vh-160px)]` — nombre magique qui ne correspond ni au header 56px ni au Footer `py-12` (bien plus haut, surtout empilé au phone).
`[locale]/layout.tsx:124`, `Footer.tsx:39`. *Cause* : offset rigide décorrélé des vraies hauteurs.

**RESP-A-09** · G · **390** · Mineur · S · Faible
Polices sous le plancher de lisibilité mobile (~12px) dans le chrome : marque `text-[10px]`, badges/disclaimers footer `text-[10px]`/`[10.5px]`/`[11px]`, lignes conformité chat `text-[10.5px]`, labels onglets `text-[11px]`. Des **disclaimers légaux** à 10–11px.
`Footer.tsx:52,60,111,121`, `ChatPanel.tsx:105,154,178`, `MobileWorkspace.tsx:115`, `Nav.tsx:83`. *Cause* : tailles fixes sub-12px sans bump responsive.

**RESP-A-10** · A · **834, 1024** · Majeur · M · Moyen
Rail chat desktop `md:sticky md:top-6 md:h-[calc(100vh-7rem)]` : dès 768 il réclame une colonne 360px pleine hauteur à côté du centre écrasé (aggrave A-01). L'offset `7rem` (112px) ≠ header 56px → écart.
`AppWorkspace.tsx:227`. *Cause* : sticky/hauteur activés à `md` avec piste fixe 360px non rétractable, offset magique.

**RESP-A-11** · A/E · **tous** · Mineur · S · Faible
Les deux nav et l'AccountMenu ne s'appuient que sur `hover:bg-accent` (aucun retour tactile) ; cibles `py-1.5` (~30px) et icônes `h-9` (36px) < 44px.
`Nav.tsx:95,110,116,122`, `AppHeader.tsx:46-66`, `AccountMenu.tsx:58`. *Cause* : hover-only + cibles sub-44px.

*Note viewport* : le meta viewport est bien formé (`viewport-fit:cover`, `maximumScale:5` — ne bloque pas le zoom, bon pour l'a11y, themeColor posé). Le seul manque transversal est la non-consommation du safe-area bottom (A-07).

---

### Surface B — Graphique

**RESP-B-01** · B · **390, 1024** · Majeur · S · Faible
Hauteur figée `h-[280px] sm:h-[340px]`, jamais fonction de la hauteur d'écran. À 390px le plot 280px tasse tous les overlays SMC ; à 1024 il est inutilement court. `autoSize:true` (`:515`) fait du conteneur la seule autorité de hauteur — gelée à deux valeurs px.
`ReadingChart.tsx:1103` ; placeholders `ReadingColumn.tsx:36`, `LandingReadingChart.tsx:29`. *Cause* : classes de hauteur fixes, pas de `svh`/`clamp()`.

**RESP-B-02** · B · **390** · Majeur · M · Moyen
La gouttière de l'axe des prix (mono `tabular-nums`, ex. `3421.55`) prend une part fixe : sur ~330-350px de plot, peu reste pour les bougies. L'overlay des zones est inséré de cette gouttière (`style={{ right: priceGutterWidth }}`, `:1128`).
`ReadingChart.tsx:545` (rightPriceScale non tuné), lecture `:802`, application `:1128`. *Cause* : pas de `minimumWidth`/`fontSize` responsive ; mono élargit les labels.

**RESP-B-03** · B · **390, 834** · Majeur · M · Moyen
Niveaux BOS/CHOCH/Retest = price lines avec `axisLabelVisible:true` + marqueurs flèches `text:'BOS'/'CHOCH'` + pills axe liquidité intacte : sur un plot court (280px) ces badges s'empilent et se chevauchent, les textes sur barres collent aux bougies. Chevauchement desktop connu, aggravé mobile.
`ReadingChart.tsx:696-705, 713-724, 664-668` ; `lib/chart/structureMarkers.ts:55-76`. *Cause* : aucune logique anti-collision, densité de labels non bornée sur budget vertical réduit.

**RESP-B-04** · B · **390, 834** · Majeur · M · Moyen
Labels HTML de zones (OB/FVG + « en test »/« touché ») et tags liquidité positionnés en px absolus avec `whitespace-nowrap` ; le fallback « étroit » ne teste que la largeur de la propre boîte (`r.width < 66/22`, `:1195`), pas la collision inter-labels ni la largeur du plot. Les labels liquidité n'ont aucun fallback.
`ReadingChart.tsx:1190-1251, 1307-1313, 1333-1343`. *Cause* : anti-collision inexistante, awareness largeur totale absente.

**RESP-B-05** · B · **390** · Mineur · M · Faible
Contrôles bas-gauche (zoom/fit/liquidité, jusqu'à 4× `h-11 w-11`) chevauchent la chip heure locale (`bottom-1 left-2`) et le logo TV (bottom-right) ; sur 280px la rangée occulte les dernières bougies.
`ReadingChart.tsx:1378-1404, 1108-1113, 523`. *Cause* : éléments absolus près du bord bas sans gouttière réservée.

**RESP-B-06** · B · **390** · Majeur · S · Faible
`DEFAULT_VISIBLE_BARS = 90` forcées → ~3.6px/barre à 390px : corps/mèches en cheveux, boîtes OB/FVG réduites à 1-2px. Pas de `minBarSpacing`.
`ReadingChart.tsx:330, 731-741, 554-567`. *Cause* : nombre de barres non fonction de la largeur + pas de plancher d'espacement.

**RESP-B-07** · B · **390, 834** · Mineur · S · Moyen
`vertTouchDrag:false` (`:573`) : impossible de recadrer le prix verticalement au doigt (seul l'axe-drag, petite cible sur la gouttière). Note : horizontal + pinch + kinetic sont correctement activés (`:572,577,581`).
`ReadingChart.tsx:569-581`. *Cause* : choix de config (évite le conflit avec le scroll de page) mais retire le seul geste vertical.

**RESP-B-08** · B · **390** · Cosmétique · S · Faible
Label « comblement live » `absolute right-1 top-0 whitespace-nowrap` dans la boîte FVG : sur boîte étroite il déborde à gauche. Seul label resté ancré à droite (les clusters de zones ont été déplacés à gauche). Visible uniquement en mode live-tick.
`ReadingChart.tsx:1275-1280`. *Cause* : ancrage droit hérité non refactoré.

---

### Surface C — Panneau Chat (M.I.A Agent)

**RESP-C-01** · C · **390, 834** · Majeur · M · Moyen
Chat mobile = `h-[70vh]` **dans** une page qui scrolle déjà (`div.flex-1.overflow-y-auto`). L'input n'est **pas épinglé au viewport** : ~70vh plus bas, il faut scroller pour l'atteindre et le clavier le pousse hors champ.
`MobileWorkspace.tsx:80,57`, `AppChatSidebar.tsx:229-237`. *Cause* : input non fixed/sticky, `vh` fractionnaire dans conteneur scrollant.

**RESP-C-02** · C · **390, 834** · Majeur · M · Moyen
`vh` partout (`h-[70vh]`, `min-h-[calc(100vh-4rem)]`) : sur iOS calculé contre le plus grand viewport → dépasse quand la barre d'URL est visible, et l'ouverture du clavier réduit le viewport visuel sans que le layout réagisse. Pas de `env(safe-area-inset-bottom)` sur la TabsList.
`MobileWorkspace.tsx:49,80,88`. *Cause* : `vh` au lieu de `dvh`/`svh` + pas de gestion visual-viewport/clavier.

**RESP-C-03** · C · **390, 834** · Majeur · M · Moyen
Aucun pattern plein écran / bottom-sheet : le chat partage la page avec header combo sticky + tab bar sticky. Hauteur utile ≈ 70vh − header − tab bar − input − 2 lignes légales → très peu de place ; `ChatWelcome` (`justify-center py-8`) + 3 boutons de démarrage remplissent déjà l'espace.
`MobileWorkspace.tsx:79-85`, `ChatWelcome.tsx:37-47`, `AppChatSidebar.tsx:229-237`, `ChatInput.tsx:101-105`. *Cause* : réutilisation verbatim du composant sidebar desktop dans une petite boîte fixe.

**RESP-C-04** · C · **tous** · Majeur · M · Faible
Le rendu Markdown assistant **n'a ni tables ni blocs de code** ; une réponse avec ```` ``` ```` ou `| col |` s'affiche en texte brut sans wrapper à défilement. Bulle `max-w-[88%]` **sans `break-words`/`overflow-x-auto`** : token long (URL, snippet) déborde horizontalement.
`lib/chat/markdown.tsx:65-153,34-39`, `ChatMessage.tsx:83`. *Cause* : renderer minimal + conteneur sans `min-w-0`/`break-words`.

**RESP-C-05** · C · **390, 834** · Mineur · M · Moyen
`anchorToLastUser` scrolle la question en haut ; sur zone mobile courte le `scrollTo` clampe et la réponse rend sous la ligne de flottaison.
`useChatAnchorScroll.ts:49-78`, `AppChatSidebar.tsx:189-192`. *Cause* : UX d'ancrage-haut supposant une zone haute.

**RESP-C-06** · C · **390** · Cosmétique · S · Faible
Header sidebar (avatar + bloc 3 lignes + 2 boutons ghost) sur une rangée ; libellés `sr-only sm:not-sr-only` (viewport, pas conteneur) : à 390 la ligne contexte tronquée peut heurter les boutons pour instruments longs.
`AppChatSidebar.tsx:92-137,122,134`. *Cause* : header dense hérité desktop.

**RESP-C-07** · C · **tous** · Mineur · S · Faible
Bouton Envoyer `h-9 w-9` = 36px < 44px (doublon RESP-E-06).
`ChatInput.tsx:83-99`. *Cause* : cible sub-44px.

**RESP-C-08** · C · **390** · Mineur · S · Faible
Textarea `text-sm` (14px) : iOS zoome au focus (<16px) → casse le layout, déjà dur à atteindre.
`ChatInput.tsx:81`. *Cause* : police input <16px.

**RESP-C-09** · C/A · **834** · Majeur · M · Moyen
La grille 3-col s'engage à 768 avec 600px de colonnes fixes → colonne lecture centrale écrasée à 834 (doublon fonctionnel de RESP-A-01, gardé pour la surface chat car la 3e colonne y contribue).
`AppWorkspace.tsx:209,227`. *Cause* : pas de palier tablette.

**RESP-C-10** · C · **1024** · Mineur · S · Faible
Colonne chat desktop `md:h-[calc(100vh-7rem)]` : `100vh` ne rétrécit pas au clavier → input docké couvert en frappe sur iPad.
`AppWorkspace.tsx:227`, `AppChatSidebar.tsx:229-237`. *Cause* : hauteur `100vh` ignore le visual viewport.

---

### Surface D — Panneaux denses

**RESP-D-04** · D · **390** · Majeur · S · Faible · **(seul débordement horizontal réel de la surface)**
`LiquidityList` : rangées `flex items-center gap-2` de 6 tokens (prix `tabular-nums` · côté · type · `·` · statut) **sans `flex-wrap` ni `min-w-0`** ; une ligne large (`2 651,30 · BSL sommet égal · balayée`) dépasse le panneau à 390px.
`StructureSection.tsx:333-362`. *Cause* : rangée mono-ligne de nombreux tokens, prix tabular, pas de wrap.

**RESP-D-02** · D · **390–1024** · Mineur · S · Faible
`sm:grid-cols-2` (« État structurel courant ») se déclenche à 640 alors que la colonne centrale du shell `/app` est bien plus étroite → grille 2-col dans une boîte plus serrée que 640 n'implique.
`RegimeSection.tsx:126,149-161`. *Cause* : breakpoint viewport dans un contexte à largeur de conteneur découplée.

**RESP-D-01** · D · **390** · Cosmétique · S · Faible — ComboCard `md:w-44`+`md:w-72` compriment le milieu (wrap OK, pas de débordement). `ComboCard.tsx:51,53,69,105`.
**RESP-D-03** · D · **390, 834** · Cosmétique · S · Faible — cellules composites BOS/CHOCH tassées en demi-colonne `sm:grid-cols-2`. `StructureSection.tsx:184-206`.
**RESP-D-05** · D · **390** · Cosmétique · S · Faible — indent `pl-7` du builder mange la largeur (wrap présent). `ConditionsBuilder.tsx:205,278-317`.
**RESP-D-06** · D · **tous** · Cosmétique · S · **Moyen** — ZoneTimeline pastille `-left-[1.30rem]` calée au pixel sur `ml-1 pl-4` : fragile, ne déborde pas aujourd'hui mais risque de désalignement si touché. `ZoneTimeline.tsx:22,28`, `ZoneLifecycleCard.tsx:279`.
**RESP-D-07/08/09** · D · Cosmétique — ScanResults badges longs (wrap OK), sélecteurs Zones (vérifié responsive `grid-cols-1 … lg:grid-cols-2`), en-tête lecture (wrap OK). `ScanResults.tsx:86-118`, `ZonesWorkspace.tsx:172-223`, `MarketReadingHeader.tsx:31-47`.

*Note layout* : les pages **scanner** et **zones** ne réutilisent PAS le shell 3-col ; chacune a son `container-wide` mono-colonne et est globalement responsive (scanner mono-stack ; zones `grid-cols-1 … lg:grid-cols-2`).

---

### Surface E — Cibles tactiles & survol

**RESP-E-09** · E · **tous** · Majeur · M · Moyen · **(cause systémique)**
La primitive partagée `Button` est toute sub-44px : `default` h-10 (40) / `sm` h-9 (36) / `icon` h-10 w-10 (40) / `lg` h-11 (44, OK). Tout `size="icon"` (ThemeToggle) et `size="sm"` (StrategyPanel Charger/Renommer/Dupliquer/Supprimer, items AccountMenu) hérite 36-40px.
`ui/button.tsx:21-26`, `theme-toggle.tsx:17-19`, `StrategyPanel.tsx:163,176,188`. *Cause* : sub-44px baked dans le design system → corriger le primitive relève plusieurs surfaces.

**RESP-E-01** · E · **tous** · Majeur · M · Faible
`InfoTooltip` et `LocaleToggle` = Radix Tooltip (hover/focus) : au tactile, taper le trigger n'ouvre rien → définition + lien « En savoir plus » inatteignables. Les variantes `iconOnly` (ⓘ 12px dont le libellé n'existe QUE dans le tooltip) sont **Bloquantes**.
`ui/tooltip.tsx:8`, `InfoTooltip.tsx:46-77`, `RegimeSection.tsx:113,117,141`, `StructureSection.tsx:302,382`, `LocaleToggle.tsx:27-44`. *Cause* : hover-only sans repli popover/tap.

**RESP-E-02** · E · **tous** · Majeur · M · Moyen
Explications d'état provisoire (« en test », « touché », « EN DIRECT provisoire », label zone) livrées via `title=` natif sur overlays chart → hover souris seulement, jamais au tactile.
`ReadingChart.tsx:1215,1226,1244,1273,1305,1361,1154`. *Cause* : `title` comme seule explication.

**RESP-E-03** · E · **834, 1024** · Majeur · S · Faible
`ChartControl` `h-11 w-11` au phone mais `sm:h-8 sm:w-8` (32px) dès 640 → les iPad tactiles reçoivent 32px pour zoom/fit/toggle liquidité (le commentaire promet pourtant ≥44px « mobile »).
`ReadingChart.tsx:1429-1434,1379-1402`. *Cause* : le pattern de rétrécissement `sm:` sert la plus petite taille au tactile tablette.

**RESP-E-04** · E · **tous** · Majeur · S · Faible
Bouton épingle par combo icon-only (`Pin` 14px, `px-2`), ~28-30px, libellé seulement en `title`/`aria-label`.
`InstrumentSidebar.tsx:206-226`. *Cause* : cible sub-44px.

**RESP-E-05** · E · **390** · Majeur · S · Faible
« Discussions » et « Réinitialiser » = `h-7` (28px) icônes `h-3` ; sous `sm` le label passe `sr-only` → 2 cibles icon-only 28px serrées.
`AppChatSidebar.tsx:113-136,122,134`. *Cause* : sub-44px + label masqué au phone.

**RESP-E-06** · E · **tous** · Majeur · S · Faible — bouton Envoyer 36px (= RESP-C-07). `ChatInput.tsx:83-99`.
**RESP-E-07** · E · **tous** · Mineur · S · Faible — switch auto-refresh `h-4 w-7` (16×28px), atténué par le `<label>` englobant. `AutoRefreshToggle.tsx:19-39`.
**RESP-E-08** · E · **tous** · Mineur · S · Faible — bouton « Copier » `opacity-0` révélé au `group-hover`/`focus-visible` seulement → invisible au tactile. `ChatMessage.tsx:120,134-156`.

*Constats croisés* : **aucune palette ⌘K/cmdk** n'existe (grep `cmdk`/`CommandDialog`/`⌘` = 0) — rien à équiper d'un équivalent tactile de ce côté ; AccountMenu est un dropdown au clic (tactile-safe). Deux classes de hover-only : (a) tooltips Radix + `title=` cachant du **texte essentiel** (E-01/E-02, `iconOnly` = pire), (b) action révélée au hover (E-08).

---

### Surface F — Landing

**RESP-F-02** · F · **tous** · Mineur · S · Faible
Toggle mensuel/annuel `px-4 py-1.5 text-sm` → ~30-32px < 44px ; contrôle principal de la section tarifs.
`PricingSection.tsx:85-113`. *Cause* : padding insuffisant, pas de `min-h`.

**RESP-F-01** · F · **1024** · Cosmétique · S · Faible — hero 2-col exactement à lg, colonne droite `minmax(0,1fr)` compressible ; à vérifier visuellement. `HeroLive.tsx:57`.
**RESP-F-03** · F · **tous** · Cosmétique · S · Faible — CTA « S'abonner » `disabled` (Stripe non câblé) : sur petit écran lit comme cassé. `PricingSection.tsx:159-168`.
**RESP-F-04** · F · **834** · Cosmétique · S · Faible — bullets tarif `lg:pl-4` seulement → pas de séparation en portrait. `PricingSection.tsx:116,173`.
**RESP-F-06** · F · **390** · Cosmétique · S · Faible — SVG « Before » `text-[8px]` rend à ~7px après scaling viewBox. `BeforeAfterSection.tsx:192-193,241,257,267`.
**RESP-F-09** · F · **390** · Cosmétique · S · Faible — CTA discret hero `text-xs`, cible ~16px. `HeroLive.tsx:63-73`.

*Confirmé conforme* : FAQ accordéon (≈56px, `FaqSection.tsx`/`accordion.tsx:26-38`), Before/After stacking + `LandingReadingChart` fluide `h-[260px] sm:h-[300px] w-full`, gallery `lg:grid-cols-3`, HowItWorks `sm:grid-cols-2 lg:grid-cols-4`, Footer `lg:grid-cols-[1.5fr_1fr_1fr]` — tous stackent proprement au phone.

---

## 3. Causes systémiques

Régler ces racines éteint plusieurs symptômes à la fois :

- **SYS-1 — Breakpoint binaire 768 sans palier tablette.** `useIsMobile = (max-width:767px)` + grille fixe `md:grid-cols-[240px_minmax(0,1fr)_360px]` : les deux rails (600px) + gaps ne cèdent pas avant ~1100px, donc 834 et 1024 reçoivent un centre écrasé. → **RESP-A-01, A-02, A-10, C-09, C-10, E-03** (côté « le tactile tablette tombe dans le chemin desktop »).
- **SYS-2 — Absence de navigation mobile (burger/drawer).** Ni `Nav` ni `AppHeader` n'ont de menu ; ancres larguées ou clusters qui débordent la rangée 390. → **RESP-A-03, A-04, A-05**.
- **SYS-3 — `vh` au lieu de `dvh`/`svh` + aucune gestion clavier/visual-viewport + pas de `safe-area-inset-bottom`.** → **RESP-A-07, C-01, C-02, C-03, C-10**.
- **SYS-4 — Cibles tactiles < 44px dans le design system + pattern `sm:` qui rétrécit.** La primitive `Button` (40/36/40) et le rétrécissement `sm:h-8` servent la plus petite taille précisément au tactile tablette. → **RESP-E-09 (racine), E-03, E-04, E-05, E-06/C-07, E-07, A-11, F-02**.
- **SYS-5 — Affordances hover-only sans équivalent tactile.** Tooltips Radix + `title=` natifs (texte essentiel) + `group-hover`. → **RESP-E-01, E-02, E-08, B-labels**.
- **SYS-6 — Hauteurs & densité chart figées.** Hauteur px fixe, 90 bougies forcées, labels sans anti-collision, gouttière axe non tunée. → **RESP-B-01, B-02, B-03, B-04, B-06**.
- **SYS-7 — Nombres magiques de hauteur sticky.** `calc(100vh-4rem)`, `calc(100vh-160px)`, `7rem` — tous décorrélés du header réel (56px) et du footer. → **RESP-A-06, A-08, A-10**.

---

## 4. Plan de PR ordonné

Ordre gouverné par la dépendance : la fondation shell/breakpoint doit précéder tout le reste (les surfaces se re-mesurent une fois le layout tablette correct).

### PR-1 — Shell responsive + palier tablette + navigation mobile *(fondation)*
- **Fiches** : RESP-A-01, A-02, A-03, A-04, A-05, A-06, A-07, A-08, A-10, C-09 ; SYS-1, SYS-2, SYS-7.
- **Objectif** : introduire un vrai système de paliers (phone / tablette 768–1024 / desktop), collapser la grille 3-col en 2-col (ou onglets prolongés) sur tablette, ajouter un drawer/burger de nav (marketing + produit), consommer le safe-area bottom, corriger les hauteurs sticky.
- **Pourquoi en premier** : tout le reste se mesure dans ce cadre ; corriger un panneau avant d'avoir réglé la largeur de sa colonne est du travail jeté. Le Bloquant unique (A-01) vit ici.
- **Dépendances** : aucune. Prérequis de PR-2, PR-3, PR-4.

### PR-2 — Graphique responsive + tactile
- **Fiches** : RESP-B-01, B-02, B-03, B-04, B-05, B-06, B-07, B-08, E-03 ; SYS-6.
- **Objectif** : hauteur fluide (`svh`/`clamp`), gouttière d'axe bornée, densité de labels/marqueurs déclutterée, nb de bougies fonction de la largeur + `minBarSpacing`, contrôles chart ≥44px au tactile, pan vertical tactile.
- **Pourquoi cet ordre** : le chart vit dans la colonne centrale dont PR-1 fixe la largeur ; inutile de tuner labels/gouttière tant que la colonne fait 138px.
- **Dépendances** : PR-1 (largeur de la colonne centrale).

### PR-3 — Panneau chat mobile (plein écran / bottom-sheet + clavier)
- **Fiches** : RESP-C-01, C-02, C-03, C-04, C-05, C-06, C-08, C-10, E-06/C-07 ; SYS-3.
- **Objectif** : layout chat mobile dédié (plein écran ou bottom-sheet), input épinglé au visual viewport, `dvh`/`svh`, safe-area, textarea ≥16px, bouton Envoyer ≥44px, wrapping markdown (tables/code + `break-words`/`overflow-x-auto`).
- **Pourquoi cet ordre** : dépend du pattern de navigation mobile (onglet/drawer) posé en PR-1 ; indépendant de PR-2.
- **Dépendances** : PR-1.

### PR-4 — Panneaux denses : débordements & empilement
- **Fiches** : RESP-D-04 (prioritaire), D-01, D-02, D-03, D-05, D-06.
- **Objectif** : régler le seul débordement réel (liquidité `flex-wrap`/`min-w-0`), fiabiliser les demi-grilles `sm:grid-cols-2` en logique conteneur, sécuriser la pastille timeline calée au pixel.
- **Pourquoi cet ordre** : léger, isolé ; se valide mieux une fois les largeurs de colonnes stables (PR-1).
- **Dépendances** : PR-1 (recommandé, non strict — D-04 corrigeable seul).

### PR-5 — Cibles tactiles + équivalents au survol *(transversal)*
- **Fiches** : RESP-E-09 (racine primitive `Button`), E-01, E-02, E-04, E-05, E-07, E-08, A-11, A-09.
- **Objectif** : relever `Button` (icon/sm/default) vers 44px (ou inset de cible), remplacer tooltips Radix/`title=` par des popovers tap-to-open pour le texte essentiel, révéler « Copier » au tactile, remonter les polices <12px des disclaimers.
- **Pourquoi cet ordre** : touche un primitive partagé → réaliser après que PR-1/2/3 ont figé les toolbars/headers évite de re-régler la même densité deux fois. Peut aussi tourner en parallèle si l'équipe accepte le churn.
- **Dépendances** : idéalement après PR-1/2/3 (surfaces qui consomment `Button`).

### PR-6 — Landing responsive *(polish)*
- **Fiches** : RESP-F-02, F-01, F-03, F-04, F-06, F-09.
- **Objectif** : toggle tarif ≥44px, vérif visuelle hero à 1024, petits ajustements typographiques SVG/CTA.
- **Pourquoi en dernier** : la landing est déjà responsive-first (aucun Bloquant/Majeur), findings mineurs/cosmétiques ; indépendante du shell produit.
- **Dépendances** : aucune (peut être planifiée à tout moment ; placée en dernier par priorité).

**Écart à l'ordre proposé du brief** : conforme. Une nuance — PR-5 (primitive `Button`) est *transversale* et pourrait démarrer tôt, mais on la place après les PR fonctionnelles pour éviter de re-toucher les mêmes toolbars/headers (churn). RESP-D-04 (débordement réel) est le seul item de PR-4 qui pourrait être extrait en hotfix isolé si le fondateur veut le corriger avant PR-1.

---

## 5. Totaux

### Par surface (défauts avérés)
| Surface | Bloquant | Majeur | Mineur | Cosmétique | Total |
|---------|:-:|:-:|:-:|:-:|:-:|
| A — Shell & Nav / Transversal | 1 | 6 | 3 | 0 | 10 |
| B — Graphique | 0 | 5 | 2 | 1 | 8 |
| C — Chat | 0 | 5 | 4 | 1 | 10 |
| D — Panneaux denses | 0 | 1 | 1 | 7 | 9 |
| E — Tactile & survol | 0 | 7 | 2 | 0 | 9 |
| F — Landing | 0 | 0 | 1 | 5 | 6 |
| **Total** | **1** | **24** | **13** | **14** | **52** |

*(+ 3 confirmations « sans défaut » en surface F : F-05, F-07, F-08.)*

### Par sévérité
- **Bloquant (1)** : RESP-A-01.
- **Majeur (24)** : A-02, A-03, A-04, A-05, A-06, A-07, A-10 · B-01, B-02, B-03, B-04, B-06 · C-01, C-02, C-03, C-04, C-09 · D-04 · E-01, E-02, E-03, E-04, E-05, E-06, E-09.
- **Mineur (13)** : A-08, A-09, A-11 · B-05, B-07 · C-05, C-07, C-08, C-10 · D-02 · E-07, E-08 · F-02.
- **Cosmétique (14)** : B-08 · C-06 · D-01, D-03, D-05, D-06, D-07, D-08, D-09 · F-01, F-03, F-04, F-06, F-09.

### Recoupements à ne pas double-corriger
- **RESP-C-09 ≡ RESP-A-01/A-02** (même grille tablette) → réglé en PR-1.
- **RESP-C-07 ≡ RESP-E-06** (bouton Envoyer 36px).
- **RESP-C-02 / A-07** partagent le manque de safe-area bottom + `dvh`.
- **RESP-E-01/E-02** partagent la racine hover-only (SYS-5).

---

*Fin de l'audit. Aucun correctif appliqué — en attente de validation du fondateur pour lancer les PR une par une.*

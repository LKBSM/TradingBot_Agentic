# AUDIT — LP-3 · Refonte créative de la page d'accueil

Branche : `feat/lp-3-refonte-accueil` · worktree `C:/MyPythonProjects/wt-lp-3`
Base : `origin/feat/lp2s-accueil-stats` (LP-2S) + merge `origin/main` @ `af0c4cf`
Modèle : Opus 5 · Date : 2026-09-10

---

## 0. Discipline d'audit

### 0.1 Écart contre `origin/main`

```
git fetch --all --prune
HEAD local (TradingBOT_Agentic)  e0dc69c  2026-08-20   →  79 commits de retard
origin/main                      af0c4cf  2026-09-09
```

Tout le diagnostic a été établi **contre `origin/main`** (`git show origin/main:<fichier>`),
jamais contre le HEAD local périmé.

### 0.2 Blocage résolu — disque plein

`git worktree add` a d'abord échoué : `No space left on device` (476 Go utilisés sur 476,
**4,3 Go libres**). 70 worktrees enregistrés, 39 avec un `node_modules` réel.

Libéré **sans toucher à une seule source ni à un `node_modules`** : suppression des
**20 dossiers `.next`** (artefacts de build, régénérés par `npm run build`), en épargnant
`wt-run-main`. **4,3 Go → 9,3 Go.** `du` reste inutilisable sur ce disque (>600 s sans finir,
cohérent avec l'incident Defender connu) : aucun chiffre par worktree n'est avancé ici.

> ⚠️ Reste à trancher : les `node_modules` des worktrees de missions **déjà mergées** sont
> le gros poste. Non touchés — un incident antérieur a vidé un store partagé par jonction
> (cf. `feedback_worktree_junction_teardown`).

### 0.3 Base de branche — LP-2S n'est pas sur `main`

PR **#203 (LP-2S) est encore OUVERTE**. La mission traite pourtant sa décision comme acquise
(« le nombre de marchés réellement en production vs. visés — déjà tranché »).

Décision prise, réversible : LP-3 est branchée sur **`origin/feat/lp2s-accueil-stats`**, puis
`origin/main` y a été mergé (19 commits d'avance, dont LP-2B/MIA-4S — **sans conflit**). LP-3
hérite donc de la ligne honnête et de son garde-fou resserré, sans rien toucher à `main`.

> 👉 **Si tu merges #203 avant LP-3, le diff de LP-3 devient propre tout seul.** Sinon la PR
> LP-3 embarquera les 3 commits LP-2S.

### 0.4 Environnement

`node_modules` par jonction depuis `wt-cln-1` (lock + `package.json` identiques au MD5).
**Piège rencontré** : `vitest --pool=forks` ne démarre aucun worker à travers une jonction
cross-worktree (« Timeout waiting for worker to respond », 60 s). **`--pool=threads` passe.**
À retirer la jonction (`cmd /c rmdir`) **avant** tout `git worktree remove`.

---

## 1. Diagnostic — ce qui faisait « généré par IA »

### 1.1 La cause structurelle, en quatre chiffres

| Mesure | Avant | Ce que dit la marque (§5) |
|---|---:|---|
| Tailles de police distinctes dans `lp1.module.css` | **30** | échelle à **6** tailles |
| Usages de `var(--fs-*)` | **0** | l'échelle est la source |
| Rayons de bordure distincts | **18** (2→30 px) | `--r: 10px` · `--r-s: 7px` |
| `style={{…}}` inline dans les 4 composants | **98** | — |

`globals.css:59-64` définit `--fs-title 26 / --fs-section 19 / --fs-body 15 /
--fs-secondary 14 / --fs-label 12 / --fs-legal 11`. **La page n'en utilisait aucune.** Elle
avait sa propre échelle à 30 crans (8,5 · 9 · 9,5 · 10 · 10,5 · 11 · 11,5 · 12 · 12,5 · 13 ·
13,5 · 14 · 14,5 · 15 · 15,5 · 16 · 16,5 · 17 · 18 · 19 · 21 · 22 · 24 · 26 · 27 · 32 · 34 ·
36 · 40 · 54). Rien n'était aligné sur rien : l'œil ne trouvait aucune règle.

### 1.2 Les motifs de la mission, localisés

| Motif | Où (avant) | Compte |
|---|---|---:|
| **Étiquettes MAJUSCULES espacées** | `.eyebrow` (`css:25`) ×8 · `.tag` (`css:82`) ×4 · `.try` (`css:197`) ×4 · `.wcLb` (`css:496`) ×3 · `.rgk`/`.pcLbl` | **5 systèmes parallèles** |
| **Monospace pour des mots** | la mention **légale entière** (`.plegal`, `css:369`), `.micro`, `.roadmap`, `.carHint`, `.illus`, `.pill`, `.stepN`… | 31 déclarations |
| **Cartes identiques à coins ronds** | `.qc` 10 · `.res` 11 · `.fq` 12 · `.zcard` 12 · `.cap` 13 · `.tile` 13 · `.wc` 13 · `.step` 14 · `.vis` 15 · `.pc` 16 · `.distinguish` 16 · `.demo`/`.final`/`.miaHero` 18 | **13 rayons pour un seul objet** |
| **Icônes génériques / dégradés** | `Spark()` (l'étoile « IA », `MiaSection.tsx:8`) · `CAP_ICONS` livre/loupe/barres/triangle (`:17`) · `TabIcon` (`DemoTabs.tsx:560`) · `.aura` dégradé 1000×540 (`css:49`) · `--lp-vio: #a78bfa` hors palette | — |
| **Faux cadres de navigateur** (3 pastilles macOS) | `VisFrame` ×3 · `ReadingCarousel` ×5 · `MiaSection` ×1 | **9 sur une page** |
| **Un mot coloré dans le titre** | `<h1>{h1a} <em>{h1b}</em></h1>` + `.h1 em { color: var(--lp-acc) }` | 1 titre coupé en deux |
| **`<b>` décoratif** | helper `rich()` | **70 chaînes sur 395** |
| **Gabarit `<b>Label</b> — explication`** | `tools.app.b1/b3/b4/b5`, `tools.mia.b4`, rendu par `Bullets` | 5 chaînes → **17 puces** |
| **`A · B` en prose** | `hero.pill`, `hero.micro`, `hero.roadmap`, `demoSection.illus`, compteur carrousel | 48 chaînes au total, dont ~10 en prose |
| **Flèche après un non-lien** | `.qlArrow` « ↗ » sur 6 puces **sans `href` ni `onClick`** | 6 |

> **Distinction faite, pas ignorée** : les `·` de l'étiquetage produit (`Or · XAU/USD ·
> 15 minutes`, `OB haussier · jamais testé`) sont la convention du **vrai produit** et sont
> conservés. Les `↑`/`↓` des zones sont de la **donnée** (direction). Seule la prose découpée
> en puces a été traitée.

### 1.3 Deux constats hors motifs

**a) 13 composants `landing/*.tsx` sont du code mort** — `BeforeAfterSection`, `HeroLive`,
`FaqSection`, `MultiMarketSection`… aucun n'est importé. `PricingSection` n'est cité que
comme *chaîne de caractères* dans un test. **Hors périmètre LP-3, signalé, non touché.**

**b) Le vrai produit n'était jamais montré.** DS-1 (PR #199) avait figé de **vraies données
produit** dans `lib/ds-samples/` (4 500 lignes depuis `market_readings.db`), et
`DesignGallery.tsx` prouvait que `MarketReadingCard`, `ZoneLifecycleCard`, `ScanResults`
rendent **sans backend**. Pendant ce temps l'accueil dessinait des imitations à la main
(`CandleSvg.tsx`) dans 9 faux cadres. **La page montrait un dessin du produit au lieu du
produit.** C'est le levier central de la refonte.

---

## 2. Ce qui a été construit

### 2.1 L'ouverture — le produit sort du cadre

`components/landing/lp1/Opening.tsx` *(nouveau)*

- Colonne droite = **`<MarketReadingCard>`**, le composant que `/app` rend, nourri de
  `SAMPLE_READING_XAU_H4` (DS-1, extrait de `market_readings.db`). Largeur
  `calc(100% + 150px)` sous un `overflow: hidden` : la carte est **rognée par le bord droit**
  — un produit continue hors cadre, une capture s'arrête au cadre. Aucun faux chrome.
- **Honnêteté de l'échantillon** : la carte porte son propre badge de fraîcheur, qui affiche
  donc « Bougie clôturée il y a 41 jours » au lieu de se faire passer pour du direct — ce que
  `faq.a5` promet précisément que le produit fait toujours. La ligne d'archive au-dessus nomme
  instrument, unité de temps et date de clôture, **lus depuis l'échantillon** (jamais un
  littéral, donc jamais de dérive).
- **La promesse est énoncée par le refus** — trois phrases sur un filet vertical, pas trois
  cartes : une seule ligne de barème `--`, jamais un `✓` vert (une coche à côté de « elle ne
  dit pas où va le prix » aurait été exactement le réflexe décoratif qu'on retire).
- Supprimés : `.aura`, `.pill`, le bandeau de 4 tuiles (**= la décision LP-2S**), la moitié
  colorée du `<h1>`, `.micro` en mono.
- **Correctif** : le bouton « Demander à M.I.A Agent » de la carte est `disabled` sans
  handler. Un contrôle mort dans le premier écran est précisément le genre d'accessoire que
  cette refonte retire → il défile maintenant vers le vrai agent (`#mia`).

### 2.2 LE moment au scroll — « la lecture s'écrit en descendant »

`components/landing/lp1/ReadingUnfolds.tsx` *(nouveau)* ·
`components/landing/lp1/structure-chart.tsx` *(extrait, partagé)*

Le graphique **réel de la démo** a été extrait de `DemoTabs` dans un module partagé : ce que
le lecteur regarde s'assembler est, au pixel près, ce dont il prend ensuite le contrôle —
mêmes bougies, mêmes bornes, même `<ZoneRect>`, même scénario
(`config/demo_illustration.json`, verrouillé par `demo-illustration-parity.test.ts`).
**Aucune version simplifiée n'a été créée.**

Trois règles tenues :

1. **Le scroll n'AJOUTE jamais que de la matière.** Arrivé au bout, toutes les couches sont
   allumées, exactement comme si on n'avait jamais défilé.
2. **Il remplace, il n'ajoute pas.** Ce bloc absorbe les 4 rangées alternées (869 mots), le
   carrousel à 5 volets (226 lignes) et l'onglet « Lire une structure ».
3. **Il dégrade en pile ordinaire.** Pas d'`IntersectionObserver`, `prefers-reduced-motion`,
   ou < 900 px → toutes les phrases rendues d'un coup, toutes les couches dessinées, puces
   vivantes dès le premier rendu. **C'est aussi le markup rendu côté serveur** : un visiteur
   sans JS ne perd que la chorégraphie.

**L'étage épinglé est aligné verticalement sur la phrase courante**
(`top: max(92px, calc(50vh - 200px))`, pas sur le haut de l'écran) : l'œil circule
latéralement entre la phrase et la couche qu'elle vient de dessiner.

**Défaut corrigé en cours de route** : la section promettait « décoche une couche : le
paragraphe ne décrit plus que ce qui reste affiché » — et l'implémentation ne le faisait pas.
Désormais, dès que le lecteur prend les puces, la phrase d'une couche éteinte est **retirée**
(pas grisée), et tout éteindre donne l'état vide honnête `demo.structure.empty`
(« elle n'invente rien pour remplir le vide »). Verrouillé par test.

### 2.3 Le refus

`components/landing/lp1/RefusalBlock.tsx` *(nouveau)* ·
`components/landing/lp1/MiaPane.tsx` *(extrait de `DemoTabs`)*

La page avait **deux** M.I.A : une conversation **scriptée** en §3, et le **vrai agent**
derrière un onglet en §4. Le chat scripté est supprimé ; il ne reste que le vrai.

Les **quatre phrases de capacité survivent verbatim** (LP-2A les avait réduites à une phrase
factuelle chacune et verrouillées sur 9 locales) — LP-3 ne change que leur rendu : texte
courant sous l'échange, plus de grille de 4 cartes, plus d'icônes.

Les **actions de vue de M.I.A pilotent le graphique de la section au scroll** : l'état
`manual` est tenu au niveau de la page, si bien que « j'ai masqué les Fair Value Gaps » déplace
le graphique que le lecteur a regardé s'assembler plusieurs écrans plus haut — au lieu de
changer d'onglet. Le CTA défile vers `#lecture`.

**Deux défauts corrigés** : `MiaPane` faisait `scrollIntoView` au montage (inoffensif quand il
vivait dans un onglet monté au clic ; sur LP-3 il **arrachait le visiteur de l'ouverture vers
le chat** au chargement) ; et sa note latérale répétait `refuse.lead`.

### 2.4 Les trois démos restantes

`DemoTabs.tsx` — la barre d'onglets à icônes est supprimée. Scanner · zone · calcul sont
**empilés et ouverts** : un visiteur qui défile voit que les trois existent, au lieu d'en
découvrir deux seulement en cliquant.

### 2.5 La structure, avant → après

| Avant (12 blocs) | Après (7) |
|---|---|
| Hero centré + aura + pilule + 4 tuiles | **Ouverture** asymétrique, produit réel rogné |
| §3 M.I.A : chat scripté + 4 cartes à icônes | *(absorbé §2.3)* |
| §4 démos en 5 onglets à icônes | **La lecture s'écrit** (§2.2) + 3 démos ouvertes |
| §5-8 : 4 rangées alternées (**869 mots**) | *(supprimé — elles racontaient ce que la démo fait)* |
| Carrousel 5 volets (368 mots, 5 faux cadres) | *(absorbé)* |
| Comment : 3 cartes 01/02/03 | *(supprimé)* |
| Pour qui : 3 cartes symétriques | *(supprimé)* |
| Ce qui distingue (carte beige, en bas) | **Promu** : matière de l'ouverture + bloc d'argument |
| — | **Le refus** : le vrai agent (§2.3) |
| Tarif (grille 2 colonnes, 1 offre) | Tarif, **1 colonne** (la grille datait de 2 offres) |
| FAQ 8 accordéons (**408 mots**) | FAQ, réponses sèches |
| CTA final générique | La phrase la plus tranchante de la page |

---

## 3. L'écriture

### 3.1 Mesures (namespace `home`, français)

| | Avant | Après | Δ |
|---|---:|---:|---:|
| Chaînes | 395 | **168** | −57 % |
| Mots | 3 583 | **1 753** | −51 % |
| Chaînes avec ` — ` | 34 | **18** | −47 % |
| Chaînes avec ` · ` | 48 | **18** | −63 % |
| Chaînes avec `<b>` | 70 | **29** | −59 % |

### 3.2 Méthode — trois niveaux de risque, séparés

1. **Suppressions** — les sections que la refonte retire (`tools`, `how`, `who`, `carousel`,
   `stats`, `demoSection`, tous les `eyebrow`). Aucun jugement : le markup qui les rendait
   n'existe plus.
2. **Déplacements mécaniques** — du texte qui **existe déjà, dans les neuf langues**, remis où
   il compte. `final.h2` est la dernière phrase (en gras) de `distinguish.p2`
   — « Tu gardes la décision. On te donne la lecture. » — sortie d'une carte en bas de page.
   `final.p` est la moitié factuelle de l'ancien. Les réponses FAQ perdent les phrases qui
   **redisent** au lieu d'ajouter, par **index de phrase, le même index dans chaque locale**
   (les neuf bundles sont structurellement parallèles : comptes de phrases identiques,
   vérifié avant toute écriture, aucune balise `<b>` orpheline). **Pas un mot n'est reformulé.**
   C'est la discipline TXT-1.
3. **Copie neuve** — l'ouverture, la section au scroll, le bloc refus. Écrite pour chaque
   locale, vérifiée contre le vocabulaire interdit de chaque langue.

### 3.3 Le ton

| Avant (générique) | Après (assumé) |
|---|---|
| « CINQ OUTILS, UN MÊME PRODUIT » | *(supprimé)* |
| « Tout ce dont tu as besoin pour lire une structure » | « Elle lit la structure. Elle ne choisit rien à ta place. » |
| « Le marché a déjà tout écrit. **MIA te le lit.** » | « Elle ne dit pas où va le prix. **Personne ne le sait.** Un outil qui l'affirme te vend une certitude qu'il n'a pas. » |
| « Arrête de tracer. Commence à lire. » | « Tu gardes la décision. On te donne la lecture. » *(déjà ta phrase, promue)* |
| `demo.scanner.emptyNoCond` | ✅ **déjà juste, gardée telle quelle** |

### 3.4 Vocabulaire interdit — vérifié sur 9 locales

```
FORBIDDEN_XLANG (setup/signal/signaal/señal/segnale/sinal/sygnał/إشارة,
                 opportun/oportun/okazja, probabilit/probabilidad/…/احتمال)
+ FORBIDDEN_FR, FORBIDDEN_EN, « moteur »/« engine »
→ AUCUNE violation, fr en de es it pt nl pl ar
```

LP-2S : `hero.roadmap` est **conservée verbatim** (son garde-fou compare au registre MKT-1) ;
aucune copie nouvelle n'énonce de décompte de marchés.

---

## 4. Le système de jetons

30 tailles → **0 valeur en px** dans la CSS, tout sur l'échelle de marque, plus **deux
extensions déclarées** (et non glissées) :

```css
--fs-display: clamp(30px, 4.4vw, 46px);  /* un cran d'affichage au-dessus de --fs-title,
                                            dont la coquille produit n'a pas besoin */
--fs-chartlabel: 9px;                    /* annotations peintes DANS un dessin : à
                                            --fs-legal elles débordent du graphique */
```

| | Avant | Après |
|---|---:|---:|
| Valeurs `font-size: Npx` | 30 | **0** |
| Valeurs `border-radius: Npx` | 18 | **1** (le filet de 2 px du refus) |
| `var(--font-mono)` | 31 | **6** — *tous des nombres* |
| Lignes de `lp1.module.css` | 509 | **382** |

`--lp-vio: #a78bfa` (violet hors palette) **supprimé** → `var(--fvg-l)` : le violet signifie
déjà « Fair Value Gap » dans ce produit, un second violet qui ne signifiait rien **était** le
problème d'accent décoratif.

**§5 appliqué à la lettre** : `.plegal` (la mention légale !), `.roadmap`, `.illus`, `.tlab`,
`.mbAction`, `.bill`, `.pcLbl` et `.try` quittent la chasse fixe — ce sont des **mots**.
`.dl` (étiquettes de graphique), `.calcRow`, `.chip`, `.mt`, `.price` la gardent — ce sont des
**nombres**.

**Purge** : 103 sélecteurs morts retirés **par analyse**, pas à la main (`prune_css.py` :
une règle survit si une seule de ses classes est encore référencée).

### 4.1 Bug préexistant trouvé et corrigé

`.mb` portait le `padding`, la `max-width` et le rayon de **toutes les bulles de chat de la
démo** — et **rien ne l'appliquait jamais** : le JSX n'utilisait que `.mbU`/`.mbA`/`.mbNo`.
Les bulles se rendaient donc bord à bord, sans marge intérieure. La purge l'a retiré comme
mort (correct) ; les propriétés sont remises là où elles devaient être depuis le début.

---

## 5. Ce qui n'a pas bougé — vérifié, pas supposé

| Invariant | Statut |
|---|---|
| Logo colonne 5 bougies, laiton, `--brand-mark`/`--brand-word` (BRD-3) | **Aucun fichier logo touché** |
| Les 4 palettes (THM-1) | la landing n'utilise **que** des jetons |
| Chasse fixe pour les nombres | conservée ; **retirée** des mots (§4) |
| Vocabulaire interdit, 9 locales | ✅ vérifié (§3.4) |
| Portée marchés (LP-2S) | `hero.roadmap` verbatim, garde-fou intact ✅ |
| « Décoche une couche » pointe la **vraie** commande (LP-2B §1) | ✅ les puces, jamais un raccourci |
| M.I.A = le vrai agent, dégradation **étiquetée** (MIA-4S) | ✅ `MiaPane` réutilisé |
| Une carte M.I.A = une phrase (LP-2A) | ✅ **garde-fou passé sans être modifié** |
| `miaSection.p` explicitement hors périmètre de coupe (LP-2A) | ✅ **déplacé verbatim, pas coupé** |
| Mentions légales (dollars US, risque de perte, 18 ans, ni conseil) | ✅ conservées, **rendues lisibles** |
| `config/demo_illustration.json` ↔ `data.ts` | non touchés, parité verte |

> **Correction en cours de route** : j'avais d'abord supprimé `miaSection.p` avec la section
> qui le portait. Le garde-fou LP-2A l'a attrapé (sa règle n°5 : « l'intro est explicitement
> HORS périmètre »). Restauré verbatim sur les 9 locales et rendu dans le bloc refus —
> **le garde-fou n'a pas été assoupli.**

---

## 6. Vérifications

| Vérification | Résultat |
|---|---|
| `tsc --noEmit` | **0 erreur** |
| `npm run build` | **vert** · accueil 13,6 kB / 200 kB first-load |
| vitest — `components/landing` + `components/gallery` + `galerie` | **109 / 109** (7 fichiers) |
| vitest — `components/landing` seul, 2 passes | **100 / 100** |
| vitest — `home.test.tsx` (réécrit) | **30 / 30** |
| vitest — `lp2a-copy.test.ts` (**non modifié**) | **8 / 8** |
| vitest — `demo-illustration-parity` (**non modifié**) | vert |
| Playwright — `lp1-accueil.spec.ts` (réécrit) | **36 / 36** (fr + en × desktop + mobile) |
| Playwright — captures 1280×800 + 390×844, fr + en | **5 / 5** |

> `--pool=threads` obligatoire : `--pool=forks` ne démarre aucun worker à travers la jonction
> `node_modules` cross-worktree.

### 6.1 Couverture du nouveau spec e2e

Le spec est en deux moitiés, **volontairement** :

- la boucle `LOCALES × VIEWPORTS` tourne en **reduced motion**, c'est-à-dire sur le chemin
  **dégradé** de la section au scroll. C'est ce que veut la plupart des assertions : c'est
  stable, et c'est ce que reçoit un visiteur en reduced motion, sans JS ou sur petit écran.
  **Rien ne doit y manquer.**
- le dernier `describe` ne réduit **pas** le mouvement : c'est le seul endroit où la
  chorégraphie épinglée est exercée, et il lui faut un vrai viewport à défiler — précisément
  ce que jsdom ne peut pas donner à `home.test.tsx`. Il vérifie que l'étape 1 n'a **qu'une**
  couche, que l'étape 4 les a **toutes** (le scroll n'a fait qu'ajouter), et que prendre une
  puce met fin à la chorégraphie **pour de bon** (remonter ne redessine pas la couche retirée).

**Défaut de spec corrigé** : mes premiers localisateurs ciblaient le texte exact des
conditions du scanner — or ce sont des `<button>` dont le texte propre inclut la puce `✓`.
Repassés sur le rôle, comme le faisait le spec d'avant LP-3.

### 6.3 Un flake réel, traité et non masqué

Deux tests de `home.test.tsx` échouaient par intermittence **quand la machine était
chargée** — jamais sur leur assertion, toujours sur le délai : « the hero renders the honest
markets line » (5 876 ms) et « every layer off yields the honest empty state » (5 043 ms),
contre un budget vitest par défaut de 5 000 ms.

Cause réelle : `<HomeLanding />` monte désormais la **vraie** `<MarketReadingCard>` (en-tête,
panneau de phase, quatre sections repliables) en plus des démos ; un rendu pleine page en
jsdom dépasse 5 s sur cette machine (Defender + `node_modules` par jonction). Le budget est
donc environnemental, et il est **déclaré** en tête du fichier (`vi.setConfig({ testTimeout:
20_000 })`) avec la raison, plutôt que laissé à flotter.

### 6.2 Captures

`docs/audits/lp-3/shots/`

```
before/  fold-{fr,en}-{desktop-1280x800,mobile-390x844}.png
         full-{fr,en}-{desktop-1280x800,mobile-390x844}.png
after/   (les mêmes)
         unfold-1-La-cassure.png · unfold-2-La-zone.png · unfold-3-La-liquidité.png
```

Les trois `unfold-*` capturent le chemin **enrichi** aux trois positions qui comptent, pour
que la chorégraphie soit lisible sans lancer l'app.

**Page entière, 1280 px : 9 301 px → 7 430 px de haut.**

> Les captures « before » sortent du build de la base (LP-2S + main) servi par `next start` ;
> les sources avaient déjà été modifiées, ce qui est sans effet — `next start` sert `.next`.
> Le badge de fraîcheur de la carte étant relatif à l'heure réelle, deux captures prises à des
> jours différents ne sont pas identiques au pixel près.

---

## 7. Réserves et points ouverts

1. **`ReadingChart` (lightweight-charts) n'a pas été mis sur la landing.** L'ouverture utilise
   `MarketReadingCard` sans `chartSlot` — DOM pur. La bibliothèque de graphiques n'est donc
   **pas** ajoutée au chargement initial de la page publique, et le problème connu « ne peint
   pas en headless » ne se pose pas. Le graphique de la section au scroll reste le SVG de la
   démo, ce que la mission demandait explicitement de réutiliser.
2. **Données d'archive réelles en production** : `lib/ds-samples/` n'est pas gated (seule la
   route `/galerie` l'est). L'ouverture affiche donc une vraie lecture d'archive sur une page
   publique, **étiquetée comme telle**. C'était l'arbitrage n°4 du plan.
3. **`--fs-display` et `--fs-chartlabel`** sont deux ajouts à l'échelle des 6 tailles. Ils sont
   déclarés et commentés en tête de `lp1.module.css`, pas glissés.
4. **Les 13 composants `landing/*.tsx` morts** ne sont pas supprimés — hors périmètre.
5. **Le libellé « PHASE DE MARCHÉ »** dans la carte de l'ouverture est en capitales espacées,
   mais il vient du **vrai composant produit** (`MarketPhasePanel`), pas de la CSS de la
   landing. C'est l'idiome du produit ; le corriger serait une mission `/app`, pas LP-3.
6. **Poids avant/après du bundle non comparé** : la sortie du premier build a été perdue
   (stdout vide). Seul l'après est mesuré : 13,6 kB / 200 kB.

---

## 8. Fichiers

**Nouveaux**
```
webapp/components/landing/lp1/Opening.tsx
webapp/components/landing/lp1/ReadingUnfolds.tsx
webapp/components/landing/lp1/RefusalBlock.tsx
webapp/components/landing/lp1/MiaPane.tsx          (extrait de DemoTabs)
webapp/components/landing/lp1/structure-chart.tsx  (extrait de DemoTabs, partagé)
webapp/tests/e2e/lp3-shots.spec.ts
docs/audits/AUDIT-lp-3-refonte-accueil.md
docs/audits/lp-3/shots/{before,after}/
```

**Supprimés**
```
webapp/components/landing/lp1/MiaSection.tsx       (chat scripté + 4 cartes à icônes)
webapp/components/landing/lp1/ReadingCarousel.tsx  (absorbé par la section au scroll)
```

**Modifiés**
```
webapp/components/landing/lp1/HomeLanding.tsx
webapp/components/landing/lp1/DemoTabs.tsx
webapp/components/landing/lp1/lp1.module.css
webapp/components/landing/lp1/__tests__/home.test.tsx
webapp/components/gallery/DesignGallery.tsx        (les 2 blocs landing retirés)
webapp/tests/e2e/lp1-accueil.spec.ts
webapp/messages/{fr,en,de,es,it,pt,nl,pl,ar}.json
```

---

## 9. Reste à faire

- [ ] **Confirmation visuelle live du fondateur** avant merge (condition de la mission).
- [ ] Trancher PR **#203 (LP-2S)** — la merger rend le diff de LP-3 propre (§0.3).
- [ ] Décider du nettoyage des `node_modules` des worktrees mergés (§0.2).

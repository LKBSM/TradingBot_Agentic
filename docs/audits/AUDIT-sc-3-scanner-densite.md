# AUDIT SC-3 — Scanner : densité de texte + cartes visuelles

## Position git

| | |
|---|---|
| Branche | `feat/sc3-scanner-density` |
| Base | `origin/main` = `54265ef` (Merge PR #200 `feat/mia-3-agent-unifie`) |
| Écart au démarrage | **0** — le worktree `wt-sc-3` est créé directement depuis `origin/main` après `git fetch` |
| Commits | 3, dans l'ordre annoncé : « cible » → auto-refresh → carte |

> Note de méthode : le diagnostic préalable avait été mené depuis un worktree principal
> **57 commits derrière** `origin/main`. Il a donc été fait intégralement par
> `git show origin/main:<path>`, jamais contre le HEAD local périmé, et l'implémentation
> se fait dans un worktree neuf branché sur `origin/main`.

---

## Dépendance : la maquette

`docs/design/scanner_v2.html` n'était **pas** dans le dépôt (absent du working tree, d'`origin/main`,
de `wt-run-main` et de tout l'historique). Le fichier a été retrouvé dans
`C:\Users\bessa\Downloads\scanner_v2.html` — le dépôt manuel s'était arrêté au téléchargement.
Il a été lu intégralement de là, puis **copié dans le dépôt au commit 3** (12 292 octets, 264 lignes,
md5 `bc280370b99a77fc6f91f1f2346d6071`, identique à l'original).

---

## Commit 1 — `2467176` · le mot interdit « cible »

**Fichiers**

| Fichier | Rôle |
|---|---|
| `src/intelligence/conditions_scanner.py` | les 15 chaînes |
| `tests/test_conditions_scanner.py` | le garde-fou étendu aux `detail` |
| `webapp/components/scanner/ComboCard.tsx` | `VALUE_ONLY_DETAIL` + composant `ConditionText` |
| `webapp/components/scanner/__tests__/ComboCard.test.tsx` | les deux formes de composition |

**La source n'était pas dans le frontend.** `ComboCard.tsx` recopie verbatim le `label` et le
`detail` émis par le backend. Un grep sur les **9** bundles i18n ne trouve « cible » que dans deux
**négations** légitimes (`fr.json:803` « ni une cible », `fr.json:1420` « pas de cible ») et un faux
positif espagnol (`es.json:767` « reha**cible** ») — d'où le mot-entier obligatoire dans le garde-fou.

**Les 8 chaînes visibles, avant → après**

| Ligne | Condition | Avant | Après |
|---|---|---|---|
| 635 | `higher_tf_agrees` | `Le 1 h va dans le même sens (haussier) — cible : même sens.` | `dans le même sens : le 1 h est haussier.` |
| 650 | `trend_is` | `Tendance structurelle observée : haussier (cible : haussier).` | `haussier.` |
| 691 | `last_event_is` | `Dernier événement : BOS ↑ il y a 3 bougie(s) (cible : BOS ↑).` | `BOS ↑, il y a 3 bougie(s).` |
| 713 | `last_event_age` | `Dernier événement (BOS ↑) il y a 3 bougie(s) — tranche « moins de 10 » (cible : « moins de 10 »).` | `3 bougie(s) — tranche « moins de 10 » (BOS ↑).` |
| 1078 | `market_phase_is` | `Phase observée : trend (cible : ranging).` | `trend.` |
| 1090 | `volatility_is` | `Volatilité observée : normale (cible : contractée).` | `normale.` |
| 1146 | `price_in_range_third` | `Prix à 7 % du range structurel […] — tiers bas (cible : tiers haut).` | `Prix à 7 % du range structurel […] — tiers bas.` |
| 1179 | `session_is` | `Session à la clôture : Londres (cible : Asie).` | `Londres, à la clôture de la bougie.` |

**Les 7 fallbacks** : lignes 592, 644, 679, 1074, 1083, 1127, 1152 —
`« Relation cible non précisée. »` → `« Relation non précisée. »`, etc.

**La composition dans la carte.** Sept de ces huit détails sont désormais une **valeur** qui complète
le libellé, rendue **sans tiret** :

```
« La tendance structurelle est » + « haussier. »  →  La tendance structurelle est haussier.
« L'unité supérieure va »        + « dans le même sens : le 1 h est haussier. »
« La phase de marché est »       + « trend. »
```

Pilotée par `VALUE_ONLY_DETAIL`, un `Set<ConditionType>` explicite et documenté dans `ComboCard.tsx`.

⚠️ **Une exception, à connaître** : `price_in_range_third` est le seul des huit à **rester au tiret**.
Son libellé porte un substitut en ligne — « Le prix est dans le tiers **…** du range » — donc une
valeur accolée après lui ne se lit pas en français. Son détail reste une phrase autonome, amputée du
seul mot interdit. C'est le seul écart mécanique par rapport à la consigne « la même logique de
composition s'applique aux 7 autres » ; il est commenté dans le code et verrouillé par un test qui
liste explicitement les 7 types inclus et 4 types exclus.

Les items de `context_against` gardent aussi le tiret : ce sont des paires *libellé — explication*,
pas *libellé + valeur*, même s'ils vivent dans le même bloc que les conditions non remplies.

Un détail énonce toujours la valeur **observée**, jamais celle demandée : la marque ✓/✗ dit déjà si
elle correspond, et la condition demandée est affichée dans « Ma stratégie ».

**Le trou de garde-fou, fermé.** `test_palette_has_no_predictive_vocabulary` ne scannait que
`type` + `label` + `description` des entrées de `PALETTE`. Les `detail` produits par les 22 fonctions
`_eval_*` — c'est-à-dire **tout** ce qu'une carte de résultat affiche — n'étaient scannés par rien.
`test_condition_details_have_no_predictive_vocabulary` parcourt maintenant chaque entrée de palette ×
chaque valeur de chaque contrôle × 3 lectures (riche / vide / indéterminée) × 4 cartes de tendance,
combinaison vide incluse pour atteindre les fallbacks : **1 872 détails, 22 types couverts**, plus une
assertion de couverture (aucun type ne peut contribuer zéro détail). Les items de `context_against`
sont scannés de même. Vérifié par mutation : replanter `(cible : x)` fait bien échouer le test.

---

## Commit 2 — `7c8a2dd` · retrait complet de l'actualisation automatique

Fonctionnalité entière, pas seulement le contrôle. **22 fichiers, +12 / −536.**

**Supprimés** (plus aucun appelant dans le dépôt, vérifié par grep) :
`components/scanner/AutoRefreshToggle.tsx`, `lib/conditions/auto-refresh-store.ts`
(préférence `mia.scannerAutoRefresh.v1`), `lib/conditions/use-candle-close-refresh.ts`,
`lib/conditions/candle-clock.ts`, et les deux fichiers de tests de ces deux derniers.

**Débranchés** : `ScanResults` (import, 2 props, rendu dans `.pghead`), `ScannerWorkspace`
(`useAutoRefreshPref`, le bloc `useCandleCloseRefresh` **et** le mémo `timeframes` qui ne servait
qu'à lui), `ConversationalScanner` (2ᵉ surface montant le même toggle), `DesignGallery` (DS-1,
2 aperçus).

**i18n** : `scanner.toggle.{label,aria}` retiré des **9** locales par découpe chirurgicale du bloc
(jamais de round-trip JSON — les bundles sont en CRLF et leur ordre de clés est relu par des
humains). Diff vérifié à **4 lignes par bundle**, avec égalité clé à clé de tout le reste.

**Ce qui ne bouge pas** : le badge « Aligné sur les clôtures » et le message « aucune nouvelle
clôture » — ce sont des constats sur la donnée, pas le mécanisme. Le bouton « Relancer le scan »
devient le seul déclencheur, avec l'enregistrement de conditions modifiées.

Les deux tests du commutateur sont remplacés par leur **inverse** : la barre d'outils ne doit plus
exposer aucun `role="switch"` ni le libellé « Actualisation auto ».

---

## Commit 3 — la carte visuelle

**Fichiers** : `webapp/components/scanner/ComboCard.tsx`,
`webapp/components/shell/pages.css`, `webapp/messages/*.json` (9 locales, +5 clés),
`webapp/components/scanner/__tests__/sc3-card.test.tsx` (nouveau),
`webapp/tests/e2e/sc3-card-density.spec.ts` (nouveau),
`webapp/components/scanner/__tests__/ComboCard.test.tsx` (1 assertion rendue agnostique du marqueur),
`docs/design/scanner_v2.html` (la maquette, versionnée).

### Bloc 1 — condensé sans amputer
La **première** condition remplie porte le poids plein (`.cl.yes.lead`, marqueur ✓, `--txt`). Les
suivantes restent dans la même famille mais reculent (`.cl.yes.sub`, `--dim`, marqueur non répété).
**Toutes** sont rendues : une carte à quatre conditions affiche quatre lignes, aucune tronquée,
aucune masquée. La densité chute par la composition (§ commit 1), pas par la suppression.

### Rangée de puces
Six puces au maximum, chacune émise **uniquement** si son champ source existe :

| Puce | Jeton | Champ source (`match.context`) | Absente quand |
|---|---|---|---|
| tendance | `--bull` / `--bear` / `--faint` | `trend` | `trend` est `null` |
| phase | `--faint` | `market_phase` | `market_phase` est `null` |
| `N OB` | `--ob-l` (rond) | `active_order_blocks` | compte à 0 |
| `N FVG` | `--fvg-l` (**carré**) | `active_fair_value_gaps` | compte à 0 |
| BOS | `--bull` / `--bear` | `bos.direction` | `bos` est `null` |
| range | `--acc` | `structural_range` | `structural_range` est `null` |

**Aucune puce de liquidité** : `ComboContext` ne porte aucun champ de poche, de sweep ni de niveaux
égaux. Pas de champ → pas de puce, conformément à la consigne. Aucun champ n'a été ajouté au backend.

**Aucun appel réseau supplémentaire** : tout vient de la réponse `POST /api/conditions-scan` que la
carte reçoit déjà.

Un compte à zéro est une vraie mesure, pas une donnée absente — mais une puce « 0 OB » ne dit rien,
donc la rangée ne garde que ce qui est présent. **Rien n'est perdu** : le bloc 3 continue d'énoncer
les comptes exacts, zéro compris (vérifié par test).

### 🔴 `--ob` / `--fvg` : « tels quels » n'était pas rendable
Dans `globals.css`, `--ob` et `--fvg` sont des **remplissages de zones de graphique translucides**
(4 à 13 % d'opacité) — invisibles sur une pastille de 7 px. Et sur le thème `schema` ils valent
**tous deux du blanc** (`rgba(255,255,255,0.05)` et `0.04`), donc indiscernables. La maquette ne les
avait d'ailleurs pas réutilisés : elle les **redéfinit en opaque** dans son propre `:root` (ligne 13).

Décision D2 appliquée, **sans créer aucun jeton** :
1. on prend les variantes « ligne » `--ob-l` / `--fvg-l` (0,22–0,5) et on les **composite sur un
   `--panel-2` opaque à l'intérieur même de la pastille** (`background: linear-gradient(tok, tok), var(--panel-2)`),
   ce qui garantit une pastille pleine quel que soit le fond ;
2. la pastille FVG est en plus **carrée** (`border-radius: 1px`) — les deux restent distinguables
   même là où le thème les peint de la même couleur, et c'est accessible aux daltoniens par surcroît.

Un test Playwright mesure les deux : chaque pastille est effectivement peinte (`background-image ≠ none`
ou couleur non transparente) et `border-radius(OB) ≠ border-radius(FVG)`.

### Bloc 2 — inchangé, jamais replié
La maquette le replie dans un `<details>` (lignes 198-203). **On s'en écarte délibérément sur ce seul
point** : « a reading you only half-read is a reading you misread » est une promesse écrite dans le
code, sur la landing (`AUDIT-lp-1-accueil.md:34`) et dans `AUDIT-lp-2.md:46`. Le bloc reste déplié,
sans `<details>` — pas même un ouvert par défaut.

Garde-fous : `sc1-results.test.tsx` (2 tests existants, intacts), `txt1-copy.test.ts` (clé protégée,
intacte), `sc1-scanner.spec.ts:122` (existant), **plus** deux nouveaux tests qui manquaient et qui
vérifient la propriété **structurelle** : `against-block` n'a **aucun ancêtre `<details>`**
(`sc3-card.test.tsx` en unitaire, `sc3-card-density.spec.ts` aux deux viewports).

À noter pour les prochaines missions : `txt1-copy.test.ts` ne gardait **pas** la non-repliabilité —
il ne vérifie que la présence de la clé i18n en fr et en en. C'est ce test structurel-là qui manquait.

### Bloc « Non évaluable ici » — inchangé
Absent de la maquette, conservé tel quel, déplié. Il ajuste le dénominateur : c'est une garantie
d'honnêteté, pas du décor.

### Bloc 3 — replié, sans rien perdre
`<details className="ctxd">` fermé par défaut. Le corps `.ctx2` est **inchangé** : chaque fait reste
à un clic, aucun n'est retiré. Le compte d'actus à fort impact monte dans le `<summary>`
(« Contexte que tu n'as pas demandé · 2 actus importantes ») pour rester lisible carte fermée —
un test verrouille que `contextBlockWithNews` **préfixe exactement** `contextBlock` dans les 9 locales,
pour qu'un traducteur ne puisse pas désynchroniser la paire.

### ⚠️ Un ajout non prévu : la grille de résultats n'avait aucun point de rupture
Mesuré à 390×844 sur le build de production : `.resgrid` restait en **deux colonnes de 181 px et
158 px**. Ce n'est pas une régression SC-3 — la règle `grid-template-columns: 1fr 1fr` n'a jamais eu
de media query — mais c'est la vraie raison pour laquelle la rangée de puces n'a nulle part où
s'enrouler sur téléphone, et 158 px n'est pas une carte lisible.

La maquette prescrit explicitement une colonne unique (`@media (max-width:860px)`). J'ai ajouté la
règle **au point de rupture du dépôt** (`767px`, celui déjà utilisé par `.scanner-actionbar` et le
pied d'avertissement mobile) plutôt qu'à 860 px, par cohérence avec le reste de `pages.css`.
Aucune règle produit ne défendait la grille à deux colonnes sur téléphone. Verrouillé par une
assertion Playwright : la carte occupe plus de 70 % de la largeur du viewport aux deux tailles.

### Typographie
Prix, horodatages, comptes OB/FVG et bornes de range portent `.mono`
(`var(--font-mono)` = JetBrains Mono + `tabular-nums`, déjà défini dans `globals.css:369`).
Le reste reste en Inter. Aucune police ajoutée.

### Non touchés
Logique de filtrage, onglets Correspondances / Presque / Non correspondants / Non évaluables,
comptage de combos, dénominateur et compte de non-évaluables : intacts.

---

## Vérifications

| Vérification | Résultat |
|---|---|
| `pytest tests/test_conditions_scanner.py` | **52 passed** (dont 4 nouveaux) |
| Mutation du garde-fou (replanter `(cible : x)`) | **échoue comme attendu** |
| vitest `components/scanner` + `components/gallery` + `txt1-copy` | **65 passed / 10 fichiers** |
| vitest `components/scanner` (dont `sc3-card.test.tsx`, 15 tests) | **50 passed / 7 fichiers** |
| `tsc --noEmit` | **0 nouvelle erreur** (3 pré-existantes dans `dictation-copy-honesty.test.ts`) |
| `next build` (CI=1) | **vert** — `✓ Compiled successfully` |
| Playwright `sc3-card-density.spec.ts` | **28 passed** — 7 tests × 2 viewports × 2 projets |
| Playwright `sc1-scanner.spec.ts` | **passe intégralement** |
| Playwright `cln-1-disclaimers.spec.ts` | 8 échecs **pré-existants**, tous sur `/compte` (voir ci-dessous) |

Playwright couvre, **à 1280×800 et 390×844** : la rangée de puces s'enroule dans sa carte sans jamais
provoquer de défilement horizontal, dans une carte de plus de 300 px (grille à 1 colonne sous 768 px,
2 au-dessus) ; chaque pastille est réellement peinte et OB ≠ FVG en forme ;
le bloc 3 est fermé au chargement et s'ouvre au clic ; le bloc 2 est visible sans interaction et hors
`<details>` ; un combo sans cassure, sans range et sans zone active n'affiche **aucune** puce vide
(ni tiret, ni « 0 OB ») ; **un seul** avertissement légal **visible** par page ; plus aucun
`role="switch"` dans la barre d'outils.

### Non-régression : `sc1-scanner.spec.ts` et `cln-1-disclaimers.spec.ts`
Lancés ensemble sur le build de production : **56 passed, 8 failed**. Les 8 échecs sont **tous** sur
`/compte`, dans `cln-1-disclaimers.spec.ts`, et sont **pré-existants et étrangers à SC-3** :
la page en rend 5 occurrences dans le HTML brut alors que le test en attend 1 visible (même flake
`/compte` déjà relevé lors de la session MIA-3 le même jour, sur ce même spec).

Preuve que SC-3 ne peut pas en être la cause : le seul fichier partagé avec `/compte` est
`pages.css`, et **tous** les sélecteurs que SC-3 y ajoute sont scopés sous `.combo`
(plus `.cl.yes.sub`, qui n'existe que sur une carte de scanner) — `/compte` ne contient aucun
`.combo`. Les commits 1 et 2 ne touchent pas du tout `pages.css`. **`sc1-scanner.spec.ts` passe
intégralement** : les invariants SC-1 (trois blocs, bloc « à l'encontre » non masquable, aucun
classement, les cinq états) tiennent avec la nouvelle carte, aux deux viewports.

### Deux détails de méthode, pour la prochaine fois
- L'avertissement légal existe en **deux nœuds DOM** (pied de rail + pied mobile) dont **un seul est
  visible** à chaque largeur — c'est le design CLN-1 §5. Compter les nœuds donne 2 et fait échouer un
  test naïf ; il faut compter les nœuds **visibles**, comme le fait `cln-1-disclaimers.spec.ts`.
- La page contient légitimement « …pas de **cible** » (la note de palette, présente aussi dans la
  maquette). L'assertion « pas de cible » doit être **scopée aux cartes** et chercher la forme
  positive `(cible : …)`, pas le mot nu : une **négation** de la chose interdite est la promesse du
  produit, pas un usage.

### Environnement
`node_modules` monté en jonction depuis `wt-ci-infra` (lock identique). Vitest échoue par intermittence
en `--pool=forks` (« Timeout waiting for worker to respond ») ; **`--pool=threads --no-file-parallelism`
passe** — et même là, une première tentative peut échouer, une relance suffit. Playwright tourné contre
un **build de production** sur un port dédié (3177) avec `E2E_BASE_URL`, jamais via `CI=1` (qui refuse
de réutiliser un serveur déjà lancé). `next build` émet un avertissement
`EPERM … symlink … .next/standalone/node_modules` : c'est l'étape de traçage de la sortie
`standalone` qui bute sur la jonction ; la compilation est verte et `next start` sert `.next`, pas
`standalone` — sans effet ici.

---

## Confirmations demandées

- ✅ **« cible » absent** : 8 chaînes visibles + 7 fallbacks retirés ; zéro occurrence restante dans
  `conditions_scanner.py` ; aucun usage positif ailleurs dans le frontend (les 2 occurrences fr sont
  des négations, l'occurrence es un faux positif de sous-chaîne) ; garde-fou backend étendu aux
  `detail`, en **mot entier**, vérifié par mutation.
- ✅ **Bloc 2 intact** : toujours déplié, dans aucun `<details>`, tous les garde-fous existants verts,
  deux nouveaux ajoutés. Écart à la maquette assumé et documenté.
- ✅ **Puce liquidité absente** : `ComboContext` ne porte aucun champ de poche / sweep / niveaux égaux.
  Aucun champ ajouté au backend. La puce ambre de la maquette est le **range structurel**
  (`context.structural_range`), peinte en `--acc` et non en `--liq`.
- ✅ **Aucun appel réseau supplémentaire** ; **aucune donnée inventée** ; **aucun jeton créé**.

---

## Reste à faire

Merge sur `main` **après confirmation visuelle live du fondateur, aux deux résolutions** —
1280×800 et 390×844, sur les 4 thèmes (`terminal`, `atelier`, `schema`, `ardoise`), le thème `schema`
étant celui qui vérifie la distinction OB/FVG par la forme.

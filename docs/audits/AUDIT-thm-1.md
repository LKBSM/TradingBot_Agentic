# AUDIT THM-1 — Palettes institutionnelles

> **Statut : IMPLÉMENTÉ après GO. Décisions du fondateur : overlay = mission séparée ·
> logo = neutre unique · libellés = clé i18n localisée.** En attente de la confirmation
> visuelle live dans les 4 thèmes avant merge sur main.
> Le diagnostic (phase STOP) est conservé ci-dessous ; le journal d'implémentation est en §10.

## 0. Discipline — position du HEAD (vérifiée AVANT diagnostic)

- `git fetch origin` effectué en 1ʳᵉ action.
- **`origin/main` = `6d658ef`** (Merge #194, fix/cln-1-nettoyage).
- Répertoire principal était sur `docs/preserve-data-1-audit` `e0dc69c` = **30 commits DERRIÈRE `origin/main`** (piège DATA-1). Diagnostic **non** fait depuis là.
- **Worktree dédié créé** : `C:\MyPythonProjects\wt-thm-1`, branche **`feat/thm-1-palettes`** partant de `6d658ef` (à jour). `HEAD = 6d658ef`, écart avec `origin/main` = **0**.
- `node_modules` : jonction depuis `wt-ci-infra` (lock md5 identique `56ba4dc8…`, `.bin` complet). À retirer avant démontage du worktree.

---

## 1. Question A — LE GRAPHIQUE SUR THÈME CLAIR (la plus importante)

**Réponse : le graphique est DÉJÀ dégradé sur le thème clair actuel (« atelier »), et le
Parchemin l'aggrave légèrement. C'est un défaut PRÉEXISTANT, indépendant de THM-1.**

Les couleurs des couches SMC sont écrites en dur dans `lib/chart/zoneOverlayPrimitive.ts`
(et les niveaux BOS/CHOCH/retest dans `ReadingChart.tsx`), pensées pour un fond sombre.
Elles ne suivent pas le thème (seul le `labelBg` de la pastille de texte l'est déjà).

### Capture

`docs/audits/thm-1-shots/overlay-legibility.png` — l'overlay redessiné avec les **constantes
exactes** du primitive (OB `#8B95A7` fill 0.12/bord 0.45, FVG `#6E84B0`, liquidité BSL
`#4FA3C7`/SSL `#C77FA3`, live `#C9A227`, labels), sur trois fonds côte à côte : Graphite
(sombre), Atelier actuel (`#f6f4ee`), Parchemin (`#FBFAF8`).

> Pourquoi une reconstitution fidèle et non une capture du vrai canvas : lightweight-charts
> **ne peint pas** son canvas dans l'environnement headless dev (le canvas existe, dimensionné
> 606×346, la donnée charge — le prix d'en-tête se met à jour — mais rien n'est peint ; la
> spec `chart2-amplitude` elle-même n'assère jamais des bougies peintes, seulement la présence
> du `<canvas>`). Les couleurs de l'overlay étant des **constantes en dur**, la reconstitution
> est exacte au pixel de couleur près. Captures du shell complet dans les 4 thèmes actuels :
> `full-{atelier,terminal,schema,ardoise}.png` (le plot y est vide, même limite headless).
> **Je referai une capture du vrai graphique en direct au moment de ta confirmation visuelle.**

### Contraste des couches sur fond clair (WCAG, seuil lisible ≈ 3.0)

| Couche (couleur en dur)     | Graphite `#0C0D0F` | Atelier ACTUEL `#f6f4ee` | Parchemin `#FBFAF8` |
|-----------------------------|:---:|:---:|:---:|
| label OB `#9AA4B8`          | 7.76 | **2.28** ✗ | **2.40** ✗ |
| FVG `#6E84B0`               | 5.18 | 3.42 | 3.60 |
| box OB base `#8B95A7`       | 6.44 | — | **2.89** ✗ |
| liquidité BSL `#4FA3C7`     | 6.85 | **2.58** ✗ | **2.72** ✗ |
| liquidité SSL `#C77FA3`     | 6.47 | — | **2.88** ✗ |
| live `#C9A227`              | 8.04 | **2.20** ✗ | **2.32** ✗ |

Et surtout, **le remplissage de boîte composité** (fill 0.12 sur fond clair) devient
quasi invisible : OB `#8B95A7`@0.12 sur `#FBFAF8` → `#EEEEEE` (contraste **1.11**), bordure
@0.45 → `#C9CDD4` (**1.53**). Sur l'image, les boîtes OB/FVG sont des rectangles gris
fantômes ; les labels sont illisibles ; les lignes de liquidité sont délavées ; seul le
front live-ambre tient à peu près.

**Recommandation : traiter la coloration de l'overlay dans une MISSION SÉPARÉE** (elle exige
de faire dériver les couleurs des couches du thème — les passer via `setData` comme le
`labelBg` l'est déjà — ce qui dépasse un simple échange de valeurs de jetons). THM-1 remplace
les palettes ; il ne peut pas rendre lisible un overlay codé en dur pour le sombre. Le
Parchemin est **livrable**, à condition d'assumer que son graphique montrera le même overlay
pâle que l'Atelier actuel jusqu'à cette mission séparée. **À toi de décider : ici ou séparé.**

---

## 2. Question B — Couleurs de DONNÉES sur les nouveaux fonds

**NE PAS MODIFIER sans ton accord.** Les palettes fournies ne spécifient AUCUN jeton de
donnée (`--bull/--bear/--ob/--fvg/--liq`) : chaque thème **conserve ses valeurs actuelles**.

Contraste des valeurs demandées (`#37b98c` / `#dd6b7a`) + liq `#d6a24a` sur les 4 fonds :

| Donnée | Graphite `#0C0D0F` | Acier `#101215` | Encre `#0A0A0B` | Parchemin `#FBFAF8` |
|--------|:---:|:---:|:---:|:---:|
| bull `#37b98c` | 7.85 | 7.58 | 7.99 | **2.37** ✗ |
| bear `#dd6b7a` | 5.98 | 5.77 | 6.09 | 3.12 |
| liq `#d6a24a`  | 8.45 | 8.15 | 8.60 | **2.21** ✗ |

⚠️ **Signal important** : ces valeurs (celles de « terminal ») échouent sur le Parchemin.
Mais le Parchemin = ancien « atelier », qui **redéfinit déjà** ses données en variantes
sombres adaptées au clair. Mesurées sur `#FBFAF8` : bull `#2f8f6b` = **3.83**, bear
`#b4564f` = **4.59**, liq `#b07d1c` = **3.47** → toutes ≥ 3, OK.
→ **À VERROUILLER : le Parchemin doit GARDER les valeurs de données claires de l'atelier ;
il ne doit JAMAIS hériter des valeurs sombres de « terminal ».** (Un simple échange de
valeurs de jetons respecte ça automatiquement, mais je le signale car c'est le seul piège.)
Les 3 thèmes sombres : leurs fonds s'assombrissent → le contraste des données **s'améliore**.

⚠️ **Second signal (identité vs sens) sur Graphite** : la donnée `--liq #d6a24a` (or) est
quasi identique au nouvel accent d'interface `--acc #C9A14A` (laiton). Sur le graphique, la
« liquidité » (sens) et l'« action » (identité) auraient la même couleur. Rien à changer
sans ton accord (data = sens), mais à noter.

---

## 3. Question C — LE LOGO : recommandation

`--brand-mark` = `#7da3ff` (sombre) / `#2962ff` (clair). Sur Graphite, un logo bleu à côté
d'un accent laiton = deux accents qui se disputent (et le problème se répète : bleu vs acier
sur Acier c'est ok, mais bleu vs laiton/bronze/encre ailleurs, non).

**Recommandation : Option (a) — couleur unique de marque sur tous les thèmes, MAIS neutre
(pas le bleu actuel, pas l'accent du thème).** Concrètement : rendre la marque dans le
**texte/premier plan neutre du thème** (ton `mono` déjà présent dans `MiaLogo`, = `currentColor`
hérite de `--txt`), pas une identité colorée et pas un second accent.

Justification :
1. **Cohérence de marque** : l'identité est la FORME du prisme, pas un bleu précis. En neutre,
   la marque a le même poids dans les 4 thèmes — plus cohérente qu'un bleu fixe qui jure sur
   certains fonds, et qu'un accent variable (qui ferait du logo un élément d'UI, pas d'identité).
2. **Règle 2 (un seul accent, rarement)** : un logo neutre ne concurrence jamais l'accent
   laiton/acier/bronze. Le logo cesse d'être un accent.
3. **Encre l'impose déjà** (accent = neutre) ; l'unifier en neutre aligne les 4 thèmes.

C'est un changement de teinte du logo (bleu → neutre). **Tu tranches.** Alternative si tu
tiens au bleu de marque : garder `#2962ff`/`#7da3ff` FIXE partout (option a stricte) — mais
alors le conflit bleu↔laiton sur Graphite demeure, ce qui est exactement le reproche initial.

---

## 4. Question E — Table de correspondance des jetons de rôle (À VALIDER AVANT APPLICATION)

Les deux vocabulaires coexistent : littéraux (`--bg/--panel/--acc…`) et rôles shadcn en HSL.
La correspondance **actuelle** (terminal) confirme que chaque rôle dérive d'un littéral par
son RÔLE ; je la conserve et je recalcule le HSL depuis les nouveaux hex.

**Correspondance de rôle (inchangée) :**

| Rôle shadcn (HSL)            | ← Littéral source | Rôle |
|------------------------------|-------------------|------|
| `--background`               | `--bg`            | fond de page |
| `--foreground`               | `--txt`           | texte principal |
| `--card` / `--popover`       | `--panel`         | surface de carte |
| `--card/popover-foreground`  | `--txt`           | texte sur carte |
| `--primary` / `--ring`       | `--acc`           | action / focus |
| `--primary-foreground`       | `--acc-txt`       | texte sur action |
| `--secondary` / `--muted`    | `--panel-2`       | surface secondaire |
| `--secondary-foreground`     | `--txt`           | — |
| `--muted-foreground`         | `--dim`           | texte atténué |
| `--accent`                   | `--panel-3`       | survol / surface d'appoint |
| `--accent-foreground`        | `--txt`           | — |
| `--border` / `--input`       | `--panel-3` (éclairci) | bordures |
| `--destructive` (+ -foreground) | *aucun littéral* | **état danger — voir §7** |
| `--radius`                   | `--r` (conceptuel, valeurs px conservées) | — |

**Valeurs HSL dérivées (hex → HSL de la source), à écrire dans chaque bloc de thème :**

| Rôle | A Graphite | B Acier | C Encre | D Parchemin |
|------|-----------|---------|---------|-------------|
| `--background`         | `220 11% 5%`  | `216 14% 7%`  | `240 5% 4%`   | `40 27% 98%` |
| `--foreground`         | `240 2% 91%`  | `216 10% 90%` | `240 2% 92%`  | `40 6% 10%`  |
| `--card`/`--popover`   | `225 9% 9%`   | `218 15% 11%` | `240 3% 8%`   | `0 0% 100%`  |
| `--primary`/`--ring`   | `41 54% 54%`  | `217 65% 57%` | `240 2% 92%`  | `41 49% 36%` |
| `--primary-foreground` | `36 33% 6%`   | `218 27% 6%`  | `240 5% 4%`   | `40 27% 98%` |
| `--secondary`/`--muted`| `220 10% 12%` | `216 14% 14%` | `240 4% 11%`  | `40 21% 95%` |
| `--muted-foreground`   | `223 3% 61%`  | `216 8% 61%`  | `240 2% 60%`  | `40 4% 40%`  |
| `--accent`             | `223 8% 16%`  | `215 12% 19%` | `240 3% 15%`  | `40 18% 90%` |
| `--border`/`--input`   | `223 8% 16%`* | `215 12% 19%`*| `240 3% 15%`* | `40 18% 90%`*|

\* bordure = `--panel-3` (actuellement légèrement éclairci vs panel-3, ~+2–4 % L ; je
reproduirai le même petit écart pour rester cohérent avec l'existant). Les **écarts de
surface resserrés** des nouvelles palettes se propagent automatiquement dans ces HSL — je ne
les élargis pas (Règle 1).

---

## 5. Tableau de contrastes — TEXTE d'interface (avant → après)

Le texte reste conforme sur les quatre thèmes (AA corps ≥ 4.5, secondaire/large ≥ 3).

| Thème | `--txt`/bg | `--dim`/bg | `--faint`/bg | `--acc`/bg | label bouton (`acc-txt`/acc) |
|-------|:---:|:---:|:---:|:---:|:---:|
| A Graphite  | **15.9** | 6.82 | 3.14 | 8.04 | 7.83 |
| B Acier     | **15.0** | 6.62 | 3.18 | 4.92 | 5.06 |
| C Encre     | **16.3** | 6.72 | 3.02 | 16.3 | 16.3 |
| D Parchemin | **16.8** | 5.32 | **2.93** | 4.67 | 4.67 |

- Tout le texte de contenu (`txt`, `dim`) : **conforme partout**.
- `--faint` (indices décoratifs / horodatages micro) : 2.93–3.18 — `faint` est du texte
  **décoratif** non soumis à l'AA ; je le signale, rien à corriger (identique en esprit à
  l'existant). Sur Parchemin `2.93` est le plus tendu.
- Labels de bouton primaire : tous ≥ 4.5 (Parchemin `4.67` et Acier `5.06` justes mais OK).

Pour mémoire, les fonds actuels (`#0a0f1c` etc.) donnaient des contrastes texte comparables
(15–17) : **pas de régression** de lisibilité, l'objectif est purement l'aspect « moins IA ».

---

## 6. Question D — Vignettes du menu de thème : DÉRIVABLES (oui, à faire)

Aujourd'hui `lib/theme/themes.ts` porte des hex STATIQUES (`swatch.bg/panel/accent/bull/bear`)
— **déjà désynchronisés** des jetons réels (ex. swatch panel `#111a2c` vs `--panel #0e1524`).
Consommés par `ThemeMenu.tsx`, `AppearancePicker.tsx`, `AccountPanel.tsx` via `style={{background: t.swatch.*}}`.

**Recommandation : dériver des jetons vivants** — rendre chaque vignette dans un conteneur
portant `data-design={t.id}` et remplacer les hex par `var(--bg)` / `var(--acc)` (et data si
affichée). Comme `[data-design='x']` définit les valeurs de jetons, un élément imbriqué avec
cet attribut résout `var(--bg)` vers CE thème quel que soit le thème actif → **la vignette
colle toujours au jeton réel, drift impossible**. On supprime alors le champ `swatch` de
`themes.ts`. (Faisable, testé conceptuellement : les custom properties cascadent par ancêtre
DOM et le sélecteur d'attribut matche l'élément imbriqué.)

---

## 7. Points à trancher / signalés (non tranchés seul)

1. **`sentinel.gold #C9A14A` vs `--acc` Graphite `#C9A14A`** (note palette A) :
   `sentinel.gold` est défini dans `tailwind.config.ts` mais **inutilisé** (0 occurrence dans
   le code). Rien à unifier. **Recommandation : garder distincts** — `--acc` = identité/action,
   `sentinel.*` = palette de données/sens. L'hex partagé est une coïncidence ; les fusionner
   coupleraient identité et sens (interdit par la règle « couleur = sens »). Optionnel : retirer
   le jeton mort `sentinel.gold` (hors périmètre).
2. **Libellés affichés** : le `name` de `themes.ts` est aujourd'hui **fixe (non localisé)**,
   une description localisée vit dans `appearance.descriptions.{id}` (9 locales). Les nouveaux
   noms (Graphite et laiton / Ardoise et acier / Encre / Parchemin) sont demandés « en fr, en,
   es ». Question : **localiser le libellé** (bouger `name` vers une clé i18n `appearance.names.{id}`,
   fr/en/es + parité 9 locales) OU garder un `name` fixe unique ? Je recommande la clé i18n
   (parité 9 locales, cf. règle projet), et j'adapterai aussi les `descriptions`. Identifiants
   internes (`terminal/ardoise/schema/atelier`) **inchangés** (préférences enregistrées).
3. **`--destructive`** (rôle danger, rouge, dans les 4 thèmes) : aucun littéral source. La règle
   « aucun rouge en interface » vise le décoratif buy/sell, pas l'état fonctionnel de danger
   (suppression). **Recommandation : conserver** `--destructive` tel quel (état, pas identité).
4. **Overlay du graphique** : ici ou mission séparée (voir §1).

## 8. Couleurs d'interface en dur (hors thème) — à corriger en implémentation

| Fichier | Valeur | Traitement |
|---------|--------|-----------|
| `app/[locale]/layout.tsx:157` | `#0a0f1c` (PWA `theme-color` sombre) | → nouveau défaut Graphite `#0C0D0F` |
| `app/[locale]/layout.tsx:156` | `#ffffff` (theme-color clair) | garder blanc / aligner Parchemin |
| `app/global-error.tsx:35–58` | `#0a0f1c`, `#e8edf7`, `#c9a227` | rendu **hors** ThemeProvider → figer sur les valeurs du défaut Graphite (`#0C0D0F` / `#E8E8E9` / `#C9A14A`) |
| `components/auth/CandleDriftCanvas.tsx:40–42` | `#37b98c`/`#dd6b7a`/`#d6a24a` | couleurs de **données** (déco login) ; option : lire les jetons. À confirmer |
| `ReadingChart.tsx:200,204–206` | `#2F9E78`/`#C2693E` (repli bougies), `#8B95A7`/`#8E84B0`/`#6E84B0` (BOS/CHOCH/retest) | **Question F** ci-dessous |

Exceptions documentées (à ne PAS toucher) : images edge/OG, favicon/manifest icons, bouton Google.

### Question F — replis du canvas

`ReadingChart.tsx` `CANDLE = { bull:'#2F9E78', bear:'#C2693E' }` diverge des jetons réels
`--bull #37b98c` / `--bear #dd6b7a`. Ce sont des **replis** (utilisés seulement si
`getComputedStyle` échoue). **Recommandation : aligner les replis sur les valeurs de jeton**
(`#37b98c`/`#dd6b7a`) — trivial, garantit qu'un rendu de repli ressemble au thème. Les
niveaux `LEVEL` (BOS/CHOCH/retest) sont de la même famille « en dur pour le sombre » que
l'overlay → **à traiter avec la mission overlay** (§1), pas ici.

---

## 9. Ce qui reste à faire (APRÈS GO)

1. Remplacer les valeurs des jetons littéraux dans `globals.css`, thème par thème (§1 palettes).
2. Écrire les HSL de rôle dérivés (§4), correspondance conservée, 0 jeton créé/supprimé.
3. Libellés fr/en/es (+ parité 9 locales) — décision point 7.2 requise.
4. Dériver les vignettes des jetons vivants (§6), supprimer `swatch` statique.
5. `MiaLogo.tsx` : `FONT` en dur → `var(--font-sans)` ; teinte logo — décision §3.
6. Corriger les couleurs en dur du §8 ; replis canvas §F.
7. Tests (4 thèmes rendent sans jeton manquant, contraste, vignettes = jetons, data
   inchangées au switch, `MiaLogo` = `var(--font-sans)`) + Playwright 1280×800 / 390×844
   dans LES QUATRE thèmes sur accueil, /app, /scanner, /zones, /actualites, connexion.
8. `tsc` + build ; staging explicite (jamais `git add -A`) ; push ; **merge main seulement
   après ta confirmation visuelle live dans les 4 thèmes**.

---

## 10. Journal d'implémentation (après GO)

**Décisions du fondateur** : overlay = **mission séparée** · logo = **neutre unique** ·
libellés = **clé i18n localisée**.

### Fichiers modifiés

| Fichier | Changement |
|---------|-----------|
| `app/globals.css` | Les 4 blocs de thème : jetons littéraux (`--bg…--acc-txt`) remplacés par les valeurs des palettes A–D ; jetons de rôle HSL dérivés (table §4) ; `--brand-mark`/`--brand-word` = **neutre** (= `--txt` de chaque thème) ; commentaires d'en-tête + par-thème mis à jour. **Inchangés** : jetons de données (`--bull/--bear/--liq/--ob/--fvg…`), `--sentinel-*`, `--destructive`, `--radius`, `--r/--r-s`, `--font-narrative`, échelle typo `--fs-*`. |
| `components/brand/MiaLogo.tsx` | `FONT` en dur → `var(--font-sans)`, appliqué via `style` (résout la var sur `<text>` SVG) ; doc « auto = neutre ». `FIXED_FILL` (color/dark/mono) inchangé (OG, favicon). |
| `lib/theme/themes.ts` | Champs `name` et `swatch` **supprimés** (nom → i18n, vignette → jetons). Reste `id` + `base`. |
| `components/theme/ThemeMenu.tsx`, `AppearancePicker.tsx`, `components/auth/AccountPanel.tsx` | Nom via `t(\`names.\${id}\`)` ; vignettes **dérivées des jetons** (conteneur `data-design={id}` + `var(--bg/--panel/--acc/--txt/--bull/--bear)`), plus aucun hex statique. |
| `messages/{fr,en,es,de,it,nl,pl,pt,ar}.json` | Ajout `appearance.names.{id}` (Graphite et laiton / Parchemin / Encre / Ardoise et acier) + `descriptions` réécrites, **9 locales** (édition chirurgicale CRLF, pas de round-trip JSON). Ids internes inchangés. |
| `app/[locale]/layout.tsx` | PWA `theme-color` : dark `#0a0f1c`→`#0c0d0f`, light `#ffffff`→`#fbfaf8`. |
| `app/global-error.tsx` | Boundary hors-thème figé sur le défaut Graphite : bg `#0c0d0f`, txt `#e8e8e9`, bouton `#c9a14a`/`#14100a`. |
| `components/app/ReadingChart.tsx` (§F) | Replis bougies `#2F9E78/#C2693E` → `#37b98c/#dd6b7a` (jetons données) ; replis `readVar` alignés sur le défaut Graphite ; `labelBg` sombre neutralisé (`rgba(20,21,24,.66)`). Couleurs d'overlay/`LEVEL` **non touchées** (mission séparée). |

### Non fait (délégué, par décision)
- **Coloration de l'overlay du graphique** (`zoneOverlayPrimitive.ts` + `LEVEL` de `ReadingChart`) →
  mission séparée. Le graphique du Parchemin garde donc l'overlay pâle documenté au §1.
- `CandleDriftCanvas.tsx` : laissé (couleurs de **données** sur la page de connexion).
- `sentinel.gold` : laissé (jeton mort, distinct de `--acc` par rôle — §7.1).

### Vérifications
- `tsc` : **0 nouvelle erreur** (3 pré-existantes dans `dictation-copy-honesty.test.ts`, hors périmètre).
- `npm run build` : **vert** (exit 0).
- vitest `MiaLogo.brand` + `ui2-copy-honesty` : **14/14**.
- i18n : 9 locales valides (JSON parse OK), CRLF préservé, `names` présents, aucun mojibake.
- Playwright captures : 4 thèmes × 6 pages × 2 viewports → `docs/audits/thm-1-shots/themes/{desktop,mobile}/`.
  (Les bougies du graphique ne se peignent pas en headless — limite d'outillage, cf. §1 —
  mais tout ce que THM-1 touche, chrome/panneaux/accent/logo/typo, est rendu.)

> **En attente de la confirmation visuelle live dans les 4 thèmes avant merge sur main.**
> Staging explicite (jamais `git add -A`), pas de force push.

# AUDIT — LP-3 « Plein cadre » (page d'accueil)

**Branche** : `feat/lp-3-plein-cadre` (worktree dédié `C:\MyPythonProjects\wt-lp-3-plein-cadre`)
**Base** : `origin/main` @ `9de5138`
**Cible visuelle** : `docs/design/v5_pleincadre.html` (versionnée dans ce commit)
**Date** : 2026-09-13
**État** : prêt pour confirmation visuelle live — **non mergé**

---

## 1. Ce qui a changé, et pourquoi

Mise en scène, pas réécriture : aucune section supprimée, aucun argument ajouté,
aucun composant de démo réinventé. Quatre commits séparés.

| Commit | Objet |
|---|---|
| `7a5bb7d` | **A** — retrait du bloc de statistiques et de la ligne « Accès anticipé » |
| `f0fb706` | **B** — héros pleine hauteur + fenêtre produit qui joue à l'arrivée |
| `3c9d754` | **C** — outils en alternance à numéros filigrane, aération, progression, révélation, appel final |
| `93d2457` | **D** — copie validée (publications / lecture d'une zone / périmètre réel), 9 locales |

### Trois choses trouvées en chemin qui n'étaient pas au programme

**(i) L'alternance gauche/droite des quatre outils ne s'appliquait pas.**
La règle s'écrivait `.featRev .txt` dans un module CSS — donc hachée — alors que
le JSX portait la chaîne littérale `className="txt"`. Le sélecteur ne matchait
rien : les quatre lignes étaient toutes texte-à-gauche. Corrigé en la liant à la
classe du module. La demande « les quatre outils alternent » était donc un
correctif, pas un restylage.

**(ii) La page annonçait un périmètre plus large que celui que le produit sert.**
La décision **DATA-1** (`docs/governance/decisions/2026-08-16_data-1_m1_retrait_perimetre.md`,
statut *TRANCHÉE*) a retiré M1 : `DISPLAY_TIMEFRAMES` en montre 5,
`enabled_combos()` en balaie 2 × 5 = 10. La page disait **6 unités et
12 combinaisons**, dans neuf langues, depuis. Rien n'avait échoué — rien ne
reliait la phrase au chiffre. C'est désormais le cas (§4).

**(iii) La maquette n'affiche pas « Données d'illustration » dans le héros.**
La mission l'exige. Ajoutée (clé existante `home.demo.illus`).

---

## 2. Le héros — ce que la fenêtre est vraiment

Ce n'est pas une imitation du produit : c'est le produit. `StructureChart` et
`StructureNarration` vivaient dans `DemoTabs` ; ils sont extraits dans
`components/landing/lp1/StructureVisual.tsx` et partagés par le héros **et** la
démo. Une seule implémentation dessine les deux — mêmes bougies, mêmes quatre
familles, même paragraphe recomposé à partir de ce qui reste affiché.

Réutilisé sans une chaîne nouvelle : la narration (`demo.structure.*`), la
question et le refus (`miaSection.chat.u3` / `.a3`), le titre de fenêtre
(`tools.app.visTitle`), la mention d'illustration (`demo.illus`).
**Trois clés ajoutées en tout** : `home.hero.stage.{loading,reading,done}`.

### La séquence (~6,5 s)

| t | ce qui se passe | statut |
|---|---|---|
| 0 → 1,17 s | les 44 bougies se dessinent de gauche à droite (22 ms d'écart) | Le graphique se charge |
| 1,2 → 2,55 s | BOS/CHOCH → OB → FVG → liquidité, 300 ms d'écart | M.I.A lit la structure |
| 2,5 → 5,3 s | la lecture s'écrit | M.I.A lit la structure |
| 5,35 s | — | lecture à jour |
| 5,7 s / 6,5 s | « tu penses que ça va rebondir ? » → M.I.A refuse | lecture à jour |

### Les trois règles, tenues par construction

**1. `prefers-reduced-motion` → état final immédiat.** Toutes les révélations sont
des animations CSS déclarées **dans** `@media (prefers-reduced-motion:
no-preference)`. Hors de cette requête, ces règles **n'existent pas** : le
serveur a déjà peint la fenêtre finie. Rien n'est ralenti ni mis en file — rien
n'est programmé. Le seul morceau piloté en JS (la frappe) est court-circuité par
le même test. Vérifié en test : `data-seq="done"` au montage, `setTimeout` jamais
appelé, et `getAnimations({subtree:true})` renvoie **0 animation en cours**.

**2. L'interaction n'est jamais prise en otage.** Les pastilles de couches sont
vivantes dès le premier rendu. Le premier `pointerdown` / touche / focus dans la
fenêtre pose `data-seq="done"` — toutes les animations sautent à leur fin — et
l'événement continue sa route vers la pastille visée. Un `ref` garde les pas
encore en vol pour qu'aucun ne réécrase le choix du visiteur.

**3. Aucun compteur animé.** Tous les chiffres (4 026,77 et les niveaux) sont
rendus à leur valeur finale par le serveur. La frappe **révèle** une phrase déjà
composée ; elle ne calcule rien.

### Deux détails qui évitent des bugs invisibles

- **Le statut est trois libellés empilés fondus en CSS**, pas un libellé permuté
  en JS. Sinon le serveur peint « lecture à jour » puis la fenêtre rembobine sous
  les yeux du visiteur.
- **La frappe dure un temps fixe (2,8 s), pas une vitesse par caractère** : la
  même phrase fait 321 signes en arabe et 449 en allemand. Une vitesse fixe
  rendrait l'arrivée une seconde plus longue dans certaines langues et la
  désynchroniserait des délais CSS qui pilotent le statut.
- **Garde-fou sans JS** : la lecture se démasque seule à 2,4 s, et chaque section
  révélée à 2,8 s. Une page publique ne doit jamais pouvoir rester blanche.

---

## 3. Décisions de copie

| | Décision | Appliquée |
|---|---|---|
| **a** | « le marché va bouger » → « les publications tombent » | ✅ 9 locales |
| **b** | « ne valent pas la même chose » → « ne se lisent pas pareil » ; « savoir ce qu'il vaut » → « connaître son histoire » | ✅ 9 locales |
| **c** | 39 $ → 39,99 $ US | ⛔ **sortie de LP-3** (voir ci-dessous) |
| **d** | 6 unités → 5, **et 12 combinaisons → 10** | ✅ 9 locales |

**(b)** — argument décisif trouvé pendant le diagnostic : la formulation proposée
**existait déjà** mot pour mot dans `demo.zones.side.desc`. La page se
contredisait entre sa section Outils (qui valorisait la zone) et sa section Démo
(qui la lisait). Les deux disent maintenant la même chose.

**(c) — pourquoi elle est sortie.** Ce n'est pas un changement de copie :

```
config/pricing.json  (amount: 39)
   ├── src/billing/pricing.py            (backend, lit le JSON)
   └── webapp/lib/pricing.generated.ts   (généré)
```

Trois obstacles : `pricing-prix-1.test.ts` assère `monthly === 39` **et** refuse
toute occurrence de `39,99` ; le fichier de config stipule « Amounts are whole
USD » ; et surtout **c'est Stripe qui débite**. Changer l'affichage seul ferait
diverger prix annoncé et prix facturé. Le changement doit commencer par Stripe.

**(d)** — étendue au-delà de la maquette : 6 → 5 impose 12 → 10 (2 marchés ×
5 unités). « de la minute au jour » tombe avec : la minute n'est plus servie,
c'est « de M5 à D1 ». 27 chaînes modifiées sur 9 locales.

---

## 4. Ce qui empêche la dérive de recommencer

Les chiffres du périmètre sont **dérivés**, plus écrits :

```ts
// lib/landing/stats.ts
markets:      ALL_MARKET_IDS.length,
timeframes:   DISPLAY_TIMEFRAMES.length,          // M1 filtré → 5
combinations: ALL_MARKET_IDS.length * DISPLAY_TIMEFRAMES.length,
```

Et `lp3-perimeter-copy.test.ts` attache enfin **la phrase au chiffre** : les
8 chaînes qui énoncent un périmètre sont vérifiées contre les figures dérivées,
sur les 9 locales, et refusent explicitement l'ancien chiffre. Le même fichier
verrouille les deux décisions de formulation par ce que la phrase n'a **plus le
droit de dire** (verbe prédictif / jugement de valeur), en 8 langues.

Si M1 est réactivé un jour, ce test échoue le premier et dit quelles phrases
réécrire.

---

## 5. Vérifications

### Tests

| Suite | Résultat |
|---|---|
| `vitest run components` | **534 / 534** (64 fichiers) |
| `vitest run lib tests` | **1 186 / 1 186** (128 fichiers) |
| `vitest run app` | **94 / 94** (15 fichiers) |
| dont `hero-stage.test.tsx` (nouveau) | 7 / 7 |
| dont `lp3-perimeter-copy.test.ts` (nouveau) | 8 / 8 |
| `tsc --noEmit` | **0** |
| `next build` | OK |
| Playwright `lp3-plein-cadre.spec.ts` | **14 / 14** |

**Total : 1 814 tests verts, 0 régression.**

Tests demandés par la mission, tous présents :

- reduced-motion → état final rendu sans animation → `hero-stage.test.tsx`
  (unitaire) **et** `lp3-plein-cadre.spec.ts` (bout en bout, avec le comptage
  d'animations en cours) ;
- un clic sur une couche pendant la séquence l'interrompt **et** s'applique →
  les deux niveaux également ;
- vocabulaire interdit sur toute chaîne nouvelle ou modifiée → `home.test.tsx`
  balaie déjà tout l'espace `home` sur les 9 locales (setup / signal /
  opportunité / probabilité / moteur…, plus les équivalents natifs des
  7 autres langues) ; `lp3-perimeter-copy.test.ts` ajoute les interdits propres
  aux décisions (a) et (b).

### Performance — LCP

Même protocole des deux côtés : build de production, `next start` à froid sur un
port dédié, LCP lu par `PerformanceObserver` dans la page.

| | 1280×800 | 390×844 |
|---|---|---|
| **avant** (origin/main) | 1 336 ms | 1 240 ms |
| **après** | 788 / 980 / 764 ms | 720 / 668 / 684 ms |

**Le script d'animation ne dégrade pas le premier rendu — il l'améliore**, ce qui
était attendu : la séquence est entièrement différée après le premier rendu,
l'état SSR est l'état final, et le bloc de statistiques retiré était l'élément
contentful le plus grand du héros.

*Honnêteté de la mesure* : trois échantillons après, **un seul avant** (le reprendre
demande de rebâtir l'état antérieur). La dispersion mesurée après est de ±110 ms
sur desktop ; l'écart avant/après (≈ 550 ms) est très au-delà. La commande exacte
pour refaire la mesure « avant » est documentée en tête de
`webapp/tests/e2e/lp3-lcp.spec.ts`.

### Captures — `docs/audits/lp-3/` (34 images)

| Famille | Fichiers |
|---|---|
| Héros avant / après | `hero-avant--{1280x800,390x844}.png`, `hero-apres--…` |
| Héros pendant la séquence | `hero-pendant--{1280x800,390x844}.png` |
| Héros interrompu par un clic | `hero-interrompu--1280x800.png` |
| Héros en reduced-motion | `hero-reduced-motion--1280x800.png` |
| Héros × 4 thèmes × 2 résolutions | `hero-theme-{terminal,atelier,schema,ardoise}--…` |
| Chaque section × 2 résolutions | `section-{mia,demo,outils,comment,honnetete,tarifs,faq}--…` |
| Page entière avant / après × 2 | `page-entiere{,-avant}--…` |

---

## 6. Points à regarder pendant la confirmation live

1. **Hauteur du héros (décision (e) appliquée).** Le graphique du héros est à
   218 px (contre 250 px dans la démo) précisément pour que **le refus — le
   moment le plus important de la séquence — reste au-dessus de la ligne de
   flottaison** à 1280×800. Vérifié par test (`toBeInViewport`). À confirmer sur
   ton écran réel.

2. **Sur mobile, la fenêtre est sous la ligne de flottaison, par construction.**
   Le héros s'empile (texte, puis fenêtre) comme dans la maquette. Conséquence
   assumée : un visiteur mobile peut arriver sur la fenêtre déjà stabilisée
   plutôt que de voir la séquence. C'est l'état complet du produit, donc une
   bonne image — mais si tu veux que la séquence démarre à l'entrée dans le
   champ plutôt qu'au chargement, c'est une mission séparée (cela casserait
   l'absence de scintillement sur desktop, où la fenêtre est visible d'emblée).

3. **Les étiquettes du graphique se chevauchent à 390 px** (OB / BOS / CHOCH /
   FVG / prix). C'est **préexistant** : le placement des étiquettes est celui de
   `StructureChart`, inchangé par LP-3 (seules la hauteur du conteneur et deux
   classes ont bougé), et la section Démo le montre de la même façon. Le héros le
   rend simplement plus visible. Candidat à une mission de suivi.

4. **Le bandeau de nav apparaît au milieu de `section-outils--*.png`** : artefact
   de capture (`locator.screenshot()` peint le header collant), pas un défaut de
   la page. La capture pleine page fait foi.

5. **Thème clair (`atelier`)** : halo laiton et ombre portée retravaillés en
   jetons (`var(--acc-dim)`, `color-mix` sur `--bg`) au lieu des valeurs sombres
   en dur de la maquette. À valider à l'œil.

---

## 7. Hygiène

- `node_modules` du worktree = **jonction** vers `wt-sc-4` (lockfile identique).
  ⚠️ **Retirer la jonction (`cmd /c rmdir`) AVANT tout `git worktree remove --force`**,
  sinon le `--force` suit le lien et vide le store partagé.
- Disque C: était **plein** au démarrage (le premier `git worktree add` a échoué
  sur `No space left on device`). Purge de caches régénérables uniquement :
  1,1 Go de `.next`, puis caches npm et pip. **7,9 Go libres** au moment du build.
- Les fichiers `messages/*.json` sont en CRLF et ordonnés à la main : toutes les
  éditions i18n sont **chirurgicales sur le texte brut** (jamais de
  `json.dump` round-trip), encodage et fins de ligne préservés, vérifiés sur
  l'arabe.
- `vitest.setup.ts` : jsdom ne fournit pas `window.matchMedia`. Un défaut
  « aucune préférence » y est posé une fois pour toutes (un test qui s'en soucie
  le stubbe et gagne). C'était un trou de l'environnement, pas un comportement.

---

## 8. Reste ouvert

| | Sujet | Qui tranche |
|---|---|---|
| (c) | 39 $ → 39,99 $ US : commencer par Stripe, puis `config/pricing.json` + le test PRIX-1 | fondateur |
| — | Démarrer la séquence à l'entrée dans le champ (bénéfice mobile, risque desktop) | fondateur |
| — | Chevauchement des étiquettes du graphique sous ~500 px (préexistant) | mission de suivi |
| — | 8 composants de l'ancienne landing pré-LP-1 sans aucun appelant (`HeroLive`, `FaqSection`, `BeforeAfterSection`…) | mission de nettoyage |

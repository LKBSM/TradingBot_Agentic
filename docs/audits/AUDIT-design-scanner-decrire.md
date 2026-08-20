# Exploration design — /scanner/decrire (« Décris ta stratégie »)

**Phase :** exploration esthétique pour commercialisation — 3 maquettes comparables.
**Branche :** `design/scanner-decrire-explorations` (worktree dédié, depuis `origin/main` @ `8e96020`).
**Statut :** exploration **non mergée**. La fonctionnalité n'est pas touchée ; aucune des 3
versions n'est appliquée à la page réelle à ce stade.

---

## 1. Dépendances vérifiées (avant build)

### Composant source réel
- Page : `webapp/app/[locale]/(product)/scanner/decrire/page.tsx`
  → `<ScannerModeToggle active="describe">` + `<ConversationalScanner>`.
- État initial « Décrire » (celui maquetté) = **`webapp/components/scanner/conversational/DescribePanel.tsx`**
  (rendu quand `mode === 'describe'` dans `ConversationalScanner.tsx`).
- Chrome au-dessus : `ScannerModeToggle.tsx` (onglets « Choisir mes conditions » / « Décrire ma stratégie »).
- Textes : `webapp/messages/fr.json` → namespaces `scannerChat.describe` / `scannerChat.dictation` / `scanner.mode`.

### Tokens de design centralisés — OUI, réutilisés (pas de système parallèle)
- **`webapp/app/globals.css`** — double vocabulaire par thème :
  1. tokens de rôle HSL shadcn (`--background`, `--primary`, `--border`, `--muted-foreground`…),
  2. tokens littéraux de référence (`--bg`, `--panel`, `--panel-2/3`, `--line`, `--txt`, `--dim`,
     `--acc`, `--r`, `--r-s`…).
- 4 thèmes : **Terminal** (dark, défaut), Atelier (clair, narration serif), Schéma (mono), Ardoise (warm dark).
- `webapp/tailwind.config.ts`, `webapp/lib/theme/themes.ts`. Numériques en `.mono` (JetBrains Mono).
- **Les 3 maquettes reprennent verbatim les valeurs du thème Terminal** (défaut) — elles sont donc
  une base d'implémentation crédible, pas des mockups déconnectés.

### ⚠️ Absence signalée
- **`docs/design/reference-scanner.html` n'existe pas** — ni aucun `docs/design/*.html`.
  Le `reference-desktop.html` cité dans les commentaires de `globals.css` n'est plus dans l'arbre
  (transposé en tokens). **Référence réelle utilisée = `globals.css`.**

---

## 2. Contenu reproduit (fidèle, rien inventé)

Identique dans les 3 versions — seule la mise en forme change :

| Élément | Texte |
|---|---|
| Onglets | « Choisir mes conditions » / « Décrire ma stratégie » (actif) |
| Titre | « Décris ta stratégie. En français. » |
| Sous-titre | « Pas de formulaire à remplir. Écris ce que tu cherches comme tu le dirais à quelqu'un. » |
| Bloc M.I.A | badge « M » · « M.I.A traduit ta stratégie » · « elle ne choisit rien à ta place · tu valides avant de lancer » |
| Textarea | placeholder « Ex. : je cherche un Order Block jamais testé… » |
| Dictée | bouton micro + note « La transcription est effectuée par ton navigateur… Aucun enregistrement audio n'est conservé. » |
| Boutons | « Traduire ma stratégie » (primaire) · « Mes stratégies » (secondaire) |
| Portée | « M.I.A traduit vers 22 conditions vérifiables. Rien d'autre. » |
| Exemples | label « Ou pars d'un exemple » + **5 puces** |
| Disclaimer | « M.I.A ne consulte pas le marché… ne classe rien, ne conseille rien… » |

### 🚩 Écart de fidélité assumé — LIGNE INVIOLABLE
- L'exemple **#4** du produit actuel — **« Dis-moi où va le prix de l'or aujourd'hui »** — est une
  **formulation prédictive** (contre-exemple pédagogique qui déclenche le refus de M.I.A).
- Décision du fondateur : **le retirer des maquettes**. Les 3 versions affichent donc **5 exemples**
  (indices 0, 1, 2, 3, 5), aucun texte inventé. À trancher séparément pour le vrai produit
  (garder le contre-exemple ou non) — hors périmètre de cette exploration.

---

## 3. Les 3 directions

Toutes sur le thème Terminal, même contenu, même accent `--acc` (#4d9de0). Ce qui les sépare :
échelle typo, densité, place accordée au chrome, et registre commercial.

### V1 — Éditorial épuré  (`scanner-decrire-v1.html`)
- **Parti pris :** colonne unique centrée (~680px), beaucoup de blanc, titre **serif** (Newsreader,
  cohérent avec `--font-narrative` du thème Atelier), sous-titre respiré.
- **Saisie = héros :** grande surface calme, texte en serif italique, **chrome minimal** (un simple
  filet sous la zone, pas de carte).
- **Exemples :** liste sobre à filet gauche (pas de boîtes), micro-label mono.
- **CTA :** primaire plein + action fantôme « Mes stratégies ».
- **Registre :** outil d'écriture premium, sérénité, clarté.

### V2 — Produit fintech mature  (`scanner-decrire-v2.html`)
- **Parti pris :** container plus large (~920px), **hiérarchie resserrée**, profondeur de surfaces
  (`--panel` < `--panel-2` < `--panel-3`).
- **Surface assistant :** carte avec **rail d'accent** vertical, dégradé subtil, avatar arrondi,
  badge mono « Traducteur » ; portée affichée en **pilule** dans la barre d'en-tête.
- **Exemples :** grille 2 colonnes de chips raffinés avec **pastille de catégorie** (accent/bull/liq).
- **CTA :** primaire avec flèche + ombre d'accent, secondaire en surface pleine.
- **Registre :** console SaaS polie (Stripe/Ramp), sérieux et maîtrise.

### V3 — Signature Terminal contrastée  (`scanner-decrire-v3.html`)
- **Parti pris :** assume l'identité « Terminal ». **Titre surdimensionné** (« En français. » en accent),
  **eyebrows mono** en majuscules tracées comme motif, grille technique de fond très légère.
- **Surface de commande :** cadre marqué, **marqueur d'invite `›` mono**, texte saisi en mono,
  point « live » vert (`--bull`), pied de console contrasté.
- **Exemples :** liste **numérotée** (`01`…`05`) façon terminal, séparateurs pleins.
- **CTA :** primaire lumineux (halo d'accent), secondaire outline.
- **Registre :** instrument pour professionnels, distinctif et ownable — le plus opiniâtre.

---

## 4. Livrables

- `docs/design/scanner-decrire-v1.html` · `-v2.html` · `-v3.html` — maquettes statiques autonomes
  (aucun backend, aucune logique ; inline CSS + Google Fonts).
- Captures Playwright (Chromium 1.60, deviceScaleFactor 2), dans `docs/audits/` :
  - `scanner-decrire-v{1,2,3}-desktop-1280x800.png`
  - `scanner-decrire-v{1,2,3}-mobile-390x844.png`

## 5. Décision & implémentation (livrée)

**Direction retenue : V3 « Signature Terminal ».** Appliquée aux DEUX entrées du scanner, en
transposant les choix visuels dans le **système de tokens produit** (échelle `--fs-*` plafonnée à
26px, tokens de rôle shadcn) — donc thémable sur les 4 skins, pas seulement Terminal — sans toucher
au comportement.

### « Décrire ma stratégie » — `components/scanner/conversational/DescribePanel.tsx`
- Eyebrow mono, titre 2 lignes (accent « M.I.A la rend vérifiable. »), surface console (marqueur `›`
  mono), phrase de portée droite (nommant Order Blocks / FVG / liquidité / structure / momentum, sans
  nombre), exemples en grille 2 colonnes (pli préservé).
- Décisions fondateur : titre **sur l'échelle** (26px, pas 52px) ; exemple prédictif #4 **conservé**
  (contre-exemple de refus).

### « Choisir mes conditions » — `components/scanner/ConditionsBuilder.tsx`
- Même tenue : eyebrow « Scanner · palette », titre 2 lignes (accent « Vois où elles sont réunies. »),
  intro remaniée (« composent ta stratégie », phrase « outil de lecture » retirée), console = la Card,
  index de famille + compte live en **mono**. Maquette : `docs/design/scanner-conditions-v3.html`.
- Vue résultats (`ScanResults`, composant partagé) non modifiée à ce stade.

### i18n
Ajouts/MAJ propres sur les **9 locales** (`describe.eyebrow`/`titleAccent`/`scope` ;
`builder.eyebrow`/`titleAccent`/`intro`) — parité de clés stricte respectée, aucun mot interdit.

### Gardes
`tsc` 0 · `next build` 0 · vitest (parité + 2 gardes vocab + composants scanner) **42/42** ·
Playwright (ui2-audit + sc1 + sc2 + sc2e) **58/58**. `ui2-audit.spec.ts` ajusté : `scope-note` nomme
les concepts (plus de compte nu) et le scan typo **exclut le texte SVG** (le mot-symbole du logo
n'est pas de la prose). Captures produit réelles : `docs/audits/PRODUCT-scanner-*.png`.

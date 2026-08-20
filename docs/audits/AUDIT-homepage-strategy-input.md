# AUDIT — Refonte bloc « Décris ta stratégie » (typographie + puces multi-sélection)

- **Branche** : `feat/homepage-strategy-input-redesign` (worktree dédié depuis `origin/main` @ `6015ae8`)
- **Date** : 2026-08-19
- **Fichier touché (source)** : `webapp/components/scanner/conversational/DescribePanel.tsx`
- **Test ajouté** : `webapp/tests/e2e/sc2-scanner-conversationnel.spec.ts` (1 cas)
- **Portée** : le bloc `describe` du scanner conversationnel, monté **uniquement** sur
  `/scanner/decrire` (via `ConversationalScanner`). **Aucune autre page ne réutilise
  `DescribePanel`** → pas de risque de régression ailleurs.

---

## 0. Note « page d'accueil » vs réalité du montage

La mission nomme ce bloc « page d'accueil ». En réalité le composant vit sur
`/scanner/decrire`. La maquette `docs/design/reference-accueil.html` l'inclut comme
onglet démo « Définir une stratégie », d'où l'appellation. Le bloc traité est bien
celui décrit (titre + textarea + M.I.A traduit + boutons Traduire / Mes stratégies +
« OU PAR UN EXEMPLE » avec puces cliquables).

---

## 1. VÉRIFICATION DE DÉPENDANCE — écart 21 vs 22 (STOP levé, chiffre INTACT)

- **Nombre réel de conditions du scanner = 22** (`CONDITION_PALETTE` dans
  `webapp/lib/conditions/palette.ts` contient exactement 22 entrées `type:`).
- Le chiffre affiché « 22 conditions vérifiables » **est CORRECT** et, sur `origin/main`,
  **déjà dérivé du code** : `t('describe.scope', { count: CONDITION_PALETTE.length })`
  (il n'est PAS codé en dur dans le composant).
- **La prémisse de la mission (« section 7 = 21 ») est erronée.** L'écart est
  mission↔réalité, pas code↔réalité. Conformément à la consigne, **le chiffre n'a pas
  été touché** (GO explicite du fondateur : « laisse 22 intact »).
- Note : la mémoire projet (UI-2) documentait déjà « 22 était JUSTE, palette=22 ».

> ⚠️ Un diagnostic initial lu dans le dépôt primaire (detached HEAD `e506735`, −13 commits)
> montrait un « 22 » codé en dur et des tailles `text-2xl`/`text-[15px]`. C'était une
> version PÉRIMÉE. Le worktree depuis `origin/main` a révélé la vraie version, déjà
> tokenisée (`fs-*`) et à comptage dérivé. Toute l'implémentation a été faite sur
> `origin/main`.

---

## 2. Typographie — réduction SUR l'échelle partagée (aucun système parallèle)

`origin/main` utilise déjà l'échelle de tokens `fs-*` de UI-2 (définie dans
`app/globals.css`) : `--fs-title 26 / --fs-section 19 / --fs-body 15 / --fs-secondary 14
/ --fs-label 12 / --fs-legal 11`. La maquette `reference-accueil.html` **ne définit
aucune échelle de tokens** (px fixes par classe) ; son panneau démo « Définir une
stratégie » calibre : titre latéral ~17px, corps ~14px, options d'exemple ~13px.

Refinement = **re-mapper vers des tokens existants plus petits**, calibré sur la maquette :

| Élément | Avant | Après | px |
|---|---|---|---|
| Titre `h1` | `fs-title` | `fs-section` | 26 → 19 |
| Textarea + placeholder | `fs-body` | `fs-secondary` | 15 → 14 |
| Puces d'exemple | `fs-secondary` | `fs-label` | 14 → 12 |
| Sous-titre | `fs-secondary` | `fs-secondary` (inchangé) | 14 |

Le sous-titre était déjà sur le petit token secondaire (= 14px, corps de la maquette) ;
la « lourdeur » perçue venait du titre à 26px, ramené à 19px. Aucune valeur arbitraire,
aucun token nouveau, aucune modification des valeurs de tokens (partagées avec tout le site).

---

## 3. Interaction des puces — sélection unique → composition additive

**Avant** : `onClick={() => onTextChange(example)}` — un clic **remplaçait tout** le
texte (sélection unique, exclusive, destructive de la saisie manuelle). Aucun état
visuel de sélection.

**Après** (`toggleExample`) :
- Clic sur puce **inactive** → **ajoute** son texte (jointure `, `).
- Clic sur puce **active** → **retire** exactement sa phrase et son séparateur.
- **Saisie manuelle préservée** : les exemples s'ajoutent **à la suite**, jamais
  d'écrasement (décision validée par le fondateur : « ajoute à la suite sans désactiver »).
- **État actif dérivé du texte lui-même** (`text.includes(example)`) → la textarea reste
  la **source unique de vérité** ; `aria-pressed` reflète l'état ; contour + anneau
  `primary` = indicateur visuel (le contour blanc de la capture).
- **Aucune fusion structurée** côté front : le texte composé reste du langage naturel
  libre, envoyé **tel quel** à « Traduire ma stratégie ».
- Les 6 phrases d'exemple sont **mutuellement non-sous-chaînes** (vérifié fr + en) → aucun
  faux-actif.

### Test d'honnêteté (mission §3)
Le cas e2e ajouté vérifie que le texte de la textarea = **concaténation exacte visible**
des puces (`ex0` → `ex0, ex1` → retrait → `ex1`), et que la saisie manuelle n'est jamais
écrasée (`« je tape moi-même »` → `« je tape moi-même, ex0 »`). Pas de transformation invisible.

---

## 4. Vérifications

- **tsc** (`npx tsc --noEmit`) : **0 erreur**.
- **build** (`npm run build`) : **vert** (compilé, lint + types OK, 10/10 pages générées).
- **Playwright SC-2** (2 projets = desktop 1280×800 + iPhone 12 mobile 390×844) : **20/20 passés**,
  dont le nouveau cas « state 1 — example chips compose additively (multi-select, toggle,
  manual preserved) » aux deux viewports.
- **vitest** : **941/941 passés** (98 fichiers + 3 fichiers relancés). L'exécution complète a
  sorti un exit 1 dû à **3 timeouts de démarrage de worker** (`Failed to start threads worker` —
  flake Defender/worker connu, sans rapport avec le changement) : les 3 fichiers
  (`CalendarEventDetail`, `viewActions`, `zoneLayout`) relancés isolément → **100/100 passés,
  exit 0**. Aucune régression.
- **Captures avant/après** : `docs/audits/homepage-strategy-shots/{before,after}/` (1280×800 + 390×844).
  Avant : titre 26px, puces 14px (repli 2 lignes). Après : titre 19px, textarea 14px, puces 12px
  (compactes, plus raffinées).

### Test d'honnêteté — texte composé = concaténation visible
Le cas e2e vérifie exactement : clic puce 0 → textarea = `ex0` ; clic puce 1 → `ex0, ex1`
(APPEND, pas remplacement) ; re-clic puce 0 → `ex1` (retrait exact) ; saisie manuelle
`« je tape moi-même »` puis clic puce 0 → `« je tape moi-même, ex0 »` (manuel préservé).
`aria-pressed` reflète l'état actif. Aucune transformation invisible : ce que « Traduire ma
stratégie » envoie = la chaîne visible.

---

## 5. Discipline

- Worktree dédié depuis `origin/main` ; staging **explicite** (jamais `git add -A`).
- **Merge sur `main` seulement après confirmation visuelle live du fondateur.**

# AUDIT — UI-2 · Scanner : échelle typographique, lisibilité, tenue professionnelle

**Branche :** `fix/ui-2-scanner-typographie` (worktree dédié `wt-ui-2-scanner`, depuis `origin/main`).
**Position au départ :** repo primaire en detached HEAD `771ccc0`, **−22 commits** derrière
`origin/main` (`e506735`). Tout le travail a été fait sur le worktree aligné sur `origin/main`.
**Périmètre :** présentation uniquement. Aucune condition ajoutée/retirée/reformulée, aucune
règle de traduction touchée, aucun appel changé, aucune page réorganisée.

**Cible visuelle :** `docs/design/reference-scanner.html` (trouvée).

## Arbitrages posés par le fondateur (avant tout code)

| Question | Décision retenue |
|---|---|
| Périmètre CSS pour ≤ 6 tailles | **Jetons partagés** `--fs-*` + preuve Playwright /app & /zones |
| Police titres (la maquette introduit Space Grotesk) | **Inter existant** — pas de 3ᵉ police |
| Règle mono | **Stricte mission** : mono = valeurs seulement |
| Unités de temps (état gaté par `onApp`) | **Présentation seule** : affordance renforcée, pas d'état actif forcé sur /scanner |

---

## Le chiffre 22 (Défaut 8) — la prémisse de la mission était inversée

La palette réelle contient **22 conditions**, pas 21.
Source de vérité backend : `src/intelligence/conditions_scanner.py` → `PALETTE` (22 entrées,
`ALLOWED_CONDITION_TYPES`), en 4 familles : structure 5, zones 8, liquidité 4, contexte 5.
Miroir frontend : `webapp/lib/conditions/palette.ts` → `CONDITION_PALETTE` (22 entrées, test
de non-dérive existant `palette.test.ts`).

**Donc « 22 » à l'écran était JUSTE.** Le vrai défaut : le nombre était **écrit en dur** dans
`messages/{fr,en}.json` (`scannerChat.describe.scope`, `scannerChat.add.hint`).

**Correctif :** le compte est désormais **dérivé** de `CONDITION_PALETTE.length` et injecté via
placeholder ICU `{count}` (DescribePanel + AddConditionPicker). La valeur affichée reste 22 mais
suivra la palette automatiquement. Vérifié par le test `condition count is derived from the palette`.

> Hors périmètre scanner : `home.tools.scanner.b1` (page d'accueil) garde un « 22 » littéral —
> surface marketing, non touchée pour ne pas déborder du scanner. Signalé, non corrigé.

---

## L'échelle typographique

Il n'existait **aucun jeton de taille** : ~18 tailles posées à la main (`shell.css`/`pages.css`
en dur, `text-[Npx]` Tailwind arbitraires, un `style` inline). Six tailles introduites, une
seule fois sur `:root` (indépendantes du thème) — `webapp/app/globals.css` :

| Jeton | px | Rôle |
|---|---|---|
| `--fs-title` | 26 | titre de page / grand nombre (compte live) |
| `--fs-section` | 19 | titre de section |
| `--fs-body` | 15 | texte courant / champ principal |
| `--fs-secondary` | 14 | sous-titres, exemples, boutons, secondaire |
| `--fs-label` | 12 | libellés, intertitres, tags |
| `--fs-legal` | 11 | mentions légales, micro-libellés, fraîcheur |

`--fs-secondary = 14px` est choisi pour coïncider avec le `text-sm` des boutons shadcn : tous
les boutons restent sur l'échelle sans surcharge. `text-xs` (12) = `--fs-label`, `text-sm` (14) =
`--fs-secondary` : les composants qui les utilisent déjà sont sur l'échelle sans changement.
Six classes utilitaires (`.fs-title` … `.fs-legal`) appliquent les jetons dans le JSX.

### Tailles rendues sur /scanner — avant / après

| Surface | Avant (échantillon des ~18) | Après (⊆ 6) |
|---|---|---|
| Titre de page (Décrire) | ~24–40 | **26** |
| Sous-titre | hérité (~16) | **14** |
| Champ de saisie | 15 | **15** |
| Boutons | 14 | **14** |
| Libellés / intertitres | 9–10 (mono) | **12** |
| Mentions (scope, transcription, disclaimer) | 10–12.5 (mono) | **11** |
| Rail — libellés `MARCHÉS`… | 9.5 | **11** |
| Rail — noms de marché / liens espace | 12.5 | **14** |
| Rail — pastilles d'unité de temps | 11.5 | **12** |
| Rail — freshbox / pied légal | 8.5–10.5 | **11** |
| Cartes combo (résultats) — `.nm` / `.blk-lbl` / `.ctx2`… | 9–13 | **11–14** |

**Test automatisé** (`ui2-audit.spec.ts`) : au plus 6 tailles distinctes sous `.app-shell`, toutes
∈ {26,19,15,14,12,11}, sur les **deux onglets**, en **1280×800 et 390×844**, **fr et en**. 7/7 vert.

> Frontière connue : l'entête de l'écran **Résultats** utilise la classe *partagée* `.pghead h1`
> (16 px, aussi employée par /zones et /compte) — laissée intacte pour ne pas modifier ces pages.
> Cet écran n'est pas l'un des deux onglets d'entrée testés.

---

## La chasse fixe — mono = valeurs seulement (Défauts 2 & 3)

`font-mono` retiré de **toutes les chaînes de langage** du scanner ; conservé sur les valeurs
(codes d'unité de temps, prix, comptes, horodatages, et l'enveloppe texte export/import qui est
du code sérialisé).

| Fichier | Élément (langage) remis en proportionnel |
|---|---|
| `DescribePanel.tsx` | sous-titre M.I.A, avertissement transcription, phrase « 22 conditions », intertitre « EXEMPLES » |
| `ConversationalScanner.tsx` | libellés « Votre demande », « CONDITIONS TRADUITES », « TA DEMANDE » |
| `AddConditionPicker.tsx` | tag « Ajouter », hint « Parcourir les 22 conditions », noms de familles |
| `EditableConditionCard.tsx` | tag famille, « valeur supposée », **boutons de valeurs des contrôles** (« Haussière », « Londres »…) |

**Restent en mono (valeurs légitimes) :** codes `M5/M15/H1/H4/D1` (`.tf`, rail), prix & âge de
bougie (`ComboCard`), horodatages, compte de conditions, enveloppe export/import.
**Test :** les nœuds `mia-sub` / `scope-note` / `transcription-note` / `examples-label` sont
proportionnels, tandis qu'un code `.tf` reste monospace. Vert.

---

## Hiérarchie (Défauts 3 & 4)

- **Avertissement de transcription** : passé en petit texte secondaire (11 px, proportionnel),
  sous le champ, une ligne — ne domine plus l'action.
- **Phrase « M.I.A traduit vers 22 conditions… »** : retirée du `ml-auto` (elle flottait à droite
  de la rangée de boutons) et **rattachée sous le bouton**, alignée à gauche.
- L'ordre visuel suit désormais : champ + bouton, puis titre/sous-titre, puis exemples, puis
  phrase palette / transcription.

## Surfaces (Défaut 6)

Cause : les cartes/champs utilisaient des modificateurs d'opacité (`bg-card/50`, `bg-background/70`)
qui les fondaient dans le fond. Passés en surfaces **pleines** à élévation croissante :
fond page (`--background`) < carte (`--card`) < champ (`--muted`), bordures pleines `border-border`.
Trois niveaux désormais perceptibles.

## Unités de temps (Défaut 7)

Cause trouvée : `isActive = onApp && …` (`ShellRail.tsx`) — sur /scanner `onApp` est faux, donc
aucune pastille n'est jamais active, et un clic **navigue vers /app**. Décision : présentation
seule. Les pastilles reçoivent une **bordure + surface au repos** (elles se lisent comme un groupe
de chips sélectionnables) et un survol renforcé ; aucun état actif n'est forcé sur /scanner.
Sur /app, l'état actif accentué (déjà géré par `.tf.on`) est plus lisible qu'avant (cf. captures).

## Titre (Défaut 5)

`scannerChat.describe.title` : « Décris ta stratégie. En français. » → **« Décris ta stratégie »**
(fr) et « Describe your strategy. In your words. » → **« Describe your strategy »** (en). Le
sous-titre porte déjà l'idée « en mots ordinaires ».

---

## Composants partagés touchés (impact hors /scanner)

Ces surfaces sont mutualisées par le châssis produit ; leurs tailles ont été portées sur les
jetons. Impact **cosmétique**, prouvé sans régression par captures /app & /zones :

- `webapp/components/shell/shell.css` — **rail** (`.rail-lbl`, `.mkt .nm/.ic/.px`, `.nl`, `.tf`,
  `.freshbox`, `.mspace-item`) + primitives partagées (`.btn`, `.input`, `.search input`,
  `.flabel`, `.livebadge`). Rendu sur /app, /zones, /compte, /actualités. Le rail grandit
  légèrement (9.5→11, 12.5→14) — plus lisible, aucune troncature.
- `webapp/components/shell/pages.css` — classes **exclusives au scanner** (`.cond`, `.scannote`,
  `.combo` + enfants, `.cl`, `.ctx2`, `.natag`). Aucune autre page ne les utilise.
- `webapp/components/shell/ShellRail.tsx` — taille inline du pied légal (8.5→`--fs-legal`).

**Non touché délibérément** (pour protéger la densité UI-1b de /app et les entêtes /zones) : les
classes de cartes centrales de /app (`.strow`, `.reg`, `.zkvc`, `.newsrow`, `.znarr`, `.zmia`…) et
la classe d'entête partagée `.pghead`.

---

## Tests

| Contrôle | Résultat |
|---|---|
| `tsc --noEmit` | **0 erreur** |
| `next build` | **vert** |
| `vitest run` | **913 passants** (3 échecs = timeouts d'E/S sous charge ; re-run isolé 19/19 vert) |
| Playwright `ui2-audit` (≤6 tailles, mono=valeurs, compte dérivé, 1ᵉʳ écran) | **7/7** |
| Playwright scanner e2e existants (`sc1`, `sc2`, `sc2e`) | **42/42** (desktop + mobile) |
| Captures avant/après | **20 + 20** (2 onglets scanner + /app + /zones, 1280×800 & 390×844, fr & en) |

Staging explicite (jamais `git add -A`). Pas de merge avant confirmation visuelle live du fondateur.

## Captures

`docs/audits/ui2-shots/{before,after}/` — nommage `‹surface›_‹locale›_‹viewport›.png` :
`scanner-decrire-initial`, `scanner-decrire-translated`, `scanner-conditions`, `app`, `zones`.

## Reste / hors périmètre

- `home.tools.scanner.b1` (accueil) garde « 22 » littéral — surface marketing, hors scanner.
- Entête écran Résultats via `.pghead` partagé (16 px) — laissée pour protéger /zones & /compte.

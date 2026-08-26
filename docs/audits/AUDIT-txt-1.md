# AUDIT TXT-1 — Alléger le texte sans perdre un seul fait

## Position du HEAD

- Répertoire principal au démarrage : `docs/preserve-data-1-audit` @ `e0dc69c` — **12 commits derrière** `origin/main` (`b4f4e75`), 1 devant. Diagnostiquer là aurait lu une version périmée.
- Travail fait dans un **worktree dédié** : `C:/MyPythonProjects/wt-txt-1`, branche **`fix/txt-1-charge-textuelle`** basée sur `origin/main` (`b4f4e75`) — **0 commit de retard**.
- Date : 2026-08-24.

## Principe directeur

« CALME PAR DÉFAUT, RICHE SUR DEMANDE ». Consigne du fondateur pour ce passage :
**couper les mots qui ne portent aucune information de marché et ne relèvent pas de
l'utilité du produit** (positionnement, remplissage) — sans seuil rigide, sans amputer
un fait.

---

## 1. Le piège honoré : ce qui a *réduit* le périmètre

Une part du texte porte l'honnêteté du produit et sa protection juridique. En protégeant
strictement cette part (section 1 de la mission), le périmètre des coupes *sûres* s'est
resserré :

- **/app** — le texte visible est fait de **faits de marché** (prix, tuiles régime,
  zones, poches), d'**honnêteté gardée par des tests** (badge « Ancrée au moteur » et
  footer « Chaque niveau cité correspond à une sortie réelle du moteur » — assertés par
  `tests/e2e/narrated-reading.spec.ts` et `components/__tests__/ui2-copy-honesty.test.ts`)
  et d'**indices d'utilité** (« Clic sur une zone = surbrillance »). Le méta est déjà
  reculé au plancher (`badge2` 9px `--faint`, `narrfoot` 9.5px `--faint`). **Aucun mot
  sûr à couper ; aucun reculer de valeur restant** sans toucher un fait gardé. → laissé
  intact, honnêtement.
- **/zones** — texte = faits + jargon (défini à la demande dans `ZoneMiaPanel`) +
  avertissement `zones.mia.disclaimer`. Pas de mot sans info à couper. Seul remède
  autorisé par le fondateur : **reculer** la ligne de distance sous la jauge (la jauge
  VZ-3 est déjà la lecture principale). → appliqué.
- **/scanner** (constructeur + décrire) — porte du **positionnement** (ce que fait le
  produit, dit à l'utilisateur) qui ne donne aucune information de marché. → **coupé**.

---

## 2. Chiffres avant / après (mots, état par défaut, FR + EN)

Comptage automatisé : `components/__tests__/txt1-copy.test.ts` échoue si une clé trimée
cesse d'être plus courte que son libellé pré-TXT-1 (les valeurs pré-TXT-1 y sont figées).
Comptage = jetons séparés par espace, balises `<b>` retirées.

### /scanner/decrire (DescribePanel)

| Clé | Mots FR avant | après | Mots EN avant | après | Nature du texte retiré |
|---|---|---|---|---|---|
| `scannerChat.describe.subtitle` | 16 | **6** | 16 | **6** | positionnement (« Pas de formulaire à remplir… comme tu le dirais à quelqu'un ») |
| `scannerChat.describe.scope` | 18 | **8** | 19 | **8** | positionnement (« décris-les avec tes mots, elle en fait des conditions exactes ») — **vocabulaire gardé** |

Sous-total /scanner/decrire : **FR 34 → 14 mots (−20, −59 %)**, EN 35 → 14 (−21).

### /scanner (ConditionsBuilder)

| Clé | Mots FR avant | après | Mots EN avant | après | Nature du texte retiré |
|---|---|---|---|---|---|
| `scanner.builder.intro` | 24 | **9** | 24 | **10** | auto-description produit (« Le scanner te montre sur quels marchés et timeframes… en ce moment ») |

Sous-total /scanner : **FR 24 → 9 mots (−15, −63 %)**, EN 24 → 10 (−14).

### /app

Aucune coupe (voir §1). Reculer déjà au plancher.

### /zones

Aucune coupe de mots. **Reculer** appliqué (voir §4).

**Total mots retirés (positionnement pur, 0 fait) : FR −35, EN −35.**

---

## 3. Tableau de classement (surfaces d'outil)

| Cat. | Sur /app | Sur /scanner | Sur /zones | Remède |
|---|---|---|---|---|
| C1 fait | prix, tuiles régime, zones, poches, dénominateurs | dénombrement conditions, fraîcheur | prix, bornes, hauteur, distance, contacts | GARDER |
| C2 protégé | `legalInline`, états marché/absence | bloc « à l'encontre », « non synchronisé », `zeroNote`, disclaimers | `mia.disclaimer`, filtres | GARDER |
| C3 jargon | titres de cartes (compacts) | descriptions de familles (repliées) | OB/FVG (définis à la demande M.I.A) | déjà à la demande |
| C4 visiteur | — (aucun) | **`describe.subtitle`, `describe.scope` (queue), `builder.intro` (2ᵉ phrase)** | — (aucun) | **SUPPRIMÉ** |
| C5 redondant | badge/footer = honnêteté gardée (conservés) | — | libellés filtres/tri (déjà minces) | conservé |
| C6 visuel | tuiles déjà chiffrées | — | **ligne de distance ↔ jauge** | RECULÉ |

---

## 4. Liste exhaustive des changements

### SUPPRIMÉ (C4 — positionnement, aucun fait, aucun protégé) — fr + en

1. `scannerChat.describe.subtitle`
   - avant (fr) : « Pas de formulaire à remplir. Écris ce que tu cherches comme tu le dirais à quelqu'un. »
   - après (fr) : « Écris ta stratégie en langage clair. »
   - motif : « Pas de formulaire à remplir » = positionnement ; « comme tu le dirais à quelqu'un » = remplissage. L'instruction d'utilité (écrire sa stratégie en langage clair) est conservée.
2. `scannerChat.describe.scope`
   - avant (fr) : « Order Blocks, Fair Value Gaps, liquidité, structure, momentum : décris-les avec tes mots, elle en fait des conditions exactes. »
   - après (fr) : « Order Blocks, Fair Value Gaps, liquidité, structure, momentum. »
   - motif : la **liste du vocabulaire** (utilité : ce qu'on peut décrire) est conservée ; la queue de positionnement, redondante avec le nouveau sous-titre, est retirée. (Garde `ui2-audit` : le scope contient toujours « Order Blocks ».)
3. `scanner.builder.intro`
   - avant (fr) : « Choisis les faits structurels **présents** qui composent ta stratégie. Le scanner te montre sur quels marchés et timeframes ils sont réunis **en ce moment**. »
   - après (fr) : « Choisis les faits structurels **présents** qui composent ta stratégie. »
   - motif : phrase 1 (instruction + présent-tense « présents ») conservée ; phrase 2 = auto-description du produit, redondante avec le CTA « Voir les marchés concernés » et la page de résultats.

Équivalents anglais coupés symétriquement (aucun repli fr/en — vérifié par test).

### RECULÉ (hiérarchie, aucun mot perdu) — /zones

- Ligne de distance de proximité (`ZoneLifecycleCard.tsx`, `data-testid="distance-line"`) :
  classe `v` → **`v faint`** (réutilise la règle existante `.zpxr .v.faint`, couleur
  `--faint`). Moins contrastée car la **jauge VZ-3 est la lecture principale** ; la chaîne
  est **inchangée** (pts + pct + bord), donc la garde VZ-3 (`gauge-gap == points`) et
  `vz-1-zones` (`.zpx.inside` visible) restent vertes. Aucune nouvelle taille de police,
  aucun nouveau CSS.

### NON TOUCHÉ (constaté honnêtement)

- /app : méta déjà reculé au plancher, reste = faits gardés → aucun changement sûr.
- Trous d'avertissement réglementaire hors périmètre (signalés, voir §6).

---

## 5. Démonstration : aucun fait perdu

| Fait / protégé | Avant | Après |
|---|---|---|
| Vocabulaire descriptible (OB, FVG, liquidité, structure, momentum) | scope | **conservé** (scope) |
| Instruction d'écriture (langage clair) | subtitle | **conservé** (subtitle) |
| Nature « présents / au présent » | intro (« présents » + « en ce moment ») | **conservé** (intro garde « présents ») |
| Sur quels marchés/TF les conditions se réunissent | intro (phrase 2) | **conservé ailleurs** : CTA « Voir les marchés concernés » + page résultats (`resultsMeta`) |
| Distance zone↔prix (pts, %, bord) | ligne texte | **conservé** (chaîne intacte, seulement reculée) + jauge |
| Bloc « ce qui va à l'encontre » (non masquable) | présent | **intact** |
| « non synchronisé » (lectures) | présent | **intact** |
| `zeroNote` (zéro condition ≠ tous les marchés) | présent | **intact** |
| Avertissements réglementaires (/app, /scanner ×2, /zones) | 1 par page | **1 par page (intacts)** |

Aucun chiffre, date, borne, dénominateur, unité ni état d'absence supprimé.

---

## 6. Signalements (hors périmètre TXT-1, non corrigés)

- **Avertissement réglementaire à 0** sur `/compte`, `/verifier-email`,
  `/mot-de-passe-oublie` et les pages d'erreur → viole « jamais zéro ». Remède = **ajouter**
  (additif), pas retirer. À traiter dans un ticket séparé (décision fondateur : signaler).
- `/inscription` et `/abonnement` : les 4 mentions légales pourraient être rendues deux
  fois sur la même page — à vérifier avant toute fusion (règle « jamais plus d'un »).

---

## 7. Tests

- `components/__tests__/txt1-copy.test.ts` (6/6) — présence des protégés fr+en ; ligne de
  distance garde `{pts}/{pct}/{side}/{edge}` ; clés trimées strictement plus courtes ; aucun
  mot interdit introduit ; aucun repli fr/en ; scope nomme toujours « Order Blocks ».
- `components/__tests__/ui2-copy-honesty.test.ts` + `lib/i18n/__tests__/locale-parity.test.ts` : verts (14/14).
- `tsc --noEmit` : seules les **3 erreurs pré-existantes** `dictation-copy-honesty.test.ts`
  (connues, hors périmètre) ; **0 erreur nouvelle**.
- Aucune spec e2e n'assertait les chaînes coupées (vérifié) ; `ui2-audit` reste vert
  (11px sur l'échelle, scope garde « Order Blocks », le pli scanner rétrécit).

---

## 9. 2ᵉ passage — /actualites/[eventId] (2026-08-25)

La fiche publication est la plus « document » du produit, mais presque chaque chaîne
porte un **fait** (dénominateur, unité, valeur, sample, période) ou de l'**honnêteté
protégée** (3 états d'absence `actualPending/Unfetched/Unavailable`, `nono` réglementaire,
`readGuide.body` « un décompte, pas une probabilité », `curve.note`). Les 4 longues
réponses mesurées et les fiches pédagogiques **portent de l'information de marché** → gardées.

Seules coupes sûres = **2 redondances (C5)** sur la même vue, en gardant l'occurrence la
plus précise :

1. **`pub.mia.subtitle`** (composant `CalendarEventDetail.tsx`, ligne du sous-titre M.I.A)
   — « Elle explique les concepts et décrit ce que le moteur a mesuré sur cette
   publication. » **retiré** : redondant avec `pub.mia.capability` (« M.I.A décrit des
   faits mesurés et explique des concepts. Elle ne dit pas… »), qui reste et **porte
   l'honnêteté de non-conseil**. Entête « Demander à M.I.A » épurée ; clé i18n conservée
   (inutilisée, parité intacte).
2. **`pub.source.onlyNote`** (fr + en) — 1ʳᵉ phrase « Ces liens mènent à l'organisme
   officiel et à lui seul. » **retirée** : déjà dite par `pub.source.intro` (toujours
   affichée). L'honnêteté unique **« MIA ne renvoie vers aucun site de commentaire ni de
   prévision : choisir un tel lien, ce serait le recommander. »** est conservée.

**Aucun fait perdu** : dénominateurs, unités, 3 états d'absence, `nono` (1 avertissement),
capability, intro source, `readGuide` — tous intacts (garde `txt1-copy` étendue : 8 tests,
dont distinction stricte des 3 états d'absence et présence de l'honnêteté source après trim).

Tests 2ᵉ passage : `txt1-copy` 8/8 ; `CalendarEventDetail` + `CalendarPublication.nw5/nw6`
37/37 ; e2e `pub-mia-chat` 3/3 ; tsc 0 erreur nouvelle. Captures avant/après
`docs/audits/txt1-shots/{before,after}/*-actualites-detail.png`.

**Reste possible (non fait, hors « coupe de mots ») :** replier les fiches pédagogiques
(C3, riche sur demande) — c'est un changement de hiérarchie, pas une coupe. → **fait au 3ᵉ
passage (§10).**

---

## 10. 3ᵉ passage — repli des fiches pédagogiques (2026-08-25, branche `feat/txt-1b-pedagogy-fold`)

La fiche pédagogique (« Ce que mesure cet indicateur ») **porte de l'information de marché**
→ on ne la coupe pas, on la **replie** (« calme par défaut, riche sur demande »).

- `CalendarEventDetail.tsx` : la carte pédagogie devient un **`<details>` natif** fermé par
  défaut. L'entête (titre + badge + chevron rotatif) est le `<summary>` cliquable ; le body
  reste **dans le DOM** (révélé à l'ouverture). Aucun contenu i18n changé.
- CSS (`calendar-pub.css`) : marqueur natif masqué, chevron `.pub-ped-chev` qui pivote à
  l'ouverture, marge d'entête nulle en fermé.
- Accessibilité native (clavier + lecteur d'écran) via `<details>/<summary>`.

**0 fait perdu** : le body est intégralement conservé, accessible d'un geste. Zéro repli
sur un fait/dénominateur/avertissement — seulement sur une définition (C3), remède
explicitement autorisé par la mission.

Tests : `CalendarEventDetail` unit 25/25 (le test `textContent` passe, `<details>` garde le
body dans le DOM) ; e2e `nw5/nw6/nw7` 26/26 — `nw6 D` mis à jour : la fiche est **présente
mais fermée** par défaut, puis **visible après clic** sur son `summary` ; `tsc` 0 erreur
nouvelle. Captures `docs/audits/txt1-shots/pedagogy-fold/{fr-collapsed,fr-expanded}.png`.

---

## 8. Discipline

- Périmètre présentationnel/rédactionnel : aucune règle métier, aucun calcul, aucun appel modifié.
- Staging explicite (pas de `git add -A`), pas de force push.
- **Merge sur `main` seulement après confirmation visuelle live du fondateur.**

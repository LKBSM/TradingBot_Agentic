# AUDIT UI-3 — Réduire le texte sans perdre ce qui fait la valeur

> **État : GO reçu → coupes sûres appliquées.** Périmètre choisi par le fondateur :
> « Safe cuts + honest report » (retraits de positionnement/étiquettes redondantes au
> niveau composant, aucune chaîne PROTÉGÉE ni PORTEUSE touchée). Captures avant/après
> dans `docs/audits/ui-3-shots/{before,after}/`.
>
> ### Découverte majeure (post-STOP) — pourquoi le plafond est ~5 %, pas 30 %
> TXT-1 (PR #189/#190, mergé il y a 2 jours) a **déjà exécuté cette mission** sur les
> surfaces scanner + publication, et a livré un **garde-fou de copie**
> (`components/__tests__/txt1-copy.test.ts`) qui verrouille comme **PROTÉGÉES** :
> `scannerChat.describe.disclaimer`, `app.desktop.legalInline`, `zones.mia.disclaimer`,
> et `zones.proximity.distanceLine` (gardée, seulement **reculée** visuellement → la
> duplication texte↔jauge que la mission cite est **volontaire**, pas un oubli).
> Mon estimation STOP de 38 % sur /decrire supposait couper `describe.disclaimer` —
> impossible (protégée). Le vrai gisement restant = pur positionnement.

## 0. Position git (discipline d'audit)

- `origin/main` = `cccf308` (PR #190 txt-1b) — **à jour**, récupéré via `git fetch`.
- Terminal d'origine (`docs/preserve-data-1-audit`) était **20 commits DERRIÈRE** origin/main (piège DATA-1) → écarté.
- Worktree DÉDIÉ créé : `C:/MyPythonProjects/wt-ui-3-densite`, branche `fix/ui-3-densite-texte`, checkout `cccf308` (= origin/main, 0 écart).
- **Contexte important** : TXT-1 (PR #189/#190) a **déjà fait une 1ʳᵉ passe d'allègement** sur `/scanner` (`describe.subtitle` −10, `describe.scope` −10, `builder.intro` −15) et `/actualites/[eventId]` (fiche pédagogique repliée en `<details>`). Le présent audit lit ces surfaces **déjà trimmées** ; le rendement marginal y est plus faible et c'est signalé.

## 1. Total de mots rendus PAR DÉFAUT, par surface (fr, desktop, sans interaction)

| Surface | Fréquence | Mots par défaut | Dont derrière interaction (non compté) | Note |
|---|---|---|---|---|
| **/app** | QUOTIDIEN | **≈ 500** (copie fixe) + 150–300 faits moteur dynamiques | ≈ 1 700 (onglets Concept/Donnée du Régime, aides `?`) | L'essentiel de la pédagogie est **déjà replié**. |
| **/scanner** (conditions) | QUOTIDIEN | **≈ 80** (4 familles repliées par défaut) | ≈ 1 500 (palette 22 conditions + définitions SMC) | Déjà trimmé par TXT-1 ; palette entière derrière dépli. |
| **/scanner/decrire** | QUOTIDIEN | **≈ 234** | ≈ 600 (traduction, refus, résultats) | Plus gros gisement des 3 surfaces quotidiennes. |
| **/zones** | QUOTIDIEN | **≈ 200** (chrome + M.I.A + 1 carte type) ; croît par carte | ≈ 400 (Détails complets par carte, réponses M.I.A) | Duplication texte↔jauge **par carte** = gain multiplié. |
| /actualites | occasionnel | ≈ 185 | DayPanel, états vides/erreur | Bloc « nono » = PORTEUSE. |
| /actualites/[eventId] | occasionnel | ≈ 955 (publication riche) | ≈ 70 (fiche repliée) | Presque tout PORTEUSE (mesures + dénominateurs). |
| Accueil / landing | UNE FOIS (vitrine) | ≈ 1 780 (+149 chrome) | FAQ fermée, démos non-actives, carrousel | Positionnement **légitime** ici. |
| /compte | occasionnel | ≈ 126 | — | Quelques ÉTIQUETTES REDONDANTES. |
| /connexion | occasionnel | ≈ 47 | — | — |
| /inscription | UNE FOIS | ≈ 175 | — | Surface non-vitrine la plus dense. |
| /inscription/google | UNE FOIS | ≈ 40 | — | — |
| /verifier-email | UNE FOIS | ≈ 25 | — | — |
| /mot-de-passe-oublie (+confirmer) | UNE FOIS | ≈ 33 / 28 | — | — |
| /abonnement | UNE FOIS/occ | ≈ 87 | — | — |
| Chrome partagé (Nav/Footer/Cookies) | chaque vue | 149 | catégories cookies | Footer disclaimer = PORTEUSE. |

## 2. Faisabilité de l'objectif « −30 % » sur /app, /scanner, /zones — HONNÊTE

| Surface | −30 % atteignable sans toucher au PORTEUSE ? | Coupe réaliste |
|---|---|---|
| **/scanner/decrire** | **OUI** (~38 %) | ≈ 90 mots (disclaimer redondant, sous-titre avatar, placeholder verbeux) |
| **/zones** | **OUI, ~30 %** | dédup texte↔jauge **par carte** (×4 cartes visibles) + trio de positionnement |
| **/scanner** (conditions) | **NON** — déjà trimmé TXT-1, défaut minuscule (~80 mots, familles repliées) | ≈ 15–18 mots (~20 %). Le reste = noms de familles = PORTEUSE. |
| **/app** | **NON sans toucher au PORTEUSE ou retirer des indices utiles** | ≈ 50–60 mots (~10–12 %) de positionnement/répétition dupliqués. Le volume est déjà replié. |

**Conclusion §2 :** je vise −30 % là où c'est atteignable proprement (**/decrire, /zones**), et je m'arrête au gain sûr sur **/scanner** et **/app** en le disant, conformément à la consigne « le chiffre est un objectif, pas une autorisation ».

## 3. Tableau de classification — candidats à la coupe (par surface)

Légende : **P** PORTEUSE (intouchable) · **R** RÉPÉTITION · **Pos** POSITIONNEMENT · **É** ÉTIQUETTE REDONDANTE · **M** MODE D'EMPLOI (signalé, non coupé d'office).

### 3.1 /app (QUOTIDIEN) — coupe ≈ 50–60 mots

| Chaîne (clé i18n) | Cat. | Constat | Décision proposée |
|---|---|---|---|
| Rail Freshbox « Lecture en direct » (`landing.hero.badgeLive`) | R | badge « live » dit **3×** (rail + AppHead `app.desktop.live` + badge graphique `chart.liveBadge`) | retirer la ligne rail ; garder AppHead (près du prix) |
| LegalBar « Lecture algorithmique éducative · ni signal ni conseil » (`app.desktop.legalInline`) | Pos | doublon de la posture dite au rail (`legal.disclaimer.chart`), au chat (`pedagogicalNote`, `complianceLine`) | garder **1 seul** énoncé légal (rail `disclaimer.chart`) ; retirer legalInline |
| Chat « Analyse pédagogique — aucun signal ni conseil » (`app.chat.pedagogicalNote`) | Pos | 4ᵉ répétition de la posture | retirer (couvert par disclaimer unique) |
| Rail Freshbox l2 « Or · 15 min » | R | instrument·TF affiché **4×** (rail actif + AppHead titre + chat header) | retirer la ligne Freshbox l2 |
| Régime sub Tendance « depuis le CHOCH … du {date} » (`sub.trendRef`) | R | même événement CHOCH que sub Maturité (`sub.mat`) | n'ancrer le CHOCH **qu'une fois** (Maturité) |
| Régime Densité « … actifs · ouverts sur {tf} » (`sub.dens`) | R | « actives » redit dans l'en-tête Structure (`struct.countFiltered`) ; `{tf}` redit partout | retirer « ouverts sur {tf} » (É) ; garder le compte |
| Badge état par ligne (Structure `badge.*`, Liquidité `badge.*`) | É | répète le « fait » de la même ligne (« Mitigée. », « A cédé. ») | **arbitrage** : le badge coloré porte une charge visuelle — à confirmer avec toi |
| « Ancrée au moteur » / « Chaque niveau cité correspond… » (`narratedBadge`, `narratedFooter`) | Pos | rassure sur la fiabilité, pas un fait marché | candidat repli/retrait — à confirmer |
| **NE PAS COUPER** : disclaimer légal unique, mesures+dénominateurs (`sub.vol` « 7 vs 20 »), états d'absence (`staleFocus`, calendrier vide), jargon SMC (BOS/CHOCH/EQH/EQL/BSL/SSL défini), description narrée moteur, mention confidentialité dictée. | P | — | intact |

### 3.2 /scanner conditions (QUOTIDIEN) — coupe ≈ 15–18 mots (~20 %, plafond honnête)

| Chaîne (clé) | Cat. | Constat | Décision proposée |
|---|---|---|---|
| Eyebrow « Scanner · palette » (`builder.eyebrow`) | É | nomme l'outil/onglet déjà actif | retrait candidat |
| Accent titre « Vois où elles sont réunies. » (`builder.titleAccent`) | Pos | fioriture, pas un fait marché | retrait/repli candidat |
| Intro « Choisis les faits structurels présents… » (`builder.intro`) | M | reformule « Compose tes conditions » ; **déjà** trimmé par TXT-1 | fondre le mot « présents » dans le titre |
| Noms + contenus de familles (Structure/Zones/Liquidité/Contexte + descriptions) | P | définissent le contenu | **intact** |
| « Zéro condition ne signifie pas “tous les marchés”… » (`zeroNote`) | P | garde conceptuelle (absence ≠ tout) | **intact** |

### 3.3 /scanner/decrire (QUOTIDIEN) — coupe ≈ 90 mots (~38 %)

| Chaîne (clé) | Mots | Cat. | Constat |
|---|---|---|---|
| `describe.disclaimer` « M.I.A ne consulte pas le marché… ne devine aucune condition… » | 39 | Pos | recoupe `miaSub` + `scope` + `titleAccent` (mêmes 3 idées sur la même vue) ; ce n'est **pas** un disclaimer légal |
| `describe.miaSub` « elle ne choisit rien à ta place · tu valides avant de lancer » | 12 | Pos | redit l'accent et le disclaimer |
| `describe.miaTitle` « M.I.A traduit ta stratégie » | 4 | R/Pos | redit le titre + accent |
| `describe.subtitle` « Écris ta stratégie en langage clair » | 6 | M | redit titre + placeholder |
| `describe.placeholder` (exemple de 27 mots) | ~20 (trim) | M | 7ᵉ exemple, doublon des puces d'exemples cliquables juste dessous |
| `describe.titleAccent` « M.I.A la rend vérifiable » | 4 | Pos | fioriture |
| `eyebrow` « Scanner conversationnel » | 2 | É | nomme le mode actif |
| **NE PAS COUPER** : 6 exemples-graines (dont les cas de refus délibérés #40/#41), `describe.scope` (périmètre palette fermée), `dictation.privacy` (seule mention légale de la vue). | — | P | intact |

### 3.4 /zones (QUOTIDIEN) — coupe ~30 % via dédup PAR CARTE

| Chaîne (clé) | Cat. | Constat | Décision proposée |
|---|---|---|---|
| Étiquette « Distance » (`proximity.distanceLabel`) | É | **exemple-phare mission** : la valeur « 4,46 pts en dessous du prix » dit déjà que c'est une distance | retirer l'étiquette, garder la valeur — **× chaque carte** |
| Ligne texte distance (`proximity.distanceLine`) **+** jauge crochet `{pts} pts` (`gauge.gap`) | R | même distance en **texte ET jauge** juste en dessous | garder **une** occurrence (jauge visuelle) + aria ; retirer le doublon texte — **× chaque carte** |
| Prix des bords de jauge (`gauge` levelLow/High) | R | dupliquent la bande de prix de l'en-tête (`card` band) une ligne au-dessus | jauge garde les positions sans réécrire les nombres |
| « Comblées » (filtre `filters.consumed` / groupe `groups.consumed` / badge `header.consumed`) | R | même état nommé **3×** simultanément | badge redondant sous l'en-tête de groupe « Comblées » |
| Groupe « Au-dessus/Sous le prix » (`groups.above/below`) | R | position déjà dans la ligne distance de chaque carte | arbitrage : séparateur de groupe utile — à confirmer |
| M.I.A sujet (`mia.subjectLabel`+valeur) + position (`mia.pos.*`) | R | redisent le tag/bande + le groupe de la carte sélectionnée | condenser l'intro M.I.A |
| Intro page queue « Aucun objectif, aucune prévision. » (`zones.intro`) + tagline « décrit · explique · ne prédit pas » (`mia.tagline`) | Pos | trio avec le disclaimer M.I.A (`mia.disclaimer`, PORTEUSE) | garder le disclaimer ; alléger intro+tagline |
| **NE PAS COUPER** : `mia.disclaimer` (honnêteté « ne prédit pas », non masquable), états d'absence (`neverReturned`, `confluence.none`, `deeplink.staleNotice`, `emptyFilter.*`), jargon SMC, aria de jauge (miroir a11y légitime), `dictationCopy.privacy`. | P | — | intact |

### 3.5 Surfaces occasionnelles / une fois — dédup surtout INTER-surfaces

- **/actualites** : `calendar.badge` « Calendrier de volatilité » (É, redit le titre) ; « Tout cocher » rendu **3×**. Le bloc « nono » et les compteurs « Ce mois-ci » = **PORTEUSE**. Coupe ≈ 10 mots.
- **/actualites/[eventId]** : `pub.curve.source` (R, 3ᵉ redite de « telle que publiée, sans conversion » — déjà dans `curve.note` + `attrib*`) ; moitié de `curve.note` (R) ; `pub.source.organismOnly` (R, « à lui seul » dit 3× dans la section Source) ; badges `curve.badge`/`qLabel` (É). Coupe ≈ 40–55 mots. **Tout le reste = mesures + dénominateurs + états d'absence = PORTEUSE.**
- **/compte** : `subtitle` (É, redit les 3 sections), `passwordMasked`/`exportValue`/`deleteValue` (É, redisent l'action du bouton). Coupe ≈ 15 mots.
- **Accueil** : positionnement **légitime** (vitrine). Seule vraie cible = le **refus M.I.A « je ne prédis pas »** écrit ~5× sur la page (`chat.a3` ≈ `tools.mia.a3` ≈ `demo.mia.refusal` ≈ `distinguish` ≈ `caps.c4`).

## 4. Ce qui est dit DEUX FOIS ENTRE surfaces (une seule place)

1. **Disclaimer légal 18+/risque/ni conseil** : écrit ~6× (`footer.disclaimer`, `pricing.legal`, `connexion.trust`, `inscription.sell.mention*`, `billing.legal*` **identique verbatim**, chat). → un par page de portée légale ; le reste = R.
2. **Refus M.I.A « ne prédit pas »** : ~5× sur l'accueil seul + chat.
3. **Liste value-prop** : `landing.tools.*` ≈ `pricing.paid.f*` ≈ `inscription.sell.value1-5`. ⚠️ **Incohérence factuelle** : `inscription.sell.value1` dit « or, indices, changes » alors que l'accueil/FAQ affirment « 2 marchés (or + euro) ». À corriger.
4. **« Comment ça marche » 3 étapes** : `landing.how.*` ≈ `inscription.sell.steps` ≈ `register.nextStep`.
5. **« Accès anticipé »** : 5 surfaces.
6. **Fiche pédagogique publication** vs `curve.explain` : le même indicateur défini 2× sur /actualites/[eventId] (fiche déjà repliée par TXT-1).

## 5. Cas MODE D'EMPLOI = défauts d'interface (à traiter AILLEURS, pas coupés ici)

- /app : « Clic sur une zone = surbrillance sur le graphique » (`struct.hint`) et « Clic sur une poche = surbrillance de son niveau » (`liq2.hint`) — si le clic doit être expliqué, c'est l'affordance qu'il faut rendre évidente.
- /app : sous-titre d'accueil chat + amorces (`welcomeSubtitleActive`, `starter_*`) — utiles au 1ᵉʳ usage, encombrants au 50ᵉ ; candidats à repli une fois la conversation entamée.
- /scanner : « Sélectionne au moins une condition » (`selectAtLeastOne`) double l'état vide « Aucune condition cochée ».
- /actualites : « Tout cocher » ×3 (un contrôle par groupe de filtres) → contrôle partagé ou icône.

## 6. Ce qui a été RETIRÉ (coupes sûres appliquées)

Toutes au **niveau composant** (le JSX qui rendait la chaîne est retiré) → la clé i18n
reste dans les 9 locales, aucune divergence de langue, aucun reword, aucune clé protégée
touchée. Le garde-fou `txt1-copy` reste **vert (8/8)**.

| Surface | Chaîne retirée (clé) | Mots fr | Catégorie | Justification |
|---|---|---|---|---|
| /app | Freshbox rail `landing.hero.badgeLive` + instrument·TF | ~6 | RÉPÉTITION | badge live + instrument déjà dans AppHead + sélecteur de marché |
| /app | `app.chat.pedagogicalNote` | 7 | POSITIONNEMENT | doublon de la ligne compliance persistante sous le champ |
| /app | `app.desktop.narratedBadge` « Ancrée au moteur » | 3 | POSITIONNEMENT | doublon de `narratedFooter` (provenance gardée) |
| /scanner/decrire | `scannerChat.describe.subtitle` | 6 | MODE D'EMPLOI | reformule le titre + le placeholder |
| /zones | `zones.proximity.distanceLabel` / `positionLabel` | 1 / carte | ÉTIQUETTE REDONDANTE | exemple-phare mission : la valeur dit déjà « X pts … du prix » |
| /zones | `zones.mia.tagline` « décrit · explique · ne prédit pas » | 7 | POSITIONNEMENT | doublon de `mia.disclaimer` (protégée, gardée) |

### Décompte AVANT / APRÈS (mots rendus par défaut, fr)

| Surface | Avant | Après | Δ mots | % | Objectif 30 % |
|---|---|---|---|---|---|
| /app | ~500 | ~484 | −16 | ~3,2 % | non atteint (voir §Découverte) |
| /scanner (conditions) | ~80 | ~80 | 0 | 0 % | intact — tenue V3 (PR #178) préservée |
| /scanner/decrire | ~234 | ~228 | −6 | ~2,6 % | non atteint — `describe.disclaimer` protégée |
| /zones | ~200 | ~189 | ~−11 | ~5,5 % | non atteint |

**Conclusion honnête :** l'objectif de 30 % n'est **pas** atteignable sur ces surfaces sans
toucher à une chaîne PROTÉGÉE (garde-fou TXT-1) ou PORTEUSE. TXT-1 a déjà récolté le gros
du texte réductible il y a deux jours. Conformément à la consigne (« arrête-toi à ce que tu
peux faire et dis-le »), seules les répétitions/étiquettes/positionnements non protégés ont
été retirés. Le plus gros volume de /app (~1 700 mots) reste, comme avant, **replié derrière
interaction** (onglets Concept/Donnée du Régime, aides `?`).

## 7. Ce que j'ai choisi de NE PAS couper — et pourquoi

- **Toutes les chaînes PROTÉGÉES** par `txt1-copy` (disclaimers /app, /decrire, /zones ;
  `distanceLine`) — verrouillées par garde-fou, honnêteté de la marque.
- **La tenue V3 « Signature » du scanner** (`builder.eyebrow`, `builder.titleAccent`,
  `describe.eyebrow`, `describe.titleAccent`) — retirer l'eyebrow/l'accent démonterait le
  design livré en PR #178, ce n'est pas une simple coupe de texte.
- **Badges d'état colorés** (Structure « Mitigée. » / Liquidité « A cédé. ») — le badge
  coloré porte une charge visuelle utile ; risque de retirer un état porteur.
- **Séparateurs de groupe /zones** (« Au-dessus / Sous le prix ») — repère de tri utile.
- **« Chaque niveau cité correspond à une sortie réelle du moteur »** — provenance =
  honnêteté (les niveaux sont réels, pas inventés), pas du pur positionnement.
- **Tout le PORTEUSE** : mesures + dénominateurs, jargon SMC défini, 3 états d'absence,
  filtres vides, bloc « ce qui va à l'encontre », mentions confidentialité dictée.

## 8. Cas MODE D'EMPLOI = défauts d'interface (à traiter ailleurs, non coupés)

- /app : `struct.hint` / `liq2.hint` (« Clic sur une zone = surbrillance… ») — si le clic
  doit être expliqué, rendre l'affordance évidente plutôt que la légender.
- /app : sous-titre + amorces du chat vide — utiles au 1ᵉʳ usage, à replier une fois la
  conversation entamée (MIA-1 a déjà commencé ce travail sur l'empty-state).
- /scanner : `selectAtLeastOne` double l'état vide « Aucune condition cochée ».
- /actualites : « Tout cocher » rendu 3× (un par groupe de filtres) → contrôle partagé.

## 9. Vérifications

- `tsc --noEmit` : 0 nouvelle erreur (seules les 3 pré-existantes `dictation-copy-honesty`).
- `next build` : vert.
- vitest (fichiers affectés) : ShellRail 3/3, **txt1-copy 8/8** (protégées intactes),
  ZoneLifecycleCard.nav 2/2, ZonesWorkspace 13/13.
- Playwright : `ui3-shots` 24/24 (avant) + 24/24 (après) ; régression `narrated-reading` +
  `ui2-audit` + `vz-1-zones` + `sc2` = 76/76.
- Captures avant/après (12 + 12) : `/app`, `/scanner/decrire`, `/zones` × {1280×800, 390×844}
  × {fr, en} dans `docs/audits/ui-3-shots/`.

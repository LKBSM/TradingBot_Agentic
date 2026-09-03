# AUDIT CLN-1 — Nettoyage ciblé (diagnostic, LECTURE SEULE)

> **STOP diagnostic.** Aucune ligne de code modifiée. Réponses à A, B, D, E ci-dessous,
> + tableau des avertissements par page, + inventaire réel des champs de publication.
> **J'attends ton GO avant toute modification.**

## Position du HEAD (discipline origin/main)

- `git fetch origin` fait.
- HEAD principal était sur `docs/preserve-data-1-audit` (`e0dc69c`), **27 commits DERRIÈRE `origin/main`** (+1 devant).
- `origin/main` = `a4244ac` (Merge PR #193 — APP-1 espace de travail).
- Worktree dédié créé : **`C:/MyPythonProjects/wt-cln-1`**, branche **`fix/cln-1-nettoyage`** partie **exactement de `a4244ac`** (écart 0/0 vs origin/main). Tout le diagnostic ci-dessous est lu sur ce worktree à jour.
- MIA-1 est déjà mergée sur main (PR #188) — mais elle a touché `AppChatSidebar` / `chat_events`, **pas** `ZoneMiaPanel`. La partie « panneau M.I.A des zones » n'est donc **pas** déjà faite.

---

## A) Les deux panneaux M.I.A (/app vs /zones) : UN composant ou DEUX ?

**DEUX composants distincts. Ils ne partagent aucun rendu.**

| | /app | /zones |
|---|---|---|
| Composant | `AppChatSidebar` | `ZoneMiaPanel` |
| Fichier | `webapp/components/app/AppChatSidebar.tsx` | `webapp/components/zones/ZoneMiaPanel.tsx` |
| Nature | Chatbot Sentinel **réseau** (backend, SSE, `useChat()` / `ChatProvider`) | Moteur **local déterministe**, zéro crédit, zéro réseau |
| i18n | namespace `app` | namespace `zones` |
| Saisie | via le composant **partagé `ChatInput`** (dictée `MicButton`) | `<form className="zmia-input">` maison |

Ce qu'affiche `ZoneMiaPanel` aujourd'hui (`ZoneMiaPanel.tsx`) :
- état vide (pas de zone) : avatar + `zones.mia.empty` (l.166-177) ;
- **4 questions préfabriquées empilées** : `TOPICS = ['whatElse','explainKind','compareUpper','lastContact']` rendues en blocs pleine largeur `.zmia-sugg > .sg` (défn l.107, rendu l.240-246) — clés `zones.mia.suggest.{whatElse,explainKind,compareUpper,lastContact}` ;
- bloc **« zone sélectionnée »** `.zmia-subj` (l.219-226) : `zones.mia.subjectLabel` + tag + bande de prix + position ;
- champ de saisie `.zmia-input` (l.249-269) : `zones.mia.input.placeholder` / `.send` ;
- mention de transcription / confidentialité via `useDictationCopy()` (l.271-287) ;
- disclaimer `zones.mia.disclaimer` (l.289).

**Recommandation (à valider) :** ces deux panneaux ont des **moteurs** légitimement différents (réseau vs local zéro-crédit) — les fusionner en un seul moteur serait hors périmètre et casserait le zéro-crédit de /zones. La bonne « unification » ici = **aligner le FORMAT** en faisant consommer à `ZoneMiaPanel` le **`ChatInput` partagé** déjà utilisé par /app (saisie + dictée + mention discrète), et **retirer les 4 questions en blocs**, la conversation prenant la hauteur. Un seul composant de *présentation* de la saisie, deux sources de données. Dis-moi si tu préfères plutôt une fusion complète en un composant unique.

---

## B) Avertissements par page — recensement AVANT

### Sources d'avertissement identifiées
| Source | Fichier | Clé | Portée |
|---|---|---|---|
| Pied du rail (menu latéral) | `components/shell/ShellRail.tsx:126` | `legal.disclaimer.chart` | **Toutes** les pages produit via `ProductShell` — **MAIS `display:none` en `max-width:767px`** (`shell.css:504-507`, cf. commentaire l.586 « the rail is hidden < 768px ») |
| Badge en-tête /app (desktop ≥1280) | `components/app/DesktopReading.tsx:411-427` (`LegalBar`) | `app.desktop.earlyAccess` + `app.desktop.legalInline` | /app desktop |
| /app mobile+tablette (<1280) | `MobileWorkspace → ReadingColumn → MarketReadingCard.tsx:88-91` | `EarlyAccessBadge` + `DisclaimerStub variant="chart"` (`legal.disclaimer.chart`) | /app <1280 |
| /scanner/decrire | `components/scanner/ComboCard.tsx:153` (via `DescribePanel`) | `scanner.combo.disclaimer` | /scanner/decrire |
| /zones | `ZoneMiaPanel.tsx:289` | `zones.mia.disclaimer` | /zones (visible uniquement quand une zone est sélectionnée ; sur mobile, panneau = bottom-sheet, donc visible seulement sheet ouverte) |
| /actualites | `CalendarMonthView.tsx:375-385` (bloc `nono`) | `calendar.nono.*` | /actualites |
| /actualites/[eventId] | `CalendarEventDetail.tsx:1233-1244` (bloc `detail.nono`) | `detail.nono.*` | page détail |

> Le texte exact de l'en-tête /app (`desktop.earlyAccess`=« Accès anticipé » + `desktop.legalInline`=« Lecture algorithmique éducative — ni signal ni conseil. ») **correspond bien** au badge visé au point 5. « ACCÈS ANTICIPÉ » n'est **pas** un avertissement légal (c'est un marqueur de statut produit).

### Compte d'avertissements légaux/éducatifs — DESKTOP (≥1280px)
| Page | Avant | Sources |
|---|---|---|
| /app | **2** | LegalBar en-tête (`desktop.legalInline`) **+** pied du rail |
| /scanner (principale) | **1** | pied du rail seul |
| /scanner/decrire | **2** | ComboCard (`scanner.combo.disclaimer`) **+** pied du rail |
| /zones | **2** | ZoneMiaPanel (`zones.mia.disclaimer`) **+** pied du rail |
| /actualites | **2** | bloc `nono` **+** pied du rail |
| /compte | **1** | pied du rail seul |

### Compte d'avertissements — MOBILE (<768px, **pied du rail masqué**)
| Page | Avant | Sources |
|---|---|---|
| /app | **1** | `DisclaimerStub chart` de MarketReadingCard |
| /scanner (principale) | **0 ⚠️** | pied masqué, aucun inline |
| /scanner/decrire | **1** | ComboCard |
| /zones | **0 ou 1 ⚠️** | ZoneMiaPanel seulement si le bottom-sheet est ouvert |
| /actualites | **1** | bloc `nono` |
| /compte | **0 ⚠️** | pied masqué, aucun inline |

**Constat structurel important (à arbitrer).** La règle « exactement un par page, ni zéro ni deux, testée à 1280×800 ET 390×844 » **n'est pas tenable avec le modèle actuel** : le pied du rail (l'avertissement « par défaut ») **disparaît sous 768px**. Donc :
- en desktop plusieurs pages sont **à 2** (doublon rail + inline) ;
- en mobile plusieurs pages tombent **à 0** (/compte, /scanner principale) parce qu'elles n'avaient que le pied.

Le point 5 (garder le pied, retirer l'en-tête) suppose « le pied = l'unique avertissement ». Pour que ce modèle tienne à **un** partout et aux **deux** viewports, il faut **soit** rendre le pied visible aussi en mobile (ou un équivalent mobile), **soit** garder un inline par page et retirer le pied. Ça touche toutes les pages + le mobile → **décision à prendre ensemble** avant que j'écrive les tests « exactement un ». Ma reco : **pied = l'unique**, rendu aux deux viewports, et suppression des inline en doublon (en-tête /app, `zones.mia.disclaimer` si on garde le pied, `scanner.combo.disclaimer`, une partie du `nono`) — mais je ne touche rien avant ton arbitrage.

Sur le point 5 lui-même : **le pied EST rendu sur /app** (desktop). Donc supprimer le texte d'avertissement du badge d'en-tête (`desktop.legalInline`) est sûr en desktop. **Attention mobile** : en <1280px /app n'a PAS de LegalBar mais garde `DisclaimerStub chart` (MarketReadingCard) — le retrait de `legalInline` ne concerne que le desktop, le mobile reste couvert. « Accès anticipé » (`desktop.earlyAccess`) : je recommande de le **garder** (statut produit utile, non légal).

---

## C) Le clic sur une zone est-il géré à un seul endroit ? Que fait un re-clic ?

- État unique : `selectedId` / `setSelectedId` (`ZonesWorkspace.tsx:214`, `useState<string|null>`), alimenté aussi par le deep-link `?zone=` (seed l.216-222).
- Un seul point de clic carte : `ZoneLifecycleCard.tsx:516-521`, `onClick={() => onSelect(zone.id)}`, avec `onSelect={setSelectedId}` (`ZonesWorkspace.tsx:329`).
- **Re-clic sur la zone déjà sélectionnée = AUCUN effet** : `setSelectedId(mêmeId)` ne change rien, la classe `zsel` reste, le bloc « zone sélectionnée » du panneau reste. **Aucune désélection possible aujourd'hui.**
- Le fix (point 3) est propre à faire : `onSelect(id)` → `setSelectedId(cur => cur === id ? null : id)`, + rendre `ZoneMiaPanel` capable de l'état « aucune zone » (il gère déjà `if (!zone)` → état vide l.166 ; il faudra juste que « aucune zone » ne réaffiche pas les 4 questions et garde la conversation). Le seed l.216-222 force `renderedZones[0]` par défaut : à revoir pour autoriser `null` sans le ré-imposer au prochain render.

---

## D) La fiche de publication a-t-elle déjà les champs demandés ? Lesquels manquent ?

**Modèle de données** : `webapp/types/calendar.ts` (interface `CalendarEvent`, l.41-74).
**Deux rendus distincts** :
1. `/actualites` — le **panneau du jour** sélectionné (`CalendarMonthView.tsx:604-664`) : liste compacte cliquable.
2. `/actualites/[eventId]` — la **fiche complète** `CalendarEventDetail.tsx` (fonction `Detail`, l.1097-1247) : **déjà très riche**.

### Champs demandés au point 4 — disponibilité RÉELLE
| Champ demandé | Existe dans les données ? | Déjà affiché ? |
|---|---|---|
| Organisme émetteur, nommé | ✅ `organism` (`calendar.ts:48`) | ✅ fiche l.1169 + panneau jour l.639 |
| Date+heure exacte avec fuseau | ✅ `scheduled_at` (ISO UTC) + `source_timezone` (l.50-51) | ✅ fiche l.1164-1166 (heure organisme + heure locale) |
| Affichée seulement si heure vérifiée | ✅ `time_confirmed` (l.52) ; indicateur « ≈ » + `timeUnconfirmed` quand faux | ✅ fiche l.1150-1155, panneau l.635-637 |
| URL de source + date de vérification (exigées par `time_confirmed=true`) | ⚠️ liens via `CalendarAttribution` (`sourceLinksFor`, l.1040) ; **date de vérification explicite : à confirmer** — pas de champ `verified_at` distinct repéré | liens : ✅ fiche l.1210 ; date de vérif : ❌ |
| Ce que mesure l'indicateur (1 phrase) | ✅ fiches pédagogiques `PEDAGOGY_FICHES` (l.100-105) | ✅ fiche l.1215-1228 (`<details>` repliable), **seulement si une fiche existe pour cet event** |
| Fréquence de parution | ✅ `periodicity` (l.49) | ✅ fiche l.1144-1148, panneau l.646-648 |
| Dernière valeur + sa date | ✅ `actual`/`actual_initial`/`previous` + `value_series[]` (l.60-73) | ✅ fiche : courbe l.1196-1199 (si `value_series` non vide) |
| Marchés rattachés (par devise motrice) | ✅ `markets[]` (l.53), libellé `calendar.affects` = « rattaché à {markets} » (**pas de verbe de causalité**, conforme) | ✅ fiche l.1120, panneau l.649 |
| Lien vers l'organisme émetteur, et lui seul | ✅ attribution organisme (l.1078-1087) | ✅ fiche l.1210 |

### Champs INTERDITS — vérifié qu'ils n'existent pas
- ❌ `impact` (retiré NW-1b, commentaire `calendar.ts:8`), ❌ `forecast`/`consensus`, ❌ severity/pastille de couleur, ❌ classement/hiérarchie. **Aucun n'est présent ni rendu.** Rien à retirer de ce côté ; il faudra juste **ne pas en réintroduire**.

### Conclusion D (honnête)
**Tous les faits demandés existent, et la quasi-totalité est DÉJÀ affichée** — mais sur la **fiche complète `/actualites/[eventId]`**, pas dans le panneau-du-jour de `/actualites`. Deux points d'ambiguïté à trancher avec toi :
1. **Où** veux-tu l'enrichissement ? Dans le **panneau du jour** de `/actualites` (aujourd'hui compact : nom, heures, organisme, périodicité, marchés) — ou considères-tu que la fiche `/actualites/[eventId]` **couvre déjà** la demande et qu'il n'y a rien à ajouter ?
2. **Placeholder de champ manquant** : le panneau du jour affiche `provenance.organismMissing` = « Organisme : non fourni par cette source » quand `organism` est absent (`CalendarMonthView.tsx:642-644`). Cela **viole** ta règle « champ absent → aucune ligne, pas de texte générique ». À corriger (masquer la ligne) — dis-moi si tu confirmes.
3. Une **date de vérification** distincte de l'heure n'existe pas comme champ dédié ; `time_confirmed` s'appuie sur les liens de source. Je n'inventerai pas de « date de vérification » affichée.

---

## E) Le calendrier distingue-t-il « pas encore chargé » de « aucune publication » ?

**Partiellement.** (`CalendarMonthView.tsx`)
- Cellule de jour : deux états visuels seulement — **avec publication** (chips) vs **`.calm-cell.empty`** (`events.length === 0`, l.496). Plus `today`/`selected`/`blank` (padding).
- **Chargement** : géré **au niveau du panneau/bandeau global**, pas par cellule (l.416-426, 333-334) — « tant que ça charge, statut d'attente distinct, jamais un “0/31 jours vides” fabriqué ». Donc **aucun compte affirmé avant chargement** ✅.
- **Mais** : il n'existe **pas d'état visuel “pas encore chargé” au niveau de la cellule**. Les cellules n'apparaissent qu'une fois les données présentes ; il n'y a donc pas les **trois** états de journée par cellule demandés au point 4 (avec-pub / sans-pub / non-chargé). Si tu veux les **trois** distinctions visuelles par cellule, il faudra ajouter un skeleton de cellule « non chargé » (présentationnel, sans compte). À confirmer.

---

## Récap des décisions que j'attends de toi (avant GO)
1. **Panneau M.I.A /zones** : aligner le *format* en réutilisant le `ChatInput` partagé (reco) — ou fusion complète en un composant unique ?
2. **Avertissements** : adopter « pied du rail = l'unique, aux 2 viewports » + suppression des inline en doublon (reco) — ou modèle inline par page ? (touche toutes les pages + le mobile ; conditionne les tests « exactement un »).
3. **/actualites point 4** : enrichir le **panneau du jour** de `/actualites`, ou considérer la fiche `/actualites/[eventId]` (déjà riche) comme suffisante ? + confirmer le retrait du placeholder « Organisme : non fourni ».
4. **Calendrier point 4** : veux-tu vraiment **trois** états visuels **par cellule** (ajout d'un skeleton « non chargé »), ou l'actuel (bandeau global + 2 états de cellule) suffit ?

Captures Playwright avant/après (1280×800 & 390×844, fr+en) : **à produire après GO**, sur les vues /zones (avec/sans zone), /actualites (calendrier + publication), /app en-tête.

---

# IMPLÉMENTATION (après GO — décisions 1/2/3/4 confirmées)

> Branche `fix/cln-1-nettoyage` (worktree `wt-cln-1`, partie d'`origin/main` `a4244ac`).
> **Pas de merge avant confirmation visuelle live.**

## §1 — /zones : phrase d'intro supprimée
`ZonesWorkspace.tsx` : la ligne `<div className="sub">{t('intro')}</div>` sous le titre est retirée ; la ligne de contexte (marché · TF · nb zones · heure prix) reste. Clé `zones.intro` supprimée des **9 locales**. Test : `ZonesWorkspace.test.tsx` « CLN-1 §1 ».

## §2 — /zones : panneau M.I.A au format /app (composant partagé)
- Nouveau `components/chat/ChatComposer.tsx` : barre de saisie **contrôlée, présentationnelle** (textarea + dictée `MicButton` + note de transcription discrète), extraite de `ChatInput`. **DOM/`data-testid` identiques** (`chat-input`, `dictation-*`, `transcription-note`) → /app et `ChatPanel` inchangés.
- `ChatInput.tsx` = fin wrapper qui câble `ChatComposer` au chat **réseau** (`useChat`). `ZoneMiaPanel.tsx` utilise le **même** `ChatComposer` câblé à son moteur **local zéro-crédit** → un seul composant de saisie, deux moteurs (pas de divergence).
- Les **4 questions préfabriquées** (`zmia-sugg`) sont retirées ; la conversation prend `flex:1` (majorité de hauteur) ; le champ reste visible ; la note de transcription = 1 ligne secondaire. Le bloc « zone sélectionnée » se **compacte** (`.zmia-subj.compact`) dès qu'une conversation démarre. Clés `zones.mia.suggest.*` supprimées des 9 locales. Test : « CLN-1 §2 ».

## §3 — /zones : désélection par re-clic
`ZonesWorkspace.tsx` : `selectZone(id)` = `setSelectedId(cur => cur === id ? null : id)` + drapeau `userTouchedSelection` pour que le seed **n'auto-réimpose pas** la 1ʳᵉ zone après une désélection délibérée. En état « aucune zone » : `ZoneMiaPanel` **ne rend pas** le bloc sujet, la conversation **n'est pas effacée** (clear uniquement à l'arrivée sur une zone *différente*), le champ reste utilisable et son placeholder passe à `mia.input.placeholderIdle` (ne prétend plus qu'une zone est choisie). Nouvelles clés `zones.mia.input.placeholderIdle` + `zones.mia.answer.noZone` (9 locales). Test : « CLN-1 §3 ».

## §4 — /actualites : calendrier seul + panneau-du-jour enrichi
`CalendarMonthView.tsx` (la vue rendue par `/actualites`) :
- Panneau de comptes (`.calm-thismonth`) **supprimé** + bloc « Ce que ce calendrier ne dit pas » (`.cal-nono`) **supprimé** ; remplacés par **une** ligne `.calm-scope` = `month.scheduledOnly` (« Ce calendrier ne recense que des publications programmées. »). Clés `calendar.month.thisMonth.*` supprimées des 9 locales. (`calendar.nono.*` **conservées** : encore utilisées par `CalendarWorkspace`, vue liste héritée non rendue par `/actualites`.)
- Panneau-du-jour enrichi : la **valeur publiée** (`actual`, uniquement si `actual_state==='published'`) est affichée (`month.panel.value`, 9 locales) ; le **placeholder « Organisme : non fourni »** est retiré (règle « champ absent → aucune ligne »). Heure non vérifiée : déjà honnête (`timeUnconfirmed` quand `!time_confirmed`), inchangé.
- États de journée (décision D = garder l'actuel) : 2 états de cellule (avec/sans pub) + bandeau global de chargement ; **aucun compte affiché avant chargement**. Tests : `CalendarMonthView.test.tsx` (4 tests « CLN-1 §4 »).

> **Point à trancher (D) — TRANCHÉ (2e passe, « aligner ») :** la **fiche complète** `/actualites/[eventId]` (`CalendarEventDetail`) n'affiche plus le placeholder `organismMissing` ni aucun texte générique de champ manquant. Détail dans la section **2e passe** ci-dessous.

## §5 — Un seul avertissement par page, aux 2 viewports (décision : pied = l'unique)
- **Pied du rail** (`ShellRail`, `legal.disclaimer.chart`) conservé (desktop ≥768). **Nouveau** pied **mobile** dans `ProductShell` (`.shell-mdisclaimer`, `<768px`, même clé) → exactement **un** avertissement visible à chaque viewport (le rail est masqué <768).
- **Doublons inline retirés** : en-tête /app `LegalBar` → `app.desktop.legalInline` retiré (garde « Accès anticipé », marqueur de statut, pas légal) ; `MarketReadingCard` → nouvelle prop `hideChartDisclaimer` posée par `ReadingColumn` (/app) — **le landing garde** son disclaimer ; `ComboCard` (scanner) → note retirée ; `ZoneMiaPanel` (/zones) → note retirée. Clés `app.desktop.legalInline`, `scanner.combo.disclaimer`, `zones.mia.disclaimer` supprimées des 9 locales.
- **Garde-fous mis à jour** : `txt1-copy.test.ts` (les 3 clés retirées de `PROTECTED`, remplacées par `legal.disclaimer.chart` — la présence de l'avertissement reste protégée, à son domicile unique) ; `ui2-copy-honesty.test.ts` (`app.desktop.legalInline` retiré de la liste scannée).
- Test Playwright `cln-1-disclaimers.spec.ts` : sur /app, /scanner, /scanner/decrire, /zones, /actualites, /compte, aux **2 viewports** et en **fr+en**, exactement **1** avertissement visible (échoue à 0 comme à 2).

### Nuance retenue (importante)
La « micro-copie » propre aux widgets — la ligne de conformité du chat /app (`chat.complianceLine` : « M.I.A Agent répond… ni signal… ») et la note M.I.A de `/scanner/decrire` (`scannerChat.describe.disclaimer`) — **n'est pas** comptée comme l'avertissement de page (formulation distincte, portée limitée au widget) et **reste en place**. La règle « un par page » porte sur l'avertissement éducatif/légal de page (« Lecture algorithmique éducative… ni signal… conseil »), désormais à sa source unique (pied). C'est cohérent avec le relevé de diagnostic et le garde-fou `ui2-pages.spec` (qui teste la ligne du chat séparément).

### Tableau des avertissements — APRÈS
| Page | Desktop ≥1280 | 768–1279 | Mobile <768 |
|---|---|---|---|
| /app | 1 (pied rail) | 1 (pied rail) | 1 (pied mobile) |
| /scanner | 1 | 1 | 1 |
| /scanner/decrire | 1 | 1 | 1 |
| /zones | 1 | 1 | 1 |
| /actualites | 1 | 1 | 1 |
| /compte | 1 | 1 | 1 |
(La ligne factuelle « publications programmées » de /actualites est une note de **portée du calendrier**, pas l'avertissement légal.)

## Tests (vitest, real node_modules)
- `ZonesWorkspace` 16/16 (dont §1/§2/§3), `CalendarMonthView` 17/17 (dont 4× §4), garde-fous `txt1-copy` + `ui2-copy-honesty` + `locale-parity` (9 locales, parité stricte) + `market-reading-components` + `ui2b-i18n-keys` = **48/48**. `tsc` propre (hors 3 erreurs pré-existantes `dictation-copy-honesty`).
- Régression composants (zones/calendar/app/scanner/shell/market-reading) + Playwright : voir section finale du rapport après exécution.

## Note environnement (pour reproduire)
Le worktree n'a pas de `node_modules` (les worktrees git ne le copient pas) ; `npm ci` échoue en ERESOLVE (repo → `--legacy-peer-deps`), et cet install-là **omettait ~180 paquets** (dont des dépendances runtime de vitest) → les workers vitest ne démarraient pas (« Timeout waiting for worker to respond »). Résolu en complétant depuis l'install complet de `wt-ci-infra` (lock **identique**). Le script i18n `webapp/scripts/cln1-i18n.mjs` (édition JSON chirurgicale, **préserve les CRLF**, round-trip byte-identique) est un utilitaire de chantier — **à retirer avant merge**.

---

# 2e passe — « aligner » (décisions 1/2/3 du suivi)

> Demande : (1) fiche détail — supprimer le placeholder « Organisme : non fourni par cette source » et appliquer « champ absent → aucune ligne » à **tous** les champs de la page ; (2) micro-copie — retirer « ni signal » de la ligne du chat /app et de la note M.I.A de `/scanner/decrire`, **le pied unique suffit** (après confirmation que le pied est bien rendu sur ces deux pages) ; (3) avertissements retirés du scanner et de /zones : **choix maintenu**, le pied unique suffit.

## 1) Fiche `/actualites/[eventId]` (`CalendarEventDetail.tsx`) — « champ absent → aucune ligne »
- **Ligne provenance en-tête (`.cald-prov`)** : n'affiche plus de placeholder. Rendue **seulement si** `organism` **ou** `value_unit` existe, en joignant les parties présentes par ` · ` ; si les deux manquent → **aucune ligne** (plus de `organismMissing`/`unitMissing`, plus de tiret).
- **Attribution de courbe** : gardée uniquement si `ev.organism` est présent (la série `series_code` reste conditionnelle en plus).
- **Ligne « valeur indisponible »** (`actual_state === 'unavailable'`) : gardée uniquement si `ev.organism` présent (sinon pas de phrase orpheline citant un organisme absent).
- **Ligne source** (`pub.curve.source`) : gardée uniquement si `ev.organism` présent.
- Test : `CalendarEventDetail.test.tsx` — le test « organisme/unité absents » est **inversé** : il asserte l'**absence** des textes `organismMissing`/`unitMissing`, `.cald-prov` nul, `.cald-head .missing` nul. **25/25**.

## 2) Micro-copie widgets — clause de conformité retirée (le pied la porte)
- **Confirmé d'abord** : le pied unique est rendu sur **/app** et **/scanner/decrire** (pied du rail ≥768 + pied mobile `.shell-mdisclaimer` <768, via `ProductShell` pour toutes les routes `(product)`) — vérifié par `cln-1-disclaimers.spec.ts` (48/48, 2 viewports, fr+en).
- **`app.chat.complianceLine`** : garde « M.I.A Agent répond à des questions sur la lecture algorithmique. » ; **retirée** la 2ᵉ phrase « Il ne donne ni signal de trading, ni recommandation personnalisée. »
- **`scannerChat.describe.disclaimer`** : garde l'honnêteté **propre au scanner** (« Elle ne classe rien et ne devine aucune condition que tu n'aurais pas exprimée. ») ; **retirée** la clause de conseil « , ne conseille rien ».
- Édition **chirurgicale** des 9 locales (préserve les CRLF, round-trip byte-identique).
- Garde-fou **`cln1-copy.test.ts`** (nouveau) : `app.chat.complianceLine` contient « lecture algorithmique » / « algorithmic reading » mais **pas** « signal »/« recommand » ; `scannerChat.describe.disclaimer` contient « ne classe rien »/« ne devine »/« orders nothing » mais **pas** « conseille »/« advises ». **2/2**.

## 3) Avertissements scanner + /zones — choix maintenu
Aucun changement : les notes inline retirées en 1re passe restent retirées ; le pied unique reste l'avertissement de page. Cf. tableau §5.

## Validation 2e passe
- `tsc --noEmit` : **0 erreur** (hors 3 pré-existantes `dictation-copy-honesty`).
- vitest : `cln1-copy` 2/2, `CalendarEventDetail` 25/25, `locale-parity` 10/10.
- `next build` : **exit 0** (10/10 pages statiques ; seuls warnings ESLint pré-existants).
- Playwright `cln-1-disclaimers.spec.ts` : **48/48** (6 pages × 2 viewports × fr/en) → exactement **un** avertissement par page tient après la 2e passe.

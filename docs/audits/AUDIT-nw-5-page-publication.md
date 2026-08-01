# AUDIT — NW-5 · Page d'une publication : contenu complet et cohérence visuelle

Branche : `feat/nw-5-page-publication` (worktree dédié, basé sur `origin/main` d90cbd3).
Maquette de référence : `docs/design/reference-publication.html` (mia-page-news-v5.html).
Discipline : **zéro modification des règles de détection.** i18n fr + en complets.

---

## 0bis — Vérification de dépendance (lecture seule, avant tout code)

**a) Les valeurs historiques des publications américaines remontent-elles ? La clé BLS est-elle
posée en production ?**
**Non.** Le fetcher BLS (`src/intelligence/calendar_providers/values/bls_values.py`) n'implémente
que `fetch()` (dernière valeur + précédente) et **pas `fetch_series()`** → il hérite du défaut de
base et renvoie une **liste vide**. De plus `BLS_API_KEY` est **absente de `render.yaml`** (seules
`ANTHROPIC_API_KEY` et `TWELVE_DATA_API_KEY` y sont). Pour les publications américaines (IPC,
emploi), **aucune courbe à 12 valeurs n'est possible**. Le seul organisme doté d'un `fetch_series`
est **Eurostat** (`eurostat_values.py` : `prc_hicp_manr`, `namq_10_gdp`, `une_rt_m`) → IPCH / PIB
zone euro. Conforme à la note NW-3 : *courbe = Eurostat, quatre questions = US CPI/or*.

**b) Les quatre mesures du moteur existent-elles ?** **Trois sur quatre**, pleinement implémentées
pour `us_cpi → XAUUSD` (`src/intelligence/publication_measures.py`) :
- ① calme avant → `_compute_calm_before` ✅
- ② état de la structure à l'instant → `_compute_structure_state` ✅
- ④ retour au calme → `_compute_return_to_calm` ✅
- ③ cycle de vie des zones créées → **DIFFÉRÉE, absente par conception** (aucun champ au schéma ; le
  module refuse explicitement d'en fabriquer un).

Conséquence appliquée (règle de la mission « une question dont la mesure n'existe pas n'est pas
rendue ») : la page rend **3 cartes de questions, pas 4** — la maquette v5 dessine pourtant une 4ᵉ
carte (zones). Écart documenté ci-dessous. Le chantier bloquant est le **calcul de la mesure #3**
(formation→mitigation des zones dans l'heure suivant la parution).

**c) Rapport sur l'emploi — combien de valeurs historiques ? La courbe en demande douze.**
**Zéro.** BLS n'a pas de `fetch_series` et pas de clé. Le Rapport sur l'emploi n'obtient **aucune
courbe**. Idem pour l'IPC. Les courbes à 12 points ne sont atteignables que pour les séries
Eurostat. Verdict : le bloc courbe **n'est pas rendu** pour les publications BLS (règle « aucune
valeur → le bloc n'est pas rendu »).

### Constat clé
La page `/actualites/{eventId}` **était déjà construite par NW-3** (mergé main, PR #109). Les six
sections A–F existent dans `CalendarEventDetail.tsx`. Le « constat » de la mission (seuls 3 blocs
visibles) reflète le **rendu à l'exécution** : blocs à données absentes non rendus (pas de série
BLS → pas de courbe ; mesures souvent vides). NW-5 n'est donc pas une construction ex nihilo mais
un **alignement visuel + partage du composant M.I.A + enrichissement des sources**.

---

## Blocs livrés

| Section | État avant NW-5 | Livré NW-5 |
|---|---|---|
| A En-tête | déjà présent | conservé |
| B Courbe 12 valeurs | présente (Eurostat only) | **grille + libellés d'axe (mois) + point à venir en pointillé sans valeur + ligne de stats (dernière valeur + période, étendue) + valeurs corrigées initiale/actuelle** |
| C Quatre questions | 3 cartes empilées | **grille 2 colonnes, en-tête « Question N », avertissement commun « comment lire » sous les cartes** ; #3 zones NON rendue (différée) |
| D M.I.A | **stub présentiel** | **composant partagé** : `AgentAvatar` (même icône chandeliers + pastille de présence que /app), questions suggérées par publication, saisie + envoi, mention de refus conservée mot pour mot |
| E Aller à la source | un lien générique | **jusqu'à 4 liens NOMMÉS** (méthodologie / dernier communiqué / série historique / calendrier officiel), organisme émetteur uniquement, licence conservée ; lien absent si URL inconnue (jamais de repli générique) |
| F Ce que mesure l'indicateur | présent (par publication) | conservé + encadré « ce que cette fiche ne dit pas » aligné visuellement |

Fichiers touchés :
- `webapp/components/calendar/CalendarEventDetail.tsx` — courbe, questions, M.I.A, sources.
- `webapp/components/calendar/calendar-pub.css` — styles alignés sur la maquette.
- `webapp/components/chat/AgentAvatar.tsx` — prop optionnelle `presence` (pastille) ; **/app inchangé**.
- `webapp/lib/calendar/sourceLinks.ts` — **nouveau** : URLs par publication, organisme uniquement.
- `webapp/messages/{fr,en}.json` — clés `pub.curve.lastValue/range`, `pub.questions.qLabel/readGuide`,
  `pub.mia.suggests.*`, `pub.source.docs.*/organismOnly/onlyNote/noneYet`.

---

## Blocs NON rendus faute de données (avec chantier bloquant)

1. **Courbe pour les publications BLS (IPC, emploi, IPP…)** — le fetcher BLS n'a pas de
   `fetch_series` et `BLS_API_KEY` n'est pas posée. *Chantier bloquant :* implémenter
   `BLSValueFetcher.fetch_series()` (API BLS v2 `timeseries/data`) + poser `BLS_API_KEY` en prod.
   Rendu correct dès qu'une série remonte (aucune modification front nécessaire).
2. **Quatrième question (cycle de vie des zones créées)** — mesure #3 différée, absente du schéma.
   *Chantier bloquant :* calcul formation→mitigation des OB/FVG nés dans l'heure suivant la parution
   (`publication_measures.py`), puis un champ au schéma. La carte apparaîtra automatiquement (le
   rendu numérote et n'affiche que les mesures présentes).

Aucun bloc n'est rendu vide, ni approximé, ni annoncé « à venir ».

---

## Discipline & tests

- **Détection : intacte** — aucun fichier `src/intelligence/*detection*` ni moteur touché.
- **tsc** : `npx tsc --noEmit` → 0 erreur.
- **build** : `npm run build` → succès (route `/[locale]/actualites/[eventId]` compilée). L'avertissement
  EPERM `standalone/node_modules` est l'incident connu jonction+`output: standalone` (affecte
  `next start`, pas le build ni `next dev`).
- **Vitest** (`components/calendar/__tests__`) :
  - `CalendarEventDetail.test.tsx` (mis à jour) + `CalendarPublication.nw5.test.tsx` (nouveau) +
    `calendar-copy-honesty.test.ts` → **40/40 verts**.
  - Garde-fous NW-5 : liens source = organisme uniquement (par clé → source → domaine) ; ≤ 4 liens ;
    **M.I.A partagé — une seule implémentation de `MiaAgentLogo`** ; mention de refus dans les deux
    langues ; point à venir sans valeur + ligne de stats ; avertissement commun rendu une fois ;
    mesure absente jamais rendue en carte vide.
  - Garde-fous NW-1/NW-3 réutilisés (toujours verts) : aucune chaîne « médiane / moyenne / bougie(s) »
    (fr + en) ; parité de clés fr/en ; aucun verbe de causalité événement→marché ; valeur initiale et
    corrigée coexistent.
- **Playwright** (`nw5-publication.spec.ts`, 1280×800 et 390×844, 3 états × 2 largeurs = 12 tests) :
  page complète (courbe + 3 questions + 4 liens source + M.I.A partagé), page sans valeurs
  historiques (bloc courbe absent, questions présentes), page dont les mesures manquent (bloc
  questions absent, courbe présente) → **12/12 verts** (serveur dédié port 3200 ; captures
  `test-results/nw5-{full,nocurve,nomeasures}-{1280,390}.png`). `calendar.spec.ts` (NW-3) mis à jour
  pour les 4 liens nommés. Note d'infra : le `node_modules` partagé (jonction) est saturé par les
  terminaux parallèles → `next dev` > 120 s au démarrage et quelques `ERR_ABORTED` de navigation ;
  résolus en préchauffant la route et en réexécutant (aucun échec d'assertion résiduel).

---

## Écarts restants avec la maquette

- **4ᵉ carte « zones » (Question 3)** : la maquette la dessine ; la mesure est différée → non rendue.
  (Décision conforme à la règle « pas de carte vide ». Levée : chantier mesure #3.)
- **Icône M.I.A** : la maquette montre une *étincelle* ; par exigence « même icône que /app » on
  utilise l'icône **chandeliers** partagée (`MiaAgentLogo`). Écart assumé et voulu.
- **Détails chiffrés visuels** (barres de répartition/dots par carte façon maquette) : le contenu
  (répartition en tranches + extrêmes datés + dénominateur) est livré en texte ; les micro-visuels
  décoratifs par ligne ne sont pas tous reproduits. Aucune information manquante.
- **Courbe US en direct** : la maquette affiche 12 valeurs IPC ; en réel la courbe US ne remonte pas
  (0bis-a). Elle se rend pour les séries Eurostat et dès que `fetch_series` BLS existera.
- **Mobile (< 768 px) — rail non replié** : la page vit dans `ProductShell`. Le repli du rail en
  tiroir mobile est **volontairement limité à `:not(.no-chat)`** (donc au seul `/app`) dans
  `components/shell/shell.css` (UI-2b : *« mobile/tablet [scanner/zones/…] hors périmètre »*). Sur
  `/actualites` (page `.no-chat`) le rail fixe (232 px) subsiste à 390 px et comprime le contenu.
  **Pré-existant, partagé par scanner/zones/compte, non introduit par NW-5** ; non corrigé ici car
  masquer le rail sans nav mobile de remplacement priverait ces pages de navigation. Le contenu de
  la publication est en une colonne et sans débordement horizontal (overflow ≤ 1 px vérifié).
  *Chantier séparé :* nav mobile pour les pages produit `.no-chat`.

---

## À valider en direct (avant merge)

Le backend complet n'est pas lançable ici en données réelles (DATA_SOURCE MT5/clés absentes). Les
tests s'appuient sur des fixtures. À vérifier sur l'environnement réel : rendu de la courbe Eurostat
(IPCH), rendu des 3 questions sur `us_cpi` avec l'historique de prix réel, et cohérence visuelle
d'ensemble avec la maquette. **Pas de merge avant confirmation live.**

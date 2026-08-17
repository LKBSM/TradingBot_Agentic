# AUDIT STR-2 — Structure : branchement CHOCH, double événement, ancrage du focus

> **Phase 1 — DIAGNOSTIC (lecture seule). Aucun code écrit. STOP avant toute correction.**

## Position git au départ

- Worktree dédié : `C:/MyPythonProjects/wt-str-2`, branche **`fix/str-2-structure`**.
- HEAD : `e506735` — **0 commit d'écart** avec `origin/main` (`git fetch` fait avant l'audit).
- Le diagnostic lit du code strictement à jour (la leçon SC-3, parti d'un HEAD détaché 22 commits en retard, est corrigée ici).

---

## DÉFAUT A — Condition branchée au mauvais champ

### La source de vérité

Le moteur expose **deux choses différentes**, pas deux versions de la même :

| Champ | Sémantique réelle | Peuplé quand | Fréquence de nullité (corpus 561) |
|---|---|---|---|
| `structure.bos` / `structure.choch` (**singulier**) | La cassure **au point présent** : une cassure *fraîche sur la toute dernière bougie lue* (`BOS_EVENT`/`CHOCH_SIGNAL ≠ 0` sur la dernière barre), ou — pour `bos` seulement — une cassure antérieure *encore vouchée par la machine de retest*. | la dernière barre lue est elle-même une barre de cassure | `bos` nul **427/561** ; `choch` nul **558/561** |
| `structure.bos_events` / `structure.choch_events` (**journal**) | L'**historique daté** de toutes les cassures de la fenêtre : `direction`, `level`, `broken_at`, `validation_status`, `bars_ago`. | à chaque lecture (via `collect_structure_events`) | riche (ex. 33 BOS / 20 CHOCH sur une lecture) |

**La source de vérité pour « une cassure est survenue dans les N dernières bougies » est le
JOURNAL (`*_events`).** Le champ singulier répond à une autre question (« y a-t-il une cassure
*à l'instant même* »). `choch_recent_confirmed` et `bos_recent_confirmed` posent une question de
**récence** mais lisent le champ **point-présent** → réponse presque toujours « rien », alors
que le CHOCH existe dans le journal. C'est le défaut A.

`structure.choch` (singulier) est un **vestige** : il ne duplique que `choch_events[0]` quand
`bars_ago == 0`, il est nul sinon, et **aucun consommateur porteur** ne l'utilise (voir liste).
`structure.bos` (singulier) **n'est PAS un pur vestige** : il porte la persistance de cassure
(retest) et alimente `retest_in_progress` — mais la condition qui le lit se trompe quand même
de champ.

### Liste complète des appelants du champ singulier (le mauvais)

Recherche `structure.bos` / `structure.choch` / `.get("bos")` / `.get("choch")` sur `src/` :

| Appelant | Fichier:ligne | Surface affamée | Voulait en réalité |
|---|---|---|---|
| **Condition scanner** `bos/choch_recent_confirmed` | `src/intelligence/conditions_scanner.py:724` (`_eval_break_recent_confirmed`, `struct_key="bos"/"choch"`) | Scanner : « CHOCH récent » renvoie faux à tort | le journal (`*_events`, `bars_ago ≤ N`) |
| **Contexte scanner** `build_context` | `conditions_scanner.py:1281-1282` (`structure.get("bos")/("choch")`) | Le bloc « contexte » du résultat de scan n'affiche jamais le CHOCH | le journal |
| **Résumé chatbot** | `src/intelligence/chatbot/signal_summary_provider.py:99-102` (`structure.bos`/`structure.choch`) | Le chatbot ne mentionne quasi jamais BOS/CHOCH | le champ courant OU le journal (à trancher) |
| **Tags** `bos_recent_*` / `choch_recent_*` | `src/intelligence/market_reading_mappers.py:2084-2087` | Tag `choch_recent_bullish` quasi jamais émis | le journal |
| **Description gabarit** | `market_reading_mappers.py:2125-2127` (`structure.bos`) | La narration modèle ne cite quasi jamais la cassure | le journal / `trend_reference` |
| **Lecture narrée** | `src/intelligence/narrated_reading.py:312-324` (`structure.choch`/`structure.bos`) | La lecture narrée ne mentionne quasi jamais le CHOCH | le journal / `trend_reference` |

**Portée du défaut A : ce n'est PAS une seule condition — c'est cinq surfaces** (scanner
condition + contexte, chatbot, tags, description gabarit, lecture narrée). Toutes affamées par le
même champ point-présent.

**Surface NON touchée (correctement branchée) :** l'affichage `/app` `StructureCard.tsx:179-180`
utilise `latestBreak(structure.choch_events, structure.choch)` — il lit le **journal** en premier,
le singulier n'est qu'un repli. `derive_structural_trend` (`market_reading_mappers.py:1833`) et
`last_event_is`/`last_event_age` (`conditions_scanner.py:662`) lisent aussi le journal. Le
`StructureStore` / `/api/structure` (`src/api/routes/structure.py`) n'est **branché à aucune
surface UI** (DG-1) — l'App lit tout via `/api/market-reading`.

### Fréquence AVANT correction (mesurée) — et estimation de l'ampleur APRÈS

Mesure : lectures régénérées via le code actuel, fenêtre glissante, **400 barres de lecture** par
série, récence **N = 10 bougies**.

| Série | CHOCH récent — AVANT (`structure.choch`) | CHOCH récent — APRÈS (journal ≤10) | BOS récent — AVANT | BOS récent — APRÈS |
|---|---|---|---|---|
| **XAU M15** (denom 400) | **0,8 %** (3/400) | **8,2 %** (33/400) | 1,8 % (7/400) | 16,5 % (66/400) |
| **EURUSD M15** (denom 400) | **2,0 %** (8/400) | **20,8 %** (83/400) | 2,2 % (9/400) | 21,2 % (85/400) |

> **La condition CHOCH passe d'≈1 % à ≈8–21 % de vrai (≈10×).** C'est exactement le bond que la
> mission (point D) annonce : après correction, le scanner se comportera différemment pour tout le
> monde. À re-mesurer APRÈS correction sur les mêmes séries.

---

## DÉFAUT B — BOS et CHOCH sur la même bougie, même sens

### La séquence reproduite (pas seulement lue)

Racine : `src/environment/strategy_features.py`, `_calculate_bos_choch_numba`, **lignes 113-128**.
La branche « renversement » écrit **délibérément deux colonnes sur la même barre** :

```
if bos_signal[i-1] == -1 and current_close > current_high_structure:   # renversement baissier→haussier
    choch_signal[i] = 1          # ligne 114
    ...
    bos_signal[i]  = 1           # ligne 118 — MAJ de l'état de tendance pour i+1
    bos_event[i]   = 1           # ligne 119 — la MÊME barre est aussi un « BOS event »
```

Ce n'est **pas** un accident d'ordre d'exécution ni « deux tests qui réussissent tour à tour »
(l'hypothèse de la mission) : c'est **une seule branche** qui pose `choch_signal[i]` **et**
`bos_event[i]`, par conception. `bos_event` est documenté comme « 1/-1 sur toute barre de
cassure réelle » — et un CHOCH *est* une cassure de renversement, donc la branche l'y compte.

**Reproduction empirique (XAU M15, 3975 barres, moteur réel) :**
- 49 barres CHOCH sur la fenêtre. **49/49 (100 %) portent aussi `BOS_EVENT` du même signe.**
- Exemples (ts, CHOCH_SIGNAL, BOS_EVENT) : `2026-03-02 14:15 (-1,-1)`, `2026-03-03 00:15 (+1,+1)`,
  `2026-03-03 05:45 (-1,-1)`, `2026-03-10 15:30 (+1,+1)`, `2026-03-11 22:45 (-1,-1)`,
  `2026-03-12 23:45 (+1,+1)`.
- Aucune barre exactement à 09:30 UTC dans cette fenêtre — mais **le phénomène est systématique
  (100 %)**, la bougie « 09:30 » n'en est qu'une instance parmi toutes les barres CHOCH.

### La règle d'arbitrage existe déjà — mais éparpillée, et jamais au niveau moteur

Le dédoublonnage « CHOCH gagne une barre partagée » est **répliqué dans trois endroits aval**,
jamais centralisé :
1. `collect_structure_events` (`market_reading_mappers.py:1035`, `drop_choch_bars=True`) — retire du
   journal `bos_events` toute barre aussi CHOCH.
2. Le mapper singulier (`market_reading_mappers.py:1416`, `fresh_break = abs(bos_event)>0 and choch_event==0`).
3. Le graphique (`webapp/lib/chart/structureMarkers.ts:93`, `if (chochTimes.has(t)) continue`).

Conséquence : le **payload MarketReading est déjà dédoublonné** (le journal et le singulier ne
portent jamais BOS+CHOCH sur la même barre). Le double événement ne « fuit » que vers les
consommateurs qui lisent les **colonnes brutes** `BOS_EVENT`/`CHOCH_SIGNAL` sans le masque.

### Rayon d'impact d'une correction de la cascade (À LIRE AVANT GO)

`bos_event` (colonne brute) est **porteur** — le supprimer sur les barres CHOCH n'est pas neutre :

| Consommateur de `BOS_EVENT` brut | Fichier:ligne | Effet si on retire `bos_event` sur les barres CHOCH |
|---|---|---|
| **Machine de retest** | `strategy_features.py:294` (`_calculate_bos_retest_numba`, `ev = bos_event[i]`) | Les cassures de **renversement (CHOCH) n'armeraient plus** le suivi de retest → change `retest_in_progress`, la persistance de `structure.bos`, `BOS_RETEST_STATE` |
| **ConfluenceDetector** (scoring hérité) | `confluence_detector.py:406-407` | Change le gating de signal (lit BOS_EVENT et CHOCH_SIGNAL) |
| **Insight v2 / règles RL / backtest** | `insight_v2/builder.py:84`, `rules_engine/rules.py:15`, `state_machine_replay.py:526` | Comportement hérité modifié |

**Deux voies possibles, à trancher au GO :**
- **Voie 1 — toucher la cascade** (la branche 113-128 n'écrit plus `bos_event` sur une barre
  CHOCH). Conforme à la lettre de la mission (« c'est la cascade qu'on corrige »), mais **rayon
  d'impact réel** : machine de retest, confluence, features RL, backtest → tout à re-valider.
- **Voie 2 — centraliser l'arbitrage au bord produit** : une seule fonction documentée
  « l'événement structurel d'une barre = CHOCH si CHOCH_SIGNAL, sinon BOS », par laquelle passent
  les 3 dédoublonnages aujourd'hui éparpillés + le champ singulier affamé. `bos_event` brut garde
  sa sémantique « toute cassure » pour la machine de retest → **zéro changement de comportement
  moteur**. C'est corriger « là où deux événements deviennent un événement produit », pas
  « l'affichage ».

Ma recommandation : **Voie 2** (ou hybride), parce que la Voie 1 déplace le risque dans la
détection sans que le produit y gagne — mais **c'est ta décision.**

**Règle d'arbitrage retenue (une phrase, pour le rapport final)** : *« Sur une bougie qui casse la
structure dans le sens opposé à la tendance en cours, l'événement est un changement de caractère
(CHOCH) ; il n'y a jamais, sur cette même bougie, un BOS de même sens en plus. »*

---

## DÉFAUT C — Le clic sur un BOS focalise un CHOCH

### La nature de l'ancrage : un horodatage, pas un identifiant

- Les événements de structure **n'ont PAS de champ `id`** dans le schéma (`BOSRecent`/`CHOCHRecent`
  = `direction`, `level`, `broken_at`, `validation_status`, `bars_ago` — aucun identifiant).
- Le front **fabrique** une clé composite au clic :
  `StructureCard.tsx:113-125` → `const id = \`${kind}:${atSec}:${ev.level}\`` où
  `atSec = isoToSec(ev.broken_at)`. **L'ancre est l'horodatage** (`broken_at` → secondes epoch),
  plus le niveau. Pas d'identifiant réel émis par le moteur.
- Le cadrage caméra (`frameEvent`, `focusController.ts`) vise correctement la bonne barre par
  `atSec`. Mais **l'emphase du marqueur** échoue : sur une barre partagée BOS+CHOCH, le graphique a
  déjà **abandonné le marqueur BOS** (`structureMarkers.ts:93`, « CHOCH gagne »), donc cliquer le
  BOS met en avant le **seul** marqueur présent — le CHOCH.

### Pourquoi B et C se nourrissent l'un l'autre

Le défaut C n'existe que parce que deux événements peuvent **partager une bougie** (défaut B) et
que l'ancrage est **par horodatage** (pas d'id). Corriger B (plus de collision) masquerait C sans
le corriger ; corriger C sans B laisserait l'ancre horodatée fragile dès que deux événements
tombent sur la même barre pour une autre raison. **Les deux doivent être corrigés et testés
séparément** — conformément à la mission.

Correction visée au GO : **émettre un `id` stable** sur `BOSRecent`/`CHOCHRecent` (schéma + mapper),
construit pour que BOS et CHOCH **ne puissent jamais entrer en collision** même sur une barre
partagée (ex. `bos_<ts>_<dir>` vs `choch_<ts>_<dir>`), et **ancrer le focus par cet id**, un id
inconnu étant **rejeté par le code** (jamais rapproché du plus proche horodatage). Ce correctif doit
tenir MÊME si deux événements partagent une bougie — testé en fabriquant ce cas, sans dépendre de B.

Absence de test aujourd'hui : `structureMarkers.test.ts` couvre le dédoublonnage graphique, mais
**aucun test** ne couvre « cliquer un BOS qui partage une bougie avec un CHOCH » ni « un id inconnu
est rejeté ».

---

## QUESTION E — indeterminate / accumulation / range (mesure, PAS décision)

Mesure sur **400 barres de lecture** par série (code actuel, `derive_structural_trend` +
`_derive_market_phase`). **NE RETIRE RIEN — c'est une mesure.**

| Série | denom | `trend=indeterminate` | `phase=accumulation` | `phase=ranging` |
|---|---:|---:|---:|---:|
| XAU M15 (rappel SC-3, denom 80) | 80 | 0,0 % | 0,0 % | 0,0 % |
| **EURUSD M15** | 400 | **0,0 %** | **0,0 %** | **0,0 %** |
| **XAU H4** | 400 | 0,0 % | 0,0 % | 0,0 % |
| **EURUSD H4** | 400 | 0,0 % | 0,0 % | 0,0 % |
| **XAU D1** | 400 | 0,0 % | 0,0 % | 0,0 % |
| **EURUSD D1** | 400 | 0,0 % | 0,0 % | 0,0 % |

Répartitions observées (exemples) : EURUSD M15 {bearish 328, bullish 72} / {trend 291, expansion 109} ;
XAU D1 {bullish 385, bearish 15} / {trend 308, expansion 92}.

**Constat, élargi à EURUSD, H4 et D1 : `indeterminate`, `accumulation` et `ranging` restent à
0 % partout** (≈ 2000 barres de lecture, 6 combinaisons instrument×unité). La cause est
structurelle : `derive_structural_trend` ne renvoie `indeterminate` que si le journal de la fenêtre
**ne contient AUCUNE cassure** BOS ni CHOCH — ce qui n'arrive jamais sur une fenêtre de plusieurs
centaines de barres. Et `accumulation`/`ranging` exigent `indeterminate` en amont. Ce sont donc des
valeurs **quasi mortes sur toute fenêtre réelle**, pas seulement sur un instrument tendanciel.
**Aucune action ici — mesure livrée, décision à toi.**

---

## Ce que je propose de faire APRÈS ton GO (rien n'est fait)

1. **Défaut A** — rebrancher `bos/choch_recent_confirmed` (+ contexte scanner) sur le journal
   `*_events` (récence en bougies via `bars_ago`). Décider avec toi si chatbot / tags / narration
   suivent le journal ou `trend_reference`. Retirer le vestige `structure.choch` (singulier) ;
   conserver `structure.bos` (porteur retest) ou le renommer si on veut lever l'ambiguïté.
2. **Défaut B** — **Voie 2 recommandée** : une règle d'arbitrage unique et documentée au bord
   produit, les 3 dédoublonnages éparpillés y passant. (Voie 1 = toucher la cascade, seulement si
   tu acceptes le rayon d'impact retest/confluence.)
3. **Défaut C** — émettre un `id` stable par événement (schéma + mapper), ancrage du focus par id,
   id inconnu rejeté. Testé indépendamment de B.

## Tests prévus (au GO)
- Une condition CHOCH sur un marché où un CHOCH existe renvoie vrai (échoue sur le code actuel).
- Recherche : aucun champ vestige `structure.choch` ne subsiste.
- Sur une bougie donnée, pas de BOS **et** CHOCH de même sens exposés ensemble.
- Cliquer l'événement X focalise l'événement X, même quand un autre partage sa bougie.
- Un id inconnu passé au focus est rejeté, pas rapproché.
- Fréquence CHOCH re-mesurée AVANT/APRÈS (attendu ≈ ×10).
- Playwright 1280×800 & 390×844, fr+en : /app événements visibles ; /scanner recherche CHOCH avec
  résultats. tsc + build verts. On ne touche pas au test AppWorkspace « skeleton » (PERF-2).

---

## STOP — j'attends ton GO

Livré : source de vérité (journal `*_events`), liste des 5 surfaces affamées, séquence 09:30
reproduite (100 % des barres CHOCH), nature de l'ancrage du focus (horodatage, pas d'id), fréquences
avant correction (≈1 % → ≈8–21 %), et les mesures de la question E. **Aucun code écrit.**

---
---

# PARTIE 2 — IMPLÉMENTATION (après GO)

> GO reçu avec trois précisions : (1) **ne pas supprimer** `structure.bos`/`choch` — les
> **renommer** en un nom « au point présent » et corriger tous les appelants ; (2) corriger
> **les cinq surfaces** avec un avant/après par surface sur un cas réel ; (3) traiter **BOS avec
> la même rigueur que CHOCH** (nul 76 % = même défaut). Ordre imposé : **A → C → B**.

## Position git (implémentation)
Worktree dédié `wt-str-2`, branche `fix/str-2-structure`, base `e506735` (= `origin/main`, 0 écart).

## Défaut A — source de vérité et renommage

**Source de vérité pour la récence : le journal `bos_events` / `choch_events`.**

**Renommage (champ conservé, pas supprimé)** : `structure.bos` → **`current_bos`**,
`structure.choch` → **`current_choch`** (schéma Pydantic + type TS), docstring explicite
« POINT-IN-TIME break state… NOT the recency history ». Le champ garde son sens (cassure à la
barre lue) ; le nom devient sans ambiguïté vis-à-vis du journal. `tsc --noEmit` = **0 erreur**
après renommage (le compilateur a garanti l'exhaustivité et distingué le champ payload du
**contexte de scan** `bos`/`choch`, laissé intact).

**Appelants corrigés** — Backend : `market_reading_schema.py`, `market_reading_mappers.py`
(2 constructeurs + tags + description), `conditions_scanner.py` (condition + `build_context`),
`chatbot/signal_summary_provider.py`, `narrated_reading.py`. Frontend : `types/market-reading.ts`,
`ReadingChart.tsx`, `StructureCard.tsx`, `RegimeCard.tsx`, `regime-facts.ts`,
`use-reading-formatters.ts`, `StructureSection.tsx`, `mockReadings.ts`, `fixtures.ts` + fixtures de test.

**Les cinq surfaces affamées, rebranchées sur le journal** (BOS ET CHOCH, même rigueur) :
condition scanner `bos/choch_recent_confirmed`, contexte de scan `build_context`, résumé chatbot,
tags `*_recent_*`, description gabarit + lecture narrée.

**Deux surfaces frontend supplémentaires trouvées affamées et corrigées** (mobile, alors que le
desktop lisait déjà le journal) : `regimeLastEvent` (→ RegimeSection) et `StructureSection`
(rangée BOS/CHOCH), via un `latestBreak` partagé exporté de `regime-facts.ts`. Le heuristique de
cohérence retest de `StructureSection` garde la sémantique point-présent (documenté). Helper mort
`formatLastStructuralEvent` (aucune surface ne le rend) : renommé pour tsc, signalé au nettoyage.

### AVANT / APRÈS par surface — cas RÉEL
Lecture régénérée XAUUSD M15, barre `2026-04-28 07:30`, prix 4629,49 — forme de production exacte :
`current_choch = None`, journal `choch_events[0]` = **CHOCH baissier, niveau 4666,41, il y a
10 bougies** (`id=choch_2026-04-28T05:00:00+00:00_bearish`).

| Surface | AVANT (`current_choch = null`) | APRÈS (journal) |
|---|---|---|
| 1. Condition scanner `choch_recent_confirmed` (≤10) | `met=False` — « Aucun CHOCH récent » (**fabriqué**) | `met=True` — « CHOCH baissier confirmé il y a ~10 bougie(s) (≤ 10) » |
| 2. Contexte de scan `context['choch']` | `None` | `{bearish, 4666.41, confirmed}` |
| 3. Résumé chatbot | pas de mention CHOCH | `…, CHOCH bearish, 7 OB actif(s), 9 FVG actif(s)` |
| 4. Tags | aucun `choch_recent_*` | `['choch_recent_bearish']` |
| 5. Lecture narrée `_collect_breaks` | 0 cassure citée | `[('choch','bearish',4666.41), ('bos','bullish',4697.57)]` |

### Fréquence de la condition — AVANT / APRÈS (dénominateurs)
400 barres de lecture par série, récence N=10 :
- **XAU M15** : CHOCH **0,8 % (3/400) → 8,2 % (33/400)** ; BOS **1,8 % → 16,5 %**.
- **EURUSD M15** : CHOCH **2,0 % (8/400) → 20,8 % (83/400)** ; BOS **2,2 % → 21,2 %**.
≈ **×10** — le scanner voit désormais les CHOCH qui existent.

## Défaut C — ancrage du focus par identifiant
Les événements n'avaient **aucun `id`** ; le front ancrait par **horodatage**.
- **Backend** : `BOSRecent`/`CHOCHRecent` portent un `id` **stable et sans collision**
  (`_event_id` → `<kind>_<broken_at_iso>_<direction>`) — le `kind` est dans l'id, donc un BOS et un
  CHOCH sur la **même bougie** ont des ids **différents**. Émis sur le journal ET le point-présent.
- **Frontend** : `eventId()` (préfère `ev.id`), `findEventById()` (**rejette un id inconnu**,
  jamais « le plus proche horodatage »). Clic ancré par id ; emphase des marqueurs par id ; un BOS
  **sélectionné** sur une bougie partagée **s'affiche** (n'est plus abandonné au profit du CHOCH) ;
  le cadrage caméra résout par id et rejette l'inconnu.
- **Testé en fabriquant** un BOS et un CHOCH sur la même bougie, **indépendamment de B** : cliquer
  le BOS met en avant le BOS, jamais le CHOCH.

## Défaut B — une règle d'arbitrage explicite et unique
**Preuve** : la branche de renversement du numba n'est **pas** un accident d'ordre d'exécution.
Une barre est **soit** un CHOCH **soit** un BOS de continuation — jamais les deux. Le `bos_event`
posé sur une barre CHOCH est le **déclencheur de retest** (`_calculate_bos_retest_*`), pas un second
événement BOS — le retirer casserait retest/confluence/RL/backtest → **Voie 2**.

**Règle (une phrase, trader)** : *« Sur une bougie qui casse la structure dans le sens opposé à la
tendance en cours, l'événement est un changement de caractère (CHOCH) ; il n'y a jamais, sur cette
même bougie, un BOS de même sens en plus. »*
- **Autorité unique et documentée** : `collect_structure_events` (le seul endroit qui mappe les
  colonnes brutes en événements de journal) porte la règle « THE SINGLE ARBITRATION AUTHORITY ». Le
  numba (2 chemins) et le mapper point-présent la référencent (bos_event = déclencheur de retest).
- **Aucun événement supprimé** pour masquer un symptôme ; l'invariant « aucune barre ne porte
  BOS+CHOCH de même sens » est verrouillé par test.

## Question E (mesure, aucune action)
`indeterminate` / `accumulation` / `ranging` = **0 % partout** (EURUSD M15, XAU/EUR H4/D1 ; ≈2000
barres). Cause : `indeterminate` exige une fenêtre sans aucune cassure. **Rien retiré.**

## Tests — ce qui a changé et POURQUOI (aucun test bâillonné)
- **Ajoutés** : garde « CHOCH récent vrai quand le journal en a un et `current_choch` est nul »
  (échoue sur le code d'avant) ; id stable & sans collision ; invariant « pas de BOS+CHOCH même
  sens sur une barre » ; focus par id robuste à une bougie partagée + rejet d'id inconnu.
- **Mis à jour (raison, pas un bâillon)** : les tests qui **encodaient le défaut** (conditions et
  lecture narrée nourries par le singulier) repointés sur le journal ; le reste = purs renommages
  de champ.
- **NON touché** : test AppWorkspace « skeleton » (PERF-2).
- **Pré-existants cassés (hors périmètre, non causés ici, non touchés)** :
  `test_tr1_structural_trend.py` (importe `_eval_mtf_aligned`, absent d'`origin/main`),
  `test_long_short_trading.py` (import cassé connu).

## Vérification
- **Backend pytest** : **462 passés** (335 + 127) sur l'ensemble des modules touchés
  (conditions scanner/endpoint, market_reading schema/mappers/assembler/endpoint, narrated,
  template, chatbot, pipeline, sentinel scanner, structure store/endpoint, liquidity, haiku,
  incremental, news, ob-rejection, perf1), 0 échec. Les deux fichiers **pré-existants cassés**
  (tr1, long_short) sont exclus — non causés par cette mission, non touchés.
- **Frontend** : `tsc --noEmit` **0 erreur** ; **vitest 916/916** (97 fichiers) ; `next build` **OK**.
- **Playwright** (chromium desktop 1280×… + iPhone 12 390×844, fr) : **26 passés, 2 skipés** —
  specs `vz-1-focus` (focus événement / défaut C), `sc1-scanner` (scanner), `rg1-regime` (régime).

**Pas de merge avant la confirmation visuelle live.**

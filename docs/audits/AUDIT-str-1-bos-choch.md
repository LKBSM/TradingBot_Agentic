# AUDIT STR-1 — BOS et CHOCH émis « simultanément » dans le même sens

> **Statut** : diagnostic en **LECTURE SEULE**. Aucune règle de détection n'a été
> modifiée, aucune branche créée. Le seul code exécuté est le harness d'audit
> MT-D1 (`scripts/audit/mt_d1/harness.py`), qui fait tourner le **vrai** moteur
> sans le changer, sur les bougies XAUUSD réellement mises en cache.

---

## Verdict (première ligne)

**C'est un défaut d'AFFICHAGE, pas une double détection.** Le moteur n'émet **qu'un
seul** événement structurel sur la bougie de 09:30 : **un CHOCH haussier**, cassure
d'**un seul** niveau. Ce même événement est enregistré par conception dans **deux
colonnes** (`CHOCH_SIGNAL` *et* `BOS_EVENT`) ; le journal transforme naïvement
chaque colonne en une ligne, d'où la fausse impression de deux événements. Ce
n'est **pas** le cas légitime « une grosse bougie casse plusieurs niveaux » — le
moteur n'en est structurellement pas capable sur une seule barre (voir §A).

L'hypothèse de la mission (« cascade à l'intérieur d'une bougie : le CHOCH bascule
la tendance, puis un BOS devient éligible ») est **infirmée** : il n'y a ni
réévaluation, ni second passage, ni second niveau. Une seule branche du code pose
les deux drapeaux.

---

## A) La mécanique réelle

### Un seul passage, une seule branche, un seul niveau

Fichier : `src/environment/strategy_features.py:100-147` (Numba) et son jumeau
Python `:202-247` (strictement équivalents). La boucle parcourt les bougies **une
seule fois**. À chaque bougie `i`, elle exécute **exactement une** des trois
branches d'un `if / elif / elif` :

```python
# CHOCH haussier (retournement) — lignes 113-120
if bos_signal[i-1] == -1 and current_close > current_high_structure:
    choch_signal[i] = 1
    bos_break_level[i] = current_high_structure      # UN seul niveau
    ...
    bos_signal[i]  = 1        # la tendance bascule à haussière
    bos_event[i]   = 1        # <-- le MÊME break est aussi noté BOS_EVENT
    last_bos_up_level = current_close

# CHOCH baissier (miroir) — lignes 121-128
elif bos_signal[i-1] == 1 and current_close < current_low_structure:
    choch_signal[i] = -1 ; bos_event[i] = -1 ; ...

# BOS de continuation — lignes 129-143, SEULEMENT si choch_signal[i] == 0
elif choch_signal[i] == 0:
    if ... : bos_event[i] = 1   # pose bos_event SANS choch_signal
```

Points de réponse à la mission :

- **Même passage ou deux ?** Un **seul** passage. Le BOS_EVENT de la bougie de
  retournement est posé **dans la branche CHOCH elle-même** (`bos_event[i] = 1`,
  ligne 119), pas par une seconde évaluation. Le `elif choch_signal[i] == 0`
  (ligne 129) **interdit** à la branche « BOS de continuation » de s'exécuter dès
  qu'un CHOCH a été posé. Il n'y a donc **jamais** de second BOS sur la barre.
- **Condition exacte de chacun** : le franchissement est **identique** (clôture
  au-delà du niveau structurel courant, sans marge, `>`/`<` strict). La **seule**
  différence est le contexte : cassure **contre** l'état de tendance courant ⇒
  CHOCH ; cassure **dans le sens** de l'état courant ⇒ BOS de continuation.
- **La tendance est-elle mise à jour pendant le passage ?** Oui : `bos_signal[i]`
  est réécrit à `+1` **dans** la branche CHOCH. Mais **un CHOCH ne peut pas rendre
  un BOS éligible sur la même bougie**, précisément parce que le `elif` bloque la
  branche BOS. La bascule de tendance n'a d'effet qu'à la bougie **suivante**
  (`bos_signal[i-1]` de l'itération d'après).
- **Règle d'exclusion / préséance existante ?** Dans le **moteur**, oui de fait :
  une bougie ne prend qu'une branche, donc jamais deux événements *sémantiques*.
  Ce qui manque, c'est une règle **au niveau de l'affichage** disant qu'une barre
  de CHOCH ne doit pas *aussi* apparaître comme « BOS » dans le journal. Cette
  règle **existe déjà pour le graphique** (voir §E) mais **pas pour le journal**.

### Ce n'est pas « plusieurs niveaux cassés »

Sur la bougie de retournement, `bos_break_level[i]` reçoit **un seul** niveau
(`current_high_structure`). Le moteur n'entretient qu'**un** sommet structurel
courant et qu'**un** creux courant ; il ne peut pas, sur une même barre,
enregistrer à la fois « j'ai cassé le dernier sommet inférieur » (CHOCH) **et**
« j'ai cassé un sommet supérieur » (BOS de continuation). Même si la grande bougie
verte engloutit visuellement plusieurs bougies, le moteur ne registre qu'**une**
cassure. **Le cas légitime multi-niveaux n'est donc pas ce qui se passe ici.**

---

## B) L'étendue du défaut (données réelles)

Mesuré en faisant tourner le vrai moteur sur les bougies XAUUSD en cache
(`docs/audits/ECHANTILLON-DETECTION-2026-07-29/_cache/`). Seules ces deux séries
sont disponibles localement ; les autres marchés/unités ne sont pas mis en cache
dans le dépôt et n'ont pas pu être comptés.

| Marché / unité | Bougies | Barres BOS_EVENT | Barres CHOCH | **Barres portant les DEUX** | même sens | sens opposé |
|----------------|--------:|-----------------:|-------------:|----------------------------:|----------:|------------:|
| XAUUSD **M15** | 6 952 | 165 | 64 | **64** | **64** | **0** |
| XAUUSD **H1**  | 2 440 | 71  | 27 | **27** | **27** | **0** |

Lecture de ce tableau :

- **Toute** barre de CHOCH est aussi une barre de BOS_EVENT (sous-ensemble strict
  vérifié : 0 CHOCH sans BOS_EVENT). Le nombre de co-émissions **est** exactement
  le nombre de CHOCH.
- **100 % des co-émissions sont de MÊME SENS.** C'est le cas contradictoire pour
  un lecteur SMC (un même break ne peut être à la fois « retournement vers le
  haut » et « continuation d'une hausse établie »).
- **Le cas SENS OPPOSÉ sur une même bougie n'existe pas — il est impossible par
  construction.** Une barre prend une seule branche ; la branche CHOCH pose
  `bos_event` **du même signe** que `choch_signal`. Il n'y a donc aucune barre où
  BOS et CHOCH pointent dans des directions opposées. (C'est une bonne nouvelle :
  le seul motif à traiter est le doublon même-sens, le moins grave.)
- **Le défaut n'est pas réservé aux grosses bougies d'expansion.** Il touche
  **chaque** bougie de retournement (chaque CHOCH), grande ou petite. La bougie
  géante de 09:30 est simplement l'exemple le plus visible ; en proportion, ~39 %
  des lignes « BOS » du journal (64/165 en M15, 27/71 en H1) sont en réalité des
  barres de CHOCH ré-étiquetées.

---

## C) Ce que dit la spécification

La spécification **connaît** ce comportement mais **ne le tranche pas** — c'est un
**trou de spécification assumé**, pas un oubli.

- `docs/architecture/SPEC-DETECTION-STRUCTURE.md` §C (« CHOCH et BOS sur la même
  bougie — le cas du 24 juillet », lignes 148-165) décrit le fait que
  `choch_signal[i]` **et** `bos_event[i]` s'allument ensemble, et le qualifie de
  « **structurellement normal et attendu** ». **Attention** : cette phrase est
  vraie **au niveau des colonnes internes** (`bos_event` = « un break a eu lieu
  ici », consommé par le `ConfluenceDetector`). Elle **ne dit pas** que le
  *journal* doive afficher **deux lignes** à l'utilisateur. La spec décrit le
  moteur, pas la couche d'affichage.
- `docs/audits/AUDIT-mt-d1-detection.md` est explicite sur le caractère **non
  tranché** :
  - §3, ligne 125 : « Toute bougie de CHOCH est aussi un BOS_EVENT … **Écart de
    convention possible (double étiquetage), choix de conception.** »
  - §7, point 6 (lignes 288-289) : « **Double étiquetage CHOCH+BOS sur la même
    bougie. Conforme au code, mais diverge de certaines conventions. Choix
    d'affichage à valider.** »

- **L'échantillon annoté** `docs/audits/ECHANTILLON-DETECTION-2026-07-29/` **ne
  contient aucun cas de co-occurrence** : sur ses 25 événements « émis », aucun
  couple (unité, horodatage) n'apparaît à la fois comme BOS et comme CHOCH — le
  générateur a présenté chaque barre de CHOCH sous **une seule** étiquette. De
  plus, les 66 lignes d'`ANNOTATION.csv` ont leur colonne *verdict* **vide** :
  l'échantillon n'a jamais été annoté. **Il ne classe donc ce motif ni comme
  correct, ni comme défaut** — il l'a masqué.

**Conclusion C** : la définition SMC (rappelée par l'audit, §3 ligne 124 : « le
CHOCH est le **premier** break contre-tendance ») implique qu'un retournement est
un CHOCH **et non** un BOS. Le double étiquetage est un choix d'affichage que la
spécification a laissé ouvert. Le trancher — CHOCH prime sur la barre partagée —
est cohérent avec la définition et doit être **écrit dans la spécification** puis
**couvert par un test**.

---

## D) La conséquence sur la tendance

**Aucune corruption de tendance.** La double émission ne dégrade l'état de tendance
nulle part :

- **État interne du moteur** : `bos_signal[i]` est réécrit **une seule fois** à
  `+1` dans la branche CHOCH (ligne 118). Le `bos_event = 1` supplémentaire ne
  touche pas `bos_signal`. L'état de tendance après 09:30 est donc **haussier,
  cohérent** — exactement ce que le CHOCH doit produire.
- **Tuile « Tendance » (Régime)** : `_derive_trend`
  (`src/intelligence/market_reading_mappers.py:1335-1348`) ne lit **que la série
  des clôtures** (première vs dernière sur la fenêtre). Elle **n'utilise ni
  `BOS_EVENT`, ni `CHOCH_SIGNAL`, ni `bos_signal`**. Une double émission ne peut
  pas la faire diverger.
- **Lecture narrée & panneau Régime** : idem — la tuile trend provient de
  `regime.trend` (issu de `_derive_trend`).
- **Condition d'alignement du scanner** (`mtf_aligned`, `trend_is`) :
  `src/api/routes/conditions_scan.py:225-236` construit son jugement à partir de
  `reading.regime.trend` — la **même** source que la tuile, donc découplée des
  événements. La double émission **ne peut pas** produire une tendance incohérente
  dans le scanner.

Le seul endroit du produit où la double émission fait surface est le **journal
d'événements** (et la liste « événements depuis » de la tuile Maturité) — c'est-à-dire
la couche qui matérialise les colonnes en lignes.

---

## E) Le défaut de focalisation (corrigeable indépendamment de la détection)

### Comment le clic résout sa cible

Fichier : `webapp/components/app/RegimeCard.tsx` (`selectEvent`, lignes 523-536 ;
`EvBtn`, lignes 150-172). L'identifiant d'un événement est
`` `${kind}:${atSec}:${level}` `` — il **contient** le type. Côté **ligne du
journal**, l'état sélectionné est donc bien distinct entre `bos:…` et `choch:…`.

Mais la cible **sur le graphique** se résout par **horodatage**, pas par cet id :
`webapp/lib/chart/structureMarkers.ts`. La couche de marqueurs **déduplique déjà**
la barre partagée (lignes 90-93) :

```js
for (const e of structure.bos_events ?? []) {
  const t = isoToSec(e.broken_at);
  ...
  if (chochTimes.has(t)) continue;   // « CHOCH already marks this bar »
```

Conséquence exacte du symptôme rapporté : quand on clique **BOS** sur la barre de
09:30, `selectEvent('bos', …)` cadre la caméra sur cette barre (`atSec`), **mais**
le marqueur BOS y a été **supprimé** au profit du marqueur CHOCH (`chochTimes.has(t)`),
et le marqueur CHOCH restant n'est **pas** mis en accent (il n'est peint en accent
que si `selected.kind === 'choch'`). Le clic « BOS » atterrit donc sur une barre
dont le **seul** marqueur visible est « CHOCH », sans marqueur BOS à éclairer →
**« le mauvais événement est ciblé »**.

C'est bien le motif que le verrou d'id devait empêcher : l'id-lock distingue les
deux **lignes**, mais la couche **graphique** confond les deux **barres** parce
qu'elle dédup­lique par horodatage. Les deux surfaces ne suivent **pas** la même
règle : le graphique dit « une seule marque (CHOCH) », le journal dit « deux
lignes (BOS + CHOCH) ».

### Nature de la correction

**Elle est purement d'affichage — elle ne touche pas la détection.** La bonne
correction aligne le **journal** sur la règle que le **graphique applique déjà** :
sur une barre portant un CHOCH, **ne pas** afficher aussi une ligne « BOS » (le
CHOCH est la lecture SMC correcte du retournement). Cela :

1. supprime le doublon contradictoire même-sens (§B) ;
2. supprime du même coup l'ambiguïté de focalisation : il ne reste qu'un seul
   événement cliquable sur la barre, dont le marqueur graphique existe et
   s'éclaire correctement.

Aucune émission n'est « supprimée » au sens de la détection : les colonnes
`BOS_EVENT` / `CHOCH_SIGNAL` restent intactes pour le `ConfluenceDetector`. On ne
change que la **projection** des colonnes en lignes de journal.

---

## Recommandation — quelle règle appliquer quand une bougie satisfait les deux conditions

**Règle proposée : préséance du CHOCH sur la barre partagée (« CHOCH wins the bar »).**
Sur toute barre où `CHOCH_SIGNAL ≠ 0`, le journal affiche **un seul** événement,
de type **CHOCH** ; la ligne « BOS » de la même barre et du même signe est
supprimée de l'affichage.

**Pourquoi cette règle plutôt qu'une suppression arbitraire :**

1. **Elle découle de la définition SMC**, pas d'un contournement de symptôme. Un
   CHOCH est le **premier** break **contre** la tendance ; c'est un
   **retournement**, catégorie mutuellement exclusive du BOS de **continuation**.
   Une barre de retournement **est** un CHOCH — l'étiquette BOS y est un artefact
   de plomberie interne (`bos_event` = « un break a eu lieu »), pas un second
   événement.
2. **Elle ne perd aucune information** : le break, sa direction, son niveau et son
   horodatage sont exactement ceux de la ligne BOS supprimée (même niveau
   `bos_break_level`, même barre). On ne retire qu'un **doublon**.
3. **Elle rend les deux surfaces cohérentes** : le journal appliquera enfin la
   même règle que `structureMarkers.ts` applique déjà au graphique (ligne 93).
   C'est un alignement, pas une invention.
4. **Elle est sûre** : le seul motif observé sur données réelles est le doublon
   **même-sens** (0 cas opposé, §B), et il est **impossible** qu'une barre porte un
   BOS et un CHOCH de sens opposés. La règle n'a donc aucun cas ambigu à arbitrer.

**Le cas multi-niveaux légitime n'existe pas dans ce moteur** (§A) : inutile de
prévoir une préséance pour « une grosse bougie casse réellement plusieurs
niveaux » — le moteur n'enregistre qu'un niveau par barre. Si un jour on voulait
exprimer des cassures multi-niveaux, ce serait un **chantier de détection** distinct
(entretenir plusieurs niveaux structurels), à spécifier séparément.

**Où écrire la règle :**

- **Spécification** : ajouter à `SPEC-DETECTION-STRUCTURE.md` §C une clause
  d'affichage explicite : « une barre de CHOCH n'engendre qu'un événement de
  journal, de type CHOCH ; `BOS_EVENT` sur cette barre est une colonne interne
  non affichée ».
- **Backend** (option recommandée, une seule source) : dans
  `collect_structure_events` (`market_reading_mappers.py:632-659`), exclure de
  `bos_events` toute barre où `CHOCH_SIGNAL ≠ 0` — la dédup se fait alors **en
  amont**, une seule fois, pour le journal comme pour la tuile Maturité. Le
  graphique n'aurait même plus besoin de sa dédup locale (elle deviendrait
  redondante mais inoffensive).
- **Test** : un test de garde vérifiant que, sur une barre de CHOCH connue (p. ex.
  H1 2026-07-23 17:00 ou M15 2026-07-14 12:15, présentes dans l'échantillon),
  `bos_events` **ne** contient **pas** cette barre, et que le journal n'expose
  qu'une ligne CHOCH.

**Correction de focalisation** : elle tombe automatiquement avec la dédup ci-dessus
(plus de ligne BOS ambiguë sur la barre partagée). Si l'on préférait garder la dédup
côté front, il faudrait au minimum que le clic « BOS » sur une barre partagée
retombe sur le marqueur CHOCH réellement présent — mais la dédup backend est plus
propre et supprime le doublon à la racine.

---

## Implémentation (après GO)

La règle de préséance a été implémentée **sans toucher la détection** :

- **Source unique — `collect_structure_events`** (`market_reading_mappers.py`) :
  toute barre portant `CHOCH_SIGNAL ≠ 0` est retirée de `bos_events` (paramètre
  `drop_choch_bars=True`) ; `choch_events` est inchangé. Le journal, la liste
  « événements depuis » (Maturité) et le champ ponctuel `structure.bos` en
  héritent d'un coup.
- **Champ ponctuel `structure.bos`** : `fresh_break` exclut désormais les barres
  de CHOCH (`fresh_break = |BOS_EVENT| > 0 and CHOCH_SIGNAL == 0`). Un break
  **persisté** (retest d'un BOS antérieur, non-CHOCH) reste affiché — c'est
  légitimement un BOS retesté.
- **Spécification** : clause « préséance du CHOCH sur une barre partagée » ajoutée
  à `SPEC-DETECTION-STRUCTURE.md` §C.
- **Cache** : `READING_LOGIC_VERSION` 5 → 6 (les lectures en cache portant le
  doublon sont invalidées et reconstruites).
- **Tests de garde** : `test_collect_structure_events_choch_precedence_drops_bos_twin`
  (une barre BOS+CHOCH n'apparaît que dans `choch_events`),
  `test_choch_bar_does_not_also_surface_point_in_time_bos`, et renfort de
  `test_choch_level_uses_break_level` (`s.bos is None`).
- **Frontend** : aucun changement — il consomme le `bos_events` déjà dédupliqué.
  La dédup de `structureMarkers.ts` reste en défense en profondeur.

### Vérifié sur données réelles (moteur réel, XAUUSD en cache)

| Unité | doublons AVANT | doublons APRÈS | tendance AVANT/APRÈS |
|-------|---------------:|---------------:|----------------------|
| M15   | 64             | **0**          | bullish / **bullish** (ancrée CHOCH) |
| H1    | 27             | **0**          | bullish / **bullish** (ancrée CHOCH) |

Chaque retournement n'apparaît plus qu'**une fois**, comme CHOCH ; aucune barre de
CHOCH ne fuit dans `bos_events` ; la **tendance est identique** avant/après
(`derive_structural_trend` s'ancre d'abord sur `choch_events`, que la règle ne
touche pas). Suites `test_market_reading_*`, `test_structure_*`,
`test_conditions_*`, `test_incremental_detection` : **175 passés, 2 skippés**.

*La détection n'a pas été modifiée : les colonnes `BOS_EVENT` / `CHOCH_SIGNAL`
sont intactes pour le moteur et le `ConfluenceDetector`. Seule la projection des
colonnes en événements d'affichage a changé.*

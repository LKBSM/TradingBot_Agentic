# Spécification de la détection de structure (BOS / CHOCH / swings / tendance)

> **Statut** : document descriptif, produit par la mission **MT-D1** en LECTURE
> SEULE. Il décrit ce que le moteur fait *aujourd'hui*, sans le juger ni le
> corriger. Chaque règle renvoie au fichier et aux lignes qui l'implémentent, pour
> qu'un trader puisse vérifier lui-même. Aucune règle n'a été modifiée pour écrire
> ce document.
>
> **Périmètre** : la logique de swings, BOS, CHOCH et de la tuile « Tendance ».
> Order Blocks, FVG, poches de liquidité et retest ne sont décrits que lorsqu'ils
> éclairent ces quatre éléments.

Le cœur de la détection vit dans **`src/environment/strategy_features.py`**
(la façade `src/intelligence/smart_money/__init__.py` ne fait que ré-exporter
`SmartMoneyEngine`, `SMCConfig`, `calculate_bos_choch_fast`). La version Numba et
la version Python de la boucle BOS/CHOCH sont **strictement équivalentes**
(vérifié : une ré-implémentation indépendante reproduit à l'identique le flux
d'événements du moteur sur 2 465 bougies H1 et 6 952 bougies M15 — voir
`AUDIT-mt-d1-detection.md`, contrôle de parité).

---

## A) Swing high / swing low (les « fractals »)

**Fichier** : `src/environment/strategy_features.py:787-834`
(méthode `_add_smc_base_features`). Paramètre : `SMCConfig.FRACTAL_WINDOW`
(`:486-490`), **valeur par défaut `2`**.

### Règle exacte, en clair

Un **sommet** (swing/up-fractal) est reconnu sur une bougie dont le **plus haut**
est le plus élevé d'une fenêtre de `2·N+1` bougies **centrée** sur elle, soit
**`N` bougies à gauche et `N` bougies à droite**. Avec le réglage par défaut
`N = 2`, cela fait une **fenêtre de 5 bougies** : la bougie centrale doit avoir le
plus haut le plus élevé des 2 d'avant, d'elle-même et des 2 d'après.

Un **creux** (swing/down-fractal) est le miroir : sa bougie doit avoir le **plus
bas** le plus bas de la fenêtre de 5.

### Points précis demandés

- **Combien de bougies de part et d'autre ?** `N = FRACTAL_WINDOW`, soit **2 de
  chaque côté** par défaut (fenêtre totale de 5 bougies). `:796-797`.
- **Sur les extrêmes ou sur les clôtures ?** Sur les **extrêmes** : le sommet
  compare des `high`, le creux compare des `low` (`:807-823`). Les clôtures
  n'interviennent pas dans la définition d'un swing.
- **Filtre d'amplitude minimale ?** **Aucun.** Il n'existe pas de seuil de taille
  minimale : un micro-sommet d'un demi-dollar est un swing au même titre qu'un
  sommet majeur, dès lors qu'il domine ses 4 voisins. (Le seul paramètre nommé
  « amplitude » du fichier, `OUTLIER_ATR_MULT` `:542-548`, est purement
  observationnel — il pose un drapeau, il ne filtre aucun swing.)
- **Confirmation causale / décalage** : un swing n'est connu que `N` bougies plus
  tard (le temps que la fenêtre de droite existe). Le moteur **décale donc le
  fractal de `N` bougies** (`shift(N)`, `:825-828`) pour ne jamais utiliser le
  futur. Concrètement : **un swing du présent n'est confirmé que 2 bougies après
  s'être produit** ; les `N` premières et `N` dernières bougies de la fenêtre ne
  peuvent porter aucun fractal (`:830-834`).
- **Égalités parfaites** : la comparaison est `high == max(fenêtre)` (`:812-816`).
  En cas d'**ex-æquo** (deux bougies au même plus haut exact dans la fenêtre),
  **les deux** satisfont l'égalité et sont donc **toutes deux marquées comme
  swing**. Il n'y a pas de règle de départage « première / dernière ». Sur des
  prix continus l'ex-æquo est rare ; il devient possible sur des données à
  faible résolution ou lors de plateaux de prix (p. ex. week-end, cf. le cas
  ouvert dans l'audit).

---

## B) BOS (Break of Structure — cassure de structure)

**Fichier** : la boucle `_calculate_bos_choch_numba` / `_calculate_bos_choch_python`
`src/environment/strategy_features.py:45-247`, appelée par
`_calculate_structure_iterative` `:942-984`. Colonnes produites :
`BOS_EVENT` (±1 uniquement sur la bougie de cassure), `BOS_SIGNAL` (état de
tendance propagé — hérité, à ne PAS confondre avec l'événement), `BOS_BREAK_LEVEL`
(niveau franchi).

### Ce qui est suivi

Le moteur entretient deux niveaux de **structure courante** :
`current_high_structure` (le sommet structurel en cours) et
`current_low_structure` (le creux structurel en cours). Ils sont initialisés puis
**mis à jour à partir des fractals** au fil des bougies (`:103-108`) — jamais à
partir du plus haut/bas d'une bougie quelconque.

### Règle exacte, en clair

- **Quel niveau doit être franchi ?** Le **sommet structurel courant**
  (`current_high_structure`) pour un BOS haussier ; le **creux structurel
  courant** (`current_low_structure`) pour un BOS baissier (`:130-143`). Ce niveau
  est un **niveau de swing (fractal)**, pas le plus haut de la bougie précédente.
- **Par mèche ou par clôture ?** Par **CLÔTURE**. La condition compare
  `current_close` (`closes[i]`) au niveau structurel : `current_close >
  current_high_structure` pour un BOS haussier, `current_close <
  current_low_structure` pour un baissier (`:130`, `:137`). **Une mèche qui perce
  le niveau mais dont la bougie reclôture en-deçà ne déclenche AUCUN BOS.**
- **Marge / seuil de franchissement ?** **Aucun.** La comparaison est un `>` (resp.
  `<`) **strict, sans tampon** : un dépassement de la clôture d'un centime au-dessus
  du niveau suffit ; il n'y a pas de « close au-delà de X ATR ». (Le paramètre
  `FVG_THRESHOLD` concerne les Fair Value Gaps, pas le BOS.)
- **Tendance préexistante exigée ?** Pour un BOS de **continuation** :
  `bos_signal[i-1] >= 0` pour un BOS haussier (état de tendance non baissier), et
  `bos_signal[i-1] <= 0` pour un baissier (`:130`, `:137`). Cet « état de
  tendance » est l'`BOS_SIGNAL` propagé — voir §D. Ce n'est donc pas une pente ni
  une moyenne mobile : c'est l'**empreinte de la dernière cassure**.
- **BOS répétés dans le même sens** : ils **ne sont pas tous émis**. Une garde
  empêche de ré-émettre un BOS sur le même niveau tant qu'un **nouveau** swing
  plus haut (resp. plus bas) ne s'est pas formé : `allow_bos_up = last_fractal_high
  > last_bos_up_level` (`:110-111`, `:130`, `:137`). Après une cassure haussière,
  `last_bos_up_level` retient le prix de clôture qui a cassé ; il faut qu'un
  **fractal plus haut que ce prix** apparaisse pour qu'un nouveau BOS haussier soit
  possible. **Conséquence importante** : le prix peut continuer à monter au-dessus
  de l'ancien sommet sans nouvel événement, tant qu'aucun nouveau sommet-fractal ne
  s'est formé au-dessus. C'est la première cause de « franchissement sans
  événement » (voir l'échantillon des non-événements).

---

## C) CHOCH (Change of Character — changement de caractère)

**Fichier** : même boucle, branches `:113-128`. Colonne `CHOCH_SIGNAL` (±1 sur la
bougie de retournement).

### Ce qui distingue exactement un CHOCH d'un BOS *dans le code*

Un **CHOCH n'est rien d'autre qu'un BOS qui casse la structure dans le sens
OPPOSÉ à l'état de tendance en cours**. Précisément :

- Si l'état de tendance précédent est **baissier** (`bos_signal[i-1] == -1`) et que
  la **clôture repasse au-dessus du sommet structurel** → c'est un **CHOCH
  haussier** (`:113-120`).
- Si l'état de tendance précédent est **haussier** (`bos_signal[i-1] == 1`) et que
  la **clôture repasse sous le creux structurel** → **CHOCH baissier** (`:121-128`).

Le franchissement lui-même (clôture au-delà du niveau structurel, sans marge)
est **identique** à celui d'un BOS. La seule différence est le **contexte de
tendance** : même sens que l'état en cours ⇒ BOS de continuation ; sens opposé ⇒
CHOCH.

### Quel état antérieur doit exister ?

Un CHOCH n'est possible que si `bos_signal[i-1]` est **strictement** de signe
opposé (`== -1` pour un CHOCH haussier, `== 1` pour un CHOCH baissier). Autrement
dit, il faut qu'une **tendance ait déjà été établie** par une cassure antérieure.
Tant que `bos_signal` vaut 0 (aucune cassure encore survenue), **aucun CHOCH ne
peut être émis** — la première cassure de l'historique est nécessairement un BOS,
jamais un CHOCH.

### CHOCH et BOS sur la même bougie — le cas du 24 juillet

**C'est structurellement normal et attendu.** Quand une bougie de retournement se
produit, la branche CHOCH pose **à la fois** `choch_signal[i] = 1` **et**
`bos_event[i] = 1` (`:114-119` ; miroir `:122-127`). Autrement dit :

> **Toute bougie de CHOCH est aussi une bougie de BOS_EVENT** (un retournement est
> une cassure). L'inverse est faux : un BOS de continuation (`:130-143`) pose
> `bos_event` **sans** poser `choch_signal`.

Sur **XAUUSD H1, le 24 juillet 18:00 UTC** (l'exemple équivalent dans les données
rechargées ; l'horodatage « 04:00 » de la mission relève d'un fuseau d'affichage
différent), une même bougie porte **BOS haussier + CHOCH haussier** au niveau
4052,33 : le prix était en état baissier, sa clôture est repassée au-dessus du
sommet structurel, ce qui est **simultanément** un changement de caractère (CHOCH)
et une cassure (BOS_EVENT). Les deux colonnes s'allument donc sur la même barre.
Ce n'est pas un doublon ni un bug **au niveau des colonnes internes** : ce sont
deux lectures de la même cassure (« la tendance a changé » **et** « un niveau a
été cassé »). `bos_event` est une colonne **interne** (consommée par le
`ConfluenceDetector`), pas un second événement destiné à l'utilisateur.

### Règle d'affichage — préséance du CHOCH sur une barre partagée (STR-1)

> **Un événement, une ligne.** Une barre de CHOCH ne produit qu'**un seul**
> événement de journal, de type **CHOCH**. La colonne `BOS_EVENT` de cette barre
> n'est **pas** affichée : la matérialiser en une seconde ligne « BOS » de même
> sens, même horodatage et même niveau serait contradictoire pour un lecteur SMC
> (un retournement n'est pas une continuation) et rendait le clic de focalisation
> ambigu (audit `docs/audits/AUDIT-str-1-bos-choch.md`).

Cette règle est appliquée **une seule fois, en amont**, dans
`collect_structure_events` (`src/intelligence/market_reading_mappers.py`) : toute
barre portant `CHOCH_SIGNAL ≠ 0` est **retirée** de `bos_events` (elle reste dans
`choch_events`). Le champ ponctuel `structure.bos` suit la même préséance (non posé
sur une barre de CHOCH fraîche). Le graphique appliquait déjà la règle jumelle
(`webapp/lib/chart/structureMarkers.ts`, « CHOCH wins a shared bar ») ; journal et
graphique partagent désormais la même règle depuis une source unique.

**Ce que la règle ne fait pas** : elle ne supprime aucune détection (les colonnes
`BOS_EVENT`/`CHOCH_SIGNAL` restent intactes pour le moteur et le
`ConfluenceDetector`) et ne concerne **que** l'affichage. Un BOS de **continuation**
(`bos_event` sans `choch_signal`) est inchangé. La **tendance** est inchangée
(§D `derive_structural_trend` s'ancre d'abord sur `choch_events`, que la règle ne
touche pas ; le repli sur `bos_events` ne joue que sans aucun CHOCH — donc sans
barre à retirer). Le cas « sens opposé sur une même barre » n'existe pas
(impossible par construction : une barre ne prend qu'une branche du `if/elif`).

---

## D) Tendance (tuile « Régime ») — dérivée de la structure (TR-1)

> **Réécrit par TR-1 (2026-07-29).** La tuile « Tendance » **ne compare plus la
> première et la dernière clôture** d'une fenêtre de 500 bougies. Elle est
> désormais **dérivée de la structure détectée par le moteur** : plus de seconde
> source de vérité. `_derive_trend` (calcul par clôtures) a été **supprimée**.

**Fichier** : `src/intelligence/market_reading_mappers.py` — `derive_structural_trend`
(dérivation) + `candles_to_regime` (assemblage), alimentée par les événements
BOS/CHOCH que l'assembleur a déjà collectés
(`market_reading_assembler.py` → `smc_features["_structure_events"]`).

### Définition exacte — def(a), dernier événement non contredit

La tendance = **le sens de la dernière cassure de structure (BOS ou CHOCH) non
contredite par une cassure de sens opposé**. C'est l'état `BOS_SIGNAL` propagé du
moteur (§B/§C), lu depuis les événements discrets :

- l'**événement d'ancrage** est le **dernier CHOCH** (le dernier changement de
  caractère qui a fixé le sens) ; à défaut de tout CHOCH, le **dernier BOS** ;
- son sens donne la tendance : `bullish` ou `bearish` ;
- **s'il n'existe AUCUN BOS ni CHOCH** dans l'historique analysé → **`indeterminate`**,
  un **état de première classe**, jamais un repli silencieux sur « neutre ».

L'ancrage est **exposé** (`regime.trend_reference` : type d'événement, sens,
niveau, `broken_at`, `bars_ago`) pour que la tuile nomme sa raison — « depuis le
CHOCH haussier du 24 juil. » — **la même histoire que la tuile Maturité** (toutes
deux ancrées sur le dernier CHOCH).

### Entrées exactes

Uniquement les **colonnes d'événements du moteur** (`BOS_EVENT`, `CHOCH_SIGNAL`,
`BOS_BREAK_LEVEL`) via `collect_structure_events`. **Aucune clôture, aucune
fenêtre fixe, aucun seuil.** La détection n'est **pas** modifiée (non-régression :
`src/environment/strategy_features.py` intact, diff vide).

### Fenêtre

**Plus de fenêtre fixe de 500 bougies.** La tendance « remonte au dernier
événement ». Mesure d'appui (XAUUSD, 60 j) : écart maximal au dernier événement =
192 bougies (H1) / 443 (M15), toujours < 500 → sur H1/M15 `indeterminate`
n'apparaît jamais ; il ne se présente qu'au tout début d'un historique ou sur
données très clairsemées.

### « ranging » retiré de la tuile Tendance (décision D2)

L'ancien état `ranging` (seuil 0,3 sur les clôtures, non documenté) **n'est plus
émis par la tendance**. Le test de consolidation a **migré vers la tuile Phase**
(`_derive_market_phase` : `indeterminate` + oscillation nette → `ranging`, sinon
`accumulation`) — fin de la redondance Tendance/Phase relevée en audit.

### Alignement multi-unités & scanner

Le biais par unité supérieure (`mtf_confluence`) est **lui aussi structurel**
(`_structural_bias_from_candle_dicts` : sens du dernier CHOCH/BOS du moteur sur
l'unité). Une unité **indéterminée** est comptée **à part** — jamais un accord ni
un désaccord — avec un **dénominateur ajusté et visible** (« N sur M »). La
condition **« 3 TF alignés »** du scanner devient un **alignement de STRUCTURE** ;
un combo dont une unité est indéterminée n'est **jamais** présenté comme aligné.

### Peut-elle encore diverger du journal ?

**Non, par construction** : la tuile et le journal lisent désormais **le même**
dernier événement. Le cas ouvert DG-1 (26-28 juil.) est réconcilié : la tendance
**reste haussière** depuis le CHOCH haussier du 24 juil., cohérente avec le
journal, tant qu'aucun creux protégé n'est cassé en clôture.

### Données persistées (décision D4)

`READING_LOGIC_VERSION` bumpée à **3** : toute lecture stockée sous l'ancienne
définition est **reconstruite avant tout affichage** (jamais deux définitions
mélangées à l'écran, sans que le client puisse le savoir).

---

## E) Paramètres de `SMCConfig` influant sur A → D

**Fichier** : `SMCConfig` `src/environment/strategy_features.py:453-581`.

### Le seul paramètre qui change l'ÉMISSION des BOS/CHOCH

| Paramètre | Défaut | Effet sur A–D | Origine documentée |
|-----------|:------:|---------------|--------------------|
| **`FRACTAL_WINDOW`** | **2** | Nombre de bougies de chaque côté pour valider un swing (§A). C'est **le seul paramètre de `SMCConfig` qui modifie la définition des swings, donc les niveaux structurels, donc l'émission des BOS/CHOCH**. Plus grand ⇒ swings plus rares et plus « majeurs » ⇒ moins de niveaux ⇒ (voir sensibilité) moins d'événements. `:486-490` | Commentaire « 2 = 5 bougies » ; pas d'autre justification écrite. |

### Paramètres qui touchent la structure MAIS pas l'émission BOS/CHOCH

| Paramètre | Défaut | Effet | Note |
|-----------|:------:|-------|------|
| `ATR_WINDOW` | 14 | Période de l'ATR. N'entre PAS dans BOS/CHOCH (aucune marge ATR n'est appliquée au franchissement). Sert au retest et au drapeau outlier. `:481-485` | Wilder standard. |
| `RETEST_TOL_ATR` | 0.5 | Tolérance de retest (machine à états post-cassure). N'affecte PAS l'émission d'un BOS, seulement l'armement du retest. `:515-520` | Audit P1-3. |
| `RETEST_INVALID_TOL_ATR` | 1.0 | Invalidation du setup de retest. Idem : postérieur à l'événement. `:521-526` | — |
| `RETEST_AWAITING_TIMEOUT` | 20 | Fenêtre d'attente du pullback. Idem. `:527-531` | — |
| `RETEST_ARMED_WINDOW` | 30 | Durée de validité d'un setup armé. Idem. `:532-540` | Dimensionné ~vie d'un trade XAU M15. |
| `OUTLIER_ATR_MULT` | 5.0 | **Drapeau** observationnel des bougies géantes. **Ne filtre rien** ; aucune logique de détection ne le lit. `:542-548` | Audit D4-3. |
| `FVG_THRESHOLD` | 0.1 | Taille min. d'un FVG (fraction d'ATR). Concerne les FVG, pas les BOS/CHOCH. `:491-495` | Audit P1-2. |
| `FVG_SESSION_GAP_MULT` | 1.5 | Supprime les FVG à cheval sur une fermeture de séance. FVG seulement. `:496-504` | — |
| `OB_REQUIRE_FVG` / `OB_FVG_BONUS` | False / 0.2 | Order Blocks. Sans effet sur BOS/CHOCH. `:505-513` | — |
| `EQ_TOLERANCE_ATR`, `EQ_TOLERANCE_PIPS_FLOOR`, `EQ_MIN_TOUCHES`, `LIQ_LOOKBACK` | 0.10 / 0.0 / 2 / 200 | **Poches de liquidité (EQH/EQL)** — descriptif seul. Réutilisent les swings existants mais **ne touchent aucune règle** BOS/CHOCH/OB/FVG (commentaire explicite `:549-555`). `:556-581` | — |
| `RSI_WINDOW`, `MACD_*`, `BB_WINDOW` | 14 / 12·26·9 / 20 | Indicateurs techniques classiques, hors structure. `:456-480` | Standards Wilder/Appel. |

### Le point à retenir de la section E

Parmi tout `SMCConfig`, **seul `FRACTAL_WINDOW` influe sur l'émission des BOS/CHOCH**.
Les trois autres « leviers » qu'un trader imaginerait — **franchissement mèche vs
clôture**, **marge/tolérance de franchissement**, **condition de tendance
préalable** — **ne sont PAS des paramètres** : ils sont **écrits en dur** dans la
boucle (`closes[i]` pour la clôture, `>`/`<` stricts pour l'absence de marge, la
mécanique `bos_signal` pour la tendance). Les faire varier suppose de toucher au
moteur ; l'audit les explore donc **hors dépôt**, en mémoire seulement (cf.
`AUDIT-mt-d1-detection.md`, sections 7-8).

---

## Récapitulatif d'une ligne par élément

- **Swing** : plus haut/bas dominant une fenêtre de 5 bougies (2 de chaque côté),
  sur les extrêmes, sans filtre d'amplitude, confirmé 2 bougies plus tard.
- **BOS** : clôture (pas mèche) au-delà du sommet/creux structurel, sans marge,
  dans le sens de l'état courant, une seule fois par niveau.
- **CHOCH** : le même franchissement mais dans le sens **opposé** à l'état courant ;
  toute bougie de CHOCH est aussi un BOS_EVENT.
- **Tendance (tuile)** : compare la **première et la dernière clôture** de ~500
  bougies, avec un test de « range » ; **totalement indépendante** des swings et
  des événements.

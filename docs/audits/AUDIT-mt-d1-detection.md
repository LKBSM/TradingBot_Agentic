# AUDIT MT-D1 — Détection de structure & calcul de tendance

> **LECTURE SEULE.** Aucune règle de détection n'a été modifiée, ajustée ni
> « corrigée ». Aucune branche de correction. Les seules « variations » (sections
> 7-8) sont calculées **hors dépôt, en mémoire**, pour le rapport ; le moteur du
> dépôt est intact. Ce document **compte et montre** ; il **ne recommande aucun
> réglage** et ne qualifie la configuration ni de trop stricte ni de correcte.
>
> **Données** : XAUUSD, rechargées depuis Twelve Data le 2026-07-29 (horodatages
> **UTC**). H1 : 2 465 bougies (18/04 → 29/07). M15 : 6 952 bougies (18/05 → 29/07).
> Couvre le cas ouvert (24-28 juillet) et deux fenêtres de 60 jours.
> Détail des règles : `docs/architecture/SPEC-DETECTION-STRUCTURE.md`.
> Reproductibilité : `scripts/audit/mt_d1/` (fetch → run_audit → generate_samples).

---

## ⚠️ Contradiction avec une règle écrite du dépôt ?

**Aucune trouvée.** La consigne était : si un comportement **contredit une règle
écrite du dépôt**, le signaler ici et s'arrêter. Après contrôle :

- **Contrôle de parité** : une ré-implémentation indépendante de la boucle
  BOS/CHOCH (mode par défaut) reproduit **exactement** le flux d'événements du
  moteur (`BOS_EVENT` et `CHOCH_SIGNAL` identiques bougie par bougie) sur H1 et
  M15. Le moteur fait donc **ce que son code dit qu'il fait**.
- Le **cas ouvert** (section 4) est **conforme** à la règle écrite (franchissement
  en clôture) : rien n'est manqué au regard de cette règle.
- Les écarts relevés dans ce rapport sont des **choix de conception** (clôture vs
  mèche, cap d'affichage, fenêtre de tendance), pas des violations de règles.

Le rapport se poursuit donc normalement. **Aucun élément ne relève d'une mission
de correction distincte à ce stade.**

---

# SECTION 6 — VERDICT : affichage ou détection ? *(à lire en premier)*

## Réponse courte

> **C'est d'abord un effet d'AFFICHAGE, pas une détection trop stricte.** Le moteur
> détecte largement plus d'événements que la surface n'en montre. Deux
> mécanismes, **aucun n'est visible pour le client**, réduisent le journal affiché :
> (1) la **fenêtre live de ~500 bougies**, et (2) un **plafond de 8 événements par
> type** (`MAX_STRUCTURE_EVENTS`).

## Quel journal alimente l'écran ?

Il existe deux journaux dans le code :

| Journal | Source | Alimente l'écran ? |
|---------|--------|--------------------|
| **Assembleur LIVE** (`/api/market-reading`) | fenêtre de `MARKET_READING_LOOKBACK` bougies, **défaut 500** (`market_reading_assembler.py:214`, `bootstrap.py:124`) | **OUI** — c'est la **seule** source à l'écran. |
| **Store profond** (`/api/structure`, `/api/coverage`) | `StructureStore` persistant, journal **non plafonné** (`incremental_detection.py:40,57-58` `_UNLIMITED`) | **NON** — délibérément **non câblé à l'UI** (décision DG-1 (b), `routes/structure.py:18-30`), verrouillé par un test-garde `single-journal-guard.test.ts`. |

- Le **panneau « Journal des événements »** (`EventsSection.tsx`) et la tuile
  **« Dernier événement »** (`RegimeSection.tsx`, formatteur
  `use-reading-formatters.ts:239-267`) sont donc alimentés par **l'assembleur
  LIVE uniquement**.
- La tuile « Dernier événement » n'affiche que **le plus récent** BOS **ou** CHOCH
  (un seul, `structure.bos`/`structure.choch`).
- Les marqueurs BOS/CHOCH du graphique proviennent des tableaux `bos_events` /
  `choch_events`, **plafonnés à 8 par type** par `collect_structure_events`
  (`market_reading_mappers.py:239` `MAX_STRUCTURE_EVENTS = 8`, tri « plus récent
  d'abord », `:649-653`).
- Le **front n'ajoute aucun plafond** : il affiche ce que le back envoie. Le
  plafond est donc **entièrement côté back** (fenêtre + cap 8), et **n'est ni
  paginé ni étiqueté** à l'écran.

## Combien le store profond contient-il vs combien la surface affiche-t-elle ?

**XAUUSD H1, 30 derniers jours** (le store profond, non plafonné, contiendrait par
construction **tous** les événements détectés ; la surface = fenêtre 500 + cap 8) :

| | BOS | CHOCH | Total |
|---|:---:|:---:|:---:|
| **Détecté sur 30 j (= ce que le store profond contiendrait)** | 17 | 7 | **24** |
| Présents dans la fenêtre live (500 bougies ≈ 20,8 j) | 14 | 5 | 19 |
| **Réellement affichés (après cap 8/type)** | 8 | 5 | **13** |
| Tuile « Dernier événement » | — | — | **1** |

**XAUUSD M15, 30 derniers jours** :

| | BOS | CHOCH | Total |
|---|:---:|:---:|:---:|
| **Détecté sur 30 j (= store profond)** | 55 | 18 | **73** |
| Présents dans la fenêtre live (500 bougies ≈ **5,2 j**) | 7 | 3 | 10 |
| **Réellement affichés (après cap 8/type)** | 7 | 3 | **10** |

## Lecture de l'écart

- **L'écart est important**, surtout en M15 : **73 détectés → 10 affichés** sur 30
  jours. Le problème observé (« trop peu d'événements ») est donc **dominé par
  l'affichage**, pas par une détection trop stricte.
- **En H1**, la fenêtre de 500 bougies (~21 j) couvre presque les 30 jours : le
  rabot vient surtout du **cap 8** (14 BOS détectés dans la fenêtre → 8 affichés).
- **En M15**, la fenêtre de 500 bougies ne couvre que **~5 jours** : c'est **la
  fenêtre** qui coupe l'essentiel (55 BOS/30 j → 7 dans la fenêtre), le cap 8
  n'ayant même pas à jouer.
- **Aucun de ces deux plafonds (fenêtre, cap 8) n'est signalé au client.** Rien à
  l'écran n'indique « 13 événements sur 24 », ni « fenêtre = 5 jours en M15 ».

**Conséquence pour la suite** : la mission prévoit que la **section 7 (sensibilité
de détection) ne soit menée que si l'affichage n'est pas en cause**. Ici
**l'affichage EST en cause**. La section 7 est néanmoins fournie ci-dessous — mais
**explicitement hors de ce gate** — parce que le fondateur a aussi posé la question
de la détection elle-même ; elle est présentée comme une mesure séparée de l'axe
« détection », sans conclusion sur l'affichage.

---

# SECTION 2 — Confrontation aux conventions SMC

Il n'existe **aucune définition canonique** du SMC : selon les auteurs, un BOS
exige une clôture ou une simple mèche, un swing se définit sur 3 ou 5 bougies, un
CHOCH dépend d'une notion de tendance elle-même variable. **On décrit ici les
écarts, on ne tranche pas.**

| Règle du moteur | Convention majoritaire | Position du moteur |
|-----------------|------------------------|--------------------|
| **Swing = fenêtre de 5 bougies (2+2), sur les extrêmes** | Le fractal « 5 bougies » (Williams) est répandu ; la variante 3 bougies existe aussi. Comparaison sur les mèches = majoritaire. | **Dans la convention majoritaire.** |
| **Aucun filtre d'amplitude de swing** | Partagé : beaucoup d'implémentations ICT ne filtrent pas l'amplitude ; certaines ajoutent un filtre ATR/points. | **Choix répandu**, mais **plusieurs conventions coexistent** (avec/sans filtre). |
| **BOS = franchissement en CLÔTURE** | **Débat non tranché.** L'école « clôture » (body close) et l'école « mèche » (wick/liquidity sweep) coexistent sans majorité claire. | **Écart assumé côté « clôture »** — c'est le levier le plus influent (section 7). |
| **Aucune marge de franchissement** (`>`/`<` strict) | Beaucoup n'appliquent aucune marge ; certains exigent un dépassement « significatif ». | **Convention répandue** (pas de marge). |
| **CHOCH = BOS de sens opposé à l'état courant** | Largement partagé : le CHOCH est le premier break contre-tendance. | **Dans la convention majoritaire.** |
| **Toute bougie de CHOCH est aussi un BOS_EVENT** | Variable selon les implémentations : certaines émettent l'un OU l'autre, d'autres considèrent aussi le CHOCH comme une cassure. | **Écart de convention possible** (double étiquetage), **choix de conception**. |
| **Garde anti-répétition** (`allow_bos` : pas de nouveau BOS sans nouveau swing) | Beaucoup ne ré-émettent pas tant que la structure n'a pas produit un nouveau point ; formulations variées. | **Esprit majoritaire**, implémentation particulière. |
| **Tendance (tuile) = 1re vs dernière clôture sur ~500 bougies** | **Hors SMC.** La tendance SMC se lit d'ordinaire via la **séquence des BOS/CHOCH** (HH/HL vs LH/LL), pas via un delta de clôtures sur fenêtre fixe. | **Écart net** : la tuile n'est pas une lecture SMC de la tendance ; c'est un indicateur de déplacement de prix. **Choix de conception à connaître.** |

> Ces écarts **peuvent être des choix légitimes**. Le tableau les rend visibles ;
> c'est au fondateur de juger lesquels il conserve.

---

# SECTION 4 — Cas ouvert DG-1 (24 → 28 juillet, XAUUSD H1)

**Question** : entre le BOS haussier du 24 juillet et le 28 juillet, le prix
descend jusqu'au bas de son range **sans aucun événement**. Dérive correcte, ou
**CHOCH baissier manqué** ?

**Réponse : dérive sans franchissement de creux — comportement CORRECT au regard
de la règle écrite. Aucun événement n'a été manqué.**

Trace horaire (script `scripts/audit/mt_d1/open_case.py`, graphique
`ECHANTILLON-DETECTION-2026-07-29/images/OPENCASE.png`) :

- Dernière cassure : **BOS haussier le 24/07 23:00 UTC** → l'état de tendance passe
  et reste **haussier** (`bos_signal = +1`) sur toute la fenêtre.
- Le **creux structurel protégé** (dernier creux-fractal) vaut **4022,09** puis
  **4012,26** (un creux plus bas enregistré le 27/07).
- Un **CHOCH baissier** exigerait une **clôture SOUS ce creux protégé** (règle §C).
- Or, sur toute la fenêtre 24→28/07, **la clôture la plus basse est 4043,43**
  (28/07 12:00 UTC) — **au-dessus** du creux protégé 4012,26. Le plus bas *touché*
  (4012,26, 27/07) n'est qu'une **mèche** ; la clôture n'est jamais passée dessous.
- **Aucune bougie ne remplit la condition de CHOCH baissier** → le moteur a
  **correctement** n'émis aucun événement.

Deux nuances factuelles pour le fondateur (sans jugement) :

1. La fenêtre **contient un week-end** (25-26/07) où les données du fournisseur
   sont **plates** (~4056, marché quasi fermé) — visible à l'œil sur le graphique.
2. Si la règle de franchissement était « **en mèche** » plutôt qu'« en clôture »,
   le résultat **serait différent** (la mèche du 27/07 a atteint 4012,26). C'est
   exactement le levier quantifié en **section 7**.

---

# SECTION 7 — Rapport de sensibilité *(hors gate — voir section 6)*

**Méthode.** Périmètre XAUUSD H1 & M15, **60 jours**. Référence = configuration
actuelle. Chaque paramètre varie **seul**, les autres restant à leur valeur
courante. `FRACTAL_WINDOW` est varié via le **vrai moteur** (paramètre public de
`SMCConfig`). Le **mode de franchissement** (mèche/clôture) et la **marge** ne sont
**pas des paramètres** (écrits en dur, cf. spec §E) : ils sont explorés via une
**ré-implémentation indépendante hors dépôt**, dont le mode par défaut reproduit le
moteur à l'identique (parité vérifiée). **Le dépôt n'est pas modifié.**

**Référence (60 j)** : H1 = **52** événements (38 BOS + 14 CHOCH) ; M15 = **188**
(136 BOS + 52 CHOCH).

### 7.1 `FRACTAL_WINDOW` — nombre de bougies de part et d'autre d'un swing *(seul vrai paramètre)*

| Réglage | H1 BOS | H1 CHOCH | H1 Δ total | M15 BOS | M15 CHOCH | M15 Δ total |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| plus permissif `N=1` (3 bougies) | 38 | 14 | **+0** | 136 | 52 | **+0** |
| **actuel `N=2` (5 bougies)** | 38 | 14 | 0 (réf) | 136 | 52 | 0 (réf) |
| plus strict `N=3` (7 bougies) | 30 | 12 | **−10** | 85 | 36 | **−67** |
| plus strict `N=4` (9 bougies) | 24 | 9 | **−19** | 80 | 36 | **−72** |

> **Conclusion `FRACTAL_WINDOW`** : influence **asymétrique**. Le rendre **plus
> strict** réduit nettement le nombre d'événements ; le rendre **plus permissif
> (`N=1`) n'ajoute AUCUN événement** (identique à l'actuel). Ce paramètre **n'est
> donc pas** ce qui limiterait le nombre d'événements : l'assouplir n'en produit
> pas davantage.

### 7.2 Mode de franchissement — mèche vs clôture *(écrit en dur, non paramétrable)*

| Réglage | H1 BOS | H1 CHOCH | H1 Δ total | M15 BOS | M15 CHOCH | M15 Δ total |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| **actuel : clôture** | 38 | 14 | 0 (réf) | 136 | 52 | 0 (réf) |
| plus permissif : mèche | 91 | 30 | **+69** | 321 | 100 | **+233** |

> **Conclusion mode de franchissement** : c'est **de loin le levier le plus
> influent**. Passer de « clôture » à « mèche » fait ~**×2,3** en H1 (52→121) et
> ~**×2,2** en M15 (188→421). Mais ce n'est **pas un paramètre** : ce comportement
> est écrit en dur (`closes[i]`, spec §B). Le faire varier suppose de toucher au
> moteur.

### 7.3 Marge de franchissement — tampon ATR avant validation *(écrit en dur = 0)*

| Réglage | H1 BOS | H1 CHOCH | H1 Δ total | M15 BOS | M15 CHOCH | M15 Δ total |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| **actuel : 0 ATR (aucune marge)** | 38 | 14 | 0 (réf) | 136 | 52 | 0 (réf) |
| plus strict : 0,10 ATR | 38 | 14 | −0 | 119 | 46 | **−23** |
| plus strict : 0,25 ATR | 33 | 14 | **−5** | 104 | 42 | **−42** |
| plus strict : 0,50 ATR | 27 | 12 | **−13** | 90 | 40 | **−58** |

> **Conclusion marge** : la configuration actuelle est **déjà au plus permissif**
> (marge nulle). Ajouter une marge **ne peut que réduire** le nombre d'événements.
> Ce n'est donc **pas** une cause du faible nombre d'événements.

### 7.4 Condition de tendance préalable & garde anti-répétition *(écrites en dur)*

La « tendance préalable » (état `bos_signal`) et la garde `allow_bos` ne sont pas
des paramètres numériques ; on ne peut pas les « faire varier » sur trois valeurs.
On **mesure leur poids** via les franchissements qu'elles écartent (60 j) :

| Motif d'écart d'un franchissement de swing | H1 | M15 |
|--------------------------------------------|:---:|:---:|
| **Niveau déjà franchi** (garde anti-répétition `allow_bos`) | **229** | **521** |
| **Mèche seule** (clôture restée en-deçà — la règle « clôture ») | 95 | 233 |
| **Total franchissements sans événement** | **324** | **754** |

> **Conclusion tendance/garde** : la garde anti-répétition (`allow_bos`) est le
> **premier** motif de « franchissement sans événement » (229/324 en H1,
> 521/754 en M15). C'est une **mécanique de conception** (ne pas ré-émettre sur un
> niveau déjà cassé sans nouveau swing), **pas** un réglage. Elle explique une part
> importante des non-événements — que le fondateur peut juger sur l'échantillon 3B.

### 7.5 Synthèse — quel paramètre est « responsable » du faible nombre d'événements ?

- **Aucun paramètre de `SMCConfig`** n'accroît le nombre d'événements en le rendant
  plus permissif : `FRACTAL_WINDOW=1` n'ajoute rien, et il n'existe pas d'autre
  levier paramétrable côté émission.
- Le **seul levier qui augmenterait fortement** le nombre d'événements — le
  **franchissement en mèche** — **n'est pas un paramètre** ; il est écrit en dur.
- Les deux plus gros filtres factuels sont donc **(a) la règle « clôture »** et
  **(b) la garde anti-répétition** — **par conception, pas par réglage**.
- **Rappel** : indépendamment de tout cela, la section 6 a montré que ce que le
  fondateur *voit* est surtout limité par l'**affichage** (fenêtre + cap 8), pas
  par ces règles de détection.

---

# SECTION 8 — Échantillon des événements supplémentaires

Le levier le plus influent (section 7) est le **franchissement en mèche**. Nous
produisons donc **15 exemples** d'événements qui **seraient émis** avec un
franchissement « en mèche » et **ne le sont pas** aujourd'hui (règle « clôture »).

- Fichiers : `ECHANTILLON-DETECTION-2026-07-29/images/EXTRA-01…15.png`.
- Chacun montre la bougie où une **mèche** a percé le niveau structurel sans que la
  **clôture** ne le confirme.
- Ils sont **ajoutés au fichier d'annotation** (`ANNOTATION.csv`, section
  `8-evenement-ajoute-si-meche`, et visualiseur `README.md`).

**Ce que le fondateur doit juger** : ces événements « manquants » sont-ils de
**vrais** signaux (le produit rate des cassures réelles) ou du **bruit** (des
mèches de balayage qu'il est sain d'écarter) ? Le rapport **ne tranche pas**.

---

# Points appelant une décision du fondateur *(sans recommandation)*

Chaque point est un **choix**, pas un défaut. Aucune direction n'est suggérée.

1. **Fenêtre d'affichage live (500 bougies).** Elle vaut ~21 jours en H1 mais
   ~5 jours en M15, et **n'est pas étiquetée**. Décision : faut-il l'afficher /
   l'adapter par unité de temps ? *(cf. reste DG-1 P5)*
2. **Plafond `MAX_STRUCTURE_EVENTS = 8` par type**, invisible au client. Décision :
   plafond visible ? plafond plus haut ? plafond levé ?
3. **Store profond non câblé à l'UI.** Un journal complet existe déjà (non
   plafonné) mais n'est pas montré (décision DG-1 (b), gelée faute de scheduler).
   Décision : le câbler (avec fenêtre étiquetée) ?
4. **Franchissement « clôture » vs « mèche ».** Levier le plus influent (×~2,3).
   Choix de conception, non paramétrable aujourd'hui.
5. **Marge de franchissement nulle.** Actuellement au plus permissif ; toute marge
   réduit les événements. Choix à assumer tel quel ou à exposer.
6. **Double étiquetage CHOCH+BOS sur la même bougie.** Conforme au code, mais
   diverge de certaines conventions. Choix d'affichage à valider.
7. **Tuile « Tendance » = delta de clôtures sur 500 bougies**, indépendante des
   BOS/CHOCH. Peut diverger légitimement du journal. Décision : garder cette
   définition, ou aligner la tuile sur la séquence structurelle SMC ?
8. **Garde anti-répétition (`allow_bos`).** Premier motif de « franchissement sans
   événement ». Choix de conception à confirmer sur l'échantillon 3B.

> Ces huit points sont **posés, pas arbitrés**. La décision appartient au fondateur.

---

## Annexe — reproductibilité

```
scripts/audit/mt_d1/fetch_data.py        # recharge H1/M15 depuis Twelve Data → _cache/
scripts/audit/mt_d1/harness.py           # exécute le VRAI moteur, extrait les événements
scripts/audit/mt_d1/variant_detect.py    # copie instrumentée (parité prouvée) — variantes hors dépôt
scripts/audit/mt_d1/run_audit.py         # sections 6-7 → results.json
scripts/audit/mt_d1/open_case.py         # section 4 (trace horaire)
scripts/audit/mt_d1/generate_samples.py  # 66 graphiques + manifest.json
scripts/audit/mt_d1/build_annotation.py  # ANNOTATION.csv + README.md
```

Chiffres bruts : `docs/audits/ECHANTILLON-DETECTION-2026-07-29/results.json`.

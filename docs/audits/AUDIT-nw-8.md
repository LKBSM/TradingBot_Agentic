# AUDIT NW-8 — Rendre les chiffres compréhensibles (vulgarisation)

Branche : `feat/nw-8-vulgarisation` (worktree dédié, depuis `origin/main` @ `28debc2`).
Statut : **DIAGNOSTIC — aucun code écrit. En attente du GO.**
Périmètre : page de publication + calendrier UNIQUEMENT.

> Rappel : la courbe et les 4 questions **fonctionnent** (NW-7). NW-8 ne les refait pas,
> il les rend lisibles. Note : **`us_cpi` affiche DÉJÀ la variation annuelle** (NW-7,
> `series_kind=yoy_percent`) ; les autres publications affichent encore leur **niveau brut**.

---

## A. LES UNITÉS PAR PUBLICATION (nature réelle)

| Publication | `value_unit` catalogue | Nature |
|---|---|---|
| us_cpi | % de variation annuelle | **DÉJÀ une variation** (an) — NW-7 |
| us_cpi_core | indice (1982-84 = 100) | **NIVEAU d'indice** |
| us_ppi | indice (nov. 2009 = 100) | **NIVEAU d'indice** |
| us_pce | indice (2017 = 100) | **NIVEAU d'indice** |
| us_employment_situation (NFP) | milliers d'emplois | **NIVEAU / compte** (emploi total) |
| us_jolts | milliers de postes | **NIVEAU / compte** (postes ouverts) |
| us_retail_sales | millions de dollars | **NIVEAU / montant** |
| us_durable_goods | millions de dollars | **NIVEAU / montant** |
| us_housing_starts | milliers d'unités (rythme annualisé) | **NIVEAU / compte** |
| us_gdp | % (variation du PIB réel) | **DÉJÀ une variation** (trimestre) |
| ea_hicp_flash | % (variation annuelle) | **DÉJÀ une variation** (an) |
| ea_gdp_flash | % (variation trimestrielle) | **DÉJÀ une variation** (trimestre) |
| ea_unemployment | % de la population active | **TAUX** (la valeur EST l'info) |
| ea_ecb_rate | taux directeur (% par an) | **TAUX** |
| us_fomc_rate | fourchette du taux directeur (%) | **TAUX** |
| us_fomc_minutes | *(absente)* | **Pas de valeur numérique** (texte) |
| us_fomc_dotplot | *(absente)* | **Pas de valeur numérique** (projections) |

**Unités absentes/à corriger** : `us_fomc_minutes` et `us_fomc_dotplot` n'ont pas de `value_unit`
(normal : pas de chiffre unique). Toutes les autres sont renseignées et exactes.

---

## B. LES VARIATIONS SONT-ELLES PUBLIÉES PAR L'ORGANISME ? (le point central)

**Réponse courte : oui, dans la grande majorité des cas — sans requête supplémentaire.**

### B.1 — Variation PUBLIÉE par l'organisme (0 calcul produit, à privilégier)

| Publication | Variation dispo | Comment | Coût requêtes |
|---|---|---|---|
| us_cpi, us_cpi_core, us_ppi | **mois + an** | BLS API v2 flag **`calculations`** → `pct_changes["1"]` (mois) & `pct_changes["12"]` (an), sur le MÊME appel de série | **0 sup.** (un flag) |
| us_employment_situation (NFP) | **variation mensuelle** (emplois créés) | BLS `calculations` → `net_changes["1"]` (∆ du compte) | 0 sup. |
| us_jolts | **variation mensuelle** | BLS `calculations` → `net_changes["1"]` | 0 sup. |
| us_gdp | **variation trimestrielle** | la série BEA est DÉJÀ la % change (Table 1.1.1) — la valeur EST la variation | 0 sup. |
| ea_hicp_flash | **variation annuelle** | la série Eurostat est DÉJÀ le taux annuel (`RCH_A`) — la valeur EST la variation | 0 sup. |
| ea_gdp_flash | **variation trimestrielle** | série Eurostat DÉJÀ % change (`CLV_PCH_PRE`) | 0 sup. |

Les valeurs `calculations` de BLS **correspondent aux variations que BLS publie** dans ses
communiqués (« l'IPC a augmenté de 0,3 % sur le mois ») → attribution honnête « **publiée par BLS** ».
Idem BEA/Eurostat où la série renvoie déjà la variation → « **publiée par l'organisme** ».

### B.2 — Variation CALCULABLE depuis deux niveaux publiés (calcul produit → à attribuer)

| Publication | Variation | Méthode | Exactitude |
|---|---|---|---|
| us_pce | mois/an % | soit une **table BEA « % change »** dédiée (T2.8.7) = **+1 requête** et publiée ; soit calcul `(niveau_t / niveau_{t−1} − 1)` depuis l'indice | calcul exact sur la série CVS **hors rebasement** (rebasement PCE ~tous les 5 ans, hors fenêtre 12 mois) |
| us_retail_sales, us_durable_goods, us_housing_starts | mois % | calcul `(niveau_t / niveau_{t−1} − 1)` depuis deux niveaux **CVS** publiés | exact vs la % change CVS publiée par Census (mêmes niveaux CVS) |

**Recommandation** : pour PCE, préférer la **table BEA % change** (publiée) à un calcul ; pour Census,
le calcul depuis deux niveaux CVS est exact et sans requête sup — attribué « **calculée à partir de
deux niveaux publiés** ». **Piège à respecter** : toujours la série **corrigée des variations
saisonnières (CVS)**, jamais mélanger brut et CVS ; un rebasement d'indice invalide un calcul qui
enjambe la nouvelle base (rare, à ignorer si absent de la fenêtre affichée).

### B.3 — Pas de variation (ce sont des TAUX — la valeur EST l'information)

`us_fomc_rate`, `ea_unemployment`, `ea_ecb_rate` : la valeur est un **taux**, pas un niveau
d'indice. Pas de variation à afficher — on garde le taux **avec son unité + la phrase
d'explication** (§2.B). (Une variation d'un taux = points de base ; hors périmètre, à ne pas inventer.)

### B.4 — Pas de valeur numérique (pas de courbe)

`us_fomc_minutes`, `us_fomc_dotplot` : releases textuelles / projections → **pas de courbe, pas de
variation**. Seule la fiche + les 4 questions s'appliquent.

---

## C. L'EXISTANT (où vit le code)

- **Rendu de la courbe** : `CurveCard` dans `webapp/components/calendar/CalendarEventDetail.tsx`
  (≈ l.205-382). Trace `ev.value_series` ; chaque point est `{period, value}` et `value` est la
  seule valeur tracée/étiquetée (l.234, 296-297).
- **Modèle de données** : `SeriesPoint(period, value)` (`values/base_value.py:33`) →
  `CalendarSeriesPoint(period, value)` (`calendar_schema.py:65`). **Ne porte qu'UNE valeur** →
  **devra être étendu** pour porter niveau + variation(s) + leur attribution.
- **Libellés d'unité** : `value_unit` (catalogue → schéma → en-tête `.meta` + contexte courbe).
- **Texte des 4 questions** : i18n `calendar.pub.questions.*` dans `webapp/messages/*.json`
  (fr+en natifs + 7 locales), rendu par `QuestionsSection` (CalendarEventDetail.tsx).
- **Calendrier** : `CalendarMonthView.tsx` + `CalendarPreview.tsx` référencent `actual`/`value_unit`
  → surfaces à aligner (§4 cohérence).

---

## SYNTHÈSE POUR LE STOP

**Ce que je livre au STOP** (demandé) :
1. **Tableau des unités** (§A) — 15 publications avec valeur, 2 sans (FOMC minutes/dotplot).
2. **Variation publiée / calculable / indisponible** (§B) :
   - **PUBLIÉE, 0 requête sup.** : IPC, IPC sous-jacent, IPP, NFP, JOLTS (BLS `calculations`), PIB US
     (BEA % change), IPCH, PIB flash zone euro (Eurostat % change). → **8 publications.**
   - **CALCULABLE** (ou table % dédiée) : PCE (BEA), ventes de détail, biens durables, mises en
     chantier (Census). → **4 publications** (calcul CVS exact, à attribuer « calculée »).
   - **SANS OBJET (taux)** : FOMC taux, chômage EA, taux BCE. → **3 publications** (la valeur reste seule).
   - **SANS VALEUR** : FOMC minutes, dot plot. → **2 publications** (pas de courbe).

**Décision qui t'appartient (avant tout code)** :
- Pour les 4 « calculables » (PCE + 3 Census) : **calculer** depuis deux niveaux CVS (0 requête,
  attribué « calculée »), ou **récupérer** la table % publiée quand elle existe (PCE : +1 requête,
  attribué « publiée ») ? Ma reco : **PCE = table publiée** ; **Census = calcul CVS** (pas de table %
  simple côté API). Mais je ne calcule rien sans ton GO.
- Portée : livre-t-on la variation pour **toutes** les publications éligibles d'un coup, ou d'abord
  les **8 « publiées »** (le plus sûr) puis les 4 « calculées » ?

**Travail à prévoir (au GO)** : extension du modèle (`SeriesPoint` → niveau + variation mois/an +
drapeau publiée/calculée) ; `CurveCard` trace la variation (axe zéro si négatif possible), niveau en
survol/second plan ; en-tête : variation en évidence + niveau plus petit ; **17 phrases d'explication**
rédigées (une par publication, fr+en) ; attribution de la variation ; relecture vulgarisation des 4
questions ; cohérence calendrier. Aucune valeur inventée ; point à venir toujours vide.

---

# PARTIE 2 — LIVRÉ (Batch 1 : les 8 publications à variation publiée)

Décisions fondateur : **PCE = table publiée, Census = calcul CVS** (différés) ; **les 8 « publiées »
d'abord**. Publications couvertes : **IPC, IPC sous-jacent, IPP** (index_change), **NFP, JOLTS**
(count_change), **PIB US, IPCH, PIB flash zone euro** (published_change).

## Modèle (backend)
- Catalogue : `series_kind` posé sur les 8 (`index_change` / `count_change` / `published_change`) ;
  **`value_unit` = unité du NIVEAU** (l'IPC repasse de « % » à « indice (1982-84 = 100) »).
- `SeriesPoint` / `CalendarSeriesPoint` étendus : `value` (variation tracée) + `level` (niveau brut)
  + `change_mom` (variation mensuelle secondaire). `CalendarEvent` : `variation_kind` +
  `variation_published`. Threadé dans `calendar_service.get_event`.

## Variations — toutes PUBLIÉES par l'organisme (aucune recalculée)
- **BLS** (`bls_values.fetch_series`) via le flag `calculations`, **0 requête sup.** :
  · index_change → `value`=`pct_changes["12"]` (an), `level`=indice, `change_mom`=`pct_changes["1"]` (mois) ;
  · count_change → `value`=`net_changes["1"]` (variation mensuelle absolue), `level`=total.
  Un mois sans la variation-clé n'a pas de point (jamais fabriqué).
- **BEA / Eurostat** (published_change) : la série renvoie déjà le %, `value`=%, pas de niveau.

## Rendu (frontend)
- `CurveCard` : la courbe **trace la variation** (`value`) ; **axe zéro visible** si négatif ;
  **niveau brut au survol** (`<title>` sur les points) ; **point à venir toujours vide** (règle intacte).
- En-tête **variation en évidence** + **niveau en second plan** (`.pub-var-level`), formaté lisible
  (séparateur de milliers, signe) : `varIndex` (mois · an), `varCount` (variation mensuelle + total),
  `varPublished` (le %). Publications **hors Batch-1** (vk=null) : rendu inchangé (niveau brut).
- **Attribution** obligatoire (`.pub-curve-attrib`) : « Variation publiée par {organisme} (série {X}) »
  (gabarit `attribCalculated` prêt pour les futures calculées).
- **Phrase d'explication** (`.pub-curve-explain`) : rendue **seulement** pour une publication whitelistée
  (`CURVE_EXPLAIN`, les 8) ; jamais de texte générique.

## Vocabulaire & i18n
- 8 phrases + libellés variation/attribution **fr + en natifs** ; 7 autres locales : mêmes clés
  (**parité stricte OK**), valeurs EN en repli (dette de valeur documentée, cf. `home`/`regimePanel`).
- Vocabulaire interdit absent (garde `calendar-copy-honesty` verte : « means » verbe reformulé en
  « indicates »). Aucune « médiane/moyenne/bougie ».

## Les 4 questions (§3) & calendrier (§4)
- Les 4 questions respectent déjà les règles (NW-7 : pas de bougie/médiane, mitigation & zone
  expliquées, dénominateurs portés). **Relecture faite ; aucune correction requise ce batch.**
- **Calendrier** (`CalendarMonthView`, `CalendarPreview`) **n'affiche AUCUNE valeur numérique**
  (nom + heure + lien) → cohérence §4 trivialement respectée, rien à aligner.

## Vérification
- **pytest** 216 verts (calendar/measures/value) — 2 échecs pré-existants hors périmètre
  (`test_tr1_structural_trend`, `test_enricher_flags_revision_across_cycles`).
- **vitest 870/870** (dont 3 tests NW-8 : index/count/published) · **parité 9 locales** verte ·
  **tsc vert** · **build vert** · **Playwright nw8 12/12** (variation publiée / sans variation /
  sans phrase, 1280×800 & 390×844 ×2 projets).

## Restes (différés, signalés)
- **PCE** (table BEA % publiée) + **Census** (retail/durables/mises en chantier, calcul CVS) — Batch 2.
- **Taux** (FOMC, chômage EA, BCE) : valeur = taux, à doter d'une **phrase d'explication** (Batch 2).
- **FOMC minutes / dot plot** : pas de valeur numérique.
- 9 phrases d'explication restantes (les 9 publications hors Batch-1).

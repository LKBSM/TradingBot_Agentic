# AUDIT TR-1 — Tendance dérivée de la structure SMC — PHASE 1 (DIAGNOSTIC)

> **STATUT : diagnostic LECTURE SEULE, avant décision et avant tout code.**
> Aucune règle de détection (BOS/CHOCH/swing) n'a été modifiée. `_derive_trend`
> n'a pas été touchée. Les chiffres ci-dessous sont produits par une évaluation
> **hors moteur** qui LIT les sorties du vrai moteur (`UP_FRACTAL`, `DOWN_FRACTAL`,
> `BOS_EVENT`, `CHOCH_SIGNAL`) sur les bougies réelles.
>
> **Données** : XAUUSD, cache MT-D1 rechargé Twelve Data le 2026-07-29 (UTC).
> H1 = 2 464 bougies, M15 = 6 976 bougies. Fenêtre d'évaluation = **60 derniers
> jours** (H1 : 1 441 bougies ; M15 : 5 761 bougies), avec ≥500 bougies d'historique
> en amont pour la fenêtre glissante.
> Repro : `scripts/audit/tr_1/diag.py` + `episodes.py` (réutilisent le harness MT-D1).
>
> **Les sections 1.A / 1.C / 1.D ne tranchent pas** — décisions de produit.

---

## Rappel du problème

Aujourd'hui la tuile Tendance = `_derive_trend` (`market_reading_mappers.py:1329-1342`) :
première vs dernière clôture sur ~500 bougies, garde-fou de range au seuil 0,3.
Elle n'utilise **ni swings, ni BOS, ni CHOCH**. C'est une 2ᵉ source de vérité,
contraire au principe « le moteur est la source de vérité, au singulier ».
Objectif TR-1 : la tendance devient une **lecture dérivée** de la structure détectée.

Deux définitions structurelles candidates :

- **(a) Direction du dernier événement (BOS/CHOCH) non contredit.** Haussière tant
  qu'un CHOCH baissier n'est pas survenu. C'est exactement l'état `BOS_SIGNAL`
  propagé par le moteur.
- **(b) Séquence des swings.** Sommets **et** creux successivement plus hauts =
  haussier ; successivement plus bas = baissier ; sinon (HH+LL ou LH+HL) = indéterminé.

---

## 1.A — Comparaison chiffrée des deux définitions (60 j, XAUUSD)

Évaluation **bougie par bougie** : à chaque barre, chaque définition est calculée
sur les 500 bougies précédentes (miroir de la fenêtre live), puis on compare.

| Mesure | H1 | M15 |
|---|---|---|
| Barres évaluées (60 j) | 1 441 | 5 761 |
| **def(a)** déterminée | **100,0 %** | **100,0 %** |
| **def(a)** indéterminée | **0,0 %** | **0,0 %** |
| **def(b)** déterminée | 61,6 % | 63,2 % |
| **def(b)** indéterminée (swings mixtes) | **38,4 %** | **36,8 %** |
| **Accord quand les DEUX tranchent** | **61,0 %** | **55,5 %** |
| Désaccord (les deux tranchent, sens opposé) | 39,0 % | 44,5 % |
| **Bascules de tendance sur 60 j — def(a)** | **14** | **52** |
| **Bascules de tendance sur 60 j — def(b)** | **65** | **235** |

**Lecture.**
- **def(a) est stable** : ~14 bascules en 60 j (H1) ≈ un changement d'avis tous les
  ~4 jours. **def(b) bascule ~4,5× plus** (65 en H1, 235 en M15) — une tuile qui
  change d'avis plusieurs fois par jour en M15.
- Même **quand les deux tranchent, elles se contredisent ~40 % du temps** (H1) et
  ~45 % (M15). Ce ne sont pas deux formulations du même fait : ce sont deux mesures
  différentes.
- **def(a) n'est jamais indéterminée** sur H1/M15 (cf. 1.B). **def(b) l'est ~37 %**
  du temps.

### Trois épisodes de divergence datés (les deux tranchent, sens opposé)

Tous se logent sur des **fenêtres de week-end / consolidation** — signal important
(voir la mise en garde plus bas). H1 :

1. **05→06 juil. (21 barres, 20 h).** def(a)=**haussier** (dernier événement : BOS
   haussier le 03/07 10:00). def(b)=**baissier** : les 2 derniers sommets valent
   4175,09 **à l'identique** et les 2 derniers creux 4174,84 à l'identique (plateau
   week-end) → la comparaison de swings dégénère.
2. **26→27 juil. (18 barres, 17 h) — c'est le CAS OUVERT DG-1.** def(a)=**haussier**
   (BOS haussier du 24/07 23:00, non contredit). def(b)=**baissier** sur le pullback
   (sommets et creux du 26/07 plus bas). La tuile actuelle disait aussi baissier ;
   def(a) réconcilierait la tuile avec le journal, def(b) non.
3. **19→20 juil. (17 barres, 16 h).** def(a)=**haussier** (BOS haussier du 18/07
   01:00). def(b)=**baissier** (plateau ~4010 le 19/07, sommets/creux ex-æquo).

En M15 les mêmes épisodes durent 73–75 barres (18 h) — mêmes causes, amplifiées.

> **⚠️ Mise en garde de données.** La majorité des divergences se logent sur les
> **plateaux de week-end** où le fournisseur renvoie des bougies quasi plates. Ces
> plateaux produisent des **fractals ex-æquo** (sommets/creux à valeur identique),
> et la lecture « 2 derniers swings » de def(b) y devient **bruitée**. C'est un
> défaut inhérent à def(b) sur données basse résolution — à connaître avant de la
> choisir. def(a), qui ne lit que le dernier événement, y est **insensible**.

---

## 1.B — Fréquence de l'état « indéterminé » (question d'honnêteté centrale)

L'indéterminé = **aucun ancrage structurel dans la fenêtre**. Il ne se comporte pas
pareil selon la définition :

| | H1 | M15 | D1 / W1 |
|---|---|---|---|
| **def(a)** indéterminée (aucun BOS/CHOCH dans 500 bougies) | **0,0 %** | **0,0 %** | non mesurable ici* |
| **def(b)** indéterminée (pas de séquence de swings nette) | **38,4 %** | **36,8 %** | non mesurable ici* |

Écart au dernier événement (def(a)) : médiane **32 barres** (H1) / **48 barres**
(M15) ; p90 **99 / 212** ; **max 192 (H1), 443 (M15)** — **toujours < 500**. D'où
0 % d'indéterminé pour def(a) : sur ces unités il y a **toujours** un événement dans
la fenêtre.

**Conséquence produit — c'est décisif :**
- Si on retient **def(a)** : l'indéterminé est **quasi inexistant** sur M15/H1. La
  tuile aura presque toujours une direction. UX peu changée. (L'indéterminé
  n'apparaîtrait qu'au tout début d'un historique, ou éventuellement sur D1/W1.)
- Si on retient **def(b)** : la tuile dira **« indéterminé » ~1 fois sur 3**.
  Honnête, mais **change complètement l'expérience** — à voir avant de s'engager,
  exactement comme la mission le prévoit.

*(\*) D1/W1 : le cache actuel (102 j en H1) ne donne pas 60 j + 500 bougies
d'historique en D1/W1. Mesurer l'indéterminé sur ces unités exige un backfill
profond (cf. LB-1). À faire si la décision penche vers def(b) et qu'on veut couvrir
les 6 unités.*

---

## 1.C — L'état « ranging »

- **Origine du seuil 0,3** : littéral en dur dans `_derive_trend` (`:1340`,
  `pct_move < rng_pct * 0.3`). **Aucun commentaire, aucune justification** dans le
  code ni ailleurs. Constante magique non documentée.
- **Redondance avec la tuile PHASE : CONFIRMÉE.** `_derive_market_phase`
  (`:1412-1417`) fait `if trend == "ranging": return "ranging"`. La Phase **recopie**
  l'état ranging issu de la Tendance. Pire : la Phase est **entièrement** fonction de
  (trend, volatilité) — elle n'a **aucune source indépendante**
  (`bullish/bearish`→`trend`/`expansion`, `ranging`→`ranging`, `neutral`→
  `accumulation`). L'information « consolidation » vit donc **déjà** dans la Phase.
- **Option (à trancher, pas tranchée)** : (i) supprimer « ranging » de la Tendance et
  le laisser à la Phase ; (ii) le remplacer par un équivalent structurel défini
  (p. ex. « swings mixtes » = l'indéterminé de def(b)) ; (iii) statu quo. À noter :
  une définition structurelle ne **produit pas** « ranging » nativement — c'est un
  artefact du calcul par clôtures actuel.

---

## 1.D — Fenêtre

- **def(a) n'a pas besoin d'une fenêtre fixe.** Elle « remonte au dernier événement ».
  Comme l'écart max au dernier événement est **< 500 bougies** partout (192 H1,
  443 M15), une fenêtre de 500 et un « depuis le dernier événement » donnent le
  **même** résultat ici — mais « depuis le dernier événement » est **plus honnête et
  auto-descriptif** (on nomme la date de l'événement, pas une fenêtre arbitraire).
- **def(b) a besoin d'assez de swings** (≥2 sommets + 2 creux). En pratique une
  profondeur d'événements, pas une durée calendaire.
- **Si une fenêtre subsiste, elle doit être NOMMÉE en durée calendaire** à l'écran
  (rappel TF-1 : 500 bougies = ~5 j en M15, ~21 j en H1, ~2 ans en D1).

---

## 1.E — Inventaire des consommateurs (avec fichier:ligne)

> Point de vérité : le passage à une définition structurelle **touche tout ce qui
> suit**. L'état « indéterminé » doit être représentable partout (colonne « repr. ? »).

### Producteur (à remplacer)
| Fichier:ligne | Rôle |
|---|---|
| `market_reading_mappers.py:1329-1342` `_derive_trend` | **LE calcul à remplacer** (clôtures→structure). |
| `market_reading_mappers.py:1420-1425` `_derive_bias_from_candles` | Dérive le biais par unité via `_derive_trend` → alimente `mtf_confluence`. |
| `market_reading_mappers.py:1428-1458` `candles_to_regime` | Assemble le régime (appelle `_derive_trend`). |
| `market_reading_mappers.py:1412-1417` `_derive_market_phase` | **Consomme** `trend` ; le `else` attraperait « indéterminé » → à gérer. |

### Backend — consommateurs
| Fichier:ligne | Rôle | Indéterminé repr. ? |
|---|---|---|
| `market_reading_schema.py:37` `TrendValue = Literal[...]` | Type. **Ajouter « indeterminate ».** | à créer |
| `market_reading_schema.py:244` `MarketReadingRegime.trend` | Champ modèle. | via type |
| `market_reading_schema.py:39` `MarketPhase` | Phase (voir 1.C). | indirect |
| `conditions_scanner.py:43-50` palette `mtf_aligned` | **Condition scanner « 3 TF alignés »** (libellé/desc). | **à mettre à jour (1.D scanner)** |
| `conditions_scanner.py:303-361` `_eval_mtf_aligned` | **Évalue l'alignement** en lisant chaque `regime.trend`. Convertit en axe up/down/flat. | **doit compter « indéterminé » à part** |
| `conditions_scanner.py:305` `_TREND_ADJ` | Libellés fr des valeurs de trend. | à étendre |
| `conditions_scanner.py:483` `trend_is` | Condition « tendance actuelle = X ». | à étendre |
| `api/routes/conditions_scan.py:225-236` | Construit `trends_by_instrument[instrument][tf]=trend` puis appelle l'évaluateur. | à propager |
| `narrated_reading.py:186-236` `ReadingFacts.trend`, `_mtf_relation` | Narration + relation multi-unités. | à gérer |
| `narrated_reading.py:337-400,490-613` | Faits/prompt LLM (Tendance : …). | à gérer |
| `haiku_description_engine.py:178` | `facts.trend` → prompt Haiku. | à gérer |
| `chatbot/signal_summary_provider.py:87` | `"trend": reading.regime.trend` → **faits M.I.A Agent**. | à gérer |
| `market_reading_mappers.py:1504,1542-1551` | Tags `trend_*` + description template fr. | à étendre |

### API / persistance
| Fichier:ligne | Rôle |
|---|---|
| `api/routes/market_reading.py:34` `GET /api/market-reading` | Expose `regime.trend` (via response_model). |
| `storage/market_readings_store.py:113-190` | **Persistance** : `payload_json` (JSON complet, trend inclus) en SQLite `market_readings`. `save_reading` / `get_latest_reading`. |
| `market_reading_assembler.py` (persist) | Écrit le reading assemblé (trend dans le régime). |

### Frontend — consommateurs
| Fichier:ligne | Rôle | Indéterminé repr. ? |
|---|---|---|
| `webapp/types/market-reading.ts:32` `TrendValue = 'bullish'\|'bearish'\|'neutral'\|'ranging'` | Type TS. **Ajouter l'état.** | à créer |
| `webapp/types/market-reading.ts:227-235,384-390` | `MarketReadingRegime`, helpers `isBullishTrend`… | à étendre |
| `webapp/components/app/RegimeCard.tsx:360-365` | **Tuile Tendance** (valeur/sous-ligne). | à gérer |
| `webapp/components/app/RegimeCard.tsx:785-804` | **Texte Concept 3 temps** de la tuile. | **à réécrire (1.E traçabilité)** |
| `webapp/components/app/RegimeCard.tsx:384-400,859-885` | **Tuile Alignement multi-unités** (compte alignés/total, flèches). | **dénominateur ajusté** |
| `webapp/lib/market-reading/formatters.ts:155-166` `TREND_LABEL`/`formatTrend` | Libellés + ton. | à étendre |
| `webapp/lib/market-reading/mtf-trend.ts:37-51,105-139` | Glyphes + `classifyMtfAlignment`. | à étendre |
| `webapp/components/market-reading/sections/RegimeSection.tsx:135-183` | Badges alignement + encart désaccord. | à gérer |
| `webapp/lib/conditions/palette.ts:23-37,149-154` | **Conditions scanner** `mtf_aligned` / `trend_is` + `TREND_OPTIONS`. | à mettre à jour |
| `webapp/components/scanner/ConditionsBuilder.tsx:50-145` | UI de sélection de tendance. | à étendre |
| `webapp/lib/conditions/types.ts:32` `TrendChoice` | Type condition. | à étendre |
| `webapp/messages/*.json` | **i18n** : `reading.labels.trend_*`, `reading.concept.trend.*`, `regimePanel.*`, `scanner.palette.mtf_aligned_*`/`trend_is_*`, `tags.trend_*` (9 locales). | **clés à créer pour l'indéterminé** |

**Point le plus sensible (mission 1.E) : le SCANNER.** `_eval_mtf_aligned`
(`conditions_scanner.py:308-361`) lit aujourd'hui `regime.trend` = accord entre
**déplacements de clôtures**. Le client qui coche « 3 TF alignés » croit cocher un
**alignement de structure**. Le sens de la condition **change** avec TR-1 : ce n'est
pas un détail technique, c'est **une fonctionnalité différente sous le même nom**.

---

## 1.F — Impact sur les données persistées

- `market_readings_store` stocke le **JSON complet** du reading ; les lectures
  déjà en base portent une tendance **ancienne définition** (clôtures).
- `get_latest_reading` les relit telles quelles → un client pourrait voir une
  ancienne tuile « clôtures » et une nouvelle tuile « structure » **mélangées**
  sans le savoir.
- **Décision (à remonter)** : recalculer / marquer (versionner la définition dans le
  payload) / purger / laisser expirer. Contrainte dure : **jamais deux définitions
  mélangées à l'écran sans que le client puisse le savoir.**

---

## Ce que ce diagnostic établit (sans trancher)

1. **def(a) = tuile stable et honnête**, réconcilie Tendance ↔ journal (dont le cas
   DG-1 du 26/07), indéterminé quasi nul sur M15/H1.
2. **def(b) = plus « SMC canonique » mais instable** (×4,5 de bascules), indéterminée
   ~37 % du temps, et **fragile aux plateaux de week-end** (fractals ex-æquo).
3. **« ranging » est redondant** avec la tuile Phase et repose sur un seuil 0,3 non
   documenté.
4. **Le scanner** est le seul endroit où le sens d'une fonctionnalité **demandée par
   l'utilisateur** change.
5. La qualité de la tendance **héritera** de la détection BOS/CHOCH — non validée
   (échantillon MT-D1 non annoté). TR-1 **expose** cette dépendance au lieu de la
   masquer.

## Décisions attendues avant GO (aucune prise ici)
- **D1.** def(a) ou def(b) ? *(ou hybride : def(a) pour la direction + un drapeau
  « swings mixtes » informatif)*
- **D2.** État « ranging » : supprimer / remplacer / laisser à la Phase ?
- **D3.** Fenêtre : « depuis le dernier événement » (nommé) ou fenêtre fixe nommée ?
- **D4.** Données persistées : recalculer / versionner / purger / laisser ?

---

# PHASE 2 — IMPLÉMENTATION (livrée)

> GO fondateur : « go avec ce qui est recommandé ». Décisions retenues et câblées.

## Définition retenue et pourquoi
- **D1 = def(a)** — tendance = **sens de la dernière cassure BOS/CHOCH non
  contredite** (état `BOS_SIGNAL` propagé), ancrée sur le dernier CHOCH (à défaut,
  dernier BOS). Retenue car : **stable** (14 bascules/60 j vs 65 pour def(b)),
  **réconcilie tuile ↔ journal** (corrige DG-1 : reste haussière depuis le CHOCH
  du 24 juil.), **indéterminé quasi nul** sur M15/H1, **insensible aux plateaux de
  week-end** qui bruitent def(b).
- **`indeterminate` = état de première classe** (aucune cassure dans l'historique
  analysé), jamais un repli silencieux « neutre ».
- **D2** — « ranging » **retiré** de la Tendance ; test de consolidation migré à la
  tuile **Phase** (fin de la redondance).
- **D3** — pas de fenêtre fixe : la tendance **remonte au dernier événement**,
  **nommé** à l'écran (`trend_reference` → « depuis le CHOCH haussier du 24 juil. »),
  histoire commune avec la tuile Maturité.
- **D4** — `READING_LOGIC_VERSION` **3** : toute lecture ancienne définition est
  **reconstruite avant affichage** → jamais deux définitions mélangées à l'écran.

## Fréquence du cas indéterminé (rappel, mesuré)
def(a) : **0 %** sur H1/M15 (60 j) ; def(b) aurait donné ~37 %. Le choix def(a)
rend donc l'indéterminé marginal en pratique tout en le représentant partout.

## Consommateurs modifiés
- **Backend** : `market_reading_schema` (`TrendValue`/`MTFBiasValue` =
  bullish/bearish/indeterminate, nouveau `TrendReference` + `regime.trend_reference`) ;
  `market_reading_mappers` (`_derive_trend` supprimée → `derive_structural_trend` +
  `_structural_bias_from_candle_dicts` ; `_derive_market_phase` porte le test de
  range ; `candles_to_regime` reçoit les événements) ; `market_reading_assembler`
  (passe `_structure_events`, bump version) ; `conditions_scanner` (`_eval_mtf_aligned`
  compte l'indéterminé à part + dénominateur visible ; `TREND_VALUES`, `_TREND_ADJ`,
  palette « structure ») ; `conditions_scan` route (`TrendChoice`). Narration : rien à
  changer (indéterminé → « flat », libellé « indéterminée », zéro vocabulaire de force).
- **Frontend** : `types/market-reading` (types + `TrendReference`) ; `formatters`
  (labels) ; `mtf-trend` (glyphe `–`, classification indéterminé) ; `RegimeCard`
  (tuile Tendance sous-ligne = événement d'ancrage ; onglet Données structurel ;
  alignement dénominateur ajusté + glyphes indéterminé/indispo distincts) ;
  `conditions/palette` + `conditions/types` (« structure », options) ; i18n 9 locales
  (Concept réécrit, `sub.trendRef/trendNone`, `value.kind*`, `align*`, `data.trendAnchor*`,
  `trendAdj_indeterminate`, `tags.trend_indeterminate`, `regime.mtfNeutral`).

## Écarts de comportement visibles par l'utilisateur
1. **Tuile Tendance** : plus jamais « en range » ni « neutre » ; « haussière /
   baissière / indéterminée », avec la **raison datée** (« depuis le CHOCH … »).
   La tuile ne peut plus contredire le journal.
2. **Tuile Alignement** : une unité indéterminée est montrée (`–`) et **exclue du
   dénominateur** (« N sur M »), jamais comptée comme accord/désaccord.
3. **Scanner** : « 3 TF alignés » devient **alignement de STRUCTURE** (libellé +
   texte mis à jour) ; un combo avec une unité indéterminée n'est **jamais** aligné ;
   l'option de tendance « Range/Neutre » disparaît au profit de « Indéterminée ».
4. **Tuile Phase** : porte désormais seule la consolidation (« ranging »).

## Vérifications
- **Détection intacte** : `git diff origin/main -- src/environment/strategy_features.py
  src/intelligence/smart_money/` = **vide**.
- Tests backend : suites market-reading / scanner / narration vertes + nouveau
  `tests/test_tr1_structural_trend.py` (17). Frontend : tsc 0, suites impactées vertes.
- Échecs `test_smoke_e2e` (scanner v1, 503) : **pré-existants** sur `origin/main`,
  hors périmètre TR-1.

## Restes non implémentés (non bloquants)
- Traduction native des 7 locales non fr/en pour les nouvelles clés (repli EN posé).
- D1/W1 : fréquence de l'indéterminé non mesurée (cache insuffisant ; def(a) ⇒
  marginal attendu).

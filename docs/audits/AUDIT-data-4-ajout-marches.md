# DATA-4 — Ajout des marchés : 2 → 80, validés contre le vrai flux

> Suite directe de [DATA-3](AUDIT-data-3-budget-55.md). Le forfait **Grow est actif**
> (`plan_category: "grow"`, `plan_limit: 55` — relevé sur `/api_usage` le 2026-09-14).
> Base : `origin/main` @ `c478f35`, branche `feat/data-4-ajout-marches`.

---

## RÉPONSE EN PREMIÈRE LIGNE

**80 marchés sont ajoutés au registre, chacun vérifié par un appel réel au flux.**
Pointe mesurée : **28 crédits/minute**, soit **49 % de marge** sous les 55 du forfait.

**Les 20 indices sont volontairement laissés de côté**, pour deux raisons factuelles
(§4). Ce n'est pas un oubli : c'est la seule partie que je ne peux pas livrer
correctement aujourd'hui, et elle est spécifiée ci-dessous.

| | Avant | Après |
|---|---|---|
| Marchés au registre | 2 | **80** |
| Combinaisons suivies en direct | 8 | **320** |
| Crédits/jour | 254 | **10 160** |
| **Pointe crédits/minute** | 8 | **28** *(plafond 55)* |

---

## 1. D'où vient la liste — et pourquoi elle n'était pas utilisable telle quelle

Le dépôt contenait déjà `config/market_catalog_ux_test.json` : 100 marchés populaires.
Ce fichier dit lui-même ce qu'il vaut :

> « Liste constituée à partir des conventions usuelles du marché […] **Ce ne sont PAS
> des marchés validés par un fournisseur de données : ni la couverture, ni les symboles
> ne sont garantis.** »

C'est exactement le point à trancher avant d'ajouter quoi que ce soit : **un marché que
le fournisseur ne sert pas devient un marché mort** — une page vide, un scanner qui ne
trouve rien, aucun message utile. J'ai donc validé les 100 candidats un par un.

---

## 2. Validation — ce que Twelve Data sert vraiment

Deux étapes, la seconde étant la seule qui fasse foi.

**a) Catalogues de référence** (`/forex_pairs`, `/cryptocurrencies`, `/indices`…) —
gratuits, aucun crédit. 98/100 candidats y figurent. **Mais figurer au catalogue ne
prouve rien** : `NAS100` y correspondait à `494300`, un code d'ETF.

**b) Appel réel `time_series`** — 1 crédit par candidat. C'est le seul juge.

| Classe | Candidats | Servis | Verdict |
|---|---|---|---|
| FX majeures | 7 | **7** | ✅ |
| FX mineures | 21 | **21** | ✅ |
| FX exotiques | 20 | **20** | ✅ |
| Métaux | 8 | **8** | ✅ |
| Crypto | 24 | **24** | ✅ *(MATIC renommé **POL** — ajouté sous `POLUSD`)* |
| **Indices** | **20** | **12** | ❌ voir §4 |

Chaque marché retenu porte son **dernier cours relevé** au moment de la validation,
archivé dans le rapport — un contrôle de vraisemblance, pas seulement un « HTTP 200 ».

---

## 3. Ce qu'il a fallu corriger pour que 80 marchés FONCTIONNENT

Ajouter des lignes au registre ne suffisait pas. Trois choses se seraient cassées en
silence — c'est-à-dire sans erreur, en affichant des données fausses.

### 3.1 Les heures de séance étaient une table écrite à la main

`market_calendar.INSTRUMENT_HOURS` listait **7 symboles**. Tout le reste retombait sur
« paire FX » (24h/5j) via une regex de nom ne reconnaissant que `BTC|ETH|USDT|USDC`.

Conséquences si on avait ajouté les marchés sans y toucher :

- **22 des 24 cryptos gelées tout le week-end** alors qu'elles cotent — le moteur aurait
  affiché la bougie de vendredi comme courante jusqu'au lundi ;
- **6 des 8 métaux sans leur pause de rollover** (17h–18h NY), donc une bougie attendue
  pendant une heure où le marché ne cote pas, et une fraîcheur perpétuellement « en retard ».

**Correctif** : les heures dérivent de la **classe d'actif déclarée au registre**
(`fx` / `metal` / `crypto`), la table par symbole devenant un simple jeu d'exceptions —
aujourd'hui vide. Ajouter une crypto la rend 24/7 sans toucher à ce module.

### 3.2 Les préréglages de volatilité étaient un dictionnaire de 6 entrées

L'invariant que les tests verrouillent est **registre ⊆ préréglages** : un marché servi
sans configuration de prévision n'aurait aucune volatilité à annoncer. À 80 marchés
l'invariant tombait.

**Correctif** : même principe. Les préréglages calés à la main (XAUUSD, EURUSD, BTCUSD,
US500, GBPUSD, USDJPY) restent prioritaires ; tout autre marché reçoit le **défaut de sa
classe**. Le code le dit explicitement : les multiplicateurs SL/TP calés à la main sont
un **résultat de backtest** pour CET instrument ; ceux d'un marché nouveau sont
**conventionnels et ne prétendent pas être optimisés**. À affiner marché par marché quand
un rejeu le justifie.

### 3.3 Le résumé injecté dans chaque message de M.I.A n'était pas borné

`SignalSummaryProvider` parcourait **tout** le périmètre et injectait le résultat dans le
prompt système. À 2 marchés : 10 combinaisons. À 80 : **400** — payées **à chaque tour de
conversation**, puisque ce bloc est la partie variable du prompt (MIA-2).

**Correctif** : plafond à 12 combinaisons (`SENTINEL_SUMMARY_MAX_COMBOS`), dans l'ordre du
registre — les marchés qui ouvrent la colonne restent en ligne, le reste est à un appel
d'outil `get_market_reading`, ce qui est précisément la raison d'être de cet outil. Le
comportement à 2 marchés est **inchangé** (10 ≤ 12).

---

## 4. Les 20 indices — pourquoi ils ne sont pas dans ce lot

Deux obstacles factuels, l'un et l'autre vérifiés :

**a) Les tickers ne se résolvent pas de façon fiable.** Sur 20, **12 seulement** rendent
des bougies, et parmi eux au moins un est **le mauvais indice** : `US2000` (Russell 2000)
ne résout que vers `RUA`, qui est le Russell **3000**. `VIX` ne résout que vers `VIX2`.
Huit n'ont aucun ticker fonctionnel (`EU50`, `ITA40`, `JP225`, `HK50`, `AUS200`, `CHN50`,
`IND50`, `CAN60`) alors que leurs noms figurent bien au catalogue `/indices`.

**b) Il n'existe aucun profil d'horaires « indice ».** Un indice actions cote ~6 h 30 par
jour, **dans le fuseau de sa place** (Tokyo, Hong Kong, Francfort, New York…). Traité
comme une paire FX 24h/5j, le moteur attendrait des bougies la nuit et afficherait « en
retard » en permanence hors séance. Ces horaires sont une donnée par place que je ne peux
pas inventer.

**Ce que demande leur ajout** (mission séparée) : un ticker confirmé par appel réel ET un
contrôle de vraisemblance du niveau pour chacun ; un profil d'horaires par place avec son
fuseau ; l'extension de `_HOURS_BY_TYPE` au type `index`. Le type `index` existe déjà dans
le registre et dans `_CLASS_DEFAULTS` — la place est faite, il manque les faits.

---

## 5. Budget de crédits — mesuré sur les décalages RÉELS du scheduler

```
combinaisons suivies en direct : 320   (80 marchés × M15/H1/H4/D1)
total/jour                     : 10 160 crédits
moyenne/minute                 : 7,1
POINTE/minute                  : 28        <-- ce qui décide du forfait
minutes actives                : 1 101 / 1 440
```

| Plafond | Verdict |
|---|---|
| **40** (cible de sécurité DATA-3) | ✅ passe, 30 % de marge |
| **55** (forfait Grow) | ✅ passe, **49 % de marge** |

Les 27 crédits/minute restants absorbent ce que l'étalement ne couvre pas : démarrage à
froid, sonde jour férié, re-tentatives. Un test verrouille ce chiffre
(`test_live_perimeter_fits_the_grow_plan`) : **il échoue si un ajout de marchés fait
passer la pointe au-dessus de 40**, avant que la facture ne le dise.

---

## 6. Outils livrés

`tools/data_budget/` — réutilisables pour le lot suivant (les indices).

```
check_provider_coverage.py   Croise les candidats avec les catalogues de REFERENCE
                             de Twelve Data. Gratuit, aucun credit.

validate_markets.py          Le juge : un appel time_series par candidat (1 credit),
                             verifie que des bougies reviennent, releve le dernier
                             cours, et ecrit les entrees de registre pretes.
                               python tools/data_budget/validate_markets.py --groups crypto

apply_validated_markets.py   Applique ces entrees a config/markets.json et
                             config/event_market_map.json par ecriture TEXTUELLE
                             (pas de json.dump qui reformaterait tout le fichier).
                               python tools/data_budget/apply_validated_markets.py --dry-run valides.json
```

**Ne jamais ajouter un marché sans passer par `validate_markets.py`.** C'est la seule
étape qui distingue un marché servi d'un marché mort.

---

## 7. Garde-fous ajoutés

`tests/test_data4_market_perimeter.py` — 21 tests. Chacun verrouille ce qui rend un
marché réellement servi plutôt que simplement déclaré :

- tout marché du registre résout vers un symbole fournisseur, et **deux marchés ne
  partagent jamais un ticker** (ce serait le même marché affiché deux fois, et payé deux fois) ;
- tout marché a un préréglage de volatilité et une règle de rattachement des actualités ;
- la crypto cote le week-end, les métaux gardent leur pause, le FX ferme — **vérifié pour
  chaque marché, pas sur un échantillon** ;
- un symbole hors registre reste supposé **fermé** (annoncer un marché ouvert à tort est
  la pire des deux erreurs) ;
- la pointe de crédits tient sous la cible de sécurité ;
- le bloc injecté dans chaque message de M.I.A reste borné ;
- le module frontend généré ne dérive pas du registre.

---

## Annexe — ce que ce lot ne change pas

- **Aucune profondeur d'historique modifiée** : les nouveaux marchés se peuplent par le
  chemin normal (première lecture) ; l'amorçage profond reste manuel et optionnel.
- **Aucun réglage de détection touché** : mêmes règles SMC, mêmes seuils.
- **Le catalogue d'affichage reste distinct du registre** : il continue de servir à tester
  la tenue de l'interface, il ne décide pas de ce que le moteur suit.
- **Les défauts de DATA-3 sont inchangés** : sans variable d'environnement, le limiteur
  reste au palier gratuit et le tick reste séquentiel.

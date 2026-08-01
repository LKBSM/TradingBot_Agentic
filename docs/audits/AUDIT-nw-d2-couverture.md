# AUDIT NW-D2 — Couverture des publications économiques (Or & EUR/USD)

**Date :** 2026-08-01
**Nature :** LECTURE SEULE. Aucune correction, aucune branche, aucun fichier moteur modifié.
**Périmètre audité :** la branche de production `origin/main` (le calendrier « Actualités programmées »
NW-1 / NW-1b / NW-1c / NW-3). ⚠️ La branche de travail locale
(`fix/mt-d1-journal-display-window`) est **45 commits en retard sur main** et ne contient
pas le flux de valeurs ni la vue mensuelle : cet audit décrit **ce que voit un client aujourd'hui**,
donc `main`.

---

## 0. LE CHIFFRE, EN PREMIER

Contre la liste de référence des **29 publications déterminantes** pour l'Or et l'EUR/USD
(section 1), voici ce que couvre MIA en production :

| Catégorie | Compte | Ce que ça veut dire |
|---|---:|---|
| **(a) PRÉSENTE ET COMPLÈTE** | **3** | Événement + heure + organisme + unité + valeur publiée + précédente. **Les 3 sont européennes.** |
| **(b) PRÉSENTE MAIS INCOMPLÈTE** | **10** | L'événement, l'heure, l'organisme, l'unité sont là — **la valeur chiffrée manque**. Ce sont, pour l'essentiel, **les publications américaines majeures** (IPC, NFP, PCE, PIB, FOMC…). |
| **(c) ABSENTE mais récupérable** | **8** | Un organisme officiel la publie, on pourrait l'ajouter (inscriptions chômage, JOLTS, procès-verbaux FOMC, dot plot, IPC allemand…). |
| **(d) HORS DE PORTÉE** | **8** | Aucun organisme public — publiée par un privé (ISM, Conference Board, Michigan, PMI S&P/HCOB, IFO, ZEW, ADP). |

**Lecture directe :** le calendrier connaît la plupart des bons *moments*, mais pour l'Or —
piloté par la macro **américaine** — il n'affiche **presque aucun chiffre publié** (les valeurs
US ne sont pas récupérées, cf. §3). Les seules fiches réellement complètes concernent l'euro.
C'est une asymétrie structurelle qui pénalise justement le marché primaire du produit (XAU/USD).

---

## 1. LISTE DE RÉFÉRENCE — les publications qui déplacent l'Or et l'EUR/USD

Établie à partir de sources publiques sérieuses (BLS, BEA, Census, Federal Reserve, ECB,
Eurostat, plus pages pédagogiques de courtiers / Investopedia / BabyPips). **OFFICIEL** =
agence d'État ou banque centrale ; **PRIVÉ** = société / association / université / institut
(distinction décisive : la décision en vigueur interdit les sources non officielles en prod).

### Emploi américain
| Publication | Organisme | Off./Privé | Périodicité | Heure (ET) | Marché |
|---|---|---|---|---|---|
| **Rapport sur l'emploi** (NFP + taux de chômage + salaires horaires — **une seule parution**) | **BLS** | OFFICIEL | mensuel | 1er vendredi, 08:30 | Les deux |
| ADP (emploi privé) | ADP + Stanford | **PRIVÉ** | mensuel | ~08:15 | Les deux |
| **Inscriptions hebdo au chômage** | **DOL / ETA** | OFFICIEL | **hebdomadaire** | jeudi, 08:30 | Les deux |
| JOLTS (offres d'emploi) | **BLS** | OFFICIEL | mensuel | 10:00 | Les deux |

### Inflation américaine
| Publication | Organisme | Off./Privé | Périodicité | Heure (ET) | Marché |
|---|---|---|---|---|---|
| **IPC** (headline) | **BLS** | OFFICIEL | mensuel | 08:30 | Les deux |
| **IPC sous-jacent** (core) | **BLS** (même parution, série distincte) | OFFICIEL | mensuel | 08:30 | Les deux |
| **PCE / PCE sous-jacent** (jauge préférée de la Fed) | **BEA** | OFFICIEL | mensuel | 08:30 | Les deux |
| **IPP** (prix à la production) | **BLS** | OFFICIEL | mensuel | 08:30 | Les deux |

### Réserve fédérale (toutes OFFICIELLES, meuvent les deux marchés)
| Événement | Organisme | Périodicité | Heure (ET) |
|---|---|---|---|
| **Décision de taux (FOMC)** | Fed / FOMC | 8×/an | 14:00 |
| **Conférence de presse** | Président de la Fed | après chaque réunion | 14:30 |
| **Projections économiques / dot plot** (SEP) | FOMC | trimestriel (mar/juin/sep/déc) | 14:00 |
| **Procès-verbaux (minutes)** | FOMC | 8×/an, 3 sem. après | 14:00 |
| Auditions du président (Humphrey-Hawkins) | Fed → Congrès | semestriel (fév/juil) | matin |
| Jackson Hole | Fed de Kansas City | annuel, fin août | variable |

### Croissance & activité américaines
| Publication | Organisme | Off./Privé | Périodicité | Heure (ET) |
|---|---|---|---|---|
| **PIB** | **BEA** | OFFICIEL | trimestriel | 08:30 |
| **Ventes au détail** | **Census** | OFFICIEL | mensuel | 08:30 |
| ISM Manufacturier | **ISM** | **PRIVÉ** | mensuel | 10:00 |
| ISM Services | **ISM** | **PRIVÉ** | mensuel | 10:00 |
| **Commandes de biens durables** | **Census** | OFFICIEL | mensuel | 08:30 |
| Confiance des consommateurs | **Conference Board** | **PRIVÉ** | mensuel | 10:00 |
| Sentiment du consommateur | **Univ. du Michigan** | **PRIVÉ** | mensuel | 10:00 |

### Zone euro
| Publication | Organisme | Off./Privé | Périodicité | Heure (CET) |
|---|---|---|---|---|
| **IPCH flash** (inflation zone euro) | **Eurostat** | OFFICIEL | mensuel | **15:00** ⚠️ (cf. §3) |
| **Décision de taux BCE** | **BCE** | OFFICIEL | 8×/an | 14:15 |
| **Conférence de presse BCE** | Présidente BCE | 8×/an | 14:45 |
| **PIB zone euro** (flash) | **Eurostat** | OFFICIEL | trimestriel | ~11:00 |
| IPC allemand flash (alimente le flash euro) | **Destatis** | OFFICIEL | mensuel | ~14:00 |
| PMI zone euro / Allemagne | **S&P Global / HCOB** | **PRIVÉ** | mensuel | ~10:00 |
| IFO (climat des affaires all.) | **Institut ifo** | **PRIVÉ** | mensuel | 10:00 |
| ZEW (sentiment économique all.) | **Centre ZEW** | **PRIVÉ** | mensuel | 11:05 |

> À noter : ISM, Conference Board, Université du Michigan, S&P Global/HCOB, ifo et ZEW sont
> **privés** — donc **hors périmètre** de la décision « sources officielles uniquement ». Ce
> n'est pas un défaut du produit, c'est une conséquence assumée de la règle.

---

## 2. CE QUE LE PRODUIT COUVRE — la comparaison, ligne par ligne

Le catalogue de production (`config/calendar_catalog.json`, version 1) contient **13 publications
récurrentes** issues de 6 organismes officiels : BLS, BEA, Census, Federal Reserve, Eurostat, BCE.

**Contexte de production déterminant pour la catégorie a/b** (`render.yaml`) :
`CALENDAR_ICS_LIVE=1`, `CALENDAR_VALUES_LIVE=1`. Mais le flux de valeurs
(`src/intelligence/calendar_providers/values/`) ne câble en dur que **3 récupérateurs** :
ECB (sans clé), Eurostat (sans clé), et **BLS uniquement si `BLS_API_KEY` est défini** — or cette
clé **n'est pas dans `render.yaml`**. Il n'existe **aucun** récupérateur pour **BEA, Census, ni la
Fed**. Conséquence : en production, **seules les valeurs européennes (Eurostat + BCE) remontent** ;
toutes les valeurs américaines restent à l'état honnête `unfetched` (« publiée mais non récupérée »,
cf. modèle `compute_value_state`, NW-1c §3B).

| # | Réf. (§1) | Clé catalogue | Cat. | Détail |
|---|---|---|---|---|
| 1 | Rapport emploi (NFP) | `us_employment_situation` | **(b)** | Daté, heure/org/unité OK. **Valeur non récupérée** (BLS clé-gaté, clé absente). NFP/chômage/salaires **fusionnés en 1 fiche** : chômage et salaires n'ont pas de valeur propre. |
| 2 | ADP | — | **(d)** | Privé. |
| 3 | Inscriptions chômage hebdo | — | **(c)** | DOL/ETA **officiel**, absent du catalogue. Récupérable. |
| 4 | JOLTS | — | **(c)** | BLS **officiel**, absent. Récupérable (même récupérateur BLS). |
| 5 | IPC | `us_cpi` | **(b)** | Daté, OK. Valeur non récupérée (BLS). Série headline seule. |
| 6 | IPC sous-jacent | — | **(b/c)** | Non distingué de l'IPC (série `CUUR0000SA0L1E` non exposée). Récupérable. |
| 7 | PCE / PCE core | `us_pce` | **(b)** | Daté, OK. **Aucun récupérateur BEA** → valeur absente. Core non distingué. |
| 8 | IPP | `us_ppi` | **(b)** | Daté, OK. Valeur non récupérée (BLS). |
| 9 | Décision FOMC | `us_fomc_rate` | **(b)** | Daté, heure/org OK. `series_code=null` → état `unavailable` : **le taux directeur lui-même n'est pas affiché**. |
| 10 | Conférence de presse FOMC | — | **(c)** | Pas d'événement distinct (même moment que la décision). Récupérable. |
| 11 | Projections / dot plot | — | **(c)** | Fed **officiel**, absent. Récupérable (dates fixes). |
| 12 | Procès-verbaux FOMC | — | **(c)** | Fed **officiel**, absent. Récupérable. |
| 13 | Auditions du président | — | **(c)** | Fed **officiel**, absent (dates irrégulières). |
| 14 | Jackson Hole | — | **(c)** | Fed KC **officiel**, absent (annuel). Priorité faible. |
| 15 | PIB | `us_gdp` | **(b)** | Daté, OK. **Aucun récupérateur BEA** → valeur absente. |
| 16 | Ventes au détail | `us_retail_sales` | **(b)** | Daté, OK. **Aucun récupérateur Census** → valeur absente. |
| 17 | ISM Manufacturier | — | **(d)** | Privé. |
| 18 | ISM Services | — | **(d)** | Privé. |
| 19 | Commandes biens durables | `us_durable_goods` | **(b)** | Daté, OK. Aucun récupérateur Census → valeur absente. |
| 20 | Confiance conso (CB) | — | **(d)** | Privé. |
| 21 | Sentiment Michigan | — | **(d)** | Privé. |
| 22 | IPCH flash zone euro | `ea_hicp_flash` | **(a)\*** | Daté, unité OK, **valeur Eurostat récupérée**. **MAIS heure encodée 11:00 alors que le calendrier officiel BCE du flash indique 15:00 CET** → défaut qualité (§3). |
| 23 | Décision BCE | `ea_ecb_rate` | **(a)** | Daté, heure/org/unité OK, **valeur BCE récupérée** (taux, série SDMX). Complète. |
| 24 | Conf. de presse BCE | — | **(c)** | Pas d'événement distinct. Récupérable. |
| 25 | PIB zone euro flash | `ea_gdp_flash` | **(a)** | Daté, unité OK, valeur Eurostat récupérée. Complète. |
| 26 | PMI zone euro / all. | — | **(d)** | Privé (S&P Global / HCOB). |
| 27 | IFO | — | **(d)** | Privé. |
| 28 | ZEW | — | **(d)** | Privé. |
| 29 | IPC allemand flash | — | **(c)** | Destatis **officiel**, absent. Récupérable. |

**Bonus présents mais hors liste de référence stricte :** `us_housing_starts` (mises en chantier,
Census, valeur non récupérée) et `ea_unemployment` (chômage zone euro, Eurostat) — ce dernier
**n'a aucune date curée dans `calendar_schedule.json`** et n'apparaîtra que si le flux ICS Eurostat
le date : risque de non-affichage silencieux.

### Compte final
- **(a) Complète : 3** — `ea_ecb_rate`, `ea_gdp_flash`, `ea_hicp_flash` (cette dernière avec un
  défaut d'heure à corriger avant de la considérer réellement complète → **2 propres + 1 à vérifier**).
- **(b) Incomplète : 10** — les 9 fiches américaines majeures (valeur manquante) + `ea_unemployment`.
- **(c) Absente récupérable : 8**.
- **(d) Hors de portée : 8**.

### Les absentes récupérables, par effort croissant

1. **Débloquer les valeurs US existantes** *(plus haut retour, coût le plus faible)* — ce n'est
   pas un ajout d'événement mais un ajout de **récupérateur de valeur**, sur le patron déjà
   éprouvé de `values/eurostat_values.py` :
   - **BEA** (`apps.bea.gov/api/data`, clé gratuite) → débloque **PIB + PCE**. Un fichier `bea_values.py`.
   - **Census** (`api.census.gov/data/timeseries/eits`, sans clé) → débloque **ventes au détail,
     biens durables, mises en chantier**. Un fichier `census_values.py`.
   - **BLS** : le récupérateur **existe déjà** — il suffit de poser `BLS_API_KEY` (gratuit) dans
     le dashboard Render → débloque **NFP, IPC, IPP**. *Effort : une variable d'environnement.*
2. **JOLTS** — série BLS, récupérateur BLS déjà là : ajouter 1 ligne au catalogue + la série.
3. **IPC sous-jacent** — ajouter la série core BLS (`CUUR0000SA0L1E`) comme fiche distincte.
4. **Procès-verbaux FOMC + dot plot (SEP)** — Fed, dates fixes : 2 lignes catalogue + dates
   `calendar_schedule.json`. Événements sans valeur numérique (comme la décision FOMC aujourd'hui).
5. **IPC allemand flash (Destatis)** — 1 organisme + adaptateur de dates ; alimente le flash euro.
6. **Inscriptions hebdo au chômage (DOL/ETA)** — hebdomadaire (volume ×4/mois) ; endpoint ETA
   (`oui.doleta.gov/unemploy/claims.asp`). Effort moyen (fréquence).
7. **Conf. presse FOMC/BCE, auditions, Jackson Hole** — événements sans valeur, dates connues :
   entrées « moment seul ». Effort faible mais utilité marginale (même instant que la décision).

---

## 3. QUALITÉ DE CE QUI EST AFFICHÉ — trois fiches examinées champ par champ

### Fiche A — `ea_ecb_rate` (Décision de taux BCE)
| Champ | Valeur | Verdict |
|---|---|---|
| Nom affiché | « Décision de taux (BCE) » | ✅ Reconnaissable par un trader. |
| Heure | 14:15 `Europe/Berlin` → conversion UTC + locale via IANA (DST géré) | ✅ Correcte (14:15 CET est l'heure officielle historique). |
| Unité | « taux directeur (% par an) » | ✅ Conforme. |
| Valeur / précédente | Récupérées en direct (`ecb_values.py`, SDMX sans clé, série `FM.D.U2.EUR.4F.KR.MRR_FR.LEV`, « previous » = dernière valeur **distincte** — juste pour une série en escalier) | ✅ Présentes, telles que publiées. |
| Marchés | EUR → XAUUSD ? Non (XAU driver=USD). Rattachée à **EURUSD** seule. | ✅ Pertinent. |
| Attribution / licence | « Réutilisation libre si source citée et statistiques non modifiées — Source: ECB » | ✅ Exacte. |

**Verdict : complète et honnête.** Aucun champ inventé.

### Fiche B — `us_cpi` (IPC)
| Champ | Valeur | Verdict |
|---|---|---|
| Nom affiché | « Indice des prix à la consommation (IPC) » | ✅ Reconnaissable. |
| Heure | 08:30 `America/New_York`, DST géré | ✅ Correcte. |
| Unité | « indice (1982-84 = 100) » | ✅ Conforme à la série CUUR0000SA0. |
| Valeur / précédente | **Non récupérées** → état `unfetched` (BLS clé-gaté, clé absente en prod) | ⚠️ **Chiffre manquant** — mais **honnêtement signalé** (« publiée, non récupérée », pas un « — » ambigu ni un nombre inventé). |
| IPC sous-jacent | **Non distingué** — seule la série headline existe | ⚠️ Manque fonctionnel (le core est souvent plus suivi que le headline). |
| Marchés | USD → **XAUUSD + EURUSD** | ✅ Pertinent. |
| Attribution | « Domaine public (17 U.S.C. §105) — U.S. Bureau of Labor Statistics » | ✅ Exacte. |

**Verdict : incomplète mais non trompeuse.** Le modèle à quatre états (`published / pending /
unfetched / unavailable`) est une vraie réussite d'honnêteté : l'absence de valeur est **nommée**,
jamais maquillée. Reste que, pour un trader Or, l'IPC sans chiffre a peu de valeur d'usage.

### Fiche C — `ea_hicp_flash` (IPCH flash zone euro)
| Champ | Valeur | Verdict |
|---|---|---|
| Nom affiché | « Inflation IPCH zone euro (estimation rapide) » | ✅ Reconnaissable. |
| **Heure** | **11:00 `Europe/Luxembourg`** encodée dans le catalogue | ❌ **PROBABLE ERREUR.** Le calendrier officiel de la BCE pour l'*estimation rapide* de l'IPCH indique **15:00 CET** (ex. parution du 31/07/2026 à 15:00). L'heure de 11:00 correspond plutôt à la publication *complète* de mi-mois. **À revérifier sur le calendrier Eurostat avant facturation.** `time_confirmed=true` alors que l'heure est douteuse aggrave le risque : une heure fausse est pire qu'une heure absente. |
| Unité | « % (variation annuelle) » | ✅ Conforme. |
| Valeur / précédente | Récupérées (`eurostat_values.py`, JSON-stat sans clé, `prc_hicp_manr`, filtres EA20/CP00/RCH_A) | ✅ Présentes. |
| Marchés | EUR → EURUSD | ✅ Pertinent. |
| Attribution | « CC BY 4.0 — Source: Eurostat » | ✅ Exacte. |

**Verdict : la seule anomalie de champ inventé/approché de l'échantillon.** L'heure 11:00, marquée
« confirmée », contredit la source officielle. C'est exactement le type de défaut que la mission
qualifie de « pire qu'un champ absent ». **À corriger et re-vérifier** (revoir aussi `ea_gdp_flash`
et `ea_unemployment`, dont les 11:00 sont probablement une hypothèse par défaut).

**Synthèse §3 :** aucun chiffre fabriqué (le modèle d'états l'empêche par construction), mais
**une heure officielle probablement fausse** sur l'IPCH flash, et un **manque de valeurs** massif
côté US (conséquence de l'absence de récupérateurs BEA/Census et de la clé BLS).

---

## 4. LES ÉVÉNEMENTS NON PROGRAMMÉS — la limite structurelle

**Constat :** cette limite **n'est écrite nulle part** dans le produit. Le bloc d'avertissement du
calendrier (`reading.calendar.nono`, fr & en) dit ce que le calendrier **ne prédit pas** (direction,
prévision d'analystes, hiérarchie d'impact) — mais **jamais** qu'il ne couvre que le programmé et
ne verra jamais l'imprévu (décision hors calendrier, choc géopolitique, faillite, déclaration
inattendue). Un client peut légitimement croire qu'être abonné au calendrier, c'est être averti de
tout. C'est une tromperie par omission à corriger avant facturation.

**Formulation proposée** (affirmative, non défensive — à AJOUTER, ne pas implémenter ici) :

- **FR :** « Ce calendrier annonce des moments **programmés**. Une décision hors calendrier, un choc
  géopolitique ou une déclaration inattendue n'y figurent pas : ils n'ont pas d'heure connue à
  l'avance. »
- **EN :** « This calendar announces **scheduled** moments. An unscheduled decision, a geopolitical
  shock or an unexpected statement will not appear here: they have no time known in advance. »

**Emplacement recommandé :** troisième item du bloc `reading.calendar.nono` (à côté de `noForecast`
et `noRanking`), dans `webapp/messages/fr.json` et `en.json`, rendu par
`webapp/components/calendar/CalendarWorkspace.tsx`. C'est le seul endroit vu de tout utilisateur du
calendrier, quelle que soit la vue (liste ou mois).

---

## 5. EXTENSIBILITÉ À UN NOUVEAU MARCHÉ

**a) Le rattachement est-il configurable ?** **Oui, entièrement par configuration.** La table vit
dans `config/event_market_map.json` (règle documentée dans le fichier lui-même) : un événement est
rattaché à un marché si la **devise** de l'événement figure dans les `driver_currencies` du marché.
Le code (`calendar_service.attach_markets`) ne fait que lire cette table — rien n'est codé en dur.

```json
"markets": {
  "XAUUSD": { "driver_currencies": ["USD"] },
  "EURUSD": { "driver_currencies": ["USD", "EUR"] }
}
```

**b) Ajouter un marché = code ou config ?** **Config seule**, à une condition : que les devises
motrices du nouveau marché soient déjà couvertes par des événements du catalogue (USD, EUR). Sinon,
il faut aussi des événements pour la nouvelle devise (donc du catalogue, voire un adaptateur de source).

**c) Cas du Bitcoin.**
- **Rattacher les publications américaines existantes à BTCUSD : trivial et propre.** Une seule
  ligne — `"BTCUSD": { "driver_currencies": ["USD"] }` — et **toutes** les fiches USD (FOMC, IPC,
  NFP, PCE…) s'attachent automatiquement à BTCUSD. **Aucune duplication** : le modèle `markets` est
  une liste (relation plusieurs-à-plusieurs) ; le même `us_cpi` sert XAUUSD, EURUSD **et** BTCUSD.
  C'est exactement ce qu'il faut, et le design le permet déjà.
- **Échéances de décision SEC :** ce sont des **événements officiels programmés** (publiés au
  Federal Register). Ils entrent **partiellement** dans le schéma : `organism = SEC`, date connue,
  `markets = [BTCUSD]` via USD. Mais **pas de valeur numérique** (issue binaire approbation/rejet/
  report, pas une série) → `series_code = null` → état `unavailable`. **Le schéma tolère déjà les
  événements sans valeur** (c'est le cas de la décision FOMC actuelle). Il faut donc juste ajouter
  SEC comme organisme (entrée `sources` + adaptateur de dates ou planning curé) : **config + petit
  adaptateur**.
- **Le halving :** déterministe et vérifiable sur la chaîne, mais **sans émetteur institutionnel**.
  Les champs `source` / `organism` / `license_label` / `series_code` présupposent tous un
  **publieur** ; le halving n'en a pas. La **date** s'insère (le schéma date+marchés suffit), mais
  l'attribution ne mappe pas : il faudrait soit un **type de source « protocole / déterministe »**
  (attribution = la chaîne elle-même), soit un type d'événement distinct. **C'est le seul cas qui
  demande une extension de schéma**, pas juste de la config.

**d) Pour qu'ajouter un marché soit purement de la config :**
- La table marché→devises l'est déjà. ✅
- Rattacher des événements **existants** à un nouveau marché (Bitcoin via USD) : déjà purement config. ✅
- Ce qui manque : (1) un **type de source non institutionnel** pour les événements sans publieur
  (halving) ; (2) l'ajout d'organismes comme la SEC reste config + petit adaptateur de dates. Une
  fois ce type ajouté, l'extension redeviendrait de la config.

**Verdict :** l'architecture de rattachement est saine et déjà extensible par configuration pour
tout marché piloté par des devises déjà couvertes. Le seul angle mort conceptuel est l'événement
**sans émetteur** (halving), qui ne rentre pas dans le modèle « un organisme publie une valeur ».

---

## 6. FRAÎCHEUR ET FIABILITÉ

- **Fréquence de rafraîchissement :** paresseux, à la requête, TTL **120 s**
  (`CalendarService.DEFAULT_TTL_SECONDS`). Un échec de fournisseur est **capté** ; le cache existant
  **n'est jamais effacé** (`_maybe_refresh`, commentaire « provider failure keeps stale cache »).
- **Dernier succès visible ?** **Partiellement.** Chaque source porte un `last_success` ; une source
  est marquée `stale` si elle n'a pas été rafraîchie au dernier cycle **ou** si son dernier succès
  remonte à > 24 h. Les sources périmées sont affichées avec leur date de dernier succès
  (`attribution.stale`). **Mais** quand tout est frais, **aucun horodatage positif « mis à jour à… »
  n'est montré** : le client voit l'absence de problème, pas la preuve de fraîcheur. Petit manque.
- **Organisme injoignable / calendrier partiel :** bien géré. Les données existantes sont conservées ;
  le drapeau `coverage.partial` prévient quand la fenêtre demandée dépasse la couverture réelle de la
  source, et `stale_sources` liste les sources en retard. **Un calendrier partiel ne peut donc pas
  paraître complet** — c'est un vrai point fort du design NW-1b/1c.
- **Heure qui change :** point faible. Les dates des sources **BLS/BEA** se rafraîchissent via leur
  flux `.ics` (report de shutdown pris en compte). Mais pour les sources **datées uniquement par le
  planning curé** (`calendar_schedule.json` : Fed, Census, Eurostat, BCE), une parution reportée
  **reste figée à l'heure/date initialement inscrite** jusqu'à une re-vérification manuelle — le
  `_doc` du fichier le reconnaît explicitement (« les calendriers officiels bougent, ex. reports de
  shutdown… à revérifier périodiquement »). De plus, la couverture curée s'arrête vers **oct.–déc.
  2026** ; au-delà, seules les sources à flux ICS continuent, le reste bascule en `partial`.

---

## 7. AVIS FRANC — un trader rate-t-il des publications qui comptent ?

**OUI.**

Deux manques, de nature différente :

1. **Des publications officielles entièrement absentes** qu'on pourrait récupérer : inscriptions
   hebdomadaires au chômage, JOLTS, **procès-verbaux du FOMC**, **projections/dot plot**, IPC
   allemand flash. Le dot plot et les minutes, en particulier, déplacent l'Or et l'EUR/USD autant
   qu'une décision de taux — et ils sont simplement absents.

2. **Surtout : les valeurs chiffrées des grandes publications américaines ne sont pas affichées.**
   Pour l'**Or**, dont le prix est mû par la macro **américaine**, le calendrier montre le *moment*
   de l'IPC, du NFP, du PCE, du PIB et de la décision FOMC — mais **pas le chiffre publié ni le
   précédent** (aucun récupérateur BEA/Census ; clé BLS non posée). Un calendrier économique dont on
   ne peut pas lire le chiffre de l'inflation US le jour de sa sortie ne remplit pas sa promesse
   pour un trader Or. Les seules fiches réellement complètes sont européennes — l'asymétrie tombe du
   mauvais côté du marché primaire du produit.

**À décharge**, ce qui est affiché est **honnête** : le modèle à quatre états ne fabrique jamais un
chiffre, l'attribution et les unités sont exactes, la couverture partielle est signalée. Le socle est
sain. Le problème n'est pas la tromperie, c'est la **complétude** — et un défaut d'heure isolé sur
l'IPCH flash à corriger.

**Le chemin le plus court vers « prêt à facturer » n'est pas un gros chantier :** poser `BLS_API_KEY`
(une variable), écrire deux récupérateurs de valeurs (BEA, Census) sur le patron existant, ajouter
quatre événements Fed/BLS déjà couverts par des récupérateurs, corriger l'heure de l'IPCH flash, et
ajouter la phrase sur les événements non programmés (§4). Après cela, la réponse pourrait devenir
« non ».

---

*Fin de l'audit NW-D2. Lecture seule — aucune modification apportée au produit.*

### Sources de la liste de référence (§1)
BLS (Employment Situation, CPI/PPI, JOLTS), BEA (PIO/PCE, GDP), U.S. Census (retail, durable goods),
DOL/ETA (jobless claims), federalreserve.gov (FOMC calendars, SEP, minutes, testimony),
KC Fed (Jackson Hole), ISM, Conference Board, Univ. of Michigan, ECB (Governing Council & HICP
release calendars), Eurostat (HICP/GDP), Destatis, S&P Global/HCOB, ifo, ZEW ; corroboration
pédagogique Investopedia / BabyPips / glossaires courtiers. Heure IPCH flash vérifiée sur le
calendrier de diffusion BCE (`ecb.europa.eu/press/calendars/statscal/ges/html/sthicp.en.html`,
consulté 2026-08-01 → **15:00 CET**).

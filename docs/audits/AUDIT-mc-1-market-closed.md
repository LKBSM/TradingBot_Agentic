# AUDIT MC-1 — État « Marché fermé » : arrêt honnête de l'analyse et des appels de données

Branche : `feat/mc-1-market-closed-state` (worktree dédié depuis `main` @ `c494b8f`).
Date : 2026-07-26. Backend + frontend. Le fait d'abord, l'explication ensuite.

---

## 1. Symptôme et verdict

**Symptôme observé** : marché fermé (week-end / férié) → le produit continue à analyser,
consomme l'API de données, et présente les structures existantes (Order Blocks, poches de
liquidité) comme si elles étaient de **nouvelles** détections « en direct ».

**Verdict racine** : ce n'est **pas** un problème d'identifiants instables (l'hypothèse
« suspect n°1 » du brief). C'est un **déclencheur d'analyse aveugle au calendrier** combiné à
l'**absence de verrou serveur sur la (ré)émission** d'une lecture. La malhonnêteté est de
présentation (badge « En direct » + narration au présent sur des données figées de vendredi),
pas une invention du détecteur.

---

## 2. Point A — stabilité des identifiants : PROUVÉE, cause écartée

Test concret exigé par le brief : deux passages du moteur sur des **bougies identiques**
(fixtures golden versionnées `tests/fixtures/ob_golden/`), comparaison des identifiants émis.

```
XAUUSD_M15: n=27 zones   run1 == run2 -> True   only-in-run1=[]  only-in-run2=[]
XAUUSD_H4 : n=32 zones   run1 == run2 -> True   only-in-run1=[]  only-in-run2=[]
EURUSD_H1 : n=29 zones   run1 == run2 -> True   only-in-run1=[]  only-in-run2=[]
```

Les identifiants sont **100 % déterministes**, dérivés du contenu de la zone (type + direction
+ horodatage de la bougie de formation), jamais d'un `uuid4`/index/horloge :

| Entité | Format d'id | Source |
|---|---|---|
| Order Block | `OB_{direction}_{created_at:%Y%m%d%H%M%S}` | `market_reading_mappers.py:1138` |
| FVG | `FVG_{direction}_{created_at:%Y%m%d%H%M%S}` | `market_reading_mappers.py:1170` |
| Poche de liquidité | `LIQ_{side}_{kind}_{created_at:%Y%m%d%H%M%S}` | `market_reading_mappers.py:1198` |
| Signal de confluence | `sha1(symbol\|bar_ts\|type\|score)[:12]` | `confluence_detector.py:364` |

`created_at` provient toujours de `enriched.index[k]` (l'horodatage de la bougie). Régénéré
non-régression : `tests/test_ob_golden_nonregression.py` reste vert (jeu d'OB identique
avant/après). **Aucun pansement « fermé » n'est posé sur les ids — ils sont déjà corrects.**

Garde permanente ajoutée : `tests/test_mc1_market_closed_wiring.py` + le golden non-régression.

---

## 3. Cause racine réelle (mécanique)

1. **Déclencheur aveugle au calendrier.** `MarketReadingScheduler.tick()` (60 s) appelle
   `_needs_regeneration()`, qui comparait le `candle_close_ts` stocké à
   `expected_last_candle_close(tf, now)` — où `now` **avance en continu**. Le week-end, il croit
   donc perpétuellement qu'une bougie plus récente aurait dû clôturer → régénère à chaque tick →
   appel Twelve Data + re-détection + re-narration, sur des bougies figées de vendredi. Le même
   chemin « lazy » existait dans `MarketReadingAssembler.get_or_generate()`.

2. **Aucun verrou serveur sur la nouveauté.** Il n'existe aucun émetteur explicite « nouvelle
   zone » : la nouveauté est *inférée* du statut `active` + `created_at`. Rien ne changeant, les
   zones gardent leurs horodatages de vendredi — mais le cadre « En direct » et la narration au
   présent les font passer pour de l'activité courante.

3. **Twelve Data Forex v2 est 24/7 et n'expose aucun état de marché** — le week-end il répète la
   dernière bougie de vendredi. On ne peut donc pas déléguer l'état « fermé » au fournisseur, et
   chaque appel du week-end est du quota gaspillé.

---

## 4. Correctif — une source de vérité unique, côté serveur

### 4.1 `src/intelligence/market_calendar.py` (nouveau)
Source de vérité unique. Combine :
- **le FAIT (autorité)** : âge de la dernière bougie clôturée vs cadence de la TF ;
- **le CALENDRIER (explication)** : horaires **par instrument**, fériés versionnés, `ZoneInfo`.

États : `open · closed_weekend · closed_holiday · daily_break · data_lagged`.

Horaires **par instrument** (jamais une constante globale), tous en `America/New_York`,
DST-safe (17:00 NY = 21:00 UTC en été / 22:00 UTC en hiver, géré par la bibliothèque) :

| Instrument | Ouverture | Fermeture | Pause quotidienne |
|---|---|---|---|
| EURUSD (forex) | dim. 17:00 NY | ven. 17:00 NY | — |
| XAUUSD (métal comptant) | dim. 18:00 NY | ven. 17:00 NY | 17:00–18:00 NY (lun.–jeu.) |
| BTCUSD (crypto) | 24/7 | — | — |

`market_aware_expected_close(instrument, tf, now)` : la clôture de la dernière bougie qui a
**réellement tradé** ≤ `now` — la borne d'horloge quand le marché est ouvert, **gelée** à la
dernière borne pré-fermeture sinon. C'est la primitive du verrou : gelée = le stocké correspond
toujours = aucune régénération, aucun appel, aucune ré-émission ; et comme la valeur gelée
**égale** la vraie dernière bougie, l'horodatage stocké reste honnête (jamais un timestamp de
week-end sur des données de vendredi).

### 4.2 Fériés versionnés — `config/market_holidays.json` (nouveau)
Chaque entrée : date + marché(s) + source en commentaire. Champ `covered_through_year`. **Au-delà
de la couverture → repli sur l'âge des bougies, jamais « ouvert » par défaut** : un férié inconnu
donne `data_lagged`, pas de la fausse activité.

### 4.3 Verrou d'émission (le vrai point d'émission)
`get_or_generate()` et `scheduler._needs_regeneration()` utilisent désormais
`market_aware_expected_close`. Quand le marché est fermé, la lecture stockée correspond → **pas de
rebuild, pas de fetch Twelve Data, pas de nouvelle narration**. C'est un verrou serveur, pas un
filtre d'affichage.

### 4.4 Reprise + sécurité
- **Reprise automatique à la réouverture** : à `next_open`, la borne « market-aware » se dégèle
  d'elle-même au tick suivant. Zéro code dédié.
- **Sonde de sécurité peu fréquente** (`refresh_if_reopened`, `scheduler._should_safety_probe`) :
  **uniquement sur fériés** (les week-ends/pauses sont déterministes), à basse fréquence
  (`safety_poll_seconds=1800`), et ne reconstruit **que si le fournisseur renvoie réellement une
  bougie plus récente**, estampillée sur la vraie clôture. Un week-end fait **zéro appel sortant**.

### 4.5 API
- `MarketReading.market_status` (calculé à chaque réponse, jamais persisté — il dépend de `now`).
- `GET /api/market-status?instrument&timeframe` : l'état serveur brut, lu par le front, le scanner
  et l'agent — jamais l'horloge du client.

### 4.6 Frontend
- `webapp/lib/market-reading/status.ts` : dérive l'état serveur (`deriveMarketStatus`) et le
  formate en heure de New-York (`formatNyTimestamp`). L'ancien `useMarketClosed` (horloge client,
  borne 22:00 UTC fixe = bug DST) n'est plus qu'un **repli** pour les surfaces sans état serveur.
- Badges : « Marché fermé » (week-end/férié), « Pause quotidienne » (daily_break),
  « Données en retard » (data_lagged), pouls « En direct » éteint hors `open`.
- Sous-ligne factuelle : « Dernière bougie clôturée {jour} {heure} (New York) · Réouverture … ».
- Scanner : note honnête « Aucune nouvelle bougie clôturée à analyser… » quand tout est périmé.
- M.I.A Agent : reçoit `market_status` dans le contexte + règle système « jamais décrire une
  structure comme récente/nouvelle quand l'état n'est pas `open` ».

---

## 5. Consommation d'API évitée

Free tier : **8 req/min, 800 req/jour**, cache fournisseur 60 s. Périmètre chaud : **6 combos**
(`SCAN_COMBOS`), tick **60 s**.

- **Avant** : le week-end, les 6 combos se croient périmés à chaque tick → jusqu'à 6 appels/min,
  plafonnés par le limiteur à **800 appels/jour**. Un week-end (~48 h, ven. 17:00 → dim. 17:00/18:00
  NY) brûle donc **≈ 1 600 appels Twelve Data** — l'enveloppe gratuite complète de deux journées —
  entièrement sur des données figées de vendredi, plus toute la re-narration LLM associée.
- **Après** : `market_aware_expected_close` gelée → **0 appel sortant** tout le week-end (vérifié
  par mock, `test_scheduler_tick_makes_zero_provider_calls_on_weekend`). S'ajoute la pause
  quotidienne XAU (~1 h/jour ouvré).

**Gain net : ≈ 1 600 appels Twelve Data / semaine évités** (le budget gratuit d'un week-end entier),
plus la charge LLM correspondante.

---

## 6. Tests (horloge figée, jamais l'heure système)

`tests/test_market_calendar.py` + `tests/test_mc1_market_closed_wiring.py` couvrent :
- samedi 12:00 NY → `closed_weekend` ; vendredi 16:59 → open / 17:01 → fermé ;
- dimanche 17:59 → fermé / 18:01 → open (XAU 18:00, EUR 17:00) ;
- une date en heure d'été (21:00 UTC) et une en hiver (22:00 UTC) → bon basculement ;
- férié inscrit → `closed_holiday` ; férié futur inconnu → **pas** supposé ouvert ;
- pause quotidienne XAU 17:00–18:00 → `daily_break` (EUR reste `open`) ;
- deux passages moteur sur bougies identiques → **identifiants identiques** ;
- marché fermé + relance → **aucun rebuild / aucune ré-émission** (pas de `save`) ;
- marché fermé → **aucun appel sortant** vers le fournisseur (mock) ;
- calendrier ouvert mais dernière bougie trop ancienne → `data_lagged`.

Frontend : `webapp/lib/market-reading/__tests__/status.test.ts` (dérivation + badges +
format NY + **honnêteté de la copy** : aucune formulation prédictive dans les nouvelles chaînes
fr + en).

`tsc` vert, build vert, suites back + front vertes (hors 2 smoke pré-existants dépendant de
`SENTINEL_TESTING_MODE` non lié à MC-1).

---

## 7. Décisions & limites

- **Horaires par instrument** dès maintenant (`INSTRUMENT_HOURS`) — ajouter un marché = une entrée.
  Le fait (âge des bougies) reste l'autorité, donc le système reste correct même si l'horaire
  configuré est légèrement faux.
- **Fériés** : `config/market_holidays.json` couvre 2026–2027 (Noël, Nouvel An — les fermetures
  pleines FX/métaux). À étendre annuellement ; au-delà, repli honnête sur `data_lagged`.
- **Pause quotidienne** : appliquée à XAU uniquement, libellé distinct « Pause quotidienne ».
- **Sonde de sécurité** : volontairement limitée aux fériés pour ne pas risquer de réintroduire de
  l'analyse le week-end si le fournisseur produit des bougies synthétiques. Les week-ends restent
  déterministes.
- **À valider live** (un week-end, ou horloge figée en semaine) avant merge — cf. §Discipline du brief.

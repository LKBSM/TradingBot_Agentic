# AUDIT NW-4 — Couverture et fiabilité du calendrier avant facturation

**Date :** 2026-08-01
**Branche :** `feat/nw-4-couverture-calendrier` (worktree dédié, depuis `origin/main` d90cbd3).
**Fondé sur :** `docs/audits/AUDIT-nw-d2-couverture.md`.
**Nature :** livraison. 6 chantiers, 6 commits. Merge sur main **seulement après confirmation live**.

---

## 0. LE CHIFFRE, EN PREMIER

Contre la liste de référence des **~29 publications déterminantes** (Or & EUR/USD) de NW-D2 :

| Catégorie | NW-D2 (avant) | NW-4 **aujourd'hui** (sans clés) | NW-4 **une fois les clés posées** |
|---|---:|---:|---:|
| **(a) PRÉSENTE ET COMPLÈTE** (valeur publiée + précédente) | **3** | **3** (les 3 EUR) | **11** |
| **(b) PRÉSENTE, valeur en attente / sans valeur par nature** | 10 | **11** | **3** |
| **(c) ABSENTE mais récupérable** | 8 | **6** | **6** |
| **(d) HORS DE PORTÉE** (organisme privé) | 8 | **8** | **8** |

**Deux lectures, parce que la vérité en a deux :**

- **Aujourd'hui, sans clés :** les 3 fiches européennes (IPCH flash, décision BCE, PIB flash EA)
  sont complètes — valeurs récupérées sans clé. Les 8 grandes publications américaines à valeur
  (NFP, IPC, IPC core, IPP, JOLTS via BLS ; PIB, PCE via BEA ; ventes, durables via Census) sont
  **présentes, datées, à l'heure exacte, avec organisme et unité** — mais leur **chiffre reste en
  attente** de la clé de leur organisme (état honnête `unfetched`, jamais un nombre inventé).
- **Une fois les 3 clés gratuites posées** (BLS, BEA, Census — cf. §2), ces 8 basculent en
  **complètes** : **11 publications complètes**, dont les plus déterminantes pour l'Or.

Le passage de 3 à 11 complètes ne demande **aucun code supplémentaire** — trois variables
d'environnement et une validation live.

---

## 1. CHANTIER 1 — Le drapeau de confiance qui mentait *(priorité absolue)*

### Audit des heures curées (les 13 événements portaient `time_confirmed=true`)

| # | Événement | Heure encodée | Heure officielle | Source consultée (2026-08-01) | Concordance |
|---|---|---|---|---|---|
| 1 | `us_employment_situation` | 08:30 ET | 08:30 ET | [BLS schedule](https://www.bls.gov/schedule/news_release/empsit.htm) | ✅ |
| 2 | `us_cpi` | 08:30 ET | 08:30 ET | [BLS CPI](https://www.bls.gov/news.release/cpi.htm) | ✅ |
| 3 | `us_ppi` | 08:30 ET | 08:30 ET | [BLS PPI](https://www.bls.gov/news.release/ppi.htm) | ✅ |
| 4 | `us_gdp` | 08:30 ET | 08:30 ET | [BEA schedule](https://www.bea.gov/news/schedule) | ✅ |
| 5 | `us_pce` | 08:30 ET | 08:30 ET | [BEA Personal Income & Outlays](https://www.bea.gov/news/schedule) | ✅ |
| 6 | `us_retail_sales` | 08:30 ET | 08:30 ET | [Census indicators](https://www.census.gov/economic-indicators/calendar-listview.html) | ✅ |
| 7 | `us_housing_starts` | 08:30 ET | 08:30 ET | Census indicators | ✅ |
| 8 | `us_durable_goods` | 08:30 ET | 08:30 ET | Census indicators | ✅ |
| 9 | `us_fomc_rate` | 14:00 ET | 14:00 ET (std. 2013) | [Fed FOMC calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | ✅ |
| 10 | **`ea_hicp_flash`** | **11:00** | **15:00 CET** | [Calendrier stat. BCE (sthicp)](https://www.ecb.europa.eu/press/calendars/statscal/ges/html/sthicp.en.html) | ❌ **corrigé → 15:00** |
| 11 | `ea_gdp_flash` | 11:00 | 11:00 (standard, non ancré) | [Eurostat euro-indicators](https://ec.europa.eu/eurostat/en/web/main/news/euro-indicators) | ⚠️ **→ `time_confirmed=false`** |
| 12 | `ea_unemployment` | 11:00 | 11:00 (standard, non ancré) | Eurostat euro-indicators | ⚠️ **→ `time_confirmed=false`** |
| 13 | `ea_ecb_rate` | 14:15 CET | 14:15 CET (depuis 21/07/2022) | [BCE — changement 2022](https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.pr220627~73acedf868.en.html) | ✅ |

**La cause racine, plus grave que l'heure :** les 3 événements Eurostat portaient le même
`time_source_url` (`.../about-us/impartiality-protocol`) — une page de gouvernance **qui ne
documente aucune heure**. Le drapeau « vérifié à la source » était posé sur une hypothèse.

### Correctifs (commit 441d664)
- `ea_hicp_flash` → **15:00**, `time_source_url` = calendrier statistique BCE, re-vérifié.
- `ea_gdp_flash` / `ea_unemployment` → **`time_confirmed=false`** (11:00 supposé, non ancré).
- **Garde** : `load_catalog` **rétrograde** tout `time_confirmed=true` dépourvu de `time_source_url`
  **et** `time_last_verified` → *un drapeau de confiance sans preuve devient impossible*. Test sur
  le catalogue livré + downgrade au chargement.
- **Affichage** : une heure non confirmée est **visiblement distincte** d'une heure vérifiée
  (marqueur « heure non confirmée » en vues liste, détail, mois — grille + panneau — et aperçu).

---

## 2. CHANTIER 2 — Débloquer les valeurs américaines *(commit 332ef7c)*

### Clés à créer (gratuites — tu les poses en dashboard Render, aucune dans le dépôt)

| Variable | Créer la clé | Débloque |
|---|---|---|
| `BLS_API_KEY` | https://data.bls.gov/registrationEngine/ | NFP, IPC, IPC core, IPP, JOLTS |
| `BEA_API_KEY` | https://apps.bea.gov/API/signup/ | PIB, PCE |
| `CENSUS_API_KEY` | https://api.census.gov/data/key_signup.html | ventes au détail, biens durables, mises en chantier |

**Ajoutées à `render.yaml` en `sync:false` et documentées dans `.env.example`.**

- **`values/bea_values.py`** (nouveau) — API NIPA GetData. Débloque PIB (T10101, ligne 1) et PCE
  (T20804, ligne 1). Ligne headline auditable ; série inconnue → `None` (jamais une ligne devinée).
- **`values/census_values.py`** (nouveau) — API EITS. Débloque ventes / durables / mises en chantier.
- **`build_value_fetcher`** — BEA/Census enregistrés **seulement si leur clé est présente** ; sinon
  la source reste honnêtement `unfetched`.

**Découverte qui corrige NW-D2 :** l'API Census EITS **exige désormais une clé** même pour les
métadonnées (page « Missing Key », vérifié 2026-08-01). L'audit la croyait sans clé — Census est en
fait key-gated comme BLS/BEA. Corrigé dans le code et ici.

**Garanties communes :** source injoignable / forme inattendue / page « Missing Key » → `None`
(état `unfetched`, cache jamais détruit) ; valeur **telle que publiée** (virgule de milliers retirée,
aucune conversion, aucun réarrondi) ; valeur précédente ≠ révisée (distinction au niveau du store).

> **À valider live après pose des clés** (avant merge) : les codes de dimension Census pour les
> mises en chantier (RESCONST) et les biens durables (ADVM3) sont les meilleurs connus mais non
> validables sans clé ; le code retail (MARTS 44X72/SM) est documenté. Le premier run live confirme.

---

## 3. CHANTIER 3 — La limite des événements non programmés *(commit d6bcdc7)*

Le produit n'écrivait nulle part qu'il ne couvre que le programmé. Ajouté comme **3ᵉ item** de
`reading.calendar.nono`, **traduit nativement dans les 9 locales**, visible sur **toutes** les vues
(liste **et** mois) :

- **FR :** « Ce calendrier annonce des moments programmés. Une décision hors calendrier, un choc
  géopolitique ou une déclaration inattendue n'y figurent pas : ils n'ont pas d'heure connue à
  l'avance. »
- **EN :** « This calendar announces scheduled moments. An unscheduled decision, a geopolitical
  shock or an unexpected statement will not appear here: they have no time known in advance. »

Test de copy : phrase présente en fr et en ; le scan d'honnêteté existant la couvre (aucun verbe de
direction).

---

## 4. CHANTIER 4 — Publications officielles absentes *(commit 6b88db9)*

Ajoutés (chômage hebdo DOL/ETA **différé** sur ta décision) :

| Événement | Organisme | Statut | Note |
|---|---|---|---|
| **JOLTS** (`us_jolts`) | BLS | daté 2026-08-04, **10:00 ET** (≠ 08:30), valeur via BLS | ouvertures de postes |
| **IPC sous-jacent** (`us_cpi_core`) | BLS | daté (parution IPC), série propre `CUUR0000SA0L1E` | co-daté avec le headline via `match_ics_keys` |
| **Projections / dot plot** (`us_fomc_dotplot`) | Fed | daté 2026-09-16 + 2026-12-09 (14:00 ET) | réunions trimestrielles, **sans valeur** |
| **Procès-verbaux FOMC** (`us_fomc_minutes`) | Fed | **non daté** (14:00 ET), **sans valeur** | la Fed ne publie pas encore les dates 2026 → **signalé non datable**, jamais une date inventée |

- **`match_ics_keys`** (tous les matchs) : un VEVENT « Consumer Price Index » date le headline **et**
  le core ; le PPI reste désambiguïsé.
- **Invariant de forme relâché honnêtement** : l'unité n'est exigée que pour un événement **avec
  série** (mesurable) ; un événement « moment seul » (minutes, dot plot) n'en a pas — jamais d'unité
  bidon.

---

## 5. CHANTIER 5 — Deux défauts de l'audit *(commit 721ea9c)*

- **5A — événement non datable, jamais silencieux.** `OfficialCalendarProvider.undatable_events()`
  **signale** (retourné + loggé au fetch) tout événement qu'aucune source ne peut dater, au lieu de
  le laisser disparaître. Cas actuels : **`ea_unemployment`** (Eurostat sans `ics_feed` + sans date
  curée) et **`us_fomc_minutes`** (dates Fed non encore publiées). Règle générale posée + testée.
- **5B — preuve de fraîcheur permanente.** Ligne « **Dernière mise à jour : {date}** » (max des
  `last_success`) affichée **en permanence** en vue liste **et** vue mois — le client voit la preuve
  de fraîcheur, pas seulement l'absence de drapeau « périmé ». 9 locales.

---

## 6. CHANTIER 6 — Extension à un nouveau marché *(documentation seule, commit fee6609)*

`docs/architecture/calendar-market-extension.md` :
- Rattachement = **configuration pure** (`config/event_market_map.json`, devise → marché).
- Ajouter un marché mû par une devise déjà couverte = **1 ligne JSON**.
- **Bitcoin** : rattacher les publications US existantes à `BTCUSD` = **1 ligne, zéro duplication**
  (le champ `markets` est plusieurs-à-plusieurs). Échéances **SEC** = config + petit adaptateur
  (événement sans valeur, déjà toléré).
- **Halving** = le **seul angle mort** : événement **sans émetteur**, hors du modèle « un organisme
  publie une valeur ». Coût d'un type de source « protocole » (attribution = la chaîne) chiffré (~2,5 j).

---

## Les absentes qui restent (par effort, après décisions)

1. **Inscriptions hebdo au chômage (DOL/ETA)** — **différé sur ta décision** (hebdomadaire → nouvel
   organisme source + volume ×4).
2. **IPC allemand flash (Destatis)**, **conférences de presse FOMC/BCE distinctes**, **auditions**,
   **Jackson Hole** — **exclus sur ta consigne** (« NE FAIS PAS »).

Ces 6 restent en catégorie (c) : officielles, récupérables, non implémentées par choix.

---

## Défauts de qualité — état

| Défaut NW-D2 | État NW-4 |
|---|---|
| IPCH flash à 11:00 marqué confirmé | ✅ corrigé 15:00 + garde anti-drapeau-sans-preuve |
| Valeurs US absentes (pas de récupérateur BEA/Census) | ✅ récupérateurs écrits ; clés à poser |
| Limite « non programmé » non écrite | ✅ phrase ajoutée, 9 locales, toutes vues |
| `ea_unemployment` disparaît en silence | ✅ signalé (diagnostic + log) |
| Pas de preuve de fraîcheur quand tout va bien | ✅ « Dernière mise à jour » permanente |
| Aucun chiffre inventé (modèle 4 états) | ✅ préservé et étendu (BEA/Census défensifs) |

---

## Vérifications

- **tsc** : clean. **build** (`next build`) : vert (toutes routes, dont `/actualites` + `/actualites/[eventId]`).
- **pytest calendrier** : 110 verts (providers, service, values, schedule, endpoint, ics, garde
  temps, fetchers, cache, nw3).
- **vitest calendrier** : verts (mois, aperçu, détail, copy-honnêteté). 3 échecs **pré-existants sur
  `main`** dans `market-reading` (« Marché en range » — sans lien avec le calendrier, reproduits sur
  `wt-run-main`).
- **Playwright** : à exécuter en environnement live (backend + `next dev`) — 4 scénarios : calendrier
  avec valeurs, source injoignable, fiche à heure non confirmée, fiche sans valeur.

---

## AVIS FRANC — un trader rate-t-il encore des publications qui comptent ?

**Aujourd'hui, avant que tu poses les clés : partiellement, mais plus pour longtemps.** Les 3 fiches
européennes sont complètes ; les 8 grandes publications américaines sont **présentes, à l'heure
exacte, avec organisme et unité**, mais leur **chiffre attend la clé** de leur organisme. Un trader
Or voit donc le *moment* exact de l'IPC, du NFP, du PCE — le chiffre remonte dès la clé posée.

**Une fois les 3 clés gratuites posées : non, pour l'essentiel.** 11 publications complètes, dont
toutes les grandes américaines qui pilotent l'Or, plus JOLTS, l'IPC sous-jacent, le dot plot et les
procès-verbaux (ces derniers dès que la Fed publie leurs dates). Restent hors couverture, **par
décision assumée** : le chômage hebdo (différé), l'IPC allemand, les conférences de presse
distinctes, les auditions, Jackson Hole — et, structurellement, tout ce qui est **privé** (ISM,
Conference Board, Michigan, PMI, ifo, ZEW) ou **non programmé** (désormais **écrit** noir sur blanc).

Le socle était honnête ; il est maintenant **complet à la hauteur de la promesse**, à trois variables
d'environnement près. Le drapeau qui mentait a été éteint, et rien n'affiche plus un chiffre faux ni
une heure supposée sans le dire.

---

*Fin de l'audit NW-4. Merge sur main uniquement après confirmation live.*

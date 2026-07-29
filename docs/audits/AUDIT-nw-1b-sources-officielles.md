# AUDIT NW-1b — Intégration des sources officielles

_Calendrier de volatilité « Actualités programmées » — branchement des organismes
officiels, suppression du consensus et du classement d'impact, révisions visibles._

Date : 2026-07-28 · Branche : `feat/nw-1b-sources-officielles`

---

## 0. Décisions structurantes

1. **Le consensus disparaît.** Aucun organisme public ne publie de prévision
   d'analystes (sondage privé et payant). Le champ `forecast`, sa carte, ses
   chaînes i18n et ses tests sont retirés. Aucune estimation de substitution.
2. **Le classement d'impact disparaît.** Aucun organisme ne grade ses
   publications (élevé/moyen/faible est une convention d'agrégateur). Le champ
   `impact`, le filtre et le badge sont retirés. Les filtres deviennent
   factuels : **organisme, marché, périodicité**.
3. **Les révisions deviennent visibles.** Le schéma porte la valeur
   initialement publiée (`actual_initial`), la valeur courante (`actual`) et la
   date de révision (`revised_at`). Une valeur jamais révisée l'indique.
4. **FRED / ALFRED écarté comme source runtime.** Sa politique (MAJ juin 2024)
   interdit l'usage pour l'entraînement de LLM ET la mise en cache/archivage —
   incompatible avec un produit cache-first et LLM-driven. FRED ne sert que de
   référence de spécification ; valeurs et vintages viennent des sources
   originales de domaine public et des API européennes.

---

## 1. Tableau comparatif par organisme

| Rubrique | BLS | BEA | US Census | Federal Reserve | Eurostat | BCE / ECB |
|---|---|---|---|---|---|---|
| **API valeurs** | `api.bls.gov/publicAPI/v2` (POST JSON) | `apps.bea.gov/api/data` (GET JSON) | `api.census.gov/data/timeseries/eits` (JSON) | H.15 / NY Fed API (pas d'API FOMC) | REST JSON-stat 2.0 + SDMX 2.1 | SDMX 2.1 `data-api.ecb.europa.eu` |
| **Auth** | clé gratuite | UserID gratuit | clé gratuite (>500 req/j/IP) | aucune | aucune | aucune |
| **Débit** | 500 req/j, 50 séries/req | 100 req/min | 500 req/j/IP sans clé | n/a (flux) | 50 catégories/req | non publié |
| **Calendrier par API** | Non — `.ics` (`bls.ics`) | Oui (JSON) + `.ics` | Non — page/PDF + RSS `indicator.xml` | Non — HTML + RSS `press_monetary.xml` | Non — `.ics` release-calendar | Non — HTML calendrier GC |
| **Heure via API** | Non ; 8h30 ET (doc.) | Partiel ; 8h30 ET | Non ; 8h30 ET | Non ; 14h00 ET | Non ; 11h00 CET | Non ; 14h15 CET |
| **ID série stable** | CUUR0000SA0, CES0000000001, LNS14000000, WPSFD4 | NIPA T10101, T20804 | MARTS, RESCONST, ADVM3 | URL `monetary…a` | prc_hicp_manr, namq_10_gdp, une_rt_m | FM.D.U2.EUR.4F.KR.MRR_FR.LEV |
| **Unité déclarée** | indice / % / milliers / $·h⁻¹ | % / indice Fisher | M$ / milliers SAAR | fourchette % | % / indice / M€ | `PCPA` (% par an) |
| **Révisions** | API = dernière valeur ; vintages en fichiers | API = dernière ; pas de vintage | advance vs final = séries séparées | statements versionnés (URL) | triangles de révision (téléch.) | `includeHistory=true` (API) |
| **Licence** | Domaine public (17 U.S.C. §105) | Domaine public §105 | Domaine public §105 + non-endorsement | Domaine public | **CC BY 4.0** (attribution obligatoire) | **Réutilisation si citée ET non modifiée** |

Vintages requêtables US = uniquement via ALFRED/FRED (écarté). Côté euro,
Eurostat et BCE exposent leurs vintages nativement.

---

## 2. Licences — texte et URL par organisme

| Organisme | Libellé de licence (affiché par enregistrement + bloc d'attribution) | URL de politique |
|---|---|---|
| **BLS** | Domaine public (17 U.S.C. §105) — citation : U.S. Bureau of Labor Statistics | https://www.bls.gov/opub/copyright-information.htm |
| **BEA** | Domaine public (17 U.S.C. §105) — citation : U.S. Bureau of Economic Analysis | https://www.bea.gov/help/faq/147 |
| **Census** | Domaine public (17 U.S.C. §105) — non approuvé ni certifié par le Census Bureau | https://www.census.gov/data/developers/about/terms-of-service.html |
| **Federal Reserve** | Domaine public — citation : Federal Reserve Board | https://www.federalreserve.gov/disclaimer.htm |
| **Eurostat** | CC BY 4.0 — attribution obligatoire : Source: Eurostat | https://ec.europa.eu/eurostat/help/copyright-notice |
| **BCE** | Réutilisation libre si la source est citée et les statistiques non modifiées — « Source: ECB » | https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html |

Textes verbatim clés :
- **Eurostat** : « Reuse … for commercial or non-commercial purposes is authorised
  provided the source is acknowledged. » (CC BY 4.0)
- **BCE** : « All publicly available ESCB statistics may be reused free of charge
  on the condition that the source is quoted … and that the statistics (including
  metadata) are not modified. » → d'où l'affichage **tel que publié, sans
  conversion ni réarrondi** (`asPublished()` = `String(valeur)`, aucune
  `toLocaleString` qui arrondirait).

Le bloc d'attribution (`CalendarAttribution`) nomme chaque source servie et lie
sa politique ; un test échoue si un enregistrement est rendu sans son attribution.

---

## 3. Verdict sur les heures — origine par événement

Aucun organisme n'expose l'heure par l'API de données ; elle vient du flux `.ics`
(BLS/BEA/Eurostat) ou d'une page officielle stable (Census/FOMC/BCE). Toutes sont
documentées et stables → aucune n'est approximée. La table d'heures est **figée,
versionnée et auditable** dans `config/calendar_catalog.json` (champs
`release_time_local`, `source_timezone`, `time_source_url`, `time_last_verified`).
Une publication dont l'heure ne serait pas confirmée porte `time_confirmed=false`,
est marquée « heure non confirmée » et **exclue des mesures d'amplitude** (NW-2).

| Événement | Heure officielle | Fuseau (DST) | Source de l'heure |
|---|---|---|---|
| BLS — Emploi / IPC / IPP | 8h30 | America/New_York | en-tête communiqué « 8:30 a.m. (ET) » + `bls.ics` |
| BEA — PIB / PCE | 8h30 | America/New_York | embargo « 8:30 a.m. EDT » + schedule |
| Census — retail / housing / durables | 8h30 | America/New_York | PDF « FOR RELEASE AT 8:30 A.M. » |
| FOMC — décision | 14h00 | America/New_York | communiqué 2013 (stable depuis) |
| Eurostat — HICP / PIB / chômage | 11h00 | Europe/Luxembourg | protocole d'impartialité (« 11 am CET ») |
| BCE — décision | 14h15 | Europe/Berlin (CET) | communiqué 2022-06-27 (stable depuis 21/07/2022) |

En l'état, les 13 événements couverts sont tous `time_confirmed=true`.

---

## 4. Événements couverts (13)

| Clé | Organisme | Série | Unité | Périodicité | Marchés | Heure |
|---|---|---|---|---|---|---|
| us_employment_situation | BLS | CES0000000001 | milliers d'emplois | mensuel | Or, EUR/USD | 8h30 ET |
| us_cpi | BLS | CUUR0000SA0 | indice (1982-84=100) | mensuel | Or, EUR/USD | 8h30 ET |
| us_ppi | BLS | WPSFD4 | indice (2009=100) | mensuel | Or, EUR/USD | 8h30 ET |
| us_gdp | BEA | NIPA-T10101 | % | trimestriel | Or, EUR/USD | 8h30 ET |
| us_pce | BEA | NIPA-T20804 | indice de prix | mensuel | Or, EUR/USD | 8h30 ET |
| us_fomc_rate | Federal Reserve | (URL décision) | fourchette % | 8×/an | Or, EUR/USD | 14h00 ET |
| us_retail_sales | Census | MARTS-RSAFS | M$ | mensuel | Or, EUR/USD | 8h30 ET |
| us_housing_starts | Census | RESCONST-HOUST | milliers SAAR | mensuel | Or, EUR/USD | 8h30 ET |
| us_durable_goods | Census | ADVM3-DGORDER | M$ | mensuel | Or, EUR/USD | 8h30 ET |
| ea_hicp_flash | Eurostat | prc_hicp_manr | % a/a | mensuel | EUR/USD | 11h00 CET |
| ea_gdp_flash | Eurostat | namq_10_gdp | % t/t | trimestriel | EUR/USD | 11h00 CET |
| ea_unemployment | Eurostat | une_rt_m | % | mensuel | EUR/USD | 11h00 CET |
| ea_ecb_rate | BCE | FM.D.U2.EUR.4F.KR.MRR_FR.LEV | % par an | 8×/an | EUR/USD | 14h15 CET |

Rattachement marché : règle existante `config/event_market_map.json`
(USD → Or + EUR/USD ; EUR → EUR/USD).

---

## 5. Architecture livrée

- **Schéma** (`calendar_schema.py`) : retrait `impact`/`forecast`/
  `previous_before_revision` ; ajout `periodicity`, `time_confirmed`,
  `actual_initial`, `revised_at` ; ajout `CalendarAttribution` +
  `coverage.last_success`/`stale_sources`.
- **Interface provider** (`base.py`) : `ProviderEvent` neutre (sans impact/
  consensus) + `ProviderAttribution` par source. Aucun format propre à une
  source ne franchit la frontière.
- **Un adaptateur par organisme** (`official_sources/organisms.py`) : `BLSProvider`,
  `BEAProvider`, `CensusProvider`, `FederalReserveProvider`, `EurostatProvider`,
  `ECBProvider`, partageant `OfficialSourceProvider` (jointure catalogue +
  planification DST-safe + attribution). Composés par `OfficialCalendarProvider`
  (agrégateur, source par défaut « official »).
- **Catalogue** (`config/calendar_catalog.json`) : 13 événements, auditable ligne
  par ligne (série, organisme, licence, unité, fuseau, heure + URL + vérif).
- **Planning** (`config/calendar_schedule.json`) : dates datées, auditables,
  livré **vide** (aucune date fabriquée). Le seam de date est une seule fonction
  injectable : brancher un feed live (`.ics` / API données) ne touche que
  l'adaptateur.
- **Store v2** (`calendar_cache_store.py`) : migration reconstruisant la table
  (cache régénérable) ; logique de révision initiale-vs-courante datée ;
  `source_last_success()`.
- **Service** : bloc d'attribution (sources réellement servies), fraîcheur par
  source, échec provider → cache conservé.
- **Front** : filtres organisme/marché/périodicité ; révisions à l'écran ;
  bloc d'attribution (liste + liens de politique) ; marque « heure non
  confirmée » ; bloc « ce que cette page ne dit pas » + 2 lignes (pas de
  prévision, pas de classement) ; valeurs affichées telles que publiées.

---

## 6. Révisions — modèle

Cycle : événement programmé (`actual=None`) → première publication (`actual=X`,
`actual_initial=X`, `revised=false`) → révision (`actual=Y≠X`, `actual_initial`
inchangée, `revised=true`, `revised_at=<date>`). Rendu : « Valeur initiale X,
révisée le <date> — valeur actuelle Y ». Aucune qualification (« importante »,
« à la hausse », « surprise »). Une valeur jamais révisée : « Ce chiffre n'a pas
été révisé depuis sa première publication ». Le schéma NW-1 suffit augmenté de
`revised_at` + `actual_initial` ; pas de table de vintages séparée nécessaire au
périmètre visé.

---

## 6bis. Données réelles peuplées — prêt pour les clients

Le planning `config/calendar_schedule.json` est **peuplé de 23 parutions
vérifiées** sur les pages officielles (2026-07-28), couvrant les 9 événements aux
plus gros impacts FX/or. La page affiche donc de vraies parutions dès le
déploiement (le provider par défaut n'est plus vide). Fenêtre front élargie à
30 jours (cap API).

| Événement | Dates vérifiées (Aoû–Déc 2026) | Source de la date |
|---|---|---|
| Emploi US (NFP) | 07/08 | bls.gov/schedule/news_release/empsit.htm |
| IPC (CPI) | 12/08 | bls.gov/schedule/news_release/cpi.htm |
| PIB (GDP) | 26/08, 30/09, 29/10 | bea.gov/news/schedule |
| Revenus/dépenses (PCE) | 26/08, 30/09, 29/10 | bea.gov/news/schedule |
| Décision FOMC | 16/09, 28/10, 09/12 | federalreserve.gov/…/fomccalendars.htm |
| Ventes au détail | 14/08, 16/09, 15/10 | census.gov/economic-indicators/calendar-listview.html |
| Mises en chantier | 18/08, 17/09, 20/10 | census.gov/…/calendar-listview.html |
| Biens durables | 26/08, 25/09, 27/10 | census.gov/…/calendar-listview.html |
| Décision BCE | 10/09, 29/10, 17/12 | ecb.europa.eu/press/calendars/mgcgc |

**Non couverts (dates laissées absentes, jamais fabriquées)** : `us_ppi` (calendrier
BLS perturbé par le *lapse in appropriations* 2025-26), `ea_hicp_flash`,
`ea_gdp_flash`, `ea_unemployment` (calendrier Eurostat servi en JS/PDF, non
extractible de façon fiable). Ces types d'événements restent dans le catalogue ;
leurs dates seront ajoutées à la prochaine vérification ou par le feed `.ics`.

**Maintenance** : chaque ligne porte `last_verified`. Les calendriers officiels
bougent (reports de shutdown) → re-vérification périodique nécessaire. Le seam
`date_source` injectable permet de brancher un **feed live `.ics`/API** par
source (auto-rafraîchi) sans toucher au reste — étape d'automatisation suivante.

## 7. Hors couverture (assumé)

- **Feed live réseau auto-rafraîchi** : le seam `date_source` est en place et
  testé ; le branchement HTTP/`.ics` par source (BLS/BEA/Eurostat) + clés BLS/BEA
  automatiserait le rafraîchissement (aujourd'hui : planning curé versionné,
  re-vérifié manuellement). BLS bloque le fetch bot (403) — à valider serveur.
- **PPI + trio Eurostat** : dates à vérifier (voir 6bis).
- **Valeurs/amplitude d'historique moteur** : « mesures à venir » (NW-2).
- **Aperçu /app** : `CalendarPreview` prêt, placement sur /app différé (comme NW-1).
- **Vintages ALFRED** : écartés (politique FRED). Vintages euro exploitables
  directement (Eurostat triangles, BCE `includeHistory`).

---

## 8. Vérifications

- Back : 52 tests calendrier verts (providers, service, store, endpoint, schedule),
  dont l'intégration HTTP avec le **vrai** provider par défaut + le planning réel.
- Front : tests calendrier verts (workspace, detail, copy-honesty), suite front complète.
- i18n : fr + en complets, parité stricte 9 locales (7 autres = fallback EN),
  aucune chaîne prédictive (garde copy-honesty), clés impact/consensus supprimées.
- tsc 0, build vert, Playwright.

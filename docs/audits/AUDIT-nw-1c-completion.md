# AUDIT NW-1c — Complétion de la page « Actualités programmées »

Date : 2026-07-29 · Branche : `feat/nw-1c-completion-actualites`

But : rendre la page montrable à un client payant (Section 1) + compléter les
valeurs et leurs états (Section 3). Rappel de ligne : pas de consensus, pas de
classement d'impact ; la page annonce des MOMENTS, jamais des DIRECTIONS.

---

## Section 1 — les 4 défauts retirés (commit 36f5245)

- **A) Verbe de causalité** — « affecte {markets} » → « rattaché à {markets} »
  (« attached to » en), 9 locales. Le rattachement est une convention
  d'affichage, pas une cause. Test d'honnêteté : échoue si un verbe de causalité
  (affecte / impacte / influence / agit sur / joue sur / pèse sur) lie un
  événement à un marché.
- **B) Référence interne** — « (NW-2) » retiré. Test : échoue si une chaîne i18n
  contient un code de mission/ticket (hors clés `_comment`, non rendues).
- **C) Cartes/slots vides** — carte « Ce que le moteur a mesuré » non rendue tant
  qu'aucune mesure n'existe (composant `EngineMeasuresCard`, rend `null`) ; slot
  amplitude de la liste retiré ; ligne amplitude retirée du bloc « ne dit pas ».
- **D) Unité réelle** — PCE « indice de prix (PCE) » (description) → « indice
  (2017 = 100) » (base déclarée par le BEA pour NIPA-T20804).

**Autres références internes signalées (non touchées)** : clés `_comment` des
fichiers de messages (noms de fichiers dev, jamais rendues) ; une réponse
scriptée du chatbot emploie « affected » à propos de la VOLATILITÉ (amplitude des
bougies), pas d'un lien événement→prix — contenu légalement porteur, laissé
verbatim.

---

## Section 3 — état du flux de valeurs, par organisme

| Organisme | Série | Liaison | Flux de valeurs |
|---|---|---|---|
| BLS (emploi, CPI, PPI) | CES0000000001 / CUUR0000SA0 / WPSFD4 | stable | Adaptateur à écrire (clé v2 gratuite) → aujourd'hui `unfetched` |
| BEA (PIB, PCE) | NIPA-T10101 / T20804 | stable | Adaptateur à écrire (UserID gratuit) → `unfetched` |
| Census (retail, housing, durables) | MARTS/RESCONST/ADVM3 | stable | Adaptateur à écrire (clé gratuite) → `unfetched` |
| Federal Reserve (FOMC) | — | **aucune série** | `unavailable` (décision = fourchette dans le communiqué, pas une série) |
| Eurostat (HICP, PIB, chômage) | prc_hicp_manr / namq_10_gdp / une_rt_m | stable | Adaptateur à écrire (sans clé) — dates non couvertes |
| **BCE (taux)** | FM.D.U2.EUR.4F.KR.MRR_FR.LEV | stable | **Implémenté + validé réseau** (SDMX-JSON sans clé, taux MRO 2,4) |

**Verdict** : le flux de valeurs est **un seam complet + un fetcher réel (BCE)**.
Le fetcher se branche par `series_code` (jamais par titre), opt-in
`CALENDAR_VALUES_LIVE=1` (défaut OFF pour le déterminisme), gracieux (un organisme
injoignable ne détruit rien → l'événement reste `unfetched`). Les adaptateurs
BEA/BLS/Census (clés gratuites) et Eurostat s'enregistrent dans le même registre
quand leurs clés sont fournies.

**Aujourd'hui, toutes les parutions du planning sont FUTURES** → toutes en état
`pending` (« non encore publiée »), ce qui est correct. Les valeurs se
rempliront à mesure que les dates passent (et que les adaptateurs par clé sont
branchés).

### Liaison événement → série
12/13 événements portent un `series_code` stable ; **`us_fomc_rate` n'en a pas**
(décision = fourchette, pas de série) → listé à part, jamais rattaché « au jugé ».
La liaison DATE→événement du feed `.ics` reste par titre (fragile) — signalé.

### Les trois absences (cœur de la mission) — distinctes dans le MODÈLE
`compute_value_state(series, actual, scheduled_at, now)` → 4 états :
`published` · `pending` (futur) · `unfetched` (passé, série, sans valeur — état
PRODUIT, avec date du dernier essai) · `unavailable` (pas de série récupérable,
organisme nommé). Distincts dans le schéma ET à l'écran (détail + liste). Une
valeur non récupérée n'est **jamais** confondue avec une valeur inexistante.

### Révisions
Valeur initiale + valeur actuelle affichées ensemble quand elles diffèrent, avec
la date. Jamais révisée → l'indique. Aucune qualification. Détection par
snapshot-au-relâchement (une valeur qui change au re-fetch est marquée révisée,
l'initiale préservée) ; vintages natifs pour BCE (`includeHistory`) / Eurostat.

### Fraîcheur visible — seuil retenu : **24 h**
Date du dernier rafraîchissement réussi affichée **par organisme** dans le bloc
d'attribution ; au-delà de 24 h (ou si l'organisme a raté le dernier cycle), il
est marqué « non rafraîchi depuis le <date> ». Un retard n'est jamais silencieux.
Un organisme injoignable conserve ses données (l'upsert ne supprime jamais).

---

## Section 4 — revue « prête pour la commercialisation »

| Critère | État |
|---|---|
| Aucune référence interne visible | ✅ (test i18n mission-code) |
| Aucune promesse de fonctionnalité future | ✅ (amplitude/mesures retirées ; test « à venir ») |
| Aucune case vide sans explication | ✅ (3 états + unité/organisme « non fourni ») |
| Aucun verbe de causalité événement→marché | ✅ (test causalité) |
| Chaque valeur porte organisme + unité | ✅ |
| Attribution de chaque source + lien de politique | ✅ (test) |
| États vides des filtres explicites | ✅ |
| Fonctionne sans aucun événement dans la période | ✅ (e2e période vide) |
| fr + en complets, aucune clé manquante/en dur | ✅ (parité 9 locales, test « no raw key ») |
| Rendu correct en 390 px | ✅ (e2e 390, overflow ≤ 1) |

### Ce qui reste imparfait (franchement)
1. **Valeurs live** : seul le fetcher **BCE** est implémenté+validé (sans clé).
   BEA/BLS/Census (clés gratuites) + Eurostat restent des seams à écrire →
   leurs événements passés afficheront `unfetched` jusque-là. (Sans effet visible
   aujourd'hui : tout est `pending`.)
2. **Dates Eurostat** (HICP/PIB/chômage) toujours absentes (calendrier JS) →
   via feed `.ics` euro-indicators (URL à confirmer) ou vérification manuelle.
3. **7 locales non-fr/en** : le namespace `calendar` est en **repli anglais**
   (pas de traduction native de/es/it/pt/nl/pl/ar).
4. **« previous » sur séries en escalier** (taux) = avant-dernière observation,
   souvent égale à l'actuelle ; révisions US par snapshot (pas de vintages ALFRED,
   écarté). 
5. **FOMC `unavailable`** : la Fed publie une fourchette (numérique) ; on la
   qualifie « pas de valeur chiffrée unique » — défendable mais nuancé.
6. **Feed `.ics` BLS** : bloque le bot (403) → repli sur le planning curé (les
   valeurs ne changent pas ; seules les dates viennent du curé pour BLS).

---

## Vérifications
- Back : 73 tests calendrier verts (providers, service, store, endpoint, schedule,
  ics, **values** : 4 états, enrichisseur, révision, seuil 24 h, parseur ECB).
- Front : 693 tests verts (dont calendrier : 3 états, liste, fraîcheur).
- Playwright 1280×800 + 390×844 : détail avec valeur, détail sans valeur
  (pending/unfetched/unavailable), détail avec révision, liste, filtres vides,
  **période sans événement**, fraîcheur.
- tsc 0, build vert. i18n fr+en complets, parité 9 locales.

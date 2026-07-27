# AUDIT RG-1 — Panneau « Régime de marché » enrichi (Phase 1 : diagnostic lecture seule)

> Branche : `feat/rg-1-regime-enrichi` (worktree `wt-rg-1`, depuis `origin/main` `c0cdd4a` — **MC-1 mergé, PR #83**).
> Frontend uniquement. Le moteur de détection n'est PAS modifié. **Aucun code produit avant GO.**
> Cible visuelle : `docs/design/reference-desktop.html` v9 (posée dans ce worktree — `pxbtn`/`lvlmini` présents, 16 occurrences).

Ce document est le livrable du **STOP** : tableau des dix mesures, réponse sur le journal d'événements,
réponse sur le tracé de niveau, fuseau retenu pour le « jour », et les conflits qui exigent une décision.

---

## 0. Sources vérifiées (fichiers réels sur `origin/main`)

| Donnée | Fichier | Champ / fonction |
|---|---|---|
| Contrat API MarketReading | `src/intelligence/market_reading_schema.py` | `MarketReadingRegime`, `MarketReadingStructure` |
| Dérivations regime | `src/intelligence/market_reading_mappers.py:1251-1333` | `_derive_trend`, `_derive_volatility`, `_derive_market_phase`, `candles_to_regime` |
| Pools de liquidité (bornes) | `src/intelligence/market_reading_mappers.py:754-888` | `collect_liquidity_pools` → `range_high`/`range_low` |
| Calendrier marché (MC-1) | `src/intelligence/market_calendar.py` | `calendar_state`, `compute_market_status`, `INSTRUMENT_HOURS`, `next_open` |
| Jours fériés (MC-1) | `config/market_holidays.json` | versionné |
| Helpers regime front (existants) | `webapp/lib/market-reading/regime-facts.ts` | `deriveTrendMaturity`, `formatLastStructuralEvent`, `countActiveZones` |
| UI regime actuelle | `webapp/components/market-reading/sections/RegimeSection.tsx` | accordéon (mobile) ; tuiles = `DesktopReading` |
| Contrat view-actions (verrou d'id) | `webapp/lib/chart/viewActions.ts` | `coerceViewAction`, `GEOMETRY_KEYS`, `ChartViewState` |
| Bougies côté client | `webapp/lib/market-reading/api-client.ts:170+` | `fetchCandles` (`GET /api/candles`, cap 500) |

---

## 1. Journal d'événements — RÉPONSE : **il existe.**

`MarketReadingStructure` expose **l'historique** des cassures, pas seulement la dernière :

```python
bos_events:   list[BOSRecent]   = Field(default_factory=list)  # most-recent first, capé
choch_events: list[CHOCHRecent] = Field(default_factory=list)
```
Chaque entrée porte `direction`, `level` (niveau réellement franchi), `broken_at` (horodatage ISO-8601 UTC),
`validation_status`. Commentaire du schéma (audit 2026-06-16 « sous-surfaçage ») : 88 BOS / 40 CHOCH détectés sur
6 combos, ≤1 surfacé auparavant via `bos`/`choch`. **Les listes portent le niveau réel + l'horodatage honnête de
chaque cassure, lus depuis les colonnes d'événements du moteur, jamais recalculés.**

**Conséquence** : Maturité (§6) et Dernier événement (§7) disposent bien de leur moitié « journal ».
`regime-facts.ts:deriveTrendMaturity` consomme déjà `choch_events`. **Aucune mission moteur séparée n'est requise
pour le journal.**

---

## 2. Tracé d'un niveau de référence — RÉPONSE : **chemin séparé obligatoire, le verrou n'est pas touché.**

Le « verrou d'id » est `coerceViewAction` (`viewActions.ts`). Sa ligne inviolable, documentée en tête de fichier :

> · Le vocabulaire d'action est un ensemble FERMÉ. Aucun verbe create/place/move/resize.
> · **Aucune action ne porte de champ prix/niveau/géométrie.**
> · focus_zone / highlight_zone référencent un id de zone détectée EXISTANTE.

`GEOMETRY_KEYS` bannit explicitement `level`, `price`, `high`, `low`, … et `highlight_zone`/`focus_zone` rejettent
tout id absent de l'écran. **Un niveau de référence est un prix (géométrie) qui n'est pas une zone détectée : il ne
peut pas — et ne doit pas — passer par ce contrat**, sous peine d'affaiblir la protection du chatbot.

**Proposition (à valider)** : ajouter un champ dédié à `ChartViewState`, p. ex.
`referenceLevel: { price: number; label: string } | null`, **posé directement par l'UI Régime** (callback local,
JAMAIS via le pipeline `coerceViewAction`/chatbot), et peint par une primitive/price-line distincte, visuellement
différente des zones et des poches de liquidité (trait à l'accent, pointillé). Un seul repère à la fois. Le
vocabulaire view-action, `GEOMETRY_KEYS` et le verrou d'id de zone restent **inchangés octet pour octet**, et une
zone détectée reste soumise à la validation existante. Le repère et la zone empruntent deux chemins nets et séparés.

---

## 3. Fuseau du « jour » — RÉPONSE : **à trancher (recommandation ci-dessous).**

- Le moteur horodate tout en **UTC** (`candle_close_ts`, `broken_at`, `to_dict` → ISO `…Z`).
- MC-1 raisonne en **America/New_York** pour la fenêtre hebdomadaire (ouverture dim. 17:00/18:00 NY, clôture
  ven. 17:00 NY, rollover métaux 17:00–18:00).
- Le front affiche les horaires en **fuseau local du lecteur** (`localTime.ts`).
- Aucun endroit ne définit « l'ouverture du jour ».

**Recommandation** : définir « jour » = **la bougie D1 telle que livrée par le flux de données** (`GET /api/candles?tf=D1`),
et « semaine » = la bougie W1. Ainsi l'ouverture du jour = `open` de la D1 courante, le haut/bas de la veille =
`high`/`low` de la D1 précédente, etc. **Une seule définition, la même que le graphique**, documentée ici. Le
référentiel D1/W1 du fournisseur est aligné sur le rollover courtier (17:00 NY pour XAU/FX). À confirmer.

---

## 4. Tableau des dix mesures

Légende : **Disponible** = valeur + preuve directes du moteur · **Adaptée** = disponible mais la façade/preuve v9
doit être corrigée pour rester honnête · **Décision** = fork produit à trancher · **Partielle** = manque un morceau
de la preuve v9.

| # | Mesure | Verdict | Source réelle | Écart avec la v9 à corriger |
|---|---|---|---|---|
| 1 | Phase de marché | **Adaptée** | `regime.market_phase` (dérivé de trend+vol, `_derive_market_phase`) | Panneau « Donnée » v9 (bornes franchies, cassures depuis CHOCH, retour intérieur) **fictif** : la phase ne vient PAS de là. Preuve honnête = trend + volatilité + règle. Vocab moteur = trend/expansion/ranging/accumulation (« distribution » jamais émis). |
| 2 | Tendance | **Décision** | `regime.trend` (`_derive_trend`) | ⚠️ `_derive_trend` = **déplacement close-à-close vs amplitude** (`pct_move < 0.3·rng_pct`), PAS une lecture de sommets/creux. Le panneau « Donnée » v9 (4 swings datés) et le Concept « succession des sommets et creux » **décrivent un autre moteur**. |
| 3 | Volatilité | **Décision** | `regime.volatility_observed` (catégoriel seul) | ⚠️ Le moteur n'expose PAS les intermédiaires numériques. Baseline réelle = `TR[:-7]` (toutes sauf 7 dernières), PAS « 20 précédentes » ; métrique = True Range (haut−bas). Seuils réels **0,70 / 1,30** (pas 0,75/1,35). La preuve chiffrée exigée n'existe pas côté API. |
| 4 | Position dans le range | **Adaptée** | bornes = `liquidity_pools` `range_high`/`range_low` (extrêmes de fenêtre, émis **conditionnellement**) | Ce sont des **extrêmes de fenêtre** (max/min), pas « dernier sommet / dernier creux ». Le % est une **arithmétique front** (le moteur ne le calcule pas). Fallback = extrêmes des bougies. Peut être absent si le cluster extrême est déjà un EQH/EQL. |
| 5 | Alignement | **Disponible** | `regime.mtf_confluence` (dict par TF) + `useMtfTrends` + `classifyMtfAlignment` | Aucun. La TF en désaccord est déjà identifiée (`disagreement`). |
| 6 | Maturité | **Disponible** | `choch_events` (ancrage = dernier CHOCH), `deriveTrendMaturity` existant | Ancrage daté (date+heure via `formatBreakTimestamp`), bougies dérivées, « événements depuis » = `bos_events` postérieurs. Conforme. |
| 7 | Dernier événement | **Partielle** | `structure.bos`/`choch` (+ historique events) | « Clôture de confirmation » (prix) **RETIRÉE** : pas de champ distinct du `level`. « Zone créée par le mouvement » **RETIRÉE** : aucun lien moteur événement→OB. « not available » = `bos`/`choch` point-in-time nuls (seulement posés sur la dernière barre) → à robustifier via l'historique. |
| 8 | Densité | **Disponible** | `order_blocks[]`/`fair_value_gaps[]` + `status` | Inventaire par type ET état complet (OB active/mitigated/invalidated ; FVG active/partially_filled/filled). « Ouverts » = active **+** partially_filled ; partiels **inclus** dans le total — dérivable proprement. |
| 9 | Session | **Décision** | MC-1 `market_calendar` : fenêtre hebdo + rollover + fériés + tz | ⚠️ MC-1 **ne modélise PAS** les sessions intraday Asie/Londres/NY ni les chevauchements. La tuile v9 (« New York », « Chevauchement Londres/NY ») exigerait une **2e source d'horaires** — interdit par la mission. |
| 10 | Niveaux de référence | **Adaptée** | calculables via `GET /api/candles` (D1/W1) | Pas dans le payload MarketReading. Calcul front depuis les bougies. Cap 500 barres ⇒ « semaine précédente » hors fenêtre en M15 si calculé sur la TF affichée → **calculer sur D1/W1** (même endpoint). Fuseau = §3. Tracé = §2. |

---

## 5. Mesures / lignes RETIRÉES faute de donnée (règle « pas de donnée, pas de tuile/ligne »)

- **Dernier événement → « Clôture de confirmation » (prix)** : le moteur stocke `level` (extrême franchi) +
  `broken_at`, pas le prix de la clôture qui a confirmé. Ligne retirée.
- **Dernier événement → « Zone créée par le mouvement »** : aucun lien moteur entre un BOS/CHOCH et l'OB qu'il a
  engendré. Ligne retirée.
- **Phase → panneau « Donnée » v9 (bornes/cassures/retour)** : ne correspond pas à la dérivation réelle. Remplacé
  par la preuve honnête (trend + volatilité + règle), pas retiré mais **reconstruit**.
- **Session (tuile entière)** et **Volatilité (preuve chiffrée)** et **Tendance (preuve swings)** : voir §6 —
  décisions requises avant tout rendu.

---

## 6. Conflits STRUCTURELS entre la v9 et le moteur réel — DÉCISIONS REQUISES

La v9 a été dessinée en supposant un moteur plus riche que l'actuel. Trois mesures opposent frontalement deux
règles de la mission : « reprends **mot pour mot** les textes v9 » **et** « chaque valeur/preuve vient d'une sortie
**réelle** du moteur ». On ne peut pas satisfaire les deux à la fois pour Tendance, Phase et Volatilité.

1. **Tendance & Phase — le récit SMC ≠ le calcul réel.** Le moteur lit la tendance par **déplacement close-à-close
   rapporté à l'amplitude** (seuil 0,3), et la phase par **(trend, volatilité)**. Les textes/preuves v9 parlent de
   sommets/creux et de bornes franchies. → Choisir : (a) présenter la **vraie** règle honnêtement (s'écarte du
   texte v9) ; (b) garder le récit v9 (malhonnête vs moteur) ; (c) autoriser une petite extension moteur exposant
   les swings/inputs (viole « pas de diff moteur »).

2. **Volatilité — la preuve chiffrée n'existe pas.** La mission veut « moyenne récente, moyenne de référence,
   rapport, seuils… un sceptique doit pouvoir refaire l'opération ». Le moteur n'expose que `low/normal/elevated`.
   → Choisir : (a) **exposer les intermédiaires** (`recent_avg`, `baseline_avg`, `ratio`, `thresholds`,
   `baseline_n`) = **petite mission moteur/API séparée** ; (b) **recomputer côté front** avec la même règle (risque
   de divergence : la fenêtre de bougies du regime ≠ celle du chart) ; (c) tuile **catégorie + règle décrite**,
   sans chiffres live. Rappel : les seuils réels sont **0,70/1,30**, baseline = « toutes sauf 7 dernières », métrique
   = True Range — le libellé v9 « 7 vs 20 » et « 0,75/1,35 » est **faux**.

3. **Session — pas de sessions intraday sans 2e source.** MC-1 donne la fenêtre hebdo + rollover + fériés, pas
   Asie/Londres/NY. → Choisir : (a) **retirer** la tuile (la règle « nombre impair → dernière pleine largeur » sert
   exactement à ça) ; (b) **re-scoper** la tuile en « Horaire du marché » adossée à MC-1 (ouvert/fermé, clôture
   hebdo, rollover, heure locale du marché, prochaine ouverture) — réutilise MC-1, zéro 2e source ; (c) autoriser
   une source de sessions intraday (**interdit par la mission telle qu'écrite**).

**Recommandations** : 1(a) présenter la vraie règle + reformuler le Concept en conséquence ; 2(a) mini-mission
moteur pour exposer les intermédiaires vol (sinon 2c) ; 3(b) re-scoper Session sur MC-1 ; §3 fuseau = D1/W1 du flux ;
§2 tracé via champ `referenceLevel` séparé.

---

## 6bis. DÉCISIONS ARRÊTÉES (fondateur, STOP du 2026-07-26)

1. **Tendance & Phase → Vérité moteur + reformuler.** On présente la **vraie règle** du moteur (tendance =
   déplacement close-à-close vs amplitude, seuil 0,3 ; phase = fonction de trend+vol) et on **reformule les textes
   Concept** en conséquence. Le verbatim v9 cède devant l'honnêteté sur ces deux tuiles.
2. **Volatilité → Mini-mission moteur/API.** On **expose les intermédiaires** (`recent_avg`, `baseline_avg`,
   `ratio`, seuils 0,70/1,30, `recent_n`=7, `baseline_n`) pour une preuve pleinement rejouable. ⚠️ Cela introduit
   une petite modification **moteur/schéma** — hors du périmètre « frontend only » de RG-1 → **séquençage à
   confirmer** (voir plan). Corriger aussi les libellés faux (« 7 vs 20 », « 0,75/1,35 »).
3. **Session → Re-scoper sur MC-1.** Tuile « Horaire du marché » adossée à `market_calendar`/`market_status` :
   ouvert/fermé, clôture hebdo, rollover (métaux), heure locale du marché, prochaine ouverture. **Zéro 2e source
   d'horaires.** Pas de sessions Asie/Londres/NY.
4. **Fuseau « jour » → Bougie D1/W1 du flux.** « Jour » = bougie D1 livrée par le fournisseur ; « semaine » = W1.
   Même référentiel que le graphique (rollover courtier ~17:00 NY). Niveaux de référence calculés sur D1/W1 via
   `GET /api/candles`, jamais sur la TF affichée (évite le trou « semaine précédente » du cap 500 barres en M15).

---

## 8. IMPLÉMENTATION (Phase 2) — livré

**Backend (contenu, minimal, décision 2)**
- `MarketReadingRegime.volatility_detail` (optionnel) : `recent_avg`, `baseline_avg`, `ratio`,
  `recent_n`, `baseline_n`, `threshold_low`=0,70, `threshold_high`=1,30. Calcul factorisé
  (`_volatility_from_candles`), rétro-compatible, miroir type TS.
- `/api/candles` sert **D1/W1** (déjà en cache via l'assembleur MTF) — conséquence mécanique du
  fuseau « bougie D1/W1 du flux ». Gate freemium inchangée (no-op OFF ; D1/W1 dégradent proprement
  pour le tier gratuit quand ON — à noter).

**Frontend**
- `RegimeCard` : 6 → **10 tuiles**, panneau de détail **Donnée / Concept** (défilement borné,
  clavier + `aria-expanded`/`role=tab`), **une seule ouverture** dans toute la page (canal `openHelp`
  partagé, clés `rg:<tuile>:<onglet>`). Règle **nombre impair → dernière pleine largeur** (`.span2`).
- **Onglet Donnée = valeurs LIVE du moteur** (décision). Volatilité : preuve chiffrée complète depuis
  `volatility_detail`. Position, Maturité, Dernier événement, Densité, Alignement : dérivés réels.
- **Niveaux de référence** : prix cliquables → `ChartViewProvider.setReferenceLevel` (canal **séparé**,
  hors `coerceViewAction`), trait plein à l'accent, re-clic = retrait. Fetch D1/W1 via `/api/candles`.
- **Session** re-scopée « Horaire du marché » sur MC-1 (`market_status` + heure locale NY).
- i18n `regimePanel.*` **fr + en complets**, 7 autres locales = EN (convention UI-2c). Concept
  trend/phase/vol **reformulés** (décision 1) ; align/pos/mat/last/dens/lvl/regime verbatim v9.

**Tests** : back mappers + schéma + `/api/candles` D1/W1 ; front `reference-levels` (11), copy-honnêteté
`regimePanel` 9 locales (surface assertive bannie ; Concept = phrases promo bannies + bloc « ne dit pas »
obligatoire), composant RG-1 (tuile→Donnée, ?→Concept, un seul panneau, impair→span2, mesure sans
donnée non rendue, clic prix trace/retire, id inventé toujours rejeté, géométrie hors whitelist).
Playwright 1280×800 (structurel sans backend + interactions si reading présent).

## 9. Écarts assumés vs la maquette v9 (à l'écran)
- **Tendance / Phase / Volatilité** : textes Concept **reformulés** pour dire la vraie règle du moteur
  (décision 1). Le libellé vol « 7 vs 20 · seuils 0,75/1,35 » de la v9 était **faux** → corrigé
  (7 vs *les précédentes* ; seuils réels **0,70/1,30** ; chiffres réels dans l'onglet Donnée).
- **Tendance – Donnée** : pas de tableau de sommets/creux (le moteur n'expose pas ces intermédiaires) —
  la règle et le résultat, honnêtement. **Alignement – Donnée** : direction par TF (pas le chip
  « CHOCH ↓ date » par TF — non fourni par `useMtfTrends`).
- **Dernier événement** : lignes « Clôture de confirmation » et « Zone créée par le mouvement »
  **RETIRÉES** (§5, pas de donnée moteur).
- **Position** : bornes = **extrêmes de structure de la fenêtre** (`range_high`/`range_low`), pas
  « dernier sommet/creux » ; % en arithmétique front bornée.
- **Session → Horaire du marché** : pas de découpage Asie/Londres/NY (MC-1 ne le modélise pas ;
  §6-3). Ouvert/fermé, clôture hebdo, heure locale, prochaine ouverture.
- **Surface mobile** (`RegimeSection` accordéon) inchangée — la mission cible le desktop 1280×800.

## 7. Ce qui est prêt à implémenter sans décision (après GO)

Mesures **5 (Alignement)**, **6 (Maturité)**, **8 (Densité)** : entièrement adossées au moteur, helpers déjà
présents. Mesure **7** amputée de 2 lignes. Mesure **4** en arithmétique front honnête (extrêmes de fenêtre).
Mesure **10** via D1/W1 une fois le fuseau tranché. Mesure **1** avec preuve reconstruite. Tuiles **2, 3, 9** :
bloquées sur les décisions §6.

*(Fin du diagnostic Phase 1 — en attente de GO et des arbitrages §6.)*

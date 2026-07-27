# AUDIT RG-1b — Régime de marché : conformité stricte (Phase 1 : diagnostic)

> Branche `feat/rg-1b-regime-conformite` (worktree `wt-rg1b`, depuis origin/main `73223fd`).
> Frontend. **Zéro diff moteur** (section 3). **Aucun code avant GO.**
> ⚠️ **La maquette v11 est INTROUVABLE** (Downloads ne contient que jusqu'à v9). Il me la faut
> avant l'implémentation (Phase 2) pour atteindre « l'état v11, tuile par tuile, dans son ordre ».

État actuel constaté : 8 tuiles au lieu de 10 (Phase présente mais dérivée ; **Position** et
**Niveaux de référence** ne s'affichent pas — voir §B).

---

## A) PHASE DE MARCHÉ — VERDICT : **aucune donnée moteur → tuile à SUPPRIMER**

`regime.market_phase` est **dérivée** de la tendance et de la volatilité, rien d'autre :
`_derive_market_phase(trend, volatility)` (`market_reading_mappers.py:1327`) — bullish/bearish →
`expansion` si vol élevée sinon `trend` ; ranging → `ranging` ; sinon `accumulation`.

Le moteur **n'expose AUCUNE phase calculée à partir du franchissement des bornes de structure**
(expansion = borne cédée en clôture sans retour ; consolidation = bornes testées sans clôture au-delà).
Recherché dans tout `src/intelligence/` (smart_money, confluence_detector, sentinel_scanner, insight_v2) :
rien de tel n'est calculé ni surfacé dans le `MarketReading`.

**→ Conformément au brief (A, cas NON) : la tuile Phase est SUPPRIMÉE.** Pas remplacée par une
dérivation (une valeur dérivée de deux tuiles n'est pas une mesure, et le « ? » global l'interdit).
**Ajout moteur possible** (hors périmètre « zéro diff moteur ») : exposer une vraie phase
expansion/consolidation depuis le suivi des bornes (broken-and-held vs tested-and-held), ancrée sur
le dernier CHOCH — ce serait une mission moteur séparée.

Conséquence mise en page : une tuile en moins → **nombre impair → dernière tuile pleine largeur**.

---

## B) TUILES ABSENTES — pourquoi Position et Niveaux ne s'affichent pas

### Position dans le range — **alimentable, mais la source actuelle est trop stricte**
- Bornes actuelles = pools `liquidity_pools` de `kind` `range_high`/`range_low` (`structureRange`).
  Ces pools sont les **extrêmes de la fenêtre** (max des swing highs / min des swing lows via
  UP_FRACTAL/DOWN_FRACTAL, `collect_liquidity_pools` `market_reading_mappers.py:755`), mais ils sont
  **émis CONDITIONNELLEMENT** : `range_high` est omis si un cluster `equal_highs` externe siège déjà à
  l'extrême (idem `range_low`). Quand ils sont omis → la tuile n'a pas de bornes → **elle disparaît**.
- **Il n'existe PAS de champ « dernier sommet / dernier creux » distinct** dans le `MarketReading`.
- **Fix front proposé (zéro moteur)** : dériver les deux bornes structurelles des pools **externes** —
  borne haute = plus haut niveau des pools BSL `is_external` (que le kind soit `range_high` OU
  `equal_highs`), borne basse = plus bas niveau des pools SSL `is_external`. Robuste dès qu'il y a de
  la structure. Ce sont bien des bornes **structurelles** (swings retenus), pas des extrêmes de bougies
  bruts. Le % reste une arithmétique front bornée.

### Niveaux de référence — **NON alimentable en l'état** (D1/W1 pas en cache)
- La tuile a besoin des bougies **D1/W1** (ouverture jour/semaine, extrêmes veille/semaine). RG-1 a
  ouvert `/api/candles` aux D1/W1, **mais rien ne peuple D1/W1 dans le cache** : le seed
  (`scripts/seed_twelve_data.py` → M15/H1/H4 seulement), l'assembleur et le scheduler ne fetchent que
  la TF demandée ; le fournisseur MTF **lit** d1/w1 du cache sans jamais les **écrire**. Donc
  `/api/candles?tf=D1` renvoie **404** → la tuile disparaît.
- **Fuseau du « jour » — à trancher, non supposé.** La décision RG-1 (« bougie D1/W1 du flux ») est
  caduque puisque D1/W1 ne sont pas peuplées. Il faut vraiment choisir (voir décisions).

---

## C) SOURCES INCOMPLÈTES

### Volatilité — le dénominateur existe dans le moteur mais n'est PAS affiché, et il vaut ~493, pas 20
- Sous-ligne actuelle : « 7 dernières bougies vs **les précédentes** » — dénominateur non nommé
  (rendu invérifiable). C'est un régression de UI-3a (la sous-ligne avait été rendue sans paramètre).
- Le moteur EXPOSE tout dans `regime.volatility_detail` : `recent_n`=7, `baseline_n`, `recent_avg`,
  `baseline_avg`, `ratio`, `threshold_low`=0,70, `threshold_high`=1,30 (`VolatilityDetail`).
- ⚠️ **`baseline_n` réel ≈ 493**, PAS 20 : `_volatility_from_candles` compare les 7 dernières à
  **TOUTES les précédentes** de la fenêtre (`baseline_n = len(trs) - 7`), et la fenêtre regime = **500
  bougies** (`DEFAULT_LOOKBACK=500`). La maquette « 7 vs 20 précédentes » suppose une fenêtre de
  référence bornée à 20 — **le moteur ne fait pas ça**. « Valeurs réelles du moteur » (§2) donne donc
  « 7 dernières vs 493 précédentes », ce qui contredit la maquette et la règle « zéro diff moteur »
  interdit de borner la fenêtre à 20. **→ décision requise.**

### D) Audit des autres sous-lignes (nomme-t-elle sa source ?)
| Tuile | Sous-ligne actuelle | Nomme sa source ? |
|---|---|---|
| Tendance | « mesurée sur {tf} » | ✅ (l'unité de temps) |
| Volatilité | « 7 dernières bougies vs les précédentes » | ❌ **dénominateur manquant** (→ « vs {baseline_n} précédentes ») |
| Alignement | « H4 ↓ · H1 ↓ · M15 ↓ » | ✅ (les TF) |
| Maturité | « depuis le CHOCH du {date} » | ✅ (l'ancrage daté) — **mais format « 24/07 », à passer en « 24 juil. »** |
| Dernier événement | « {date} » | ⚠️ donne l'heure, pas la nature de la source ; format « 24/07 » aussi |
| Densité | « ouverts sur {tf} » | ✅ (état + TF) |
| Session | « {transition} dans {délai} » | ✅ |
| Position / Niveaux | (absentes) | — |

### Format de date — actuellement numérique « JJ/MM à HH:MM »
`formatLocalDayHm` (`lib/time/localTime.ts:33`) utilise `{day:'2-digit', month:'2-digit'}` → « 08/07 à 09:30 ».
Le brief veut **« 24 juil. à 09:45 »** (mois court localisé, format long, s'inversant fr↔en). Fix front
(attention : helper partagé — soit le changer globalement + ajuster les tests, soit ajouter un format long
dédié au panneau).

---

## SYNTHÈSE DES DÉCISIONS À TRANCHER (STOP)

1. **v11 introuvable** → me fournir `mia-markets-reference-v11.html` (Downloads) avant Phase 2.
2. **Phase** → SUPPRIMÉE (verdict acté, sauf si tu veux plutôt planifier l'ajout moteur d'une vraie phase).
3. **Volatilité** → afficher le **vrai** `baseline_n` (~493) [zéro moteur] OU autoriser une petite modif
   moteur bornant la fenêtre de référence (ex. 20) pour coller à la maquette [diff moteur].
4. **Niveaux de référence** → (A) peupler D1/W1 dans le cache [data/back], (B) calculer côté front
   depuis les bougies de la TF affichée + fuseau documenté [front, limité : M15 ne couvre pas la semaine
   précédente], ou (C) supprimer la tuile. + **fuseau du « jour »**.
5. **Position** → OK avec le fix front (bornes = pools externes). À confirmer.

---

## DÉCISIONS ARRÊTÉES (STOP du 2026-07-27)
1. **Phase de marché → SUPPRIMÉE** (aucune donnée moteur ; ajout moteur possible plus tard).
2. **Volatilité → borne moteur à 20** : fenêtre de référence = les **20 bougies précédant les 7 récentes**
   (au lieu de toutes les ~493). Sous-ligne « 7 dernières vs 20 précédentes » = vraies valeurs.
3. **Niveaux de référence → peupler D1/W1** ; fuseau = **la bougie D1/W1 du flux** (source unique).
4. **Position → bornes = pools de liquidité externes** (fix front, à faire Phase 2).

## IMPLÉMENTATION BACKEND (faite — indépendante de v11)
- **Volatilité** : `_VOL_BASELINE_N = 20` ; `_volatility_from_candles` compare les 7 dernières aux
  20 précédentes (ou moins si fenêtre courte). `volatility_detail.baseline_n` = 20 en régime normal.
  ⚠️ change `volatility_observed` dans tout le produit (voulu). Tests mappers 59 verts.
- **D1/W1 à la demande** : `MarketReadingAssembler.warm_candles(instrument, tf)` (fetch+cache candles
  seuls, market-aware + idempotent, aucune lecture SMC). `/api/candles` peuple D1/W1 sur cache-miss
  (`REFERENCE_TIMEFRAMES`), M15/H1/H4 restent en 404 honnête (scheduler les chauffe). Tests endpoint +
  assembleur verts. Fuseau = bougie du flux (Twelve Data D1/W1).
- Régression backend : 190 verts sur le lot regime/scanner/narration ; 2 échecs `test_smoke_e2e`
  (health/scanner 503) **pré-existants** (identiques sur l'arbre propre `73223fd`, environnementaux).

## RESTE (Phase 2 — BLOQUÉ sur la maquette v11)
Frontend : retirer la tuile Phase ; sous-ligne Volatilité « 7 vs 20 » + Donnée calcul complet ;
Position (bornes = pools externes) ; Niveaux (façade 2 repères + panneau 6 prix cliquables → tracé) ;
format date long « 24 juil. à 09:45 » (fr/en) ; règle impair→pleine largeur ; vérifier bloc « ne dit
pas » partout ; i18n fr+en ; ordre/écarts vs v11. **Il me faut `mia-markets-reference-v11.html`.**

## IMPLÉMENTATION FRONTEND (faite — v11 posée dans reference-desktop.html)
- **Phase → SUPPRIMÉE** de la grille (jamais rendue). Grille passe à **9 tuiles** → règle
  **impair → dernière pleine largeur** active.
- **Volatilité** : sous-ligne « 7 dernières vs 20 précédentes » (vraies valeurs `volatility_detail`) ;
  Donnée = calcul complet (moyennes, rapport, seuils) déjà présent.
- **Position** : `structureRange` retombe sur les pools **externes** (plus haut BSL / plus bas SSL)
  quand `range_high`/`range_low` ne sont pas émis → la tuile s'affiche de façon robuste.
- **Niveaux de référence** : inchangée côté front (façade 2 repères veille + panneau 6 prix cliquables →
  tracé via le canal séparé) ; s'affiche dès que D1/W1 sont peuplés (backend à la demande, live).
- **Date longue localisée** : `formatLocalDayLong` → « 24 juil. à 09:45 » (fr) / « Jul 24 at 09:45 » (en) ;
  utilisée par le panneau (maturité, dernier événement, journal). Le chart garde son format numérique.
- **Session** : sous-ligne + heure locale « … · 15:20 NY » (conforme v11).
- **Concept** : bloc « ce que ça ne dit pas » présent sur les 9 mesures + le « ? » global ; global
  reformulé « neuf mesures ».

## ÉCARTS ASSUMÉS vs la maquette v11 (à l'écran)
1. **Tuile Phase absente** alors que v11 la montre (« Expansion · depuis le CHOCH ») — le moteur n'a
   PAS de phase fondée sur les bornes ; règle A → supprimée, pas dérivée. Réactivable par un ajout moteur.
2. **Tendance — Concept & Donnée** : v11 décrit « la succession des sommets et creux » ; le moteur lit la
   tendance par **déplacement close-à-close vs amplitude** (`_derive_trend`). On garde la formulation
   HONNÊTE (décision RG-1 « vérité moteur »), pas le récit swings de v11. Donnée = règle + résultat, pas
   un tableau de swings (non exposés).
3. **Niveaux de référence** : n'apparaît qu'en live (D1/W1 peuplés à la demande) ; absente en e2e/mock
   sans backend feed. Fuseau = bougie D1/W1 du flux.
4. **Position** : bornes = extrêmes structurels des pools externes (pas un champ « dernier sommet/creux »
   dédié, inexistant côté moteur).

*(Phase 1 + 2 livrées ; MERGE après confirmation live.)*

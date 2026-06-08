# STRUCTURE_DEFINITIONS_AUDIT — Comparaison aux définitions canoniques SMC

> **Audit de validation algorithmique — Phases 1.2 + 1.3**
> Date : 2026-06-07 · Branche : `audit/validation-algorithmique-detection-structures`
> Réfère à : `STRUCTURE_DEFINITIONS_CURRENT.md` (définitions du code).
>
> ⚠️ **Posture de l'audit** : ce document **identifie et classe** les écarts entre le code et les conventions SMC/ICT généralement admises. Il **ne décrète pas** qu'une définition est « objectivement correcte » — la définition canonique SMC est elle-même expert-dépendante (ICT vs The Strat vs communauté). Les sources de référence sont mobilisées **de mémoire générale**, sans citation d'autorité unique. La validation experte externe reste une action opérateur (founder).

---

## 1. Échelle de sévérité

| Sévérité | Définition | Action recommandée |
|---|---|---|
| 🟢 **Mineure** | Variation de paramètre (lookback, seuil) sans changer la nature de la détection | OK — laisser tel quel |
| 🟡 **Modérée** | Variation méthodologique (close vs wick, englobant vs displacement) — la détection reste défendable mais diverge d'une convention répandue | **Documenter sur `/methodology`** |
| 🔴 **Majeure** | Définition fondamentalement différente de toutes les conventions, ou le champ publié ne correspond pas à ce que le moteur calcule (bug de câblage) | **Corriger ou retirer le champ du visible client** |

---

## 2. Table comparative — détection brute

| Concept | Définition CODE actuel | Convention ICT/SMC répandue | Divergence | Sévérité |
|---|---|---|---|---|
| **Swing / Fractal** | Williams 5 bougies (N=2), causal `shift(N)` | Swing = pivot avec n bougies de chaque côté (n variable 2-5 selon l'école) | Choix de N=2 (plus réactif, plus de bruit) | 🟢 Mineure |
| **BOS bullish** | `close > current_high_structure` (structure = max des fractals), break par **close** | Prix dépasse le dernier **swing high majeur**, idéalement clôture de **corps** (pas wick) | Le code casse l'**extrême accumulé** des fractals depuis reset, pas nécessairement « le dernier swing high majeur ». Pas de notion majeur/mineur. | 🟡 Modérée |
| **BOS = continuation** | Oui, BOS continue la tendance ; renversement = CHOCH | ICT : BOS = continuation de tendance. ✔️ aligné | Conforme | 🟢 Mineure |
| **CHOCH** | = BOS de renversement, **même bougie** (`bos_signal` change de signe) | ICT : CHOCH = première cassure d'un swing **interne opposé** signalant un changement de caractère AVANT le BOS de la nouvelle tendance. Souvent CHOCH ≠ BOS (deux events distincts) | Le code fusionne CHOCH et BOS-de-renversement en un seul event. Pas de CHOCH « interne » distinct. | 🟡 Modérée → 🔴 selon exigence |
| **Order Block** | Motif **englobant 2 bougies** + nouveau high/low. Zone = bougie `i-1` | ICT : **dernière bougie opposée avant un displacement / un BOS** (pas un simple englobant). Filtre d'impulsion (displacement) requis | Définition fondamentalement différente : englobant ≠ OB ICT. Pas de filtre displacement. **Déjà connu P0-2.** | 🔴 Majeure |
| **FVG** | Gap 3 bougies `low[i] > high[i-2]` (bull), seuil 0.1×ATR | ICT : FVG = imbalance 3 bougies, `candle1.high < candle3.low` (bull). ✔️ formule alignée | Formule canonique. Seuil 0.1×ATR ≈ spread XAU (laxe). **P1-2.** | 🟢 Mineure (formule) + 🟡 (seuil) |
| **FVG fill/mitigation** | ❌ non suivi (statut toujours `active`) | FVG se remplit partiellement/totalement quand le prix revient | Le statut `filled/partially_filled` du schéma n'est jamais alimenté | 🟡 Modérée |
| **OB mitigation** | ❌ non suivi (statut toujours `active`) | OB « mitigé » quand le prix le reteste | Idem | 🟡 Modérée |
| **Retest** | State machine ATR (tol 0.5×ATR) | Retest = pullback vers la zone cassée. Concept aligné | Tolérance 0.5×ATR ≈ spread XAU. **P1-3.** | 🟢 Mineure |

---

## 3. Divergences de CÂBLAGE (engine → MarketReading) — au-delà des définitions

Ces écarts ne concernent pas la *définition* de la détection mais le fait que **le champ publié au client ne reflète pas ce que le moteur calcule**. Ils sont 🔴 par nature (risque « le produit affiche une valeur fausse »).

| Réf | Constat | Fichier | Sévérité | Reco |
|---|---|---|---|---|
| **F1** | `BOS.level` lit `BOS_PRICE_LEVEL` (inexistant) → retombe **toujours** sur `current_price`. Le vrai niveau (`BOS_BREAK_LEVEL`) est ignoré. | `market_reading_mappers.py:128` vs `strategy_features.py:737` | 🔴 Majeure | Corriger la clé OU retirer `level` du visible client |
| **F2** | `CHOCH.level` lit `CHOCH_PRICE_LEVEL` (inexistant) → **toujours** `current_price`. | `market_reading_mappers.py:140` | 🔴 Majeure | Idem F1 |
| **F3** | Niveaux OB/FVG publiés = `current_price ± ATR/2` (proxy), alors que l'engine connaît les **vraies zones** (`BULLISH_OB_HIGH/LOW`, et les bornes FVG). | `market_reading_mappers.py:157-158, 174-175` | 🔴 Majeure | Câbler les vraies zones OU ne pas afficher de niveaux chiffrés |
| **F4** | 1 OB et 1 FVG max par reading (signal de la bougie courante), pas une liste des zones **actives non mitigées**. | `market_reading_mappers.py:146-180` | 🟡 Modérée | Documenter « OB/FVG le plus récent » OU construire un registre de zones |
| **F5** | Le LLM Haiku ne reçoit jamais la `structure` (seulement `tags`+`regime`). Il ne peut pas mentionner les niveaux de prix et déduit la présence OB/FVG/BOS uniquement des tags. | `market_reading_assembler.py:265` | 🟡 Modérée | Conscient/voulu (sobriété) — documenter |
| **F6** | `BOS.direction` vient de `BOS_SIGNAL` (état **propagé**) : un BOS peut être affiché comme « récent » alors qu'il date de plusieurs bougies (l'état reste à ±1). `broken_at` = clôture courante, pas la date réelle du break. | `market_reading_mappers.py:121-131` | 🟡 Modérée | Distinguer « état de tendance » de « BOS récent (event)» ; utiliser `BOS_EVENT` pour la fraîcheur |

> **Implication produit niveau 1.5** : F1/F2/F3 signifient que **tout niveau de prix chiffré attaché à un BOS/CHOCH/OB/FVG dans le MarketReading est un proxy, pas une mesure**. Cohérent avec la note mémoire 2026-05-27 (« ne PAS dire Order Block institutionnel », champs probabilistes retirés). **Recommandation forte : ne pas exposer de niveaux chiffrés BOS/CHOCH/OB/FVG au client tant que F1-F3 ne sont pas corrigés**, ou les présenter explicitement comme « zone approximative autour du prix courant ».

---

## 4. Audit du moteur de PHASE de marché (Étape 1.3)

### 4.1 Nature
- **Custom, rule-based**, sans indicateur standard (pas d'ADX, pas d'EMA crossover) ni ML.
- Trend = momentum close-à-close normalisé vs amplitude de la fenêtre (`_derive_trend`).
- Volatilité = ratio TR récent (7) / baseline (`_derive_volatility`).
- Phase = table trend × volatilité (`_derive_market_phase`).

### 4.2 Justification / faiblesses
| Point | Évaluation |
|---|---|
| Simplicité, déterminisme, reproductibilité | 🟢 Fort (auditable, pas de boîte noire) |
| Dépendance à la **fenêtre** (200 bougies par défaut) | 🟡 Le verdict trend/vol dépend mécaniquement de la longueur de lookback — non documenté côté produit |
| Seuils `0.3`, `0.7`, `1.3` non calibrés empiriquement | 🟡 Choix heuristiques, à documenter comme tels sur `/methodology` |
| Valeur `distribution` au schéma jamais produite | 🟡 Code mort de vocabulaire — retirer du schéma OU implémenter |
| Divergence possible avec les 4 HMM/filtres présents mais non câblés | 🟢 Acceptable tant que le produit n'affiche qu'une source ; à NE PAS mélanger |

### 4.3 Risque de glissement niveau 1.5 (langage prédictif)
- **MarketReading régime** : 100 % **descriptif présent**. `_derive_*` n'emploie aucun terme prédictif. ✔️
- **Templates de description** (`_build_description`, `_engine_template_fallback`) : verbes descriptifs uniquement (« Tendance X, volatilité Y, phase Z »). ✔️
- **System prompt Haiku** (`haiku_description_engine.py:42-49`) : interdit explicitement conseil/jugement, impose le présent descriptif. ✔️
- ⚠️ **Seul langage forward-looking trouvé** : docstrings de `regime_gate.py:13-35` (« we are likely in or **about to enter** a structural shift », « BOCPD tells you we are *entering* a new regime »). **Mais `RegimeGate` n'est PAS câblé au MarketReading** → pas exposé client. À surveiller si un jour câblé. Sévérité actuelle : 🟢 (interne, non-client).

---

## 5. Comptage des divergences

| Sévérité | Détection brute | Câblage | Régime | Total |
|---|---|---|---|---|
| 🟢 Mineure | 4 | 0 | 1 | **5** |
| 🟡 Modérée | 4 | 2 (F4, F5) | 4 | **10** |
| 🔴 Majeure | 1 (OB) | 3 (F1, F2, F3) | 0 | **4** |

> **Aucune divergence ne déclenche un STOP** (cas critique #2 = définition >30 % différente de **toutes** les sources sur la détection elle-même). La seule définition franchement non-canonique est l'**OB englobant** (déjà documentée P0-2). Les 3 autres 🔴 sont des **bugs de câblage** (champ publié ≠ valeur calculée), pas des divergences de définition — donc corrigeables sans toucher au moteur de détection.

---

## 6. Recommandations Phase 1

### 🔴 Avant bêta (corriger OU retirer du visible)
1. **F1/F2/F3** : soit câbler `BOS_BREAK_LEVEL` et les vraies zones OB/FVG, soit **ne pas exposer de niveaux chiffrés** BOS/CHOCH/OB/FVG. (Patch mapper trivial et documenté, à valider founder — touche `market_reading_mappers.py`, pas le moteur.)
2. **OB englobant ≠ ICT** : ne **pas** écrire « Order Block institutionnel » côté client (déjà acté mémoire 2026-05-27). Documenter la définition réelle « englobant directionnel » sur `/methodology`.

### 🟡 À documenter sur `/methodology`
3. BOS sur close (corps), structure = extrêmes de fractals accumulés.
4. CHOCH = BOS de renversement (pas de CHOCH interne distinct).
5. FVG fill / OB mitigation non suivis → statut toujours « actif » = « détecté », pas « encore valide ».
6. Régime = formule custom close-à-close + ratio TR ; seuils heuristiques ; dépend de la fenêtre.
7. Valeur de schéma `distribution` non produite.

### 🟢 Nice-to-have
8. Calibrer `FVG_THRESHOLD` et `RETEST_TOL_ATR` (≈ spread XAU) — Sprint 2.
9. Distinguer « état de tendance » (BOS_SIGNAL) et « BOS récent event » (BOS_EVENT) pour `broken_at` (F6).

---

*Suite : Phase 2 — génération du dataset de 60 MarketReadings pour validation manuelle (`docs/audits/VALIDATION_DATASET_2026_06_06.md`).*

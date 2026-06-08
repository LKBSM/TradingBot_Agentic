# RD_AUDIT_D2 — Qualité des heuristiques de détection

> **Audit R&D exploratoire — Dimension 2/5**
> Date : 2026-06-08 · Branche : `audit/rd-exploratoire-moteur-detection`
> Sources de comparaison mobilisées **de mémoire générale** (ICT/SMC, LuxAlgo Pine, détecteurs communautaires GitHub). Aucune autorité unique citée ; la validation experte externe reste une action founder.
> ⚠️ Complémentaire de `STRUCTURE_DEFINITIONS_AUDIT.md` (audit précédent) qui classait les divergences vs canonique. Ici on évalue **la qualité méthodologique + faux positifs/négatifs + sensibilité paramètres**, et on propose **uniquement du Petit/Moyen**.

---

## 2.1 — Analyse heuristique par structure

### A. FRACTALS (swing points) — `_add_smc_base_features` (`strategy_features.py:597-710`)

**Heuristique** : fractale de Williams vectorisée, fenêtre `2N+1` (N=`FRACTAL_WINDOW`=2 → 5 bougies), causale via `.shift(N)`. `UP_FRACTAL[i]` si `high[i]==max(high[i-2:i+2])`.

**Forces** : 🟢 Causale (anti look-ahead, validée par un garde-fou runtime lignes 692-696), vectorisée, déterministe, canonique.

**Faiblesses** :
- **Fractale sur le wick** (`high`/`low`), alors que BOS compare sur le `close`. Asymétrie : une **mèche longue** crée un fractal très haut → relève `current_high_structure` → un BOS ultérieur exige un close au-dessus de la mèche (**faux négatif**). Cf. edge case 2.2-D.
- **Égalité stricte** `high[i]==rolling_max` : sur des plateaux (deux bougies de même high), aucune des deux n'est marquée fractale (`==` échoue car le max est partagé… en réalité les deux égalent le max → **les deux** marquées). Sur prix très répétitifs (faible volatilité, instruments à tick large), sur-marquage possible.

**Sensibilité paramètre** : N=2 réactif/bruité ; N=3 (7 bougies) plus lisse mais plus laggé. Devil's advocate : ni l'un ni l'autre n'est « juste » — c'est un curseur réactivité/bruit → **paramètre, pas bug**.

**→ Améliorations** :
- **D2-6** *(Petit / impact Faible-Moyen / risque Faible)* : rendre `FRACTAL_WINDOW` **TF-aware** (N=2 sur M15, N=3 sur H4/D1) au lieu d'un défaut global. Touche le wiring de config, pas l'algo.

---

### B. BOS — Break of Structure — `_calculate_bos_choch_*` (`strategy_features.py:32-234`)

**Heuristique** : break par **close** de l'extrême structurel (`current_high_structure` = max des fractals accumulés depuis le dernier reset). Anti-repeat via `allow_bos_up = last_fractal_high > last_bos_up_level`, où `last_bos_up_level = close` du dernier break.

**Forces** :
- 🟢 Break sur **corps (close)**, pas sur wick → conforme à la convention « close confirmation » la plus défendable.
- 🟢 Anti-« 100 %-firing » : exige un **nouveau** fractal au-delà du dernier break pour re-fire (régression couverte par `test_bos_no_repeated_fire.py`). C'est le bon design SMC « break → pullback → new swing → break ».

**Faiblesses** :
- **Pas de distinction majeur/mineur** : `current_high_structure` est le **max de tous les fractals** depuis reset. Un seul fractal-mèche aberrant fixe une barre trop haute → suppression de BOS légitimes (**faux négatifs**) jusqu'au prochain reset.
- **Anti-repeat indexé sur le `close`** (`last_bos_up_level = closes[i]`, ligne 107/123) et non sur le **niveau cassé**. Conséquence subtile : le re-fire est gated par « nouveau fractal > close du dernier break » — un close de break très étendu (grande bougie) relève artificiellement la barre du prochain BOS.
- **Pas de buffer ATR sur le break** : un close à **1 tick** au-dessus du niveau déclenche un BOS. Sur instrument à spread large (XAU), un break « cosmétique » de quelques cents passe (**faux positifs** sur bruit).
- **Mémoire structurelle = fenêtre de 200 bougies** (lookback assembleur, cf. D3/D4) : la structure ne « voit » pas au-delà ; le seed ne lit que les 50 premières (`min(50,n)`, ligne 77).

**Sensibilité** : un buffer ATR de break déplacerait l'équilibre faux+/faux−. Devil's advocate : buffer trop grand = retard + breaks ratés → c'est pourquoi je propose un défaut **0** (neutre) exposé en config.

**→ Améliorations** :
- **D2-7** *(Petit / impact Moyen / risque Faible)* : exposer un `BOS_BREAK_BUFFER_ATR` (défaut **0.0** = comportement actuel **inchangé**), permettant de filtrer les micro-breaks sur instruments à spread large après calibration. Risque nul tant que défaut=0.
- **D2-7b** *(Moyen / impact Moyen / risque Moyen)* : option « structure = dernier swing majeur » (fractal le plus récent confirmé d'un swing pivot net) plutôt que max-de-tous-fractals, pour réduire la suppression par mèche. **Gated sur annotation** (peut changer le taux de BOS). À ne tenter qu'après mesure d'exactitude.

---

### C. CHOCH — Change of Character — (`strategy_features.py:100-115`)

**Heuristique** : CHOCH = **BOS de renversement**, **même bougie** (l'état `bos_signal` passe de −1 à +1 ou inverse).

**Forces** : 🟢 Déterministe, simple, cohérent avec BOS.

**Faiblesses** : diverge de l'ICT « CHOCH interne précède le BOS de la nouvelle tendance » (deux events distincts). Ici CHOCH et BOS-de-renversement sont **fusionnés**. Pour un trader ICT strict, l'absence de CHOCH « interne » est une lacune sémantique (déjà notée 🟡 dans l'audit précédent).

**→ Améliorations** :
- **D2-8** *(Petit = documentation / impact Faible)* : documenter sur `/methodology` « CHOCH = première cassure structurelle en sens inverse de la tendance (= renversement) ». Implémenter un **vrai CHOCH interne distinct** = **Hors-portée** (nouvelle logique de swing interne, >5j, risque de régression sur tout le pipeline) → archive V1.2+.

---

### D. ORDER BLOCK — `_add_smc_order_blocks` (`strategy_features.py:756-815`)

**Heuristique** : motif **englobant directionnel 2 bougies** (bougie `i-1` opposée + bougie `i` directionnelle + nouveau high/low). Zone = high/low de la bougie `i-1`. Force = `taille(i-1)/ATR (+0.2 si FVG adjacent)`.

**Comparaison** : ICT définit l'OB comme **la dernière bougie opposée avant un displacement / un BOS** (impulsion requise). L'englobant simple **n'est pas** un OB ICT. **Divergence 🔴 majeure déjà connue (P0-2)**, et explicitement actée en mémoire (« ne PAS dire Order Block institutionnel »).

**Forces** : 🟢 Capture des zones d'absorption plausibles ; vectorisé ; strength normalisée ATR.

**Faiblesses** :
- **Aucun filtre d'impulsion (displacement)** : tout englobant compte, même sans mouvement consécutif → **beaucoup de faux positifs** (le test `test_more_obs_without_fvg_requirement` confirme « ~5-10× plus d'OB » sans filtre FVG).
- **Non ancré à un BOS** : un OB ICT précède un break structurel ; ici l'OB est détecté indépendamment de toute structure → zones non significatives.
- **Force mélange deux unités** (D2-2) : `base_strength` (ratio taille/ATR, continu ~0..N) **+** bonus FVG **+0.2 plat**. Additionner un ratio et une constante de prime brouille l'échelle, puis le mapper bucketise en `importance` (seuils 0.75/0.4) → la prime FVG peut faire basculer un OB de `low`→`medium` artificiellement.
- **Pas de mitigation** (cf. E/FVG).

**→ Améliorations** :
- **D2-1** *(Moyen / impact Fort / risque Moyen — GATED annotation)* : ajouter un **filtre displacement** simple — ne retenir l'OB que si la bougie `i` (ou `i`+1) a un corps `> k×ATR` (impulsion), **ou** rattacher l'OB au `BOS_EVENT` suivant dans une fenêtre courte. Rapproche de l'ICT, réduit fortement les faux positifs. **Ne pas livrer avant l'annotation manuelle** (change le volume d'OB) — sinon risque de casser la précision mesurée. Impact honnêteté/qualité **Fort**.
- **D2-2** *(Petit / impact Faible / risque Faible)* : ne **pas** additionner la prime FVG à un ratio (séparer `OB_STRENGTH_NORM` brut et un flag `OB_HAS_FVG` booléen), pour que le bucket `importance` reste interprétable. Touche le moteur (champ) **et** le mapper (bucket) → à coordonner avec l'autre terminal pour la partie mapper.

---

### E. FAIR VALUE GAP — `_add_smc_base_features` (`strategy_features.py:646-679`)

**Heuristique** : gap 3 bougies canonique (`low[i] > high[i-2]` bull). Normalisé ATR. Signal si `|FVG_SIZE_NORM| > FVG_THRESHOLD` (défaut **0.1** ATR).

**Forces** : 🟢 **Formule canonique ICT** (validée audit précédent). Vectorisé.

**Faiblesses** :
- **Seuil 0.1×ATR ≈ spread XAU** → laisse passer des micro-gaps = bruit (P1-2). Sur XAU M15, 10 % d'ATR est de l'ordre du spread+slippage → des « FVG » sans signification.
- **Pas de suivi de remplissage** : `FVG_SIGNAL` est instantané ; le statut `filled/partially_filled` du schéma n'est **jamais** alimenté → un FVG « actif » peut être déjà comblé.
- **Gap de séance/weekend assimilé à un FVG** (cf. 2.2-A) : `low[dimanche] > high[vendredi]` produit un énorme FVG fantôme.

**→ Améliorations** :
- **D2-3** *(Petit / impact Moyen / risque Faible — calibrer via annotation)* : relever le **défaut** `FVG_THRESHOLD` à ~**0.25-0.35×ATR** (filtre le bruit spread). Devil's advocate : trop haut rate les vrais petits gaps → **calibrer** sur le dataset d'annotation avant de figer. Changement de défaut = trivial ; la valeur exacte est l'objet de la validation empirique.
- **D2-4** *(Moyen / impact Moyen / risque Faible)* : ajouter une **passe de suivi de remplissage** FVG/OB (voir ci-dessous, partagée).

---

### F. RETEST — state machine (`strategy_features.py:259-334`)

**Heuristique** : après un `BOS_EVENT`, AWAITING (±1) → ARMED (±2) si le low/high revient à `tol×ATR` du niveau ; invalidation si close dépasse `invalid_tol×ATR`. Tol=0.5, invalid=1.0, awaiting_timeout=20, armed_window=30.

**Forces** : 🟢 State-machine propre, déterministe, l'invalidation **précède** le retest (évite de classer un rejet violent comme confirmation — commentaire lignes 296-299, bon réflexe). Bien testée.

**Faiblesses** :
- `RETEST_TOL_ATR=0.5` ≈ spread (P1-3) → un retest peut être déclenché par le bruit.
- **Incohérence de magic number historique** : `armed_window=5` (défaut fonction ligne 420) vs `RETEST_ARMED_WINDOW=30` (config ligne 510). Le chemin produit passe la config (30) ; mais l'API basse de la fonction garde 5. Source de confusion (P1-1).

**→ Améliorations** :
- **D2-5** *(Petit / impact Faible / risque Faible)* : aligner le défaut de `calculate_bos_retest_fast(armed_window=...)` sur `RETEST_ARMED_WINDOW` (ou retirer le défaut pour forcer l'appelant). Calibrer `RETEST_TOL_ATR` avec l'annotation.

---

### G. RSI DIVERGENCE — `_detect_rsi_divergence` (`strategy_features.py:817-879`)

**Heuristique** : compare chaque fractal au précédent de même type dans `lookback=5` fractals ; émet `CHOCH_DIVERGENCE` ±1.

**Constat majeur** : **ce champ n'est PAS consommé par le `MarketReading`** (le mapper ne le lit pas). Il est calculé à **chaque** `analyze()` (boucle Python O(n·k), non chronométrée) **pour rien** dans le chemin produit. Historique de bug d'indexation (P0-15).

**→ Améliorations** :
- **D2-9** *(Petit / impact Faible-Moyen / risque Faible)* : **ne pas calculer** la divergence RSI dans le chemin produit (paramétrer `analyze(compute_divergence=False)` ou la sortir de la séquence produit). Économise une boucle O(n·k) à chaque reading et retire du code mort du chemin client. Si la divergence doit un jour être exposée, la wirer proprement d'abord.

---

## 2.2 — Cas limites mal couverts (heuristiques)

| # | Cas limite | Comportement actuel | Comportement idéal | Gap (sévérité) | Coût fix |
|---|---|---|---|---|---|
| A | **Gap weekend/holiday** | `low[dim] > high[ven]` → **FVG géant fantôme** ; structure peut sur-fire. Aucune détection de Δt. | Détecter discontinuité temporelle, neutraliser FVG/BOS sur la bougie de gap. | 🔴 Élevée (live feed) | **Moyen** (D4-1) |
| B | **News majeure (NFP/FOMC)** | Bougie à très haute volatilité = gros corps → BOS/OB déclenchés ; ATR encore bas (lag Wilder 14) → strength & seuils sous-estiment. | Idéalement neutraliser/flagger la fenêtre news (le pipeline news existe mais n'alimente pas le détecteur). | 🟡 Modérée | Moyen→HP |
| C | **Doji / corps minuscule** | OB exige `close>open` strict ; un doji (`close==open`) ne déclenche pas d'OB (OK). BOS sur close indifférent au corps. | OK globalement. | 🟢 Faible | — |
| D | **Mèche longue (wick > 5× corps)** | Fractale posée sur la **mèche** ; BOS exige close au-delà → **faux négatif** ; ou OB zone = range mèche-inclus très large. | Option fractale « corps » ou clamp wick. | 🟡 Modérée | Moyen (D4-4) |
| E | **Consolidation extrême (micro-bougies)** | Peu de fractals → warning loggé (lignes 706-710) ; structure gelée ; FVG quasi nuls. | Acceptable (le warning existe). | 🟢 Faible | — |
| F | **Flash crash / fat finger (>3σ)** | Bougie aberrante crée fractal + possible BOS + OB ; aucune détection d'outlier. | Flagger (ne pas altérer) la bougie aberrante. | 🟡 Modérée | Petit-Moyen (D4-3) |

> Détail complet et stratégies défensives : **`RD_AUDIT_D4_CAS_LIMITES.md`**.

---

## 2.3 — Synthèse Dimension 2

**État des heuristiques : 🟢 Acceptable, honnête, majoritairement canonique.** FVG et retest sont conformes ; BOS/CHOCH sont déterministes et bien protégés contre le sur-firing. La seule définition franchement non-canonique reste l'**OB englobant** (déjà actée, ne pas vendre comme « institutionnel »).

**Les faiblesses dominantes sont des seuils ≈ spread** (FVG 0.1, retest 0.5) **non calibrés**, et **l'absence de suivi de remplissage** (statut toujours « actif »). Ce sont des corrections **Petit/Moyen, additives, à faible risque** — pas une réécriture.

**Améliorations retenues (Petit/Moyen)** : D2-2, D2-3, D2-5, D2-6, D2-7, D2-9 (Petit) ; D2-1, D2-4 (Moyen). **Hors-portée** : vrai CHOCH interne (D2-8), refonte OB ICT complète au-delà du filtre displacement.

**Validation empirique requise** (ne pas figer les valeurs sans elle) : seuils FVG/retest et filtre displacement OB **dépendent de l'annotation manuelle des 60 candles** (template déjà produit par l'audit précédent).

**Cas STOP rencontrés en D2** : aucun. L'OB englobant est une **divergence de définition documentée**, pas un bug ; aucune heuristique ne produit de résultat *faux* au sens « contredit sa propre définition ».

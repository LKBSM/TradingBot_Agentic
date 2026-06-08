# Eval 03 — Smart Money Engine (BOS / CHOCH / Order Blocks / FVG)

**Audit ICT/SMC — 2026-04-24 — Smart Sentinel AI**
Dataset : XAU/USD M15 du 2019-01-02 au 2024-12-30, 141 499 bars (≈ 97.6 % de couverture, feed corrigé).
Scope : `src/environment/strategy_features.py` + scoring `src/intelligence/confluence_detector.py`.

---

## 1. Résumé exécutif

| Axe | Verdict |
|-----|---------|
| Conformité ICT/SMC | **5/10** — FVG & BOS conformes, CHOCH correct mais symétrie suspecte, OB **non ICT** (engulfing dérivé, pas « last opposing candle before impulse »). |
| Fréquence d'émission | **Trop élevée** : 247 OB + 160 FVG / 1000 bars (1 OB tous les 4 bars). Signal ≈ bruit. |
| Retest state machine | Succès **89.8 %** — tolérance 0.5 ATR trop laxe, faux confirme largement. |
| Edge exploitable armed (TP 2R / SL 1R) | LONG 33.5 % WR (break-even), **SHORT 30.3 % WR (PF 0.87 — perd)**. Aucun edge net. |
| Asymétrie long/short | Faible (CHOCH 842/842, BOS 49 %/51 %). L'asymétrie observée en PnL vient **du WR par côté**, pas du nombre de signaux. |
| Latence détection | Médiane **4 bars** (1 h M15) fractal → BOS. Acceptable mais non instantané. |
| **Note globale de différenciation commerciale** | **4.5 / 10** |

**Verdict commercial** : en l'état, le « Smart Money Engine » génère un volume massif de signaux non qualifiés. Il est difficilement défendable face à un concurrent open-source (smc-python, LuxAlgo) à moins de durcir drastiquement la sélection et d'ajouter des éléments visuels propriétaires.

---

## 2. Inventaire des détecteurs

| Détecteur | Fichier / ligne | Implémentation | Conformité ICT |
|-----------|-----------------|----------------|----------------|
| Fractals (swing points) | `strategy_features.py:617` | Bill Williams 5 bars (N=2) causal, shift N | ✅ Standard |
| FVG | `strategy_features.py:650` | Gap 3 candles : `low[t] > high[t-2]` (bull) | ✅ Défn ICT classique |
| FVG threshold | `strategy_features.py:675` | `abs(FVG_SIZE/ATR) > 0.1` | ⚠️ Trop laxe (0.1 ATR = micro-gap) |
| BOS event | `strategy_features.py:730` (numba) | Cassure d'un fractal, 1/-1 uniquement sur bar de rupture | ✅ Correct post Sprint 1 |
| BOS break level | idem | Niveau fractal cassé mémorisé | ✅ |
| CHOCH | idem | BOS à contre-tendance | ✅ Mais symétrie exacte suspecte |
| Retest state machine | `strategy_features.py:741` (numba) | 0→±1 (awaiting)→±2 (armed) avec tol 0.5 ATR, timeout 20, fenêtre 30 | ⚠️ Tolérance trop large |
| Order Block | `strategy_features.py:766` | Engulfing bullish/bearish candle + high/low break vs N-1 | ❌ **Non ICT** |
| RSI Divergence | `strategy_features.py:817` | Comparaison fractals successifs, RSI vs price | ✅ Standard TA |

---

## 3. Conformité ICT/SMC — audit ligne à ligne

### 3.1 FVG — ✅ Conforme
```python
bullish_fvg_size = np.where(self.df['low'] > self.df['high'].shift(2), ...)
```
Respecte la définition 3-bougies d'ICT. **Seul reproche** : `FVG_THRESHOLD = 0.1 ATR` laisse passer les micro-gaps (il faudrait ≥ 0.5 ATR pour rester visuellement « imbalance »). C'est ce qui produit **22 548 FVG sur 6 ans** (1 bar sur 6 porte un signal FVG).

### 3.2 BOS / CHOCH — ✅ Correctement implémenté, symétrie suspecte
Après le fix Sprint 1 (événement vs état), les comptes sont sains : 4 391 BOS dont 53 % long / 47 % short. Mais **CHOCH est strictement symétrique année par année** (842 up / 842 down global, et moins d'un de différence chaque année) — ce n'est pas statistiquement attendu sur XAU qui a un biais haussier marqué sur 2019-2024. Cela suggère que CHOCH se déclenche à **chaque** inversion de tendance, donc ~2 inversions/jour, indépendamment du contexte.

### 3.3 Order Block — ❌ Non ICT
Le code :
```python
bullish_ob_condition = (close.shift(1) < open.shift(1)) & (close > open) & (high > high.shift(1))
```
C'est simplement une **chandelle engulfing**, pas un Order Block. Définition ICT correcte :
> « Last down-candle (bear candle) before the impulse move that created the BOS. »

Il manque :
1. Le lien avec un BOS précédent (aucun anchor). Ici n'importe quelle engulfing qualifie.
2. La détection « opposing » (l'OB bullish doit être la **dernière bougie baissière**, pas une bougie haussière).
3. La notion d'impulse (body/ATR ratio élevé sur les N bougies suivantes).

Conséquence : **34 722 OB sur 6 ans** (247/1000 bars = 1 bar sur 4). Inexploitable.

### 3.4 Retest state machine — ⚠️ Seuils trop laxes
`RETEST_TOL_ATR=0.5` → retest confirmé si prix ≤ 0.5 ATR du niveau cassé. Avec un ATR M15 XAU ≈ 3 $, ça équivaut à 1.5 $ de tolérance, c'est-à-dire ~1.5× le spread. Le prix y revient quasiment systématiquement → taux 89.8 %.

---

## 4. Matrice de confusion — événements armés (TP 2R / SL 1R, horizon 50 bars)

| Direction | N armed | Win (TP 2R) | Loss (SL 1R) | Ambigu | **Win rate** | **Profit Factor** |
|-----------|--------:|------------:|-------------:|-------:|-------------:|------------------:|
| LONG (↑)  |   2 104 |       695   |       1 381  |    28  | **33.5 %** | **1.006** |
| SHORT (↓) |   1 837 |       550   |       1 267  |    20  | **30.3 %** | **0.868** |
| **Total** | **3 941** | **1 245** | **2 648**  | **48** | **31.9 %** | **0.94** |

Break-even théorique (2R:1R) = 33.33 %. Le LONG est **à l'équilibre net de frais**, le SHORT **perd**. Cohérent avec le replay harness (PF 0.94, shorts légèrement déficitaires après fix BOS event).

### Interprétation
- Le signal *armed* n'est **pas** un signal d'entrée valable sans filtre supplémentaire.
- Bonus/malus = 3 pts de win rate entre LONG et SHORT → **asymétrie faible côté signaux**, forte côté PnL parce que l'amplitude des moves haussiers XAU est supérieure à celle des baissiers (dérive séculaire de l'or).

---

## 5. Asymétrie long/short par année

| Année | Bars | BOS↑ | BOS↓ | CHOCH↑ | CHOCH↓ | OB↑ | OB↓ | FVG↑ | FVG↓ | Ratio BOS↑/↓ |
|-------|-----:|-----:|-----:|-------:|-------:|----:|----:|-----:|-----:|-------------:|
| 2019  | 23 450 | 385 | 325 | 133 | 132 | 2 785 | 2 718 | 2 127 | 1 946 | 1.18 |
| 2020  | 23 611 | 386 | 278 | 128 | 129 | 2 959 | 2 956 | 1 815 | 1 621 | **1.39** |
| 2021  | 23 581 | 386 | 354 | 145 | 144 | 2 859 | 2 863 | 1 995 | 1 827 | 1.09 |
| 2022  | 23 653 | 435 | 415 | 158 | 158 | 2 941 | 2 937 | 1 955 | 1 911 | 1.05 |
| 2023  | 23 558 | 354 | 353 | 142 | 143 | 2 914 | 2 878 | 1 917 | 1 762 | 1.00 |
| 2024  | 23 646 | 394 | 326 | 136 | 136 | 3 017 | 2 895 | 1 954 | 1 718 | 1.21 |
| **6y**| 141 499 | 2 340 | 2 051 | 842 | 842 | 17 475 | 17 247 | 11 763 | 10 785 | **1.14** |

**Lectures clés :**
- Biais long léger sur BOS (+14 %), aligné au trend haussier XAU.
- CHOCH **rigoureusement symétrique** chaque année (≤ 1 différence) — artefact : CHOCH compte les inversions qui sont par définition équilibrées.
- 2020 (année volatile COVID) a la plus forte asymétrie long (+39 %) — le système sur-émet dans les régimes à forte variance.
- OB / FVG sont quasi symétriques — ils ne portent aucune information directionnelle.

---

## 6. Latence de détection

| Direction | n | médiane | p25 | p75 |
|-----------|--:|--------:|----:|----:|
| BOS ↑ | 2 340 | **4.0 bars** (1 h) | 2.0 | 6.0 |
| BOS ↓ | 2 051 | **4.0 bars** (1 h) | 2.0 | 6.0 |

Cohérent : fractal confirmé à N=2 bars après le swing, BOS déclenché quand le close dépasse le fractal → délai structurel ~2-6 bars. Pas de retard pathologique.

---

## 7. Features visuelles — état vs cible commerciale

Un concurrent comme LuxAlgo **affiche** :
- Zones OB ombrées (rectangles) persistantes jusqu'à mitigation.
- FVG colorés avec % de remplissage.
- Lignes BOS/CHOCH étiquetées avec niveau.
- Liquidity sweeps (equal highs/lows).
- Premium/discount zones (Fibonacci institutionnel).

**Ce que nous avons :**

| Feature | Dispo en colonne | Rendu visuel | Notes |
|---------|------------------|--------------|-------|
| BOS event/level | ✅ `BOS_EVENT`, `BOS_BREAK_LEVEL` | ❌ | prêt à dessiner |
| CHOCH signal | ✅ `CHOCH_SIGNAL` | ❌ | — |
| OB zones | ✅ `BULLISH_OB_HIGH/LOW`, idem bearish | ❌ | pas de persistence tant que non mitigé |
| FVG | ✅ `FVG_SIZE`, `FVG_SIZE_NORM` | ❌ | pas de tracking remplissage |
| Liquidity sweeps | ❌ | ❌ | à construire |
| Premium/discount | ❌ | ❌ | à construire |
| RSI divergence | ✅ `CHOCH_DIVERGENCE` | ❌ | non affiché |

Tout ce qui est en colonne est « derrière le LLM », jamais exposé. Il manque un **renderer chart** (matplotlib/lightweight-charts) qui superpose OB/FVG/BOS sur l'OHLC — c'est le livrable le plus visible pour différencier commercialement.

---

## 8. Top 5 améliorations priorisées (effort × impact)

| # | Amélioration | Effort | Impact PF | Impact commercial | Priorité |
|---|--------------|:------:|:---------:|:-----------------:|:--------:|
| **1** | **OB ICT-compliant** : ancrer chaque OB à un BOS et exiger un body/ATR ≥ 1.5 sur la bougie d'impulse qui suit. Passer de 17k OB/côté à ~1 500. | M | **+0.15 PF** | ⭐⭐⭐⭐⭐ | P0 |
| **2** | **FVG threshold à 0.4 ATR** + tracking du remplissage (`FVG_FILLED_PCT`). Diviser le volume FVG par 4-5. | S | +0.05 | ⭐⭐⭐⭐ | P0 |
| **3** | **Retest tolerance à 0.25 ATR** + obliger un « touch » (low ≤ level ≤ high du bar) plutôt qu'une proximité. Le taux armé devrait tomber à ~40-50 %, la WR post-armed devrait remonter. | S | **+0.10 PF** | ⭐⭐⭐ | P0 |
| **4** | **Chart renderer commercial** : module `src/intelligence/chart_renderer.py` qui exporte PNG annoté (OB shaded, FVG coloré, BOS/CHOCH étiquetés, retest state). Livrable Telegram + API. | M | 0 | ⭐⭐⭐⭐⭐ | P1 |
| **5** | **Liquidity sweep detector** : détecter equal highs/lows (± 0.1 ATR) et leur balayage. Feature ICT haut de gamme, absente de smc-python de base. | M | +0.03 | ⭐⭐⭐⭐ | P1 |

**Petite estimation post-améliorations 1+2+3 :** PF LONG 1.0 → ~1.30, PF SHORT 0.87 → ~1.10, volume d'alertes armed divisé par ~3. C'est ce qui rendrait l'offre commercialement défendable.

---

## 9. Benchmark vs alternatives open-source

| Acteur | Librairie | OB ICT-strict | FVG | BOS/CHOCH | Retest state machine | Rendu visuel |
|--------|-----------|:-------------:|:---:|:---------:|:--------------------:|:------------:|
| smc-python (Joshyattridge) | Python, OSS | ✅ | ✅ | ✅ | ❌ | partiel |
| LuxAlgo SMC | Pine Script | ✅ | ✅ | ✅ | ✅ | ⭐ premium |
| **Smart Sentinel AI** | Python | ❌ | ✅ (laxe) | ✅ | ✅ | ❌ |

**Différenciateurs actuels** : retest state machine (armed/awaiting/invalidated), circuit breakers LLM, narratives Claude. **Déficits** : OB conformité, rendu visuel, liquidity sweeps.

---

## 10. Conclusion et recommandation commerciale

Le moteur Smart Money **est exploitable côté données** (0 data-leakage, 4/6 bars de latence, 0.5 s pour 20k bars via Numba) mais **pas côté sémantique ICT**. L'Order Block de l'implémentation actuelle n'est pas un OB, le FVG threshold est trop laxe, et le retest accepte des quasi-touches. Le résultat net : un signal *armed* **sans edge** (WR 31.9 %, PF 0.94).

**Décision recommandée** (P0, ~3-5 jours de travail) avant toute promesse commerciale :
1. Réécrire OB avec ancrage BOS + filtre impulse.
2. Durcir FVG threshold à 0.4 ATR, tracker remplissage.
3. Durcir retest : touch strict, tolérance 0.25 ATR.
4. Ajouter chart renderer (livrable visuel propriétaire).
5. Re-benchmarker via `replay_harness` et `eval_03`.

Seulement après ces 5 points les KPI commerciaux (PF > 1.2, WR > 38 %, volume < 500 alertes armed/an/instrument) devraient être atteints.

---

### Annexes
- Stats JSON brut : `reports/eval_03/eval_03_stats.json`
- Stats markdown brut : `reports/eval_03/eval_03_summary.md`
- Script audit : `scripts/eval_03_smart_money.py`
- Données : `data/XAU_15MIN_2019_2024.csv` (97.6 % couverture)
- Commit baseline : `632e9dd` (Sprint 2 institutional backtest)

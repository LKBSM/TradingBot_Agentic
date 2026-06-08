# RD_AUDIT_D3 — Performance computationnelle

> **Audit R&D exploratoire — Dimension 3/5**
> Date : 2026-06-08 · Branche : `audit/rd-exploratoire-moteur-detection`
> **Mesures réelles effectuées** dans cet environnement (Python 3.12.6, fallback Python — numba absent). Aucune modification de code.

---

## 3.1 — Analyse de complexité (par détecteur)

| Détecteur | Fonction | Complexité | Justification |
|---|---|---|---|
| TA indicators | `_add_ta_indicators` | **O(n)** | RSI/MACD/BB/ATR via lib `ta` (vectorisé), mais **coût fixe élevé** (instanciation de 4 objets indicateurs/appel) |
| Fractals | `_add_smc_base_features` | **O(n)** | `rolling(center=True).max/min` vectorisé pandas |
| FVG | idem | **O(n)** | `np.where` vectorisé sur shifts |
| Order Blocks | `_add_smc_order_blocks` | **O(n)** | conditions vectorisées + `np.where` |
| BOS / CHOCH | `_calculate_bos_choch_*` | **O(n)** | boucle séquentielle simple (état propagé) ; numba si dispo, sinon **boucle Python pure** |
| Retest | `_calculate_bos_retest_*` | **O(n)** | state-machine séquentielle ; numba|python |
| **RSI divergence** | `_detect_rsi_divergence` | **O(n·k)** | boucle Python imbriquée sur `recent_*_fractals[-lookback:]` (k=`lookback`=5) — **jamais compilée numba**, **non chronométrée**, **non consommée par le produit** |

**Complexité globale** : **O(n·k)** au pire, dominée en réalité par (a) le **coût fixe** des objets `ta` et de pandas sur petits `n`, et (b) les **boucles Python** (BOS/CHOCH, retest, divergence) quand numba est absent.

---

## 3.2 — Mesures réelles (cet environnement)

> **Constat critique** : `import numba` échoue → `NUMBA_AVAILABLE = False`. **Le moteur tourne sur le fallback Python.** Les benchmarks vantés dans les docstrings (« 0.2-0.5 s pour 20 000 bars », « 50-100× ») supposent numba **installé** — ce n'est pas le cas ici, ni nécessairement en prod (numba n'est pas une dépendance garantie).

### Latence end-to-end (`SmartMoneyEngine.analyze()`, moyenne 3 runs, après warm-up)

| n (bougies) | Moyenne | Min | Débit |
|---|---|---|---|
| **200** (lookback produit réel) | **194,6 ms** | 170,5 ms | ~1 030 rows/s |
| 2 000 | 403,6 ms | 380,4 ms | ~4 960 rows/s |
| 20 000 | 2 404,8 ms | 2 324,6 ms | ~8 320 rows/s |

### Ventilation interne à n=200 (chemin produit)

| Étape | Temps | Part |
|---|---|---|
| `smc_features` (fractals + FVG + OB) | 89,3 ms | **43,9 %** |
| `ta_indicators` (RSI/MACD/BB/ATR) | 62,0 ms | **30,4 %** |
| `structure` (BOS/CHOCH + retest) | 31,8 ms | 15,6 % |
| `cleaning` (dropna) | 12,6 ms | 6,2 % |
| *RSI divergence* | *~8 ms (non chronométré)* | *~4 %* |
| **TOTAL** | **~204 ms** | 100 % |

### Lectures clés
1. **À 200 bougies, le coût est dominé par le fixe pandas/`ta` (74 %)**, PAS par les boucles. `n=200→195 ms` mais `n=2000→400 ms` (×2 pour ×10 de données) confirme un **gros coût constant par appel**.
2. → **numba aiderait peu au lookback réel** : il n'accélère que `structure` (16 %) et serait inutile sur `smc_features`/`ta` (vectorisés/objets). Le marketing « 50-100× » concerne le régime 20 000 bars, **pas** le régime produit 200 bars.
3. La **divergence RSI** (~4 %, O(n·k)) est **du gaspillage pur** (non consommée — cf. D2-9).

---

## 3.3 — Optimisations possibles

| ID | Optimisation | Coût | Gain estimé @200 | Risque |
|---|---|---|---|---|
| **D3-1** | **Ne pas calculer la divergence RSI** dans le chemin produit (= D2-9) | Petit | −4 % latence + retire code mort | Faible |
| **D3-2** | **Réutiliser/mémoïser les objets `ta`** ou calculer RSI/MACD/ATR en numpy direct (éviter ré-instanciation par appel) | Moyen | −15 à −25 % (cible les 30 % `ta`) | Moyen (revalider valeurs) |
| **D3-3** | **Installer numba en prod** (déjà géré par fallback) + warm-up au boot (`cache=True` existe) | Petit | −10 à −15 % @200 (le gros gain est @≥2 000) | Faible |
| **D3-4** | **Borner le lookback au strict nécessaire** par TF (200 surdimensionné pour ne sortir qu'1 ligne ; mais ↓ trop casse la mémoire structurelle — arbitrage avec D4-5) | Petit | linéaire | Moyen (qualité détection) |
| **D3-5** | Vectoriser `_detect_rsi_divergence` (si un jour wirée) | Moyen | n/a tant que non consommée | Faible |

> **Anti-reco (Hors-portée)** : ré-écrire le moteur en Polars/Cython, ou cache incrémental bougie-à-bougie. Le gain ne justifie pas le risque au volume V1 (cf. 3.4). **Piège « réinventer ».**

---

## 3.4 — Scalabilité

**Architecture mitigeante : lazy + store (cache).** `get_or_generate` ne régénère que si la dernière bougie a changé (`_payload_matches`). Donc **le nombre d'utilisateurs ne multiplie PAS les appels moteur** : un `MarketReading` par (instrument, TF) est calculé une fois par clôture de bougie, puis servi depuis le store. 🟢 Bon design.

| Scénario | Charge moteur | Verdict |
|---|---|---|
| **V1 : 2 instr × 3 TF = 6 combos** | 6 × ~195 ms ≈ **1,2 s** par vague de régénération (étalée selon les TF) | 🟢 OK |
| **10 instr × 5 TF = 50 combos** | 50 × ~195 ms ≈ **~10 s** si tout régénère **en même temps** (clôture M15 simultanée) | 🟡 **Point de tension** : rafales synchronisées à la clôture |
| **1 000 utilisateurs** | inchangé (cache partagé, 1 reading/combo) | 🟢 OK tant que le store est partagé inter-workers |
| **Bootstrap 10 000 bougies** | ~2,4 s/combo (one-shot historique) | 🟢 Acceptable (rare) |

**Points de cassure prévisibles** :
1. **Rafale de clôture** : à la clôture d'un TF, tous les combos de ce TF régénèrent en bloc. À 50 combos × 195 ms en mono-thread = ~10 s de latence sur le dernier servi. → **D3-6** *(Moyen)* : étaler/quotient la régénération (file + workers), ou pré-générer en tâche de fond avant la clôture.
2. **numba absent en prod** : si le conteneur n'installe pas numba, la latence ≥ 2 000 bougies explose (×5-10 vs annoncé). → **D3-3**.
3. **Cold start numba** (si installé) : première compilation JIT ~quelques secondes sur le 1ᵉʳ reading du process. `cache=True` amortit entre process. → warm-up au boot.

**→ Améliorations scalabilité** :
- **D3-6** *(Moyen / impact Moyen / risque Faible)* : worker pool + file pour lisser les rafales de clôture au-delà de ~20 combos (V1.1+, pas bloquant V1 à 6 combos).
- **D3-3** *(Petit)* : garantir numba dans l'image prod **ou** documenter explicitement que le fallback Python est le mode nominal et dimensionner en conséquence.

---

## 3.5 — Synthèse Dimension 3

**État performance : 🟢 Acceptable pour V1, avec deux angles morts honnêtes.**

1. Au **lookback produit réel (200 bougies)**, une lecture coûte **~170-195 ms** sur le fallback Python — **largement suffisant** pour 6 combos en mode lazy/caché. Le coût est dominé par le fixe pandas/`ta`, **pas** par les boucles → numba apporterait peu **ici**.
2. **Le discours perf des docstrings (« 0.2-0.5 s / 20k bars », « 50-100× »)** ne reflète ni l'environnement réel (numba absent) ni le régime produit (200 bougies). → à recadrer en interne pour ne pas sous-dimensionner la prod.

**Actions Petit à fort ratio** : D3-1 (retirer divergence du produit, −4 % + propreté), D3-3 (garantir numba OU assumer le fallback). **Moyen** : D3-2 (objets `ta`), D3-6 (lissage rafales, V1.1+). **Aucune réécriture.**

**Cas STOP rencontrés en D3** : aucun. (Le fait que numba soit absent est une **observation d'environnement**, pas un bug du moteur : le fallback fonctionne correctement et est explicitement prévu.)

# MANUAL_ANNOTATION_TEMPLATE_2026_06_06 — Annotation manuelle (founder)

> Audit de validation algorithmique — Phase 2.4
> **À compléter par le founder en regardant TradingView** (H1, même instrument, même heure UTC).
> Le bloc « Détection algo MIA Markets » est pré-rempli pour comparaison directe.

## Mode d'emploi

1. Ouvre TradingView sur l'instrument + H1, va à la bougie indiquée (heure d'OUVERTURE UTC).
2. Annote ce que TU vois (BOS/CHOCH/OB/FVG/phase).
3. Compare au bloc algo. Coche le verdict.
4. Reporte le verdict dans `SCORING_TEMPLATE_2026_06_06.csv` (colonnes `manual_*` + `verdict`).

**Rappel niveaux** : les niveaux de prix BOS/CHOCH/OB/FVG affichés par l'algo sont des
**proxies** (cf. findings F1-F3 de `STRUCTURE_DEFINITIONS_AUDIT.md`) — ne valide PAS les
niveaux chiffrés, valide la **présence/direction/phase**.

---

### Candle #1 — XAUUSD H1 — open 2026-01-01T23:00:00 (close 2026-01-02T00:00:00) — Close: 4348.25500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché oscille dans une bande de consolidation avec une volatilité normale et un alignement haussier sur les timeframes supérieures.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #2 — XAUUSD H1 — open 2026-01-02T00:00:00 (close 2026-01-02T01:00:00) — Close: 4354.12850

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bearish', 'fvg_active', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché oscille dans une plage de consolidation avec une volatilité normale, tandis que les timeframes supérieures affichent une orientation haussière malgré la phase de range actuelle.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #3 — XAUUSD H1 — open 2026-01-02T03:00:00 (close 2026-01-02T04:00:00) — Close: 4375.18500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **bullish**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bullish', 'choch_recent_bullish', 'fvg_active', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix oscillent dans une zone de consolidation avec des signaux heureux récents et un alignement haussier sur les timeframes supérieures.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #4 — XAUUSD H1 — open 2026-01-05T05:00:00 (close 2026-01-05T06:00:00) — Close: 4412.56850

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'retest_in_progress', 'mtf_aligned']`
- Description (haiku_generated) : *Le prix reteste une zone de support dans un contexte de tendance haussière confirmée sur les timeframes jour et 4h, avec une volatilité normale.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #5 — XAUUSD H1 — open 2026-01-06T14:00:00 (close 2026-01-06T15:00:00) — Close: 4487.78850

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bullish', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix évoluent dans une plage de consolidation avec des signaux haussiers alignés sur les graphiques quotidien et 4-heures dans un contexte de volatilité normale.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #6 — XAUUSD H1 — open 2026-01-09T05:00:00 (close 2026-01-09T06:00:00) — Close: 4468.16000

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**low**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_low', 'phase_ranging', 'bos_recent_bullish', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix oscillent dans une plage définie avec une tendance haussière confirmée sur les deux timeframes supérieurs et une volatilité réduite.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #7 — XAUUSD H1 — open 2026-01-12T00:00:00 (close 2026-01-12T01:00:00) — Close: 4588.92650

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bullish', 'fvg_active', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent dans un environnement haussier avec une volatilité élevée, les timeframes alignés confirment la phase d'expansion en cours.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #8 — XAUUSD H1 — open 2026-01-16T10:00:00 (close 2026-01-16T11:00:00) — Close: 4600.23500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent dans un contexte haussier confirmé sur les timeframes journalier et 4-heures, avec une volatilité normale et un momentum récent positif.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #9 — XAUUSD H1 — open 2026-01-20T23:00:00 (close 2026-01-21T00:00:00) — Close: 4775.94500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent dans un contexte haussier confirmé sur les timeframes journalier et 4-heures, avec une volatilité normale et un momentum récent positif.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #10 — XAUUSD H1 — open 2026-01-26T03:00:00 (close 2026-01-26T04:00:00) — Close: 5076.79500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bullish', 'mtf_aligned']`
- Description (haiku_generated) : *Les cours progressent sur plusieurs timeframes avec une volatilité élevée en phase d'expansion haussière.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #11 — XAUUSD H1 — open 2026-01-28T01:00:00 (close 2026-01-28T02:00:00) — Close: 5204.69000

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['medium'])
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bullish', 'ob_active', 'mtf_aligned']`
- Description (haiku_generated) : *La structure montre une tendance haussière alignée multi-timeframes avec volatilité élevée, phase d'expansion active et niveaux de support/résistance identifiés.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #12 — XAUUSD H1 — open 2026-01-29T15:00:00 (close 2026-01-29T16:00:00) — Close: 5182.82500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **bearish**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'choch_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *La tendance haussière persiste sur plusieurs horizons temporels avec une volatilité élevée durant la phase d'expansion, bien que des signaux baissiers récents soient présents localement.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #13 — XAUUSD H1 — open 2026-01-30T01:00:00 (close 2026-01-30T02:00:00) — Close: 5301.14000

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent dans un contexte haussier multi-timeframe avec volatilité élevée et phase d'expansion, malgré un signal baissier récent de court terme.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #14 — XAUUSD H1 — open 2026-02-02T05:00:00 (close 2026-02-02T06:00:00) — Close: 4549.61650

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['high'])
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**elevated**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_elevated', 'phase_ranging', 'bos_recent_bearish', 'ob_active', 'mtf_mixed']`
- Description (haiku_generated) : *Les prix oscillent dans une plage définie avec une volatilité accrue, le graphique quotidien affichant une direction haussière tandis que les délais plus courts demeurent en consolidation.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #15 — XAUUSD H1 — open 2026-02-05T16:00:00 (close 2026-02-05T17:00:00) — Close: 4889.06650

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['high'])
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bearish', 'ob_active', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché évolue dans une phase de consolidation avec une volatilité normale, tandis que les timeframes supérieures affichent une orientation haussière contrastant avec un signal baissier récent.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #16 — XAUUSD H1 — open 2026-02-13T17:00:00 (close 2026-02-13T18:00:00) — Close: 5021.11000

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *Les cours évoluent à la hausse avec une volatilité normale, confirmée par l'alignement haussier des timeframes journalier et 4-heures, bien qu'une structure baissière récente soit observable.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #17 — XAUUSD H1 — open 2026-02-16T12:00:00 (close 2026-02-16T13:00:00) — Close: 5002.02500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**low**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_low', 'phase_ranging', 'bos_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché oscille dans une plage étroite avec une volatilité réduite, tandis que les timeframes supérieures maintiennent une structure haussière.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #18 — XAUUSD H1 — open 2026-02-17T04:00:00 (close 2026-02-17T05:00:00) — Close: 4953.00000

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**low**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_low', 'phase_ranging', 'bos_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché oscille dans une plage étroite avec une volatilité réduite, tandis que les timeframes supérieures maintiennent une structure haussière.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #19 — XAUUSD H1 — open 2026-02-24T13:00:00 (close 2026-02-24T14:00:00) — Close: 5115.14850

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **bearish**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bearish', 'choch_recent_bearish', 'fvg_active', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché oscille dans une zone de consolidation avec une structure baissière récente, des vides de prix actifs et une tendance haussière confirmée sur les périodes supérieures.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #20 — XAUUSD H1 — open 2026-02-27T20:00:00 (close 2026-02-27T21:00:00) — Close: 5263.41500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **bullish**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'choch_recent_bullish', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent dans un contexte haussier aligné sur plusieurs timeframes avec volatilité modérée et structure en phase de tendance.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #21 — XAUUSD H1 — open 2026-03-02T06:00:00 (close 2026-03-02T07:00:00) — Close: 5387.71500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bullish', 'retest_in_progress', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché affiche une tendance haussière confirmée sur les timeframes quotidienne et 4h, en phase d'expansion avec volatilité élevée et un retesting en cours.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #22 — XAUUSD H1 — open 2026-03-02T13:00:00 (close 2026-03-02T14:00:00) — Close: 5394.02500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'retest_in_progress', 'mtf_aligned']`
- Description (haiku_generated) : *Le prix reteste une zone de support dans un contexte de tendance haussière confirmée sur les timeframes jour et 4h, avec une volatilité normale.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #23 — XAUUSD H1 — open 2026-03-02T14:00:00 (close 2026-03-02T15:00:00) — Close: 5333.60350

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **bearish**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'choch_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *La tendance haussière persiste sur plusieurs horizons temporels avec une volatilité élevée durant la phase d'expansion, bien que des signaux baissiers récents soient présents localement.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #24 — XAUUSD H1 — open 2026-03-03T10:00:00 (close 2026-03-03T11:00:00) — Close: 5248.27500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent dans un contexte haussier multi-timeframe avec volatilité élevée et phase d'expansion, malgré un signal baissier récent de court terme.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #25 — XAUUSD H1 — open 2026-03-05T21:00:00 (close 2026-03-05T22:00:00) — Close: 5083.64000

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché oscille dans une bande de consolidation avec une volatilité normale et un alignement haussier sur les timeframes supérieures.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #26 — XAUUSD H1 — open 2026-03-18T20:00:00 (close 2026-03-18T21:00:00) — Close: 4818.56500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bearish', 'fvg_active', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière avec volatilité normale, des gaps de fair value actifs et une dynamique quotidienne haussière contrariée par une consolidation en quatre heures.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #27 — XAUUSD H1 — open 2026-03-19T04:00:00 (close 2026-03-19T05:00:00) — Close: 4849.82500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bearish', 'retest_in_progress', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière sur le court terme avec une volatilité normale, tandis que le daily maintient une orientation haussière et l'H4 évolue en range.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #28 — XAUUSD H1 — open 2026-03-19T13:00:00 (close 2026-03-19T14:00:00) — Close: 4621.29000

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bearish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'fvg_active', 'mtf_divergent']`
- Description (template_fallback) : *Tendance baissière, volatilité élevée, phase d'expansion. MTF mixte.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #29 — XAUUSD H1 — open 2026-03-22T23:00:00 (close 2026-03-23T00:00:00) — Close: 4464.79000

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'mtf_divergent']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière avec volatilité élevée, divergence multi-timeframes (haussière en D1, baissière en H4) et phase d'expansion des mouvements.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #30 — XAUUSD H1 — open 2026-03-26T19:00:00 (close 2026-03-26T20:00:00) — Close: 4364.16000

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **bearish**
- Order Blocks actifs : **1** (importance ['high'])
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bearish', 'choch_recent_bearish', 'ob_active', 'mtf_divergent']`
- Description (haiku_generated) : *Les prix évoluent dans une tendance baissière avec une volatilité normale, tandis que le graphique quotidien affiche une orientation haussière contrastant avec la tendance baissière observée en périodes intraday.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #31 — EURUSD H1 — open 2025-10-01T00:00:00 (close 2025-10-01T01:00:00) — Close: 1.17359

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**low**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_low', 'phase_trend', 'bos_recent_bullish', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière en phase de trend avec une volatilité faible, tandis que le jour montre une orientation haussière et l'intraday un mouvement de consolidation.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #32 — EURUSD H1 — open 2025-10-01T01:00:00 (close 2025-10-01T02:00:00) — Close: 1.17392

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**low**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_low', 'phase_ranging', 'bos_recent_bullish', 'mtf_mixed']`
- Description (haiku_generated) : *Les prix oscillent dans une fourchette établie avec une volatilité réduite, tandis que la tendance quotidienne affiche des signaux haussiers contrastant avec la consolidation sur les périodes intermédiaires.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #33 — EURUSD H1 — open 2025-10-01T07:00:00 (close 2025-10-01T08:00:00) — Close: 1.17500

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['high'])
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**elevated**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_elevated', 'phase_ranging', 'bos_recent_bullish', 'ob_active', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché oscille dans une fourchette avec volatilité accrue, soutenu par une structure haussière en daily tandis que l'échelle 4H demeure latérale.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #34 — EURUSD H1 — open 2025-10-02T15:00:00 (close 2025-10-02T16:00:00) — Close: 1.16967

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **bearish**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'choch_recent_bearish', 'mtf_mixed']`
- Description (haiku_generated) : *Les prix se situent en tendance baissière sur le court terme avec une volatilité accrue, tandis que le graphique journalier affiche une orientation haussière et l'H4 oscille sans direction claire.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #35 — EURUSD H1 — open 2025-10-02T20:00:00 (close 2025-10-02T21:00:00) — Close: 1.17158

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bearish', 'retest_in_progress', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière sur le court terme avec une volatilité normale, tandis que le daily maintient une orientation haussière et l'H4 évolue en range.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #36 — EURUSD H1 — open 2025-10-09T03:00:00 (close 2025-10-09T04:00:00) — Close: 1.16448

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **bullish**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**low**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_low', 'phase_trend', 'bos_recent_bullish', 'choch_recent_bullish', 'mtf_mixed']`
- Description (haiku_generated) : *La tendance baissière persiste avec une volatilité basse, tandis que le daily affiche une structure haussière et l'H4 oscille en range.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #37 — EURUSD H1 — open 2025-10-09T18:00:00 (close 2025-10-09T19:00:00) — Close: 1.15464

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['high'])
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'ob_active', 'mtf_divergent']`
- Description (haiku_generated) : *Les prix affichent une tendance baissière sur divergences multi-temporelles, avec une volatilité élevée et des niveaux de surv​ente en phase d'expansion.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #38 — EURUSD H1 — open 2025-10-14T13:00:00 (close 2025-10-14T14:00:00) — Close: 1.15741

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière avec volatilité élevée, tandis que la dynamique quotidienne reste haussière et le court terme oscille sans direction claire.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #39 — EURUSD H1 — open 2025-10-15T18:00:00 (close 2025-10-15T19:00:00) — Close: 1.16333

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'mtf_mixed']`
- Description (haiku_generated) : *La tendance hebdomadaire et journalière affiche un biais haussier récent tandis que la structure horaire montre une consolidation, dans un contexte de volatilité normale et de phase de tendance baissière générale.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #40 — EURUSD H1 — open 2025-10-20T06:00:00 (close 2025-10-20T07:00:00) — Close: 1.16585

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['medium'])
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bullish', 'ob_active', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché fluctue dans une bande de consolidation avec une tendance haussière sur le timeframe quotidien, tandis que le graphique 4H demeure latéral dans un contexte de volatilité normale.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #41 — EURUSD H1 — open 2025-10-20T17:00:00 (close 2025-10-20T18:00:00) — Close: 1.16434

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **bearish**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bearish', 'choch_recent_bearish', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché oscille dans une zone de consolidation avec des signaux techniques baissiers récents tandis que la volatilité demeure habituelle.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #42 — EURUSD H1 — open 2025-10-24T12:00:00 (close 2025-10-24T13:00:00) — Close: 1.16342

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**elevated**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_elevated', 'phase_ranging', 'bos_recent_bullish', 'mtf_mixed']`
- Description (haiku_generated) : *Les prix oscillent dans une fourchette sans direction claire, avec une volatilité élevée et un contexte haussier quotidien contrasté par une consolidation en 4h.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #43 — EURUSD H1 — open 2025-10-30T20:00:00 (close 2025-10-30T21:00:00) — Close: 1.15646

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['medium'])
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bearish', 'retest_in_progress', 'ob_active', 'mtf_divergent']`
- Description (template_fallback) : *Tendance baissière, volatilité normale, phase de tendance. MTF mixte.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #44 — EURUSD H1 — open 2025-10-31T13:00:00 (close 2025-10-31T14:00:00) — Close: 1.15389

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bearish', 'fvg_active', 'mtf_divergent']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière sur H4 contrastant avec une direction haussière sur D1, avec une volatilité normale et un vide de prix actif en cours de phase de tendance.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #45 — EURUSD H1 — open 2025-11-05T17:00:00 (close 2025-11-05T18:00:00) — Close: 1.14805

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bearish', 'retest_in_progress', 'mtf_divergent']`
- Description (haiku_generated) : *Les prix testent les niveaux précédents dans un contexte de volatilité élevée, avec une tendance baissière court terme contrastant avec une orientation haussière quotidienne.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #46 — EURUSD H1 — open 2025-11-06T13:00:00 (close 2025-11-06T14:00:00) — Close: 1.15358

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **bullish**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'choch_recent_bullish', 'fvg_active', 'mtf_divergent']`
- Description (haiku_generated) : *Un marché en tendance baissière avec volatilité normale présente des divergences multi-temporelles (D1 haussière, H4 baissière), des changements de structure récents et une zone de déséquilibre active.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #47 — EURUSD H1 — open 2025-11-07T11:00:00 (close 2025-11-07T12:00:00) — Close: 1.15581

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'fvg_active', 'mtf_divergent']`
- Description (template_fallback) : *Tendance baissière, volatilité normale, phase de tendance. MTF mixte.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #48 — EURUSD H1 — open 2025-11-11T12:00:00 (close 2025-11-11T13:00:00) — Close: 1.15705

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'mtf_divergent']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière sur le court terme avec volatilité normale, tandis que la tendance haussière quotidienne diverge avec la baisse en 4H.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #49 — EURUSD H1 — open 2025-11-19T10:00:00 (close 2025-11-19T11:00:00) — Close: 1.15694

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'bos_recent_bearish', 'mtf_divergent']`
- Description (template_fallback) : *Tendance en range, volatilité normale, phase de range. MTF mixte.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #50 — EURUSD H1 — open 2025-11-25T13:00:00 (close 2025-11-25T14:00:00) — Close: 1.15584

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **bullish**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**ranging**, volatilité=**elevated**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_elevated', 'phase_ranging', 'bos_recent_bullish', 'choch_recent_bullish', 'fvg_active', 'mtf_mixed']`
- Description (template_fallback) : *Tendance en range, volatilité élevée, phase de range. MTF mixte.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #51 — EURUSD H1 — open 2025-11-28T10:00:00 (close 2025-11-28T11:00:00) — Close: 1.15593

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'mtf_mixed']`
- Description (haiku_generated) : *La tendance hebdomadaire et journalière affiche un biais haussier récent tandis que la structure horaire montre une consolidation, dans un contexte de volatilité normale et de phase de tendance baissière générale.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #52 — EURUSD H1 — open 2025-11-28T18:00:00 (close 2025-11-28T19:00:00) — Close: 1.16016

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**elevated**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_elevated', 'phase_ranging', 'bos_recent_bullish', 'mtf_mixed']`
- Description (haiku_generated) : *Les prix oscillent dans une fourchette sans direction claire, avec une volatilité élevée et un contexte haussier quotidien contrasté par une consolidation en 4h.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #53 — EURUSD H1 — open 2025-12-01T15:00:00 (close 2025-12-01T16:00:00) — Close: 1.16289

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'retest_in_progress', 'mtf_mixed']`
- Description (haiku_generated) : *Les prix progressent à la hausse avec une volatilité normale, les retests de niveaux clés se poursuivent dans un contexte de tendance haussière.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #54 — EURUSD H1 — open 2025-12-03T10:00:00 (close 2025-12-03T11:00:00) — Close: 1.16606

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché affiche une tendance haussière en tendance long terme avec volatilité normale, tandis que le graphique 4H montre une consolidation dans cette dynamique.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #55 — EURUSD H1 — open 2025-12-05T16:00:00 (close 2025-12-05T17:00:00) — Close: 1.16343

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **bearish**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bearish', 'choch_recent_bearish', 'mtf_mixed']`
- Description (haiku_generated) : *La tendance haussière se maintient sur le quotidien avec une volatilité normale, tandis que des signaux baissiers récents apparaissent sur les structures de court terme.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #56 — EURUSD H1 — open 2025-12-11T20:00:00 (close 2025-12-11T21:00:00) — Close: 1.17411

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bullish', 'retest_in_progress', 'fvg_active', 'mtf_aligned']`
- Description (haiku_generated) : *La structure affiche une tendance haussière sur plusieurs timeframes avec volatilité élevée, un gap de liquidité actif et un retestage en cours dans une phase d'expansion.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #57 — EURUSD H1 — open 2025-12-15T02:00:00 (close 2025-12-15T03:00:00) — Close: 1.17414

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (pending)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent dans un contexte haussier confirmé sur les timeframes journalier et 4-heures, avec une volatilité normale et un momentum récent positif.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #58 — EURUSD H1 — open 2025-12-16T12:00:00 (close 2025-12-16T13:00:00) — Close: 1.17779

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'bos_recent_bullish', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent dans un contexte haussier confirmé sur les timeframes journalier et 4-heures, avec une volatilité normale et un momentum récent positif.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #59 — EURUSD H1 — open 2025-12-17T04:00:00 (close 2025-12-17T05:00:00) — Close: 1.17297

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bearish** (confirmed)
- CHOCH : **bearish**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**low**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_low', 'phase_trend', 'bos_recent_bearish', 'choch_recent_bearish', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché affiche une tendance haussière confirmée sur plusieurs timeframes avec une volatilité faible, malgré des signaux baissiers récents aux niveaux locaux.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

### Candle #60 — EURUSD H1 — open 2025-12-17T15:00:00 (close 2025-12-17T16:00:00) — Close: 1.17530

**À annoter manuellement (founder)** :

- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________
- [ ] CHOCH ? (Oui/Non + direction) : ________
- [ ] Nb Order Blocks actifs : ____ — niveaux : ________
- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________
- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / Phase (trend/range/volatile/accumulation/expansion) ____

**Détection algo MIA Markets** :

- BOS : **bullish** (confirmed)
- CHOCH : **bullish**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'bos_recent_bullish', 'choch_recent_bullish', 'fvg_active', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix affichent une tendance haussière confirmée sur plusieurs horizons temporels avec une volatilité élevée, des structures techniques récentes constructives et une phase d'expansion en cours.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

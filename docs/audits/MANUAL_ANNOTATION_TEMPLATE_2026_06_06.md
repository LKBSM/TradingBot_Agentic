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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché oscille dans une plage de consolidation avec une volatilité normale tandis que les timeframes quotidienne et 4-horaire affichent des signaux haussiers.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'fvg_active', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix oscillent dans une fourchette de trading avec une tendance haussière confirmée sur les timeframes supérieures et une volatilité normale.*

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
- Description (haiku_generated) : *Les prix oscillent dans une plage de consolidation avec des signaux haussiers confirmés sur les périodes journalière et 4h, une zone de liquidité active et une volatilité normale.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'retest_in_progress', 'mtf_aligned']`
- Description (haiku_generated) : *Le cours teste à la hausse un niveau de résistance en phase de tendance haussière avec une volatilité normale et une confirmation multi-timeframe (quotidien et 4h haussiers).*

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
- Description (haiku_generated) : *Les prix oscillent dans une bande de consolidation avec un biais haussier confirmé sur les timeframes quotidienne et 4-heures, dans un contexte de volatilité normale.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**low**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_low', 'phase_ranging', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix oscillent dans une fourchette définie avec une faible volatilité tandis que les timeframes supérieurs affichent une orientation haussière.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'fvg_active', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent sur plusieurs timeframes avec une volatilité accrue durant la phase d'expansion.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché affiche une tendance haussière confirmée sur les horizons quotidien et 4-heures, avec une volatilité normale et une dynamique de tendance établie.*

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
- Description (haiku_generated) : *Le marché affiche une dynamique haussière confirmée sur les timeframes multiples avec une volatilité normale et un momentum récent positif.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'mtf_aligned']`
- Description (haiku_generated) : *Les cours progressent sur plusieurs temporalités avec une volatilité accentuée dans une phase d'expansion haussière.*

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
- Description (haiku_generated) : *Les prix progressent dans une dynamique haussière alignée sur plusieurs timeframes avec une volatilité élevée et une phase d'expansion active.*

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
- Description (haiku_generated) : *Les prix affichent une tendance haussière sur plusieurs échelles de temps avec une volatilité élevée en phase d'expansion, bien que des signaux baissiers récents soient détectés localement.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'mtf_aligned']`
- Description (haiku_generated) : *Les cours progressent sur plusieurs temporalités avec une volatilité accentuée dans une phase d'expansion haussière.*

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
- Description (haiku_generated) : *Le marché oscille dans une bande de prix avec une volatilité élevée, tiraillé entre des pressions haussières quotidiennes et un mouvement latéral sur quatre heures.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['high'])
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'ob_active', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché oscille dans une bande de prix avec une volatilité normale, aligné sur les timeframes supérieures en tendance haussière.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché affiche une tendance haussière confirmée sur les horizons quotidien et 4-heures, avec une volatilité normale et une dynamique de tendance établie.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**low**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_low', 'phase_ranging', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix oscillent dans une fourchette définie avec une faible volatilité tandis que les timeframes supérieurs affichent une orientation haussière.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**low**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_low', 'phase_ranging', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix oscillent dans une fourchette définie avec une faible volatilité tandis que les timeframes supérieurs affichent une orientation haussière.*

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
- Description (haiku_generated) : *Le marché oscille dans une plage de prix avec une volatilité normale tandis que les structures multiples temporelles affichent une orientation haussière.*

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
- Description (haiku_generated) : *Les prix évoluent à la hausse sur plusieurs timeframes avec une volatilité normale et un alignement haussier confirmé par les structures récentes.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'retest_in_progress', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché affiche une tendance haussière sur plusieurs timeframes avec une volatilité élevée en phase d'expansion, et un retest des niveaux clés est en cours.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'retest_in_progress', 'mtf_aligned']`
- Description (haiku_generated) : *Le cours teste à la hausse un niveau de résistance en phase de tendance haussière avec une volatilité normale et une confirmation multi-timeframe (quotidien et 4h haussiers).*

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
- Description (haiku_generated) : *Les prix affichent une tendance haussière sur plusieurs échelles de temps avec une volatilité élevée en phase d'expansion, bien que des signaux baissiers récents soient détectés localement.*

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
- Description (haiku_generated) : *Les cours progressent dans un contexte haussier multi-timeframe avec volatilité élevée durant la phase d'expansion, bien que les données récentes affichent une légère pression baissière.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché oscille dans une plage de consolidation avec une volatilité normale tandis que les timeframes quotidienne et 4-horaire affichent des signaux haussiers.*

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
- Description (haiku_generated) : *Les prix affichent une tendance baissière sur le court terme avec volatilité normale, tandis que le quotidien montre une dynamique haussière et l'H4 une consolidation, créant une divergence multitemps.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'retest_in_progress', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché montre une tendance baissière en phase de retrace avec volatilité normale, tandis que le cadre quotidien reste haussier et l'horaire 4H oscille entre les bornes.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bearish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_elevated', 'phase_expansion', 'fvg_active', 'mtf_divergent']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière avec volatilité élevée, divergence multi-timeframe (haussière en daily, baissière en 4h) et activité FVG durant une phase d'expansion.*

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
- Description (haiku_generated) : *La tendance baissière persiste avec une volatilité élevée tandis que la phase d'expansion se poursuit, créant une divergence entre la tendance haussière sur D1 et baissière sur H4.*

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
- Description (haiku_generated) : *La tendance baissière persiste sur les timeframes courtes avec une volatilité normale tandis que le daily affiche une orientation haussière, créant une divergence multitimeframe avec des niveaux de liquidité actifs.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**low**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_low', 'phase_trend', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière sur timeframe mixte avec volatilité réduite, tandis que le daily montre une dynamique haussière et le H4 une consolidation latérale.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**low**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_low', 'phase_ranging', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché évolue dans une plage de prix stable avec une volatilité réduite, le quotidien affichant une tendance haussière tandis que l'horizon à 4 heures reste consolidé.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['high'])
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**elevated**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_elevated', 'phase_ranging', 'ob_active', 'mtf_mixed']`
- Description (haiku_generated) : *Les cours oscillent dans une fourchette définie avec une volatilité accrue, le cadre quotidien affichant une tendance haussière tandis que l'horizon court terme reste indécis.*

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
- Description (haiku_generated) : *La tendance baissière persiste avec une volatilité élevée en phase d'expansion, tandis que le daily affiche une structure haussière contrastant avec la consolidation en 4H.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'retest_in_progress', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché montre une tendance baissière en phase de retrace avec volatilité normale, tandis que le cadre quotidien reste haussier et l'horaire 4H oscille entre les bornes.*

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
- Description (haiku_generated) : *Les prix affichent une tendance baissière en phase de trend avec une faible volatilité, tandis que les timeframes supérieures montrent un Daily haussier et un H4 en consolidation.*

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
- Description (haiku_generated) : *Les prix évoluent à la baisse avec une volatilité élevée, les divergences multitimeframe persistent tandis que la phase d'expansion continue malgré les signaux baissiers récents.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_elevated', 'phase_expansion', 'mtf_mixed']`
- Description (haiku_generated) : *La tendance baissière sur le court terme contraste avec une expansion haussière intraday, tandis que la volatilité reste élevée entre les timeframes.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché présente une tendance baissière sur le court terme avec une volatilité normale, tandis que la structure quotidienne reste haussière et l'intraday range-bound.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['medium'])
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**normal**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_normal', 'phase_ranging', 'ob_active', 'mtf_mixed']`
- Description (haiku_generated) : *Les cours oscillent dans une fourchette de prix sur le timeframe quotidien haussier tandis que le timeframe 4H demeure limité à une plage, avec une volatilité normale et un order book actif.*

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
- Description (haiku_generated) : *Le marché évolue dans une phase de consolidation avec des cassures baissières récentes, une volatilité normale et une divergence entre la tendance haussière quotidienne et le ranging sur timeframes inférieures.*

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
- Description (haiku_generated) : *Le marché oscille dans une fourchette avec une volatilité élevée, tandis que la tendance quotidienne affiche une orientation haussière et l'échelle 4H demeure dans une consolidation.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **1** (importance ['medium'])
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'retest_in_progress', 'ob_active', 'mtf_divergent']`
- Description (haiku_generated) : *Le marché affiche une tendance baissière sur H4 tandis que D1 reste haussière, avec un retesting en cours et une volatilité normale dans la phase de tendance.*

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
- Description (haiku_generated) : *La structure présente une tendance baissière avec une zone de valeur juste active, une divergence multi-timeframe où le D1 reste haussier tandis que le H4 confirme la baisse, dans un contexte de volatilité normale.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_elevated', 'phase_expansion', 'retest_in_progress', 'mtf_divergent']`
- Description (haiku_generated) : *Le marché teste à nouveau les niveaux précédents en tendance baissière sur H4 tandis que D1 affiche une dynamique haussière, avec une volatilité élevée durant cette phase d'expansion.*

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
- Description (haiku_generated) : *Les prix évoluent dans une tendance baissière avec volatilité normale, marquée par des signaux haussiers récents sur structure plus large alors que le court terme reste orienté à la baisse.*

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
- Description (haiku_generated) : *La structure présente une divergence multi-timeframe avec un mouvement baissier en phase de tendance, une volatilité normale et des zones de value gap actives sur la timeframe inférieure.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bearish, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'mtf_divergent']`
- Description (haiku_generated) : *Le marché présente une tendance baissière en phase de trend avec une volatilité normale, tandis que la divergence multi-timeframe montre une orientation haussière en daily et baissière en H4.*

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
- Description (haiku_generated) : *Le marché oscille dans une zone de consolidation avec une divergence multitemporal : tendance haussière en daily face à une orientation baissière en H4, dans un contexte de volatilité normale.*

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
- Description (haiku_generated) : *Les prix oscillent dans une plage de consolidation avec une volatilité élevée, tandis que la tendance quotidienne affiche une orientation haussière et les cassures récentes montrent une dynamique positive.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bearish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bearish', 'volatility_normal', 'phase_trend', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché présente une tendance baissière sur le court terme avec une volatilité normale, tandis que la structure quotidienne reste haussière et l'intraday range-bound.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**ranging**, volatilité=**elevated**, phase=**ranging**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_ranging', 'volatility_elevated', 'phase_ranging', 'mtf_mixed']`
- Description (haiku_generated) : *Le marché oscille dans une gamme définie avec une volatilité accrue, tandis que la tendance quotidienne affiche une direction haussière et le délai de 4h reste consolidé.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:ranging, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'retest_in_progress', 'mtf_mixed']`
- Description (haiku_generated) : *Le prix teste un niveau de résistance dans une tendance haussière avec une volatilité stable, tandis que le timeframe horaire montre une consolidation.*

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
- Description (haiku_generated) : *Les prix affichent une tendance haussière sur le graphique journalier avec volatilité normale, tandis que la période 4 heures montre une consolidation latérale dans un contexte de phase de tendance.*

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
- Description (haiku_generated) : *Les prix affichent une tendance haussière en daily avec volatilité normale, tandis que l'échelle horaire 4H montre une consolidation, avec des signaux récents baissiers locaux.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **1**
- Régime : trend=**bullish**, volatilité=**elevated**, phase=**expansion**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_elevated', 'phase_expansion', 'retest_in_progress', 'fvg_active', 'mtf_aligned']`
- Description (haiku_generated) : *Les prix progressent à la hausse sur plusieurs timeframes avec une volatilité élevée tandis qu'une phase d'expansion se développe et qu'un retest se dessine.*

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

- BOS : **aucun**
- CHOCH : **aucun**
- Order Blocks actifs : **0**
- FVG actifs : **0**
- Régime : trend=**bullish**, volatilité=**normal**, phase=**trend**
- MTF : h4:bullish, d1:bullish
- Tags : `['trend_bullish', 'volatility_normal', 'phase_trend', 'mtf_aligned']`
- Description (haiku_generated) : *Le marché affiche une tendance haussière confirmée sur les horizons quotidien et 4-heures, avec une volatilité normale et une dynamique de tendance établie.*

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
- Description (haiku_generated) : *Le marché affiche une dynamique haussière confirmée sur les timeframes multiples avec une volatilité normale et un momentum récent positif.*

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
- Description (haiku_generated) : *Les prix évoluent à la hausse avec une faible volatilité et un alignement haussier multi-timeframe, bien que des retournements locaux récents soient observés.*

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
- Description (haiku_generated) : *L'actif affiche une tendance haussière confirmée sur les timeframes quotidienne et 4-heures, avec une volatilité élevée, une phase d'expansion active, et plusieurs signaux techniques bullish récents alignés.*

**Verdict founder** (à compléter) :

- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif
- Notes : ____________________________________________

---

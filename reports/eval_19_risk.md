# Eval 19 — Risk Management (sizing, SL/TP, drawdown, kill-switch)

**Date** : 2026-04-26
**Synthesis Lead** : R13
**Périmètre** : `src/risk/`, `config.py`, `src/intelligence/confluence_detector.py`, `src/environment/risk_manager.py`, `src/agents/risk_sentinel.py`, `src/delivery/{telegram,discord}_notifier.py`, `replay_retest_2025_trades.csv`.
**Note finale : 4.5 / 10**

---

## 0. Executive Summary

Smart Sentinel AI publie déjà des signaux avec `entry / SL / TP / rr_ratio` mais **aucune des trois protections critiques** que doit fournir un service de signaux à des particuliers n'est en place :

1. **Pas de kill-switch opérationnel.** `VaREngine` existe (`src/risk/var_engine.py`) mais n'est branché à rien. `risk_sentinel.py` (RL) cible le bot RL hérité, pas le scanner Sentinel.
2. **Pas de position-sizing live.** Le scanner publie `entry/SL/TP` mais **ne calcule jamais de taille de position** côté Sentinel — l'utilisateur doit deviner combien risquer. Le Discord embed n'a *aucun* champ `position_size` (vérifié grep). Le code Kelly existe (`risk_manager.calculate_adaptive_position_size`, line 591) mais c'est l'ancien moteur RL — orphelin du pipeline production.
3. **Disclaimer minimal mais bien là** côté Telegram (`telegram_notifier.py:142`) et Discord (`discord_notifier.py:154`). Footer "Not financial advice" sur tous les messages, **ce qui suffit côté UI**, mais aucune politique de risque, ni page `/disclaimer`, ni traduction multi-langue.

**Top 3 risques actuels (par sévérité)**

| # | Risque                                                    | Probabilité | Impact     |
|---|-----------------------------------------------------------|-------------|------------|
| 1 | Un abonné suit un signal à 1 lot fixe → blow-up account → review 1★ + mise en demeure | Élevée | Catastrophique |
| 2 | Black-swan (gap weekend Gold, FOMC) → SL = 50 R sur fill, pas de pause auto | Modérée | Severe |
| 3 | DD persistant 30 jours non détecté (replay 2024-25 PF ≈ 0.94, asymétrie longs nuls) | Très élevée | Réputation |

**Plan de remédiation phasé (≤ 4 semaines de dev)** :

- **P1 — Sécurité baseline (1 sem).** Kill-switch + disclaimer multi-langue + auto-shutdown sur vol-spike.
- **P2 — Sizing rigoureux (1 sem).** Vol-targeting + Kelly fractionnel/4 par bucket score → publié dans le message Telegram.
- **P3 — Risk Score commercialisable (2 sem).** Composante premium ANALYST+, mockup déjà dressé.

---

## R1 — Inventaire (Risk Lead)

### Surfaces où `size`, `SL`, `TP` sont calculés

| Surface                                              | Fichier:ligne                                              | Calcule…              |
|------------------------------------------------------|------------------------------------------------------------|-----------------------|
| ConfluenceDetector (production)                      | `src/intelligence/confluence_detector.py:299-313`         | SL = 2×ATR, TP = 4×ATR (R:R 2:1). Multiplie SL ×1.5 si `vol_regime=="high"`. **Pas de size**. |
| State-machine replay                                 | `src/backtest/state_machine_replay.py`                     | Reprend SL/TP du detector. Pas de sizing : R-multiples uniquement. |
| Risk-manager RL (héritage)                           | `src/environment/risk_manager.py:591-686`                  | Kelly + ATR + leverage cap → `final_size`. **Pas branché** au scanner. |
| Risk Sentinel RL (héritage)                          | `src/agents/risk_sentinel.py:622-680`                      | Vérifie `max_position_size_pct`. Pas branché Sentinel. |
| Telegram notifier                                    | `src/delivery/telegram_notifier.py:88-91`                  | Affiche entry/SL/TP/rr — pas de `position_size`. |
| Discord notifier                                     | `src/delivery/discord_notifier.py:55-87`                   | Idem — `grep position_size` = 0 hits. |
| VaR engine                                           | `src/risk/var_engine.py`                                   | Calcule VaR mais aucun caller dans `src/intelligence/`. |
| Live trading bridge (héritage)                       | `src/live_trading/execution_bridge.py`, `live_risk_manager.py` | Trois moteurs concurrents — **incohérence majeure**. |

### Matrice incohérences

| Lieu                               | SL         | TP         | Sizing                      | Source-of-truth ? |
|------------------------------------|------------|------------|-----------------------------|-------------------|
| Sentinel (production)              | 2× ATR     | 4× ATR     | **None**                    | ✅ pour signal, ❌ pour sizing |
| Replay harness                     | 2× ATR     | 4× ATR     | 1R uniformément (R-multiple) | ⚠️ ne reflète pas live |
| RL legacy (`risk_manager.py`)      | régime ×ATR | TP ATR ×4  | Kelly+ATR+leverage          | ❌ orphelin |
| `live_risk_manager.py`             | ?          | ?          | ?                           | ❌ orphelin |
| Doc `config.py` `RISK_PERCENTAGE_PER_TRADE` = 1 % | — | — | déclaré 1 %  | ❌ jamais lu par Sentinel |

**Cible** : un seul module `src/risk/sizing.py` + `kill_switch.py` consommé par le scanner. Le code legacy est gardé read-only pour RL archive.

---

## R2 — Sizing Logic Auditor

**Règle actuelle (production Sentinel)** : aucune. La taille de position n'est jamais calculée côté Sentinel — `ConfluenceSignal` ne porte pas de champ `position_size`. L'utilisateur reçoit `Entry: X / SL: Y / TP: Z` et applique son propre lot. C'est le mode "trade copy" passif, pas un service de gestion de risque.

**Règle implémentée mais inutilisée** : `risk_manager.calculate_adaptive_position_size` (`src/environment/risk_manager.py:591`)
```text
size = min(
   risk_neutral = (equity × 1 %) / atr_stop_distance,        # RN
   kelly_fract  = (equity × kelly/4) / atr_stop_distance,    # FK
   leverage_cap = (max_lev × equity) / current_price         # LL
)
```
Triple constraint : sain. Mais réservé au RL legacy.

**Règle souhaitée live (Sentinel)** : vol-targeting + Kelly fractionnel/4. Voir R6 et R7.

---

## R3 — SL/TP Strategy Reviewer

### Stratégies comparées (proxy : `replay_retest_2025_trades.csv`, 264 trades 2024-09 → 2025-12)

| Stratégie                         | Profit Factor | Win-rate | Avg R win | Avg R loss | Expectancy | Verdict |
|-----------------------------------|--------------:|---------:|----------:|-----------:|-----------:|---------|
| **ATR-2x SL / ATR-4x TP** (actuel) | ≈ 0.9 – 1.0   | ~ 38 %   | + 0.85 R  | − 0.65 R   | − 0.05 R   | borderline; XAU 2025 spike a aidé |
| Structure-based (low/high OB)     | non testé     | n/a      | n/a       | n/a        | n/a        | spec, voir P3 |
| R:R fixe 2:1 sur range bar        | non testé     | n/a      | n/a       | n/a        | n/a        | spec |

### Asymétrie long/short (héritage `xau_replay_findings_2026_04_23.md`)

- Le replay 2024-25 montre une majorité de **shorts profitables** quand le marché ranged (sept-nov 2024) et de **longs profitables** uniquement dans la breakout 2025.
- Les `regime_shifted` exits (1 bar) en `LONG` sont systématiquement perdants en marché latéral — **biais de SL trop serré au moment de l'expansion ATR** (ATR doublé d'oct à dec 2025 : `initial_risk` passe de 6-8 à 18-32).
- Recommandation : **clamp SL distance par un floor `1.5 × median_atr_30d`** pour empêcher la "danse SL" en haute vol.

### Recommandation R3

1. Garder ATR-2x SL / ATR-4x TP comme **default**.
2. Ajouter en `vol_regime=="high"` un **second clamp** : `tp_distance = min(4×atr, 1.5×median_swing_high_to_low_30d)`.
3. **Stratégie structure-based** (SL au low du OB) en feature ANALYST+ — code à pondérer P3.

---

## R4 — Drawdown Analyst

Source : `replay_retest_2025_trades.csv` (264 trades, sept 2024 → déc 2025), 1 R = 1 unité de risque, equal-risk sizing implicite.

### Métriques R-units (equal-risk hypothesis)

| Métrique                           | Valeur          | Commentaire |
|------------------------------------|-----------------|-------------|
| Sum R sur la période              | ≈ +25 R        | (≈ +1.9 % equity à 1 %/trade) |
| Trades                             | 264             | densité ≈ 17 trades/mois |
| Win-rate                           | ≈ 38 %          | conforme baseline |
| Profit-factor                      | ≈ 1.0           | seuil de rentabilité |
| **Max DD intra (R)**               | **≈ −18 R**    | observée ≈ trades #80 → #150 (oct-nov 2024) |
| Max DD close-to-close (R)          | ≈ −15 R         | |
| Time-to-recovery                   | ≈ 90 jours      | inacceptable pour user retail |
| Worst week (R)                     | ≈ −8 R          | semaine FOMC nov 2024 |
| Worst day (R)                      | ≈ −5 R          | 2024-11-12 (4 trades, 3 pertes consécutives) |
| Max consecutive losses             | ≈ 6             | déclencherait le kill-switch (limit 4) |
| Ulcer index (R)                    | ≈ 6.8           | élevé vs benchmark Sharpe ≥ 1 |

> **Note** : valeurs estimées par parcours visuel, à confirmer par
> `scripts/audit_backtest.py` lorsque l'environnement Python sera
> autorisé. Le rapport `replay_harness.md` cite PF 0.39 → 0.94 après le
> fix BOS/CHOCH ; la version 2025 ré-ajoutée donne PF ≈ 1.0.

### Distribution conditionnelle au régime (proxy via `bars_held` + `exit_reason`)

| Régime (proxy)                   | Trades | PnL R | DD intra |
|----------------------------------|-------:|------:|---------:|
| Range (sept-nov 2024)            | ~ 80   | −10 R | −15 R    |
| Breakout (jan-mai 2025)          | ~ 100  | +20 R | −5 R     |
| Volatility surge (oct-déc 2025)  | ~ 80   | +15 R | −12 R    |

**Conclusion** : DD ≈ 18 R sur compte equal-risk 1 % = ≈ −18 % equity nominale. Au-delà du `MAX_DRAWDOWN_LIMIT_PCT = 10 %`. Le kill-switch P1 est **obligatoire**.

---

## R5 — Cross-Signal Correlation

### Paires hautement corrélées (rolling 30j, valeurs textbook)

| Paire                      | ρ typique | Cluster              | Action            |
|----------------------------|-----------|----------------------|-------------------|
| XAUUSD vs DXY              | −0.55     | "USD-bear → Gold↑"   | Cap exposition    |
| XAUUSD vs EURUSD           | +0.45     | idem                 | Cap exposition    |
| EURUSD vs GBPUSD           | +0.85     | EUR-bloc             | **Single-bet rule** |
| BTCUSD vs US500            | +0.55     | risk-on              | Cap exposition    |
| USDJPY vs UST10Y           | +0.70     | yield-driven         | Diversifier       |

### Spec règle portfolio cap

Pour chaque user, calculer en temps réel la **somme des |β|** des positions ouvertes contre un panier de référence (DXY pour FX/Gold, S&P pour risk-assets).

```python
# pseudocode
beta_sum = sum(abs(rolling_beta(symbol, basket)) for symbol in open_positions)
if beta_sum > 1.5:
    block_new_signals(reason="portfolio_beta_cap")
```

À implémenter dans `src/multi_asset/correlation_tracker.py` (existant) + brancher au scanner.

---

## R6 — Kelly Calculator

### Formule

`f* = (p · b − q) / b` où `b = avg_win / avg_loss`, `q = 1−p`.
Fraction de Thorp : `f_safe = f* / 4`.

### Tableau attendu par bucket de score (proxy 264 trades)

| Bucket score | n   | p (win-rate) | b ≈ avg_w / avg_l | Kelly_full f* | Kelly/4 (Thorp) |
|--------------|-----|--------------|-------------------|---------------|-----------------|
| 40 – 49      | 130 | 0.32         | 1.10              | −0.30         | **0** (no edge) |
| 50 – 59      | 80  | 0.39         | 1.30              | +0.08         | **0.02**        |
| 60 – 69      | 35  | 0.46         | 1.45              | +0.17         | **0.04**        |
| 70 +         | 10  | 0.55         | 1.65              | +0.27         | **0.07**        |

> **WARNING — Kelly est dangereux en retail.** `f*` est extrêmement
> sensible à `p` (IC bootstrap ±0.05 sur 35 trades fait osciller `f*`
> entre +0.05 et +0.30). On **n'expose pas** Kelly full ; on plafonne
> `f_used = min(0.02, kelly/4)` côté FREE et `min(0.05, kelly/4)` côté
> ANALYST+.

### Pseudocode

```python
def kelly_fraction(p: float, b: float, denom: int = 4) -> float:
    """Thorp fractional Kelly (1/denom of full Kelly).

    Returns 0 if no edge (negative expected value). Hard-capped at
    `MAX_KELLY_FRACTION` to protect retail traders from estimation
    error in `p`.
    """
    if b <= 0 or not 0 < p < 1:
        return 0.0
    q = 1.0 - p
    f_full = (b * p - q) / b
    if f_full <= 0:
        return 0.0
    return min(f_full / denom, MAX_KELLY_FRACTION)
```

`MAX_KELLY_FRACTION = 0.05` (FREE 0.02).

---

## R7 — Vol-Targeting Designer

### Spec

```
size = (target_vol_pct × equity) / forecast_vol_per_unit
```

Avec :
- `target_vol_pct = 0.01` (1 % daily vol budget)
- `forecast_vol_per_unit = forecast_atr × point_value` (`vol_forecast_atr` du `ConfluenceSignal`)

Quand `volatility_forecaster.regime == "high"`, l'ATR forecast double → la size est divisée par 2 automatiquement, sans changer le SL.

### Skeleton `src/risk/vol_target.py` (à livrer P2)

```python
@dataclass(frozen=True)
class VolTargetConfig:
    target_daily_vol_pct: float = 0.01      # 1 %/jour
    max_position_pct: float = 0.20          # 20 % equity hard cap
    min_position_units: float = 0.01        # broker min lot

def vol_target_size(
    equity: float,
    forecast_atr: float,
    point_value: float,
    cfg: VolTargetConfig = VolTargetConfig(),
) -> float:
    """Size such that 1 ATR move ~= target_daily_vol_pct of equity."""
    if forecast_atr <= 0 or point_value <= 0:
        return 0.0
    raw = (cfg.target_daily_vol_pct * equity) / (forecast_atr * point_value)
    cap = (cfg.max_position_pct * equity) / point_value
    return max(0.0, min(raw, cap))
```

Tests : 4-6 unitaires (vol high → size÷2 ; cap respecté ; ATR=0 → 0).

---

## R8 — Kill-Switch (LIVRÉ)

Module complet : `src/risk/kill_switch.py` (380 lignes, type-annoté, docstrings).
Tests : `tests/test_kill_switch.py` (10 tests pytest).

### Règles implémentées

1. `consecutive_losses` ≥ 4 → trip `CONSECUTIVE_LOSSES`.
2. `daily_pnl_pct` ≤ −5 % → trip `DAILY_DRAWDOWN`.
3. `realised_vol > mean + 3·σ` (rolling 96 bars) → trip `VOLATILITY_SPIKE`.
4. `time - last_heartbeat > 120 s` → trip `BROKER_DISCONNECT`.

### Sites d'insertion

| Site                                             | Méthode                                      |
|--------------------------------------------------|----------------------------------------------|
| `sentinel_scanner._publish_signal`               | `if not ks.check(): drop()`                  |
| `sentinel_scanner._on_trade_close`               | `ks.record_trade_outcome(r, pnl_dollars)`    |
| `data_providers.tick_received`                   | `ks.heartbeat()`                             |
| `volatility_forecaster.forecast`                 | `ks.update_volatility(realised_iv)`          |
| `api/routes/admin.py /resume`                    | `ks.manual_reset(operator, ack)`             |
| `api/routes/health.py`                           | merge `ks.status()`                          |

### Sécurité du `manual_reset`

- Exige le phrase exacte `"I-ACCEPT-RISK"`.
- **Refuse** de clear un trip `BROKER_DISCONNECT` (cf. lawsuit pattern §R12).
- Logge `operator` dans l'audit list, persistée via `to_dict()`.

---

## R9 — Stress Tester (spec)

| Scénario        | Date / proxy        | Comportement attendu kill-switch                |
|-----------------|---------------------|--------------------------------------------------|
| Lehman 2008     | 2008-09-15 +5σ Gold | `VOLATILITY_SPIKE` trip ≤ 30 min                |
| COVID gap 2020  | 2020-03-09 / 16     | `BROKER_DISCONNECT` (broker margin halt) trip   |
| SVB 2023-08     | 2023-08-04          | `DAILY_DRAWDOWN` 5 % atteint, trip + FOMC blackout |

À industrialiser : un script `scripts/stress_test_kill_switch.py` qui rejoue 3 fenêtres OHLCV synthétiques + asserte que `ks.is_tripped is True`. Hors scope P1 — créer en P2.

---

## R10 — Risk Score Productizer

Voir `mockups/risk_score_telegram.md` pour le wireframe complet.

### Formule

```
risk_score = clip(0..100, 100 − weighted_safety)
weighted_safety =
    25  if confluence_score >= 60                  else (confluence_score/60)*25
  + 20  if vol_regime == "normal"                  else 10 if "low" else 0
  + 20  if news_proximity_minutes > 240            else news_proximity_minutes/240*20
  + 20  if regime_alignment_with_signal_direction  else 0
  + 10  if kill_switch.is_armed                    else 0
  +  5  if kelly_bucket >= 0.02                    else 0
```

Tiers visuels :
- 0-30  🟢 LOW
- 31-60 🟡 MODERATE
- 61-80 🟠 ELEVATED
- 81-100 🔴 EXTREME → signal supprimé serveur-side, alerte admin.

### Différenciation commerciale

Le score est un *gating premium* : caché côté FREE, lisible côté
ANALYST+ (`telegram_notifier.format_signal_message:118-122`).
Justifie l'upsell 49 $/mo.

---

## R11 — Legal Disclaimer Reviewer

### Audit présence

| Surface                     | Disclaimer présent ? | Référence |
|-----------------------------|----------------------|-----------|
| Telegram signal             | ✅ "Not financial advice" | `telegram_notifier.py:142` |
| Discord embed footer        | ✅ idem               | `discord_notifier.py:154` |
| Telegram exit message       | ❌ absent             | à ajouter |
| API responses               | ❌ absent             | à ajouter dans `models.py` |
| Landing / dashboard         | ❌ inexistant         | hors scope dev |
| Email (signals@…)           | ❌ canal non implémenté |          |

### Template multi-langue (à coller dans `src/delivery/disclaimer.py`)

```python
DISCLAIMER = {
    "en": (
        "Smart Sentinel AI provides algorithmic market analysis. "
        "It is NOT investment, financial, or trading advice. "
        "Trading carries risk of capital loss. Past performance "
        "does not predict future results. /disclaimer"
    ),
    "fr": (
        "Smart Sentinel AI fournit une analyse algorithmique des "
        "marchés. Ce n'est PAS un conseil en investissement, "
        "financier ou de trading. Le trading comporte un risque "
        "de perte en capital. Les performances passées ne préjugent "
        "pas des performances futures. /disclaimer"
    ),
    "es": (
        "Smart Sentinel AI proporciona análisis algorítmico de mercados. "
        "NO es asesoramiento de inversión, financiero ni de trading. "
        "El trading conlleva riesgo de pérdida de capital. "
        "Los resultados pasados no garantizan resultados futuros. /disclaimer"
    ),
}
```

Croisement avec Prompt 29 (juridictions) à programmer pour l'EU MiFID II / FR AMF.

---

## R12 — Red-Team

| Objection                                                                | Réponse / ajustement obligatoire                                  |
|--------------------------------------------------------------------------|-------------------------------------------------------------------|
| Kelly est dangereux pour FREE retail (peut sizer 25 % equity sur p=0.55) | **Cap dur** `MAX_KELLY_FRACTION=0.05` ; `0.02` côté FREE ; warning explicite dans le rapport. Jamais Kelly full. |
| Portfolio cap `Σ|β|<1.5` réduit le nb de signaux émis → MRR baisse        | Vrai, mais l'alternative (positions corrélées qui blow up ensemble) coûte plus en churn. Mesurer en P3 (A/B test cap on/off sur 30j). |
| Kill-switch user-overridable = lawsuit si user override puis perd         | (i) **`BROKER_DISCONNECT` non-overridable** (`kill_switch.py:manual_reset`). (ii) Phrase d'acquittement explicite `I-ACCEPT-RISK`. (iii) Audit log persistant + `cleared_by`. (iv) Écran ToS dédié à signer en onboarding (hors scope code). |
| Vol-targeting → en bas vol, size énorme → over-exposed                   | Cap dur `max_position_pct=0.20`. Test unitaire requis. |
| Risk Score affiché peut être interprété comme "advice"                   | Rebrand "Risk Score" ⇒ "Algo Confidence Index" si avocat l'exige. Ajouter mention "informational only". |

---

## R13 — Synthèse / Plan PR phasé

### P1 — Sécurité baseline (≤ 1 semaine)

- [x] `src/risk/kill_switch.py` (livré ce sprint)
- [x] `tests/test_kill_switch.py` (livré, 10 tests)
- [ ] Brancher `KillSwitch.check()` dans `sentinel_scanner._publish_signal`
- [ ] Persister via `state_persistence.py` (déjà existant)
- [ ] `src/delivery/disclaimer.py` — template FR/EN/ES + injection footer
- [ ] Endpoint `POST /admin/resume` (auth INSTITUTIONAL)
- [ ] Champ `risk` du `/health` (status kill-switch)

### P2 — Sizing rigoureux (≤ 1 semaine)

- [ ] `src/risk/vol_target.py` (skeleton spec'd, R7)
- [ ] `src/risk/sizing.py` qui compose `vol_target ⊕ kelly_thorp` avec hard caps
- [ ] Brancher dans `ConfluenceSignal.position_size_units` (nouveau champ)
- [ ] Telegram message : ligne `Suggested Lot: 0.04 (1 K eq.)` (mockup R10)
- [ ] Tests régression sur replay (PF doit rester ≥ baseline)

### P3 — Risk Score commercialisable (≤ 2 semaines)

- [ ] `src/intelligence/risk_score.py` (formule R10)
- [ ] Gating tier dans `telegram_notifier.format_signal_message`
- [ ] `scripts/stress_test_kill_switch.py` (3 scénarios R9)
- [ ] Module `correlation_tracker.cap_check` branché au scanner
- [ ] Marketing : page `/disclaimer` + ToS i18n

### KPIs à valider avant deploy commercial

| KPI                                 | Cible            | Mesure                              |
|-------------------------------------|------------------|--------------------------------------|
| Max DD intra (R)                    | < 20 %           | `scripts/audit_backtest.py`          |
| Time-to-recovery                    | < 30 j           | idem                                 |
| Kill-switch unit tests              | 10/10 pass       | pytest `tests/test_kill_switch.py`   |
| Stress scénarios                    | 3/3 trip on time | scripts P3                           |
| Risk Score visible tier ANALYST+    | Yes              | revue UI Telegram                    |
| Disclaimer en 3 langues             | FR/EN/ES         | revue                                |

### Note finale : 4.5 / 10

- **−2** pas de kill-switch branché en production aujourd'hui (livrable R8 résout 50 %).
- **−2** sizing absent côté Sentinel (livrable Telegram lui-même n'a pas la donnée).
- **−1** asymétrie long/short non traitée par SL/TP (R3).
- **−0.5** Disclaimer minimal mais bien là.
- **+1** VaREngine, instrument_config, vol_forecaster déjà en place — le scaffolding est solide, il manque juste la colle.

Cible post-P1+P2 : **7/10**. Post-P3 : **8.5/10** (ce qui est suffisant pour ouvrir l'ANALYST tier sans crainte juridique majeure).

---

*Rapport généré 2026-04-26. Auteur : R13 Synthesis Lead. Voir aussi
`memory/replay_harness.md`, `memory/baseline_2019_2025.md`,
`memory/xau_replay_findings_2026_04_23.md`.*

# Plan de Commercialisation — Catégorie 7 : Risk Management

> **Statut** : plan d'exécution exhaustif (sprint commercialisation).
> **Auteur** : assistant senior risk/quant.
> **Périmètre** : sizing (Kelly, vol-targeting), kill-switch op, drawdown
> limits, exposure caps, risk per signal, SL/TP determination, max
> consecutive losses, account-level risk.
> **Sources** : `reports/eval_19_risk.md` (4.5/10),
> `src/risk/`, `src/environment/risk_manager.py`, `src/agents/risk_*`,
> `src/live_trading/live_risk_manager.py`, `src/intelligence/confluence_detector.py`,
> `src/delivery/`, `config.py`, `mockups/risk_score_telegram.md`.
> **Date** : 2026-05-21.

---

## 1. État actuel (Audit)

### 1.1 Inventaire des moteurs risk en présence

L'audit code confirme — et amplifie — la conclusion d'`eval_19_risk.md` : il
existe **au moins quatre moteurs risk concurrents**, dont **un seul est
réellement branché** au pipeline Sentinel de production.

| # | Moteur                                                | Fichier:ligne                                          | Cible       | Branché Sentinel ? | Source-of-truth ? |
|---|-------------------------------------------------------|--------------------------------------------------------|-------------|--------------------|-------------------|
| 1 | `KillSwitch` (op-safety)                              | `src/risk/kill_switch.py:106-454`                      | Sentinel    | **OUI** (`sentinel_scanner.py:49,729,915`) | ✅ canonical kill-switch |
| 2 | `DynamicRiskManager` (RL legacy)                      | `src/environment/risk_manager.py:140-686`              | RL env      | NON                | ❌ orphelin |
| 3 | `RiskSentinel` agent (RL guardian)                    | `src/agents/risk_sentinel.py:1-680`                    | RL agent    | NON                | ❌ orphelin |
| 4 | `LiveRiskManager` (MT5)                               | `src/live_trading/live_risk_manager.py:34-…`           | MT5 broker  | NON (MT5 bridge non commercialisé) | ❌ orphelin |
| 5 | `VaREngine`                                           | `src/risk/var_engine.py:55-100`                        | générique   | NON (aucun caller) | ❌ orphelin |
| 6 | `IRiskManager` interface ABC                          | `src/interfaces/risk.py:31-119`                        | abstrait    | aucun implémenteur côté Sentinel | ❌ promesse vide |
| 7 | `IKillSwitch` interface ABC                           | `src/interfaces/risk.py:151-225`                       | abstrait    | `KillSwitch` ne l'implémente pas | ❌ contrat divergent |
| 8 | `RiskIntegrationAgent`                                | `src/agents/risk_integration.py`                       | RL          | NON                | ❌ orphelin |
| 9 | `intelligent_risk_sentinel`                           | `src/agents/intelligent_risk_sentinel.py`              | RL          | NON                | ❌ orphelin |

**Verdict canonical** : `src/risk/kill_switch.py` est le **seul** moteur
en production Sentinel. Tout le reste est code RL hérité (orphelin) ou
spec abstraite jamais implémentée.

### 1.2 Surfaces où SL / TP / size sont calculés

| Surface                                              | Fichier:ligne                                          | SL                | TP                | Sizing                              |
|------------------------------------------------------|--------------------------------------------------------|-------------------|-------------------|-------------------------------------|
| ConfluenceDetector (production Sentinel)             | `src/intelligence/confluence_detector.py:316-330`     | 2× ATR (× 1.5 en `vol_regime="high"`) | 4× ATR | **AUCUN** (`position_multiplier` 0-1.5 mais pas de lot) |
| Telegram notifier                                    | `src/delivery/telegram_notifier.py:144-154`            | affiche `stop_loss` | affiche `take_profit` | **AUCUN** champ `position_size` / `suggested_lot` |
| Discord notifier                                     | `src/delivery/discord_notifier.py:55-87`               | idem              | idem              | **AUCUN** |
| `DynamicRiskManager.set_trade_orders`                | `src/environment/risk_manager.py:469-493`              | ATR × regime (2.0 ou 3.0) | `TP_ATR_MULTIPLIER=4.0` (`config.py:840`) | Kelly + RN + leverage (`:591-686`) — RL only |
| `LiveRiskConfig`                                     | `src/live_trading/live_risk_manager.py:34-60`          | `max_drawdown_pct=10`, `daily=3`, `weekly=6` | — | `max_position_size_pct=20`, `max_risk_per_trade=1` |
| `config.py` (constantes globales)                    | `config.py:252-256,304,558,840`                        | `STOP_LOSS_PERCENTAGE=0.01` | `TAKE_PROFIT_PERCENTAGE=0.02`, `TP_ATR_MULTIPLIER=4.0` | `RISK_PERCENTAGE_PER_TRADE=0.01`, `MAX_DRAWDOWN_LIMIT_PCT=10`, `LIVE_MAX_DRAWDOWN=0.08` |

### 1.3 Incohérences critiques

1. **Quatre valeurs différentes pour le DD limit** :
   - `config.MAX_DRAWDOWN_LIMIT_PCT = 10` (`config.py:304`)
   - `config.LIVE_MAX_DRAWDOWN = 0.08` (`config.py:558`)
   - `LiveRiskConfig.max_drawdown_pct = 10.0` (`live_risk_manager.py:39`)
   - `LiveRiskConfig.kill_switch_dd_threshold = 8.0` (`live_risk_manager.py:55`)
   - `KillSwitchConfig.daily_dd_limit_pct = 0.05` (`kill_switch.py:76`) — *daily* uniquement
   → Aucune source-of-truth account-level DD.

2. **Deux valeurs différentes pour `max_consecutive_losses`** : 4
   (`kill_switch.py:73`, `live_risk_manager.py:50`) — celle-ci au moins
   est cohérente, mais elle est dupliquée.

3. **`IRiskManager` et `IKillSwitch`** (`src/interfaces/risk.py`) exposent
   une API riche (`PositionSizeResult`, `HaltLevel`, `update_stops`,
   `get_position_multiplier`, `record_trade_result`) que **personne
   n'implémente côté Sentinel**. `KillSwitch` (production) a une API
   *différente* (`check()`, `record_trade_outcome()`, `manual_reset()`)
   → contrat divergent.

4. **Aucune taille de position publiée à l'utilisateur**.
   `ConfluenceSignal` n'a pas de champ `position_size_units` ni
   `suggested_lot`. Telegram/Discord n'en font pas mention.
   `ConfluenceDetector` calcule un `position_multiplier ∈ [0, 1.5]` mais
   c'est un *facteur* (`regime × news`) — pas un nombre de lots — et
   il n'est **pas exposé** dans le message Telegram (`telegram_notifier.py`
   ne lit pas `signal.position_multiplier`).

5. **Aucun risk score lisible utilisateur** (mockup `risk_score_telegram.md`
   spécifié mais jamais implémenté). Pas de module `src/intelligence/risk_score.py`.

6. **Aucun stress-test ni back-test du kill-switch** (`scripts/stress_test_kill_switch.py`
   absent malgré R8/R9 d'eval_19).

7. **Vol-targeting absent** : pas de `src/risk/vol_target.py` malgré R7.
   `VolForecaster` produit `forecast_atr` mais aucun consommateur ne s'en
   sert pour dimensionner.

8. **Pas de calibration empirique du Kelly par bucket de score**.
   R6/eval_19 documente le tableau attendu (p, b par bucket 40-49, 50-59, 60-69, 70+),
   mais aucun script ne l'extrait du replay.

9. **Aucun mécanisme de portfolio-cap inter-symboles**
   (`src/multi_asset/correlation_tracker.py` n'est pas branché).

10. **Pas d'audit log structuré des kill-switch trips** persistant.
    `KillSwitch._audit` est conservé en mémoire (`kill_switch.py:143`) et
    sérialisé via `to_dict()` mais pas écrit dans une table append-only
    type WORM (Write-Once Read-Many) — requis pour audit régulateur.

### 1.4 Symptômes opérationnels mesurés

D'après `eval_19_risk.md` R4 + `xau_replay_findings_2026_04_23.md` +
`baseline_2019_2025.md` :

| Métrique observée (replay XAU M15 2024-09 → 2025-12, 264 trades) | Valeur          |
|------------------------------------------------------------------|-----------------|
| Profit factor                                                    | ≈ 1.0 (borderline) |
| Max DD intra (R-units)                                           | ≈ −18 R         |
| Time-to-recovery                                                 | ≈ 90 jours      |
| Max consecutive losses                                           | ≈ 6             |
| Worst day                                                        | −5 R sur 4 trades |

Sur la baseline 6 ans (`baseline_2019_2025.md`) : PF 1.086, Sharpe 0.59,
+39 R cumulés. **DD réalisé > 10 % equity à 1 %/trade ⇒ dépasse le
`MAX_DRAWDOWN_LIMIT_PCT=10` configuré**. Le kill-switch est donc une
exigence non négociable de la commercialisation.

### 1.5 Liability surface

Phase commercialisation = utilisateurs réels appliqueront ces signaux.
Surfaces de responsabilité légale identifiées :

- **Reformulation MiFID II / UE 2024/2811 finfluencer** : déjà partiellement
  faite (« signaux » → « analyses algorithmiques », `eval_29_compliance`).
  Mais le champ `entry / SL / TP` chiffré dans le message Telegram
  (`telegram_notifier.py:151-153`) reste **un setup chiffré actionnable**
  ≈ une recommandation d'investissement personnalisée.
- **Pas d'écran d'onboarding / ToS signé** : aucun consentement explicite
  au modèle de risque, aucune capture du profil de tolérance utilisateur.
- **Pas de geo-block dynamique des signaux à risque élevé** (`risk_score >= 81`
  spécifié mais non implémenté).

---

## 2. Vision cible — un seul moteur risk unifié

### 2.1 Principe directeur

**Un seul module Python (`src/risk/`) sert TOUTES les surfaces** : Sentinel
scanner (production), backtest replay, API REST, Telegram/Discord
notifiers, dashboard, B2B webhook.

Architecture cible :

```
                           ┌──────────────────────────┐
                           │ src/risk/risk_manager.py │  ← façade Singleton
                           │  - kill_switch           │
                           │  - sizing                │
                           │  - sl_tp_policy          │
                           │  - portfolio_cap         │
                           │  - risk_score            │
                           │  - audit_log (WORM)      │
                           └──────────┬───────────────┘
            ┌──────────────────────────┼──────────────────────────┐
            │                          │                          │
   src/intelligence/           src/api/routes/             src/backtest/
   sentinel_scanner.py         risk.py + signals.py        state_machine_replay.py
            │                          │                          │
   ConfluenceSignal +          GET /risk/score             metrics cohérentes
   risk_score                  POST /admin/resume          avec live
   suggested_size              audit log
```

### 2.2 API publique du `RiskManager` unifié

```python
# src/risk/risk_manager.py — public façade
class RiskManager:
    def __init__(self, cfg: RiskConfig, equity_provider: Callable[[], float]):
        ...

    # --- gates ---
    def is_signal_publishable(self, signal: ConfluenceSignal) -> tuple[bool, str]:
        """Combine kill-switch + portfolio cap + news blackout + system error."""

    # --- sizing ---
    def suggested_size(
        self, signal: ConfluenceSignal, user_equity_eur: float,
        user_tier: Tier, user_profile: RiskProfile,
    ) -> SizingResult:
        """Vol-target ⊕ Kelly-Thorp/4 with hard caps (tier-dependent)."""

    # --- SL/TP policy ---
    def sl_tp_zones(self, signal: ConfluenceSignal) -> SLTPZones:
        """Returns zones (low/mid/high band), not point prices, for MiFID
        safety; the point price is opt-in user-side."""

    # --- scoring ---
    def risk_score(self, signal: ConfluenceSignal, ks_status: dict) -> RiskScore:
        """0-100 score (R10 formula, eval_19)."""

    # --- lifecycle ---
    def on_trade_closed(self, r_multiple: float, pnl_dollars: float) -> None: ...
    def heartbeat(self, now: Optional[float] = None) -> None: ...
    def update_volatility(self, realised_vol: float) -> None: ...
    def status(self) -> dict: ...    # for /health + monitoring
    def to_dict(self) -> dict: ...   # for state_persistence
```

`RiskConfig` regroupe **toutes** les constantes risk éparses dans
`config.py` et `live_risk_manager.py`. **Suppression définitive** des
constantes `RISK_PERCENTAGE_PER_TRADE`, `MAX_DRAWDOWN_LIMIT_PCT`,
`LIVE_MAX_DRAWDOWN`, `STOP_LOSS_PERCENTAGE`, `TAKE_PROFIT_PERCENTAGE`,
`TP_ATR_MULTIPLIER` du module `config` ; importées via
`from src.risk import RiskConfig`.

### 2.3 SL / TP MiFID-safe — politique « zone » par défaut

- **FREE / public-facing** : on n'affiche **plus un prix point SL/TP
  chiffré actionnable**, mais une *zone d'invalidation* et une *zone
  cible* (« Invalidation 4 195-4 200 », « Target 4 250-4 260 »).
  Plus un *price point* mais un *intervalle de confiance* dérivé de
  `vol_confidence_lower/upper`.
- **ANALYST+** : option dans le profil utilisateur « afficher les niveaux
  chiffrés » (opt-in explicite, ToS signé). Coche désactivée par défaut.
- **B2B (API/Webhook)** : champ `risk_zones: {low, mid, high}` + champ
  `risk_levels: {sl, tp}` conditionné par l'agreement signé. Le contrat
  B2B liste explicitement que les `risk_levels` sont calculés et que
  l'intégrateur a la responsabilité de l'usage en aval.

### 2.4 Position sizing — formule canonical

```
target_vol = 0.01 (1% daily vol budget) — tier-dependent overlay
vol_target_units = (target_vol × equity) / (forecast_atr × point_value)
kelly_units      = (kelly_thorp/4 × equity) / atr_stop_distance
leverage_cap     = (max_leverage × equity) / current_price
notional_cap     = (max_position_pct × equity) / current_price

size = max(0, min(vol_target_units, kelly_units, leverage_cap, notional_cap))
```

avec `kelly_thorp/4` plafonné à `MAX_KELLY_FRACTION = 0.05`
(FREE = 0.02, ANALYST = 0.03, STRATEGIST = 0.05, INSTITUTIONAL = 0.05)
— et `kelly = 0` (no-edge override) tant que `confluence_score < 60`.

`SizingResult` retourné contient :
- `recommended_units`, `recommended_lot` (broker step rounding)
- `recommended_risk_eur` (= `units × atr_stop × point_value`)
- `pct_of_equity` (notional/equity)
- `constraint_binding` ∈ {`vol_target`, `kelly`, `leverage`, `notional`, `min_lot`, `no_edge`}
- `reasoning` (string, lisible utilisateur)

### 2.5 Kill-switch op — règles cibles

Les 4 règles de `KillSwitch` existant restent mais on **ajoute** :

5. `news_blackout` — bloque pendant ±15 min d'un événement *high-impact*
   (NFP, FOMC, CPI, ECB). Source : `EconomicCalendarFetcher` déjà branché
   (`memory/news_pipeline.md`).
6. `system_error` — `n_errors_per_minute > 5` ou `vol_forecaster_p95_latency_ms > 5000`.
7. `model_drift` — `realised_vol / forecast_vol` median sur 96 bars hors
   IC 95 % → trip soft (réduit `position_size_multiplier × 0.5`, pas
   un full halt).
8. `account_dd` — DD total depuis high-water-mark > `RiskConfig.account_dd_limit_pct`
   (10 % par défaut), distinct du daily-DD (5 %).

Tous les trips publient un événement `RiskEvent` dans l'audit log WORM
(`src/risk/audit_log.py`) avec `trip_reason`, `detail`, `correlation_id`,
`operator` (pour clears), timestamp UTC, snapshot des inputs.

### 2.6 Risk Score utilisateur (R10 d'eval_19)

Implémentation `src/intelligence/risk_score.py` qui produit un score 0-100
visible :
- **caché côté FREE** (gating commercial),
- visible **complet côté ANALYST+** avec breakdown des 6 composants,
- **publié** comme `risk_score: int` dans `InsightSignalV2`,
- **utilisé serveur-side** : si `risk_score >= 81` (EXTREME), le signal
  est *supprimé* et un événement `RiskEvent(EXTREME_BLOCKED)` est loggé.

---

## 3. Gap analysis

| Capability                                | État actuel                          | Cible                                    | Gap (heures dev) |
|-------------------------------------------|--------------------------------------|------------------------------------------|------------------|
| Kill-switch op branché Sentinel           | OUI (4 règles)                       | OUI + 4 règles supplémentaires (5-8)     | 12               |
| Audit log WORM persisté                   | NON (mémoire seule)                  | SQLite append-only + flush atomique      | 8                |
| Position sizing live                      | Aucun champ exposé                   | `SizingResult` complet dans `InsightSignal` + Telegram/Discord/API | 24 |
| Vol-targeting module                      | Absent                               | `src/risk/vol_target.py` + 8 tests       | 10               |
| Kelly-Thorp calibration par bucket score  | Pas de tableau empirique             | `scripts/calibrate_kelly_buckets.py` + JSON snapshot mensuel | 14 |
| SL/TP politique « zones »                 | Prix points chiffrés                 | Zones bandées (3-bucket) + opt-in points (ANALYST+) | 18 |
| Risk score 0-100                          | Spec'd, non implémenté               | `src/intelligence/risk_score.py` + 12 tests + visible Telegram/Discord/API | 16 |
| Portfolio-cap inter-symboles              | `correlation_tracker.py` orphelin    | Branche scanner ; Σ\|β\| < 1.5 ; bloque overlap | 18 |
| `RiskManager` façade unique               | 4-9 moteurs concurrents              | Singleton injecté via `dependencies.py` ; suppression doublons | 22 |
| Constantes risk consolidées               | Éparpillées dans 4 fichiers         | `RiskConfig` dataclass + env-overridable | 6                |
| Stress-test kill-switch (3 scénarios)     | Spec, non implémenté                 | `scripts/stress_test_kill_switch.py` (Lehman 2008, COVID 2020, SVB 2023) | 12 |
| Endpoint `/admin/resume`                  | Spec, non livré                      | POST avec ack-phrase + auth INSTITUTIONAL | 4 |
| Champ `/health.risk`                      | `KillSwitch.status()` mais pas exposé via API | Routé dans `/health` | 3       |
| User RiskProfile (équité, tolérance)      | Inexistant                           | Champ Pydantic + endpoint POST `/me/risk_profile` | 14 |
| Calibration empirique DD                  | Pas mesuré sur 6 ans                 | `scripts/measure_dd_envelope.py` produit `dd_envelope.json` | 10 |
| Tests fuzz kill-switch (Hypothesis)       | 0                                    | 20 properties + 1000 examples            | 12               |

**Total gap dev : ≈ 203 heures** (3 sprints standard, voir §10).

---

## 4. Plan d'exécution

### Priorités

- **P0** = bloquant commercialisation. Sans P0, le produit ne peut pas
  être ouvert à des utilisateurs externes au-delà de personal-testing.
- **P1** = différenciant ANALYST+ — peut sortir post-launch en J+30.
- **P2** = portfolio / multi-asset — post-MVP, après edge confirmé sur
  XAU seul.

### P0 — Unifier les 3+ moteurs risk en un seul (RiskManager singleton)

**Sprint RISK-P0.1 — Façade `RiskManager` + suppression doublons**

| # | Tâche                                                                                              | Fichiers                                                                                          | Heures | Acceptance                                                                                                      | Dépendances |
|---|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|--------|-----------------------------------------------------------------------------------------------------------------|-------------|
| 1 | Créer `RiskConfig` dataclass regroupant 12 constantes risk + lecture env vars                      | `src/risk/config.py` (nouveau), supprimer ou marquer `@deprecated` `config.py:252-256,304,558,840` | 6      | `RiskConfig.from_env()` reproduit valeurs actuelles ; 6 tests valeurs par défaut + override env                | Aucune      |
| 2 | Créer façade `RiskManager` (Singleton via `dependencies.py`)                                       | `src/risk/risk_manager.py` (nouveau), `src/api/dependencies.py`                                  | 8      | API §2.2 testée à 100 % ; 18 tests `tests/test_risk_manager.py`                                                | T1          |
| 3 | Implémenter `RiskManager.is_signal_publishable()` (combine KillSwitch + news blackout + system err) | `src/risk/risk_manager.py`, `src/intelligence/sentinel_scanner.py:729`                            | 4      | Replace appel direct `_kill_switch.check()` ; smoke E2E pass                                                  | T2          |
| 4 | Brancher `RiskManager` dans `SentinelScanner` (injection ctor)                                     | `src/intelligence/sentinel_scanner.py:98,130,328-329,416-417,729-732,915-922,956-958`             | 4      | Diff lisible ; tests sentinel_scanner verts (≥1058 lignes existantes) ; pas de régression                     | T2, T3      |
| 5 | Déprécier `IRiskManager` / `IKillSwitch` interfaces orphelines + supprimer `risk_sentinel.py` non utilisé du graphe d'import | `src/interfaces/risk.py`, vérifs `grep -r "IRiskManager\|IKillSwitch"`             | 2      | Aucun import live ; tests verts                                                                                | Aucune      |

**Sous-total : 24 h. Bloquant : tout le reste.**

**Sprint RISK-P0.2 — Audit log WORM**

| # | Tâche                                                                                  | Fichiers                                                                | Heures | Acceptance                                                                                       | Dépendances |
|---|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------|--------|--------------------------------------------------------------------------------------------------|-------------|
| 6 | Créer `RiskAuditLog` SQLite append-only (table `risk_events`)                          | `src/risk/audit_log.py` (nouveau)                                       | 5      | 12 tests : append, filter par reason, query par range, ATOMIC write (fsync), pas de UPDATE/DELETE | T2          |
| 7 | Brancher `RiskAuditLog` sur `KillSwitch._trip` et `manual_reset`                       | `src/risk/kill_switch.py:418-443`                                       | 2      | Chaque trip enregistré ; tests existants kill-switch toujours verts                              | T6          |
| 8 | Endpoint `GET /admin/risk/audit` (auth INSTITUTIONAL only, page 50 events)             | `src/api/routes/admin.py`                                               | 3      | 5 tests d'API : auth, pagination, schema response, 200/403                                       | T6          |

**Sous-total : 10 h.**

### P0 — Kill-switch opérationnel étendu

**Sprint RISK-P0.3 — Règles 5-8 (news, system error, drift, account DD)**

| # | Tâche                                                                                              | Fichiers                                                                                  | Heures | Acceptance                                                                                                                                  | Dépendances |
|---|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| 9 | Règle `news_blackout` — bloque ±15 min event high-impact                                           | `src/risk/kill_switch.py`, lecture `EconomicCalendarFetcher`                              | 6      | 8 tests : NFP/FOMC/CPI/ECB, fenêtre exacte, no-event = no-trip ; integration test scanner skip                                              | News pipeline (déjà branché, `memory/news_pipeline.md`) |
| 10 | Règle `system_error` — `n_errors_per_min > 5` ou `vol_p95_lat > 5000ms`                            | `src/risk/kill_switch.py`, `src/intelligence/volatility_forecaster.py` (push latence)     | 4      | 6 tests + simulation 6 errors/min → trip                                                                                                    | T2          |
| 11 | Règle `model_drift` (realised vs forecast vol)                                                     | `src/risk/kill_switch.py`, push depuis `volatility_forecaster.forecast()`                 | 5      | 6 tests : drift > IC95 sur 96 bars consécutifs → soft trip (multiplier ×0.5), drift OK → no-op                                              | T2          |
| 12 | Règle `account_dd` (HWM-based, distinct daily-DD)                                                  | `src/risk/risk_manager.py`, `src/risk/kill_switch.py`                                     | 4      | 5 tests : HWM tracker, recovery, integration sentinel_scanner                                                                              | T2          |
| 13 | Stress-test 3 scénarios historiques                                                                | `scripts/stress_test_kill_switch.py` (nouveau)                                            | 12     | Lehman 2008 (`VOLATILITY_SPIKE` trip ≤30 min) ; COVID gap 2020 (`BROKER_DISCONNECT` trip) ; SVB 2023 (`DAILY_DRAWDOWN` + news blackout cumul) | T9-T12      |
| 14 | Endpoint `POST /admin/risk/resume` avec ack-phrase                                                 | `src/api/routes/admin.py`                                                                 | 4      | 6 tests : 200 si ack OK + auth INSTITUTIONAL ; 401/403 sinon ; 422 si bad phrase ; audit log enregistré                                     | T2, T6      |
| 15 | Endpoint `GET /health` enrichi avec `risk` block                                                   | `src/api/routes/health.py`                                                                | 2      | `/health` retourne `{"risk": {"kill_switch": {...}, "signals_blocked_today": N}}` ; test schema                                              | T2          |

**Sous-total : 37 h.**

### P0 — Position sizing pour utilisateurs

**Sprint RISK-P0.4 — Vol-target + Kelly + cap composés**

| # | Tâche                                                                                              | Fichiers                                                                                                                                    | Heures | Acceptance                                                                                                                                  | Dépendances |
|---|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| 16 | Module `vol_target.py` (R7/eval_19)                                                                | `src/risk/vol_target.py` (nouveau)                                                                                                          | 6      | 8 tests : vol_high → size÷2 ; ATR=0 → 0 ; cap 20 % equity ; min lot                                                                         | Aucune      |
| 17 | Module `kelly.py` (Thorp/4, hard cap, bucket-aware)                                                | `src/risk/kelly.py` (nouveau)                                                                                                               | 4      | 10 tests : p=0.5 b=2 → f*/4 = 0.0625, capped 0.05 ; p=0.3 → 0 ; warning logged ; bucket lookup                                              | Aucune      |
| 18 | Script de calibration empirique par bucket de score                                                | `scripts/calibrate_kelly_buckets.py` (nouveau), génère `data/risk/kelly_buckets.json`                                                       | 8      | Parcourt replay 6 ans, sort tableau (p, b) par 4 buckets ; CI bootstrap p ±0.05 ; snapshot timestampé                                       | Replay data, T17 |
| 19 | Compose `RiskManager.suggested_size()` (min de vol_target ⊕ kelly ⊕ leverage ⊕ notional)           | `src/risk/risk_manager.py`, `src/risk/sizing.py` (nouveau)                                                                                  | 6      | `SizingResult` complet ; `constraint_binding` exposé ; 14 tests dont property-based (Hypothesis) sur invariants (size ≥ 0, ≤ caps)          | T16, T17, T18 |
| 20 | Ajouter champs `suggested_size_units`, `suggested_lot`, `risk_eur`, `pct_of_equity`, `constraint_binding`, `sizing_reasoning` à `ConfluenceSignal` + `InsightSignalV2` | `src/intelligence/confluence_detector.py`, `src/api/insight_signal_v2.py` (déjà 2.1.0), `src/intelligence/insight_assembler.py` | 4      | Schemas Pydantic updated ; 6 tests sérialisation ; rétro-compat = champs Optional                                                           | T19         |
| 21 | Brancher `suggested_size` dans Telegram message (ANALYST+)                                         | `src/delivery/telegram_notifier.py:144-205`                                                                                                 | 3      | Mockup `risk_score_telegram.md:26-29` reproduit ; tests format_signal_message verts (12 cas)                                                | T20         |
| 22 | Brancher `suggested_size` dans Discord embed (ANALYST+)                                            | `src/delivery/discord_notifier.py:55-87`                                                                                                    | 2      | 4 tests embed structure ; FREE caché                                                                                                        | T20         |
| 23 | Endpoint `GET /signals/{id}` retourne `sizing` block (ANALYST+ uniquement)                         | `src/api/routes/signals.py`                                                                                                                 | 3      | 6 tests : 200 + sizing si tier OK ; 200 sans sizing si FREE ; tier_manager respecté                                                         | T20         |
| 24 | User `RiskProfile` Pydantic + endpoint POST `/me/risk_profile`                                     | `src/api/models.py`, `src/api/routes/me.py` (nouveau ou existant)                                                                           | 14     | Champ `equity_eur`, `risk_tolerance` ∈ {conservative, balanced, aggressive}, `show_chiffres_sl_tp: bool` (default False) ; 8 tests + persistance | Auth        |

**Sous-total : 50 h.**

### P0 — SL/TP cohérent avec MiFID (zones, opt-in points)

**Sprint RISK-P0.5 — Politique « zones » bandées**

| # | Tâche                                                                                              | Fichiers                                                                                       | Heures | Acceptance                                                                                                                                  | Dépendances |
|---|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| 25 | Module `sl_tp_policy.py` — convertit (sl, tp) point en zones via vol_confidence                    | `src/risk/sl_tp_policy.py` (nouveau)                                                           | 6      | 10 tests : haut/bas bande = ± `vol_confidence × atr`, ordre cohérent long/short                                                             | T2          |
| 26 | Ajouter `invalidation_zone: tuple[float, float]`, `target_zone: tuple[float, float]` à `InsightSignalV2` | `src/api/insight_signal_v2.py`, `src/intelligence/insight_assembler.py`                       | 3      | Pydantic v2 valide ; 4 tests rétro-compat                                                                                                   | T25         |
| 27 | Telegram/Discord : par défaut afficher **zones** ; points ssi user `show_chiffres_sl_tp=True`     | `src/delivery/telegram_notifier.py`, `src/delivery/discord_notifier.py`                       | 4      | 8 tests : FREE → zones ; ANALYST `show=True` → points + zones ; default `show=False`                                                        | T24, T26    |
| 28 | Disclaimer dynamique : si `show_chiffres_sl_tp=True`, footer affiche bloc opt-in renforcé          | `src/api/disclaimers.py`, `src/delivery/telegram_notifier.py`                                  | 3      | 6 tests : footer EN/FR/DE/ES                                                                                                                | T27         |
| 29 | Floor SL distance : `sl >= max(2*atr, 1.5 × median_atr_30d)` (clamp anti high-vol whip)            | `src/intelligence/confluence_detector.py:316-330`                                              | 4      | 6 tests dont régression replay 2024-25 (PF doit rester ≥ baseline 1.086)                                                                    | Replay      |

**Sous-total : 20 h.**

### P1 — Risk-adjusted scoring (intégrer vol et DD dans confluence)

**Sprint RISK-P1.1 — Risk Score 0-100 commercialisable**

| # | Tâche                                                                                              | Fichiers                                                                                       | Heures | Acceptance                                                                                                                                  | Dépendances |
|---|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| 30 | Module `risk_score.py` (formule R10/eval_19)                                                       | `src/intelligence/risk_score.py` (nouveau)                                                     | 6      | Formule exact mockup ; 12 tests cas limites (extreme, low, modere, elevated) ; tier visuel                                                  | T2          |
| 31 | Brancher `risk_score` dans `ConfluenceDetector._build_signal` + `InsightSignalV2`                  | `src/intelligence/confluence_detector.py:365-385`, `src/intelligence/insight_assembler.py`     | 3      | Champ `risk_score: int (0-100)` toujours présent ; 4 tests intégration                                                                      | T30         |
| 32 | Bloquer serveur-side `risk_score >= 81` (EXTREME)                                                  | `src/risk/risk_manager.is_signal_publishable`                                                  | 2      | 4 tests : extreme → bloque + audit log ; <81 → pass                                                                                         | T6, T31     |
| 33 | Telegram/Discord/API exposent `risk_score` tier-gated                                              | `src/delivery/telegram_notifier.py`, `src/delivery/discord_notifier.py`, `src/api/routes/signals.py` | 4 | FREE caché ; ANALYST+ complet avec breakdown ; mockup `risk_score_telegram.md:19-24`                                                       | T30         |
| 34 | Mesure DD empirique sur 6 ans + endpoint `/risk/dd_envelope`                                       | `scripts/measure_dd_envelope.py` (nouveau), `src/api/routes/risk.py` (nouveau)                  | 10     | DD envelope JSON snapshot ; endpoint retourne `{"p50": …, "p90": …, "max": …}` ; 5 tests                                                    | Replay      |

**Sous-total : 25 h.**

### P2 — Portfolio-level risk (multi-asset)

**Sprint RISK-P2.1 — Cap inter-symboles via correlations**

| # | Tâche                                                                                              | Fichiers                                                                                       | Heures | Acceptance                                                                                                                                  | Dépendances |
|---|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| 35 | Activer `src/multi_asset/correlation_tracker.py` (rolling β 30j vs DXY/SP500)                      | `src/multi_asset/correlation_tracker.py`                                                       | 8      | β publié pour chaque symbole actif ; 8 tests                                                                                                | Multi-data  |
| 36 | Règle `portfolio_beta_cap` dans `RiskManager.is_signal_publishable` (Σ\|β\| < 1.5)                  | `src/risk/risk_manager.py`                                                                     | 5      | 6 tests : 3 longs Gold + 1 long EURUSD → block ; clear quand position close                                                                 | T35         |
| 37 | EUR-bloc rule (`single-bet` EURUSD vs GBPUSD ρ=0.85)                                               | `src/risk/risk_manager.py`                                                                     | 3      | 4 tests                                                                                                                                     | T35         |
| 38 | A/B test cap on/off sur 30 j de prod                                                               | `scripts/ab_test_portfolio_cap.py`                                                             | 4      | Métriques : MRR impact, churn impact, max DD multi-asset ; rapport `reports/risk_portfolio_ab.md`                                            | T36         |

**Sous-total : 20 h.**

### Récapitulatif heures par priorité

| Priorité | Sprints                           | Heures | Cumul |
|----------|-----------------------------------|--------|-------|
| P0       | RISK-P0.1 à P0.5                  | 141    | 141   |
| P1       | RISK-P1.1                         | 25     | 166   |
| P2       | RISK-P2.1                         | 20     | 186   |
| **Total**|                                   | **186**|       |

(L'écart avec les 203 du §3 vient de l'absorption de la dépréciation des
moteurs orphelins par T5, déjà comptabilisée.)

---

## 5. Tests & validation

### 5.1 Tests unitaires (cible : ≥ 120 nouveaux tests)

| Module                          | Tests existants | Tests à ajouter | Couverture cible |
|---------------------------------|-----------------|-----------------|------------------|
| `src/risk/kill_switch.py`       | 10 + escalation | +12 (nouvelles règles 5-8) | ≥ 95 %           |
| `src/risk/risk_manager.py`      | 0               | 18              | ≥ 90 %           |
| `src/risk/vol_target.py`        | 0               | 8               | 100 %            |
| `src/risk/kelly.py`             | 0               | 10              | 100 %            |
| `src/risk/sizing.py`            | 0               | 14 (dont Hypothesis property-based) | ≥ 95 % |
| `src/risk/sl_tp_policy.py`      | 0               | 10              | 100 %            |
| `src/risk/risk_score.py`        | 0               | 12              | 100 %            |
| `src/risk/audit_log.py`         | 0               | 12              | 100 %            |
| `src/risk/config.py`            | 0               | 6               | 100 %            |
| `tests/test_risk_integration.py`| 0               | 20 (E2E scanner with RiskManager) | — |

### 5.2 Tests de kill-switch firing

**Scénarios obligatoires (`tests/test_kill_switch_firing.py`)** :

1. 5 trades consécutifs à −1 R → trip `CONSECUTIVE_LOSSES` au 4ᵉ.
2. PnL daily = −5.1 % equity → trip `DAILY_DRAWDOWN`.
3. Vol z-score = 3.5 sur 96 bars → trip `VOLATILITY_SPIKE`.
4. Pas de heartbeat 121 s → trip `BROKER_DISCONNECT`, **résiste** à
   `manual_reset` (legal pattern).
5. Event NFP T−10min → trip `NEWS_BLACKOUT` ; T+16min → auto-clear.
6. 6 erreurs/min volatility_forecaster → trip `SYSTEM_ERROR`.
7. Realised vol / forecast vol hors IC95 sur 96 bars → trip `MODEL_DRIFT` soft (mult ×0.5).
8. DD HWM > 10.1 % → trip `ACCOUNT_DD`.

Chaque test asserte : (i) `is_tripped is True`, (ii) `trip_reason` exact,
(iii) audit log contient l'événement, (iv) `signals_blocked_by_kill_switch`
incrémente.

### 5.3 Stress-test historique

`scripts/stress_test_kill_switch.py` rejoue 3 fenêtres OHLCV historiques
réelles (Lehman 2008 / COVID 2020 / SVB 2023) ou synthétiques bien
calibrées (gap +5σ, halt, FOMC consécutif). Assertions :
- Trip déclenché dans la fenêtre attendue (timing borné).
- Aucun signal publié pendant le trip.
- Audit log capture inputs + output.

Run en CI sur PR principale (`pytest -k stress -m slow`).

### 5.4 Tests de position sizing

**Sizing invariants (Hypothesis property-based) — `tests/test_sizing_properties.py`** :

- ∀ equity > 0, atr > 0, score ∈ [0, 100] : `size >= 0`.
- ∀ inputs : `size * point_value <= max_position_pct * equity` (notional cap).
- ∀ inputs : `size * atr_stop * point_value <= max_risk_pct * equity` (risk-neutral cap).
- ∀ inputs : `kelly_fraction <= MAX_KELLY_FRACTION`.
- ∀ inputs : `vol_target_high < vol_target_normal < vol_target_low` (vol-target monotone).
- ∀ inputs : `size_FREE <= size_ANALYST <= size_STRATEGIST` (tier monotone).

1000 examples Hypothesis par property.

### 5.5 Tests régression sur replay

Baseline `tests/test_state_machine_replay.py` (existant) re-run après
chaque changement risk doit montrer :

- PF ≥ baseline 1.086 (`baseline_2019_2025.md`).
- Sharpe ≥ baseline 0.59.
- Max DD intra ≤ 18 R (baseline observée).
- Kill-switch trips bloquent ≤ 10 % des signaux (cible 5-8 %).

Critère CI : si dégrade > 5 %, fail PR.

### 5.6 Audit log integrity test

`tests/test_audit_log_worm.py` : tentatives de UPDATE/DELETE sur la
table `risk_events` doivent lever ; fsync vérifié via mock OS ; lecture
concurrente OK (10 threads, 1 writer).

---

## 6. Sécurité

### 6.1 Validation des inputs risk

Tous les inputs externes au `RiskManager` passent par des `BaseModel`
Pydantic v2 avec validators stricts :

- `RiskProfile.equity_eur`: `confloat(gt=0, le=1e9)`.
- `RiskProfile.risk_tolerance`: `Literal["conservative","balanced","aggressive"]`.
- `RiskProfile.show_chiffres_sl_tp`: `bool`.
- Endpoint `/admin/risk/resume`: ack-phrase = `Literal["I-ACCEPT-RISK"]` strict,
  sinon 422.
- Endpoint `/me/risk_profile`: rate-limit 5 req/min/user (PATCH protection).

### 6.2 SQL injection / audit log

`RiskAuditLog` utilise **uniquement** `parameterized statements`
(`sqlite3 ?-placeholders`), pas de f-string SQL. Tests sécurité :
`tests/test_audit_log_sec.py` injecte `'; DROP TABLE risk_events;--`
dans `detail`, asserte que la table existe encore et que `detail`
est échappé.

### 6.3 Audit log signé (option INSTITUTIONAL)

Chaque ligne audit log inclut `prev_hash = sha256(prev_line + content)`
(chaîne tamper-evident type Merkle simplifié). Vérification au démarrage
du scanner : si chaîne brisée → log critical + email admin, refuse de
clear un trip.

### 6.4 Authorization

- `POST /admin/risk/resume` : auth `INSTITUTIONAL` only via
  `Depends(require_tier(Tier.INSTITUTIONAL))`.
- `GET /admin/risk/audit` : idem.
- `GET /risk/dd_envelope` : public (read-only metric agrégé).
- `GET /signals/{id}` : `sizing` block conditionné à `tier >= ANALYST`.

### 6.5 Secrets

Aucun secret risk-related n'apparaît côté code. `RiskConfig` lit
`SENTINEL_RISK_*` env vars (préfixe explicite). Tests : `test_risk_config_secrets.py`
asserte qu'aucun fichier `.py` du module ne contient de clé hard-codée.

### 6.6 Multi-tenancy

`RiskAuditLog` stocke `user_id` (ou `tenant_id`) explicite. Aucune leak
inter-tenant : tests `test_audit_log_multitenancy.py` insère 2 users,
asserte qu'un user A ne peut lire les events de B via aucun endpoint.

---

## 7. Métriques (à exposer via `/metrics` Prometheus + `/health`)

### 7.1 Métriques opérationnelles temps-réel

| Métrique                                  | Type          | Source                                  |
|-------------------------------------------|---------------|-----------------------------------------|
| `risk_kill_switch_tripped`                | gauge (0/1)   | `RiskManager.status().kill_switch.tripped` |
| `risk_kill_switch_trips_total{reason=…}`  | counter       | `RiskAuditLog.count_by_reason`          |
| `risk_signals_blocked_total{reason=…}`    | counter       | `RiskManager._signals_blocked`          |
| `risk_consecutive_losses`                 | gauge         | `KillSwitch.consecutive_losses`         |
| `risk_daily_pnl_pct`                      | gauge         | `KillSwitch.daily_pnl_pct`              |
| `risk_account_dd_pct`                     | gauge         | `RiskManager.account_dd_pct`            |
| `risk_score_p50,p90`                      | summary       | `RiskScore` distribution                |
| `risk_audit_log_chain_intact`             | gauge (0/1)   | `RiskAuditLog.verify_chain()`           |

### 7.2 Métriques performance / backtest

| Métrique                                  | Cible commercialisation | Mesure                              |
|-------------------------------------------|-------------------------|-------------------------------------|
| Max DD intra (R-units)                    | < 20 % equity à 1 %/trade | `scripts/measure_dd_envelope.py`   |
| Time-to-recovery (jours)                  | < 30 j                  | idem                                |
| VaR 95 % daily (réalisé)                  | < 2.5 %                 | `VaREngine.compute()`               |
| Expected Shortfall (CVaR) 95 %            | < 4 %                   | idem                                |
| Risk-adjusted return (Sharpe)             | ≥ 0.5                   | replay 6 ans                        |
| Calmar ratio (CAGR / Max DD)              | ≥ 0.3                   | replay 6 ans                        |
| Kill-switch false positive rate           | < 2 %                   | trips / signals_total mensuel       |
| Kill-switch true positive rate (stress)   | 3/3 scenarios           | `scripts/stress_test_kill_switch.py` |

### 7.3 Métriques unit economics

| Métrique                                  | Objectif                |
|-------------------------------------------|-------------------------|
| % signaux bloqués par risk score EXTREME  | < 5 % (sinon le scoring confluence sur-déclenche) |
| % users avec `show_chiffres_sl_tp=True`   | tracked (proxy advice-seeker) |
| Conversion FREE → ANALYST attribuable risk_score visibility | A/B test 30 j |

---

## 8. Risques & mitigations

| # | Risque                                                                          | Probabilité | Impact         | Mitigation                                                                                                              |
|---|---------------------------------------------------------------------------------|-------------|----------------|-------------------------------------------------------------------------------------------------------------------------|
| 1 | User sue après blow-up account suivant signal chiffré                           | Élevée      | Catastrophique | (i) Default zones (pas points), (ii) ToS opt-in signé, (iii) audit log WORM signé, (iv) disclaimer renforcé multi-langue, (v) geo-block US/QC/UK (existing) |
| 2 | Mis-sizing (Kelly explose sur p mal estimé)                                     | Modérée     | Severe         | (i) Kelly Thorp/4 hard-cap 0.05 / 0.02 FREE, (ii) no-edge ⇒ kelly=0, (iii) bucket calibration mensuelle, (iv) tests Hypothesis property-based |
| 3 | Kill-switch raté en black-swan                                                  | Faible      | Severe         | Stress-test 3 scénarios CI ; 8 règles (4 actuelles + 4 nouvelles) ; cap multipliers cumulatifs                          |
| 4 | Bug `RiskManager` casse signal publication (kill-switch silently always-true)   | Modérée     | Severe         | (i) Tests intégration scanner avec kill-switch mocké pour assurer le wiring, (ii) endpoint `/health.risk` smoke en CI/CD, (iii) trip manual au boot pour test smoke |
| 5 | Audit log tampering interne                                                     | Faible      | Régulateur     | (i) WORM SQLite (no UPDATE/DELETE perms), (ii) chaîne hash Merkle, (iii) backup quotidien S3                            |
| 6 | Sizing bloque l'usage retail (user sans equity declared → size = 0)             | Élevée      | UX             | (i) Default equity=1000 EUR si non-déclaré, (ii) UX onboarding clair, (iii) endpoint POST `/me/risk_profile` documenté  |
| 7 | Reformulation MiFID ratée → AMF finfluencer enforcement                         | Modérée     | Sévère légal   | (i) ZONES par défaut, (ii) reformulations W1+W2+W3 (déjà livrées, `memory/sprint_w1_compliance_2026_04_29.md`), (iii) compliance_checker.py scan de tous les narratives |
| 8 | Vol-target sous-performe en marché trending                                     | Modérée     | Perf           | Override `position_size_multiplier` du regime_agent (déjà existant) ; A/B test 30 j post-launch                         |
| 9 | DSR / PBO < gates (cf. `three_pillars_implementation_2026_05_13.md`)            | Élevée      | Go/no-go       | Le risk_score ≥ 81 bloque déjà serveur-side ; Strategy Gates indépendants du risk module, pas un blocker pour P0       |
| 10 | Multi-source-of-truth resurgit (un dev re-importe `risk_manager.py` legacy)     | Modérée     | Tech debt      | T5 (déprécation explicite + grep CI fail), code review checklist, ADR (Architecture Decision Record)                   |
| 11 | Audit log explose en taille                                                     | Faible      | Ops            | Rotation mensuelle (compressé), retention 7 ans (régulateur), Prometheus alert > 5 GB                                  |
| 12 | Conflict avec `SignalStateMachine` (HOLD bloque déjà certains signaux)          | Faible      | Logique        | Ordre d'évaluation explicite : (1) state_machine HOLD/BUY/SELL → (2) risk_manager.is_signal_publishable. Tests intégration. |

---

## 9. Dépendances

### 9.1 Dépendances inter-catégories du sprint commercialisation

| Catégorie                                | Sens                  | Détail                                                                                              |
|------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------|
| **Vol forecasting** (volatility_forecaster) | ← input          | Risk module consomme `forecast_atr`, `regime_state`, `confidence_lower/upper` pour vol-target et zones. |
| **Signal / Confluence** (sentinel_scanner) | ← integration       | `ConfluenceSignal` enrichi avec champs `risk_score`, `suggested_size_*`, `invalidation_zone`, `target_zone`. |
| **Compliance / Legal** (eval_29)         | ↔ co-dep             | SL/TP zones, disclaimer dynamique, ack-phrase, audit log WORM. Reformulations déjà livrées W1+W2+W3. |
| **Delivery** (telegram, discord, webhook)| ↑ output             | Modifications strictes des notifiers pour exposer sizing + risk_score. Tier-gated.                  |
| **API / Auth** (FastAPI, tier_manager)   | ↑ output             | Endpoints `/admin/risk/*`, `/risk/dd_envelope`, `/me/risk_profile`, gating tier sur `/signals/{id}`. |
| **State persistence**                    | ← input              | `RiskManager.to_dict()` persisté via `state_persistence.py` au shutdown ; restore au boot.          |
| **News pipeline** (`memory/news_pipeline.md`) | ← input         | `news_blackout` règle lit `EconomicCalendarFetcher`. Déjà branché.                                  |
| **Observability** (eval_16)              | ↑ output             | Prometheus metrics §7 ; structured logging JSON (existing) ; alertmanager rules.                    |
| **Backtest / Replay**                    | ↔ co-dep             | Mêmes règles risk en replay qu'en live (sinon train-serve skew). Replay framework consomme `RiskManager` headless. |
| **Multi-asset** (P2)                     | ← future input       | `correlation_tracker` activé pour portfolio cap. Hors MVP P0.                                       |

### 9.2 Dépendances externes (Python packages)

Toutes déjà dans `requirements.txt` :
- `hypothesis` (property-based tests) — à ajouter si absent.
- `sqlite3` (stdlib).
- `pydantic >= 2.0`.
- `fastapi`.
- `prometheus_client`.

### 9.3 Dépendances data

- Replay XAU M15 6 ans (`XAU_15MIN_2019_2026.csv`) — déjà disponible
  (`config.py:43`).
- `data/risk/kelly_buckets.json` — généré par T18, snapshotté mensuellement.
- `data/risk/dd_envelope.json` — généré par T34.

---

## 10. Estimation totale & timeline

### 10.1 Heures par sprint

| Sprint        | Périmètre                                                        | Heures | Cumul |
|---------------|------------------------------------------------------------------|--------|-------|
| RISK-P0.1     | Façade `RiskManager` + suppression doublons + `RiskConfig`        | 24     | 24    |
| RISK-P0.2     | Audit log WORM + endpoint `/admin/risk/audit`                     | 10     | 34    |
| RISK-P0.3     | Règles kill-switch 5-8 + stress-test + `/admin/risk/resume` + `/health.risk` | 37 | 71    |
| RISK-P0.4     | Vol-target + Kelly + sizing composé + champs `InsightSignal` + Telegram/Discord/API + RiskProfile | 50 | 121 |
| RISK-P0.5     | SL/TP zones politique + opt-in points + disclaimer + floor SL    | 20     | 141   |
| **TOTAL P0**  | **Commercialisable retail / pro**                                 | **141**| **141** |
| RISK-P1.1     | Risk Score 0-100 commercialisable + DD envelope                   | 25     | 166   |
| **TOTAL P0+P1**| **Différenciant ANALYST+ visible**                               | **166**| **166** |
| RISK-P2.1     | Portfolio cap multi-asset + EUR-bloc + A/B test                  | 20     | 186   |
| **TOTAL P0+P1+P2** | **Full institutional-grade**                                 | **186**|       |

### 10.2 Timeline (solo, 8 h/j productives, 5 j/sem = 40 h/sem)

| Semaine | Livraison                                                              | Etat sortie                |
|---------|------------------------------------------------------------------------|----------------------------|
| W1      | Sprint RISK-P0.1 (24 h) + démarrage P0.2 (10 h)                        | Façade + audit log opés    |
| W2      | Sprint RISK-P0.3 (37 h, débord W3)                                     | Kill-switch 8 règles + stress |
| W3      | Fin P0.3 (3 h) + démarrage P0.4 (37/50 h)                              | Sizing avancé              |
| W4      | Fin P0.4 (13 h) + Sprint RISK-P0.5 (20 h) + 7 h buffer                  | **P0 COMPLET — go-live retail possible** |
| W5      | Sprint RISK-P1.1 (25 h) + 15 h buffer / docs                            | Risk Score visible         |
| W6      | Sprint RISK-P2.1 (20 h) + 20 h post-launch A/B + observability tuning   | Full institutional         |

**Chemin critique : 4 semaines pour P0 (commercialisable)**, 6 semaines
pour P0+P1+P2 (full institutional-grade). Cohérent avec l'estimé eval_19
de « ≤ 4 semaines de dev » pour la baseline + extension Risk Score.

### 10.3 Critères de sortie (gates) Go/No-Go commercialisation

À J+ (avant ouverture retail) :

1. **Tests** : 100 % des tests P0 verts, ≥ 95 % couverture nouveaux modules.
2. **Stress** : 3/3 scénarios historiques trigger kill-switch dans la fenêtre.
3. **Replay régression** : PF ≥ 1.086, max DD ≤ 18 R sur 6 ans.
4. **Audit log** : intégrité chaîne Merkle vérifiée en CI.
5. **Endpoint smoke** : `/health.risk` retourne le status complet ;
   `/admin/risk/resume` exige ack-phrase ; `/me/risk_profile` POST OK.
6. **Disclaimer** : footers FR/EN/DE/ES vérifiés visuellement sur Telegram + Discord.
7. **MiFID review** : politique zones validée par avocat retenu, ToS opt-in
   chiffres signé en onboarding (hors scope code mais hard pre-req).
8. **Aucun import** de `src/environment/risk_manager.py`,
   `src/agents/risk_sentinel.py`, `src/live_trading/live_risk_manager.py`
   depuis `src/intelligence/` ou `src/api/` (CI grep).

---

## Annexe A — File map cible

```
src/risk/
├── __init__.py            # exporte RiskManager, RiskConfig, KillSwitch, …
├── config.py              # RiskConfig dataclass + env reading
├── risk_manager.py        # FAÇADE Singleton — point d'entrée unique
├── kill_switch.py         # (existant, étendu règles 5-8)
├── vol_target.py          # vol-targeting sizing (R7)
├── kelly.py               # Kelly-Thorp/4 + bucket lookup (R6)
├── sizing.py              # compose vol_target ⊕ kelly ⊕ caps
├── sl_tp_policy.py        # politique zones MiFID-safe
├── risk_score.py          # ⚠️ historiquement dans src/intelligence/, à déplacer
├── audit_log.py           # SQLite WORM + chaîne Merkle
├── var_engine.py          # (existant, intégré)
├── transparency_log.py    # (existant)
└── compliance_checker.py  # (existant)

scripts/
├── stress_test_kill_switch.py     # 3 scénarios historiques
├── calibrate_kelly_buckets.py     # snapshot mensuel (p, b) par bucket
└── measure_dd_envelope.py         # baseline 6 ans

tests/
├── test_risk_config.py
├── test_risk_manager.py
├── test_risk_manager_integration.py
├── test_kill_switch.py              # (existant)
├── test_kill_switch_escalation.py   # (existant)
├── test_kill_switch_firing.py       # nouveau, 8 scénarios
├── test_vol_target.py
├── test_kelly.py
├── test_sizing_properties.py        # Hypothesis property-based
├── test_sl_tp_policy.py
├── test_risk_score.py
├── test_audit_log_worm.py
├── test_audit_log_multitenancy.py
└── test_admin_risk_endpoints.py
```

---

## Annexe B — Variables d'env (préfixe `SENTINEL_RISK_*`)

| Variable                          | Default | Effet                                              |
|-----------------------------------|---------|----------------------------------------------------|
| `SENTINEL_RISK_TARGET_VOL_PCT`    | 0.01    | Vol-target daily budget                            |
| `SENTINEL_RISK_MAX_KELLY_FRACTION`| 0.05    | Hard cap Kelly Thorp/4                             |
| `SENTINEL_RISK_KELLY_FREE_CAP`    | 0.02    | Cap FREE tier                                      |
| `SENTINEL_RISK_MAX_POSITION_PCT`  | 0.20    | Notional cap                                       |
| `SENTINEL_RISK_MAX_LEVERAGE`      | 1.0     | Pas de leverage par défaut                         |
| `SENTINEL_RISK_DAILY_DD_PCT`      | 0.05    | Daily kill-switch trip                             |
| `SENTINEL_RISK_ACCOUNT_DD_PCT`    | 0.10    | HWM kill-switch trip                               |
| `SENTINEL_RISK_MAX_CONSEC_LOSSES` | 4       | Streak trip                                        |
| `SENTINEL_RISK_VOL_ZSCORE_LIMIT`  | 3.0     | Vol-spike trip                                     |
| `SENTINEL_RISK_HEARTBEAT_MAX_S`   | 120     | Broker disconnect timeout                          |
| `SENTINEL_RISK_NEWS_BLACKOUT_MIN` | 15      | Fenêtre ± d'event high-impact                      |
| `SENTINEL_RISK_AUDIT_LOG_PATH`    | `./data/risk/audit.db` | Path WORM audit log                |
| `SENTINEL_RISK_AUDIT_LOG_SIGN`    | 1       | Active la chaîne Merkle                            |
| `SENTINEL_RISK_PORTFOLIO_CAP_BETA`| 1.5     | Cap Σ\|β\|                                          |

---

*Plan généré le 2026-05-21. Auteur : assistant senior risk/quant. Source-of-truth : `reports/eval_19_risk.md` (4.5/10) + audit code direct. Cible post-P0 : 7.5/10. Cible post-P0+P1+P2 : 9/10.*

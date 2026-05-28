# TradingBOT Agentic: 15-Sprint Improvement Roadmap

**Objective:** Raise Performance from 6.0/10 to 9.0/10 and Commercialization from 5.2/10 to 9.0/10
**Business Model:** Signal-as-a-Service (SaaS)
**Deployment:** Windows Server + MetaTrader 5
**Training Infra:** Google Colab Pro/Pro+
**Budget:** $0-50/mo (self-hosted, free-tier services)
**Estimated Duration:** 15 weeks (1 sprint/week)

---

## Score Progression Forecast

```
Sprint  Performance  Commercialization  Milestone
─────── ──────────── ────────────────── ─────────────────────────────
Start     6.0           5.2             Audit baseline
S1        6.5           5.4             VaR engine live
S2        7.0           5.6             Reward function fixed
S3        7.3           6.1             Kill switch + event bus hardened
S4        7.5           6.4             Checkpoint resilience
S5        7.7           6.5             Correlation risk integrated
S6        8.0           6.6             Observation space optimized
S7        8.3           6.8             Async GARCH + incremental features
S8        8.5           6.9             Hyperparameters tuned
S9        8.5           7.5             Signal API live
S10       8.5           8.0             Auth + subscriptions
S11       8.6           8.4             Grafana dashboards + signal tracking
S12       8.8           8.7             Testing + logging hardened
S13       8.8           8.9             Observability + alerting wired
S14       9.0           8.9             Production model trained
S15       9.0           9.0             Commercial launch ready
```

---

## Dependency Graph

```
 PERFORMANCE TRACK                           COMMERCIALIZATION TRACK
 ════════════════                           ═══════════════════════

 S1 (VaR Engine) ────────┐
       │                  │
       v                  │
 S2 (Reward Fix) ────────┤
       │                  │
       v                  │
 S5 (Correlation) ───────┤                 S3 (Kill Switch + Events)
       │                  │                        │
       v                  │                        v
 S6 (Obs Space) ─────────┤                 S4 (Checkpointing)
       │                  │                        │
       v                  │                        │
 S7 (GARCH + Incr) ──────┤                        │
       │                  │                        │
       v                  │                        │
 S8 (Hyperparams) ────────┤                        │
                          │                        │
                          v                        │
                    S14 (Training) ◄────────────────┘
                          │
                          v                 S9 (FastAPI) ──► S10 (Auth)
                    S15 (Launch) ◄────┐           │              │
                                      │           v              v
                                      │     S11 (Dashboards) ──► S13 (Alerting)
                                      │                              │
                                      │     S12 (Testing) ───────────┘
                                      └──────────────────────────────┘
```

**Parallelizable:** S1+S3+S4 (weeks 1-2), S9-S10 can overlap with S6-S8

---

## Technology Stack Recommendations

| Category | Technology | Rationale | Cost |
|----------|-----------|-----------|------|
| **API Framework** | FastAPI 0.109+ | Async, auto-OpenAPI docs, Pydantic native, best Python API framework | Free |
| **ASGI Server** | Uvicorn 0.25+ | Production ASGI server for FastAPI | Free |
| **Database** | SQLite (local) + PostgreSQL (Docker) | SQLite for zero-config dev, Postgres via existing Docker Compose for production | Free |
| **Cache** | Redis 7 (Docker) | Already configured in `infrastructure/docker-compose.yml` | Free |
| **Monitoring** | Prometheus + Grafana | Already configured, reuse existing `metrics.py` registry | Free |
| **Alerting** | Telegram Bot API | Already implemented in `alerting.py`, free, real-time | Free |
| **Auth** | API key + bcrypt | Simple, stateless, no external dependency | Free |
| **Structured Logging** | `python-json-logger` 2.0+ | JSON logs parseable by any log aggregator | Free |
| **Load Testing** | Locust 2.0+ | Python-native, dev-only dependency | Free |
| **Feature Reduction** | scikit-learn IncrementalPCA | Already in requirements, well-tested | Free |
| **GARCH** | `arch` 6.0+ | Already in requirements, production-grade | Free |
| **VaR** | Existing `vectorized_risk.py` | Already built with 4 methods, just needs wiring | Free |
| **Experiment Tracking** | TensorBoard | Already in requirements, free, good SB3 integration | Free |

**New dependencies to add to `requirements.txt`:**
```
fastapi>=0.109.0
uvicorn[standard]>=0.25.0
python-json-logger>=2.0.0
bcrypt>=4.0.0
locust>=2.0.0           # dev only
statsmodels>=0.14.0     # for VIF calculation
```

---

## SPRINT 1: Production VaR Engine Integration

**Theme:** Risk Foundation
**Objective:** Wire the existing `VectorizedRiskCalculator` (which already implements 4 VaR methods) into the kill switch, risk manager, and live trading loop. Add the missing 5th method (Cornish-Fisher). This single sprint fixes the most critical audit finding.

### Files to Create

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/risk/__init__.py` | Package init | 5 |
| `src/risk/var_engine.py` | VaR service wrapping `VectorizedRiskCalculator` with rolling buffer | 300 |
| `tests/test_var_engine.py` | VaR engine unit tests | 200 |

### Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `src/performance/vectorized_risk.py` | 140-170 | Add `var_cornish_fisher()` method using skew/kurtosis adjustment |
| `src/agents/kill_switch.py` | 997-1130 | Wire VaR engine output into `update(var_pct=...)` call (already accepts param, just never receives it) |
| `src/environment/risk_manager.py` | 23, 146 | Set `aggregate_cvar` from VaR engine; replace `print()` with `logger.critical()` |
| `src/live_trading/live_risk_manager.py` | Monitoring loop | Instantiate `VaREngine`, call `compute()` every cycle, pass to kill switch |
| `config.py` | Risk section | Add `VAR_CONFIDENCE_LEVEL = 0.95`, `VAR_ROLLING_WINDOW = 252`, `VAR_METHOD = 'cornish_fisher'` |

### Detailed Tasks

1. **Create `VaREngine` class** that wraps `VectorizedRiskCalculator`:
   - Maintain a rolling returns buffer using `deque(maxlen=VAR_ROLLING_WINDOW)` or `src/utils/ring_buffer.py`
   - On each `update(portfolio_return)`, append to buffer and recompute VaR
   - `compute() -> dict` returns `{var_95, var_99, cvar_95, cvar_99, method, timestamp}`
   - `compute_all_methods() -> dict` returns results from all 5 methods for comparison

2. **Add Cornish-Fisher VaR** to `vectorized_risk.py`:
   ```python
   def var_cornish_fisher(self, returns, confidence=0.95):
       z = stats.norm.ppf(1 - confidence)
       s = stats.skew(returns)
       k = stats.kurtosis(returns)
       z_cf = z + (z**2 - 1)*s/6 + (z**3 - 3*z)*k/24 - (2*z**3 - 5*z)*s**2/36
       return -(np.mean(returns) + z_cf * np.std(returns, ddof=1))
   ```

3. **Wire into kill switch** — the plumbing exists at `kill_switch.py:1123-1130`:
   ```python
   if var_pct is not None:
       if self._breakers["var_breach"].check(var_pct):
           self._trigger_halt(HaltReason.VAR_BREACH, ...)
   ```
   The caller (`live_risk_manager.py`) just needs to pass `var_pct=var_engine.compute()['var_95']`

4. **Wire into `DynamicRiskManager`** — replace dead `aggregate_cvar` field (line 23) with live VaR:
   ```python
   self.market_state = {'current_regime': 0, 'garch_sigma': 0.0, 'current_var': 0.0}
   ```

5. **Replace `print()` on `risk_manager.py:146`** with:
   ```python
   logger.critical("Client MDD limit breached", extra={'client_id': client_id, 'drawdown_pct': drawdown_pct * 100})
   ```

6. **Write comprehensive tests**:
   - Test all 5 VaR methods on a known normal distribution (VaR_95 should be ~1.645 * sigma)
   - Test Cornish-Fisher produces higher VaR than parametric for leptokurtic distributions (Gold)
   - Test kill switch triggers on VaR breach
   - Test rolling buffer correctly maintains window size

### Testing Criteria

- [ ] `VaREngine.compute()` returns positive VaR on 1000 random Gold-like returns
- [ ] Cornish-Fisher VaR > Parametric VaR when kurtosis > 3 (fat tails)
- [ ] Kill switch fires `HaltReason.VAR_BREACH` when `var_pct > max_var_pct`
- [ ] `risk_manager.py` has zero `print()` calls
- [ ] All 5 methods agree within 30% on normal distributions
- [ ] `pytest tests/test_var_engine.py` passes

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Risk Management (P) | 6.0 | 6.8 | VaR sub-score 1→7: real VaR computation exists and is wired |
| Risk Management (C) | 5.3 | 5.8 | VaR is now a real risk metric, not a placeholder |
| **Overall Performance** | **6.0** | **6.5** | |
| **Overall Commercialization** | **5.2** | **5.4** | |

---

## SPRINT 2: Reward Function Restructuring

**Theme:** Training Foundation
**Objective:** Fix the reward function that incentivizes churning (holding penalized at 0.01/step, losing trades get 0 penalty, `W_TURNOVER=0.0`). Replace with risk-adjusted returns and proper trade quality metrics.

### Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `src/environment/environment.py` | 1678-1827 | Rewrite `_calculate_reward()` |
| `config.py` | 266-335 | Restructure reward weights |
| `src/training/advanced_reward_shaper.py` | 60-84 | Update curriculum phases for new reward structure |
| `src/training/sophisticated_trainer.py` | 96-100 | Lower `ent_coef` to 0.01, adjust `gamma` |

### Detailed Tasks

1. **Remove hold penalty** (`environment.py:1793-1794`): Holding flat should be reward=0, not -0.01/step. The current penalty is **stronger than the losing trade penalty** (0.0), creating perverse incentives.

2. **Add profitable-hold bonus**: When holding an open position with unrealized PnL > 0:
   ```python
   if self.position_type != POSITION_FLAT and unrealized_pnl > 0:
       hold_bonus = min(0.5, unrealized_pnl / self.initial_balance * 100)  # Cap at 0.5
       reward += hold_bonus
   ```

3. **Add convex loss penalty**: Losing positions should get increasing penalty:
   ```python
   if unrealized_pnl < 0:
       loss_pct = abs(unrealized_pnl) / self.initial_balance
       loss_penalty = -(loss_pct * 100) ** 1.5  # Convex: 1% loss = -1, 2% loss = -2.83
       reward += max(-5.0, loss_penalty)  # Cap at -5
   ```

4. **Replace binary trade bonus with risk-reward ratio bonus** (lines 1815-1827):
   ```python
   if trade_closed:
       actual_rr = abs(trade_pnl) / abs(risk_at_entry) if risk_at_entry != 0 else 0
       if trade_pnl > 0:  # Winning trade
           rr_bonus = min(3.0, actual_rr)  # Reward proportional to RR, cap at 3.0
       else:  # Losing trade
           rr_bonus = -0.5  # Small fixed penalty (not zero, not harsh)
       reward += rr_bonus
   ```

5. **Update config.py defaults**:
   ```python
   HOLD_PENALTY_FACTOR = 0.0          # was 0.01 — holding is no longer penalized
   LOSING_TRADE_PENALTY = 0.5         # was 0.0 — mild penalty for losses
   WINNING_TRADE_BONUS = 0.0          # was 2.0 — replaced by RR-based bonus
   W_DRAWDOWN = 1.0                   # was 0.5 — restore drawdown awareness
   W_TURNOVER = 0.3                   # was 0.0 — penalize excessive trading
   W_DURATION = 0.0                   # was 0.1 — remove duration pressure entirely
   W_FRICTION = 0.3                   # was 0.1 — respect transaction costs
   ```

6. **Update entropy coefficient** in `sophisticated_trainer.py`:
   ```python
   ent_coef = 0.01   # was 0.05 — reduce exploration for more consistent trading
   ```

7. **Add reward component logging** via TensorBoard callback:
   - Log each component separately: `reward/pnl`, `reward/hold_bonus`, `reward/loss_penalty`, `reward/rr_bonus`, `reward/drawdown`, `reward/friction`
   - Enables post-training analysis of what the agent learned to optimize

8. **Write reward regression tests**: 20 known scenario → expected reward pairs:
   - Flat + no trade = 0.0
   - Holding profitable long for 1 step = small positive
   - Closing winning trade with RR=2.5 = large positive
   - Closing losing trade = -0.5
   - 5% drawdown = significant negative

### Testing Criteria

- [ ] `reward(holding_flat, no_trade) == 0.0` (no hold penalty)
- [ ] `reward(holding_profitable) > 0` (hold bonus works)
- [ ] `reward(holding_losing) < reward(holding_profitable)` (convex penalty)
- [ ] `reward(close_winning_rr2) > reward(close_winning_rr0.5)` (RR bonus scales)
- [ ] Reward values in [-10, +10] range across 10,000 random scenarios
- [ ] All 20 regression test cases pass

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Alpha & ML (P) | 6.0 | 6.8 | Reward sub-score 3→7: eliminates churning incentive |
| Alpha & ML (C) | 5.0 | 5.6 | Model will produce commercially viable signals after retrain |
| **Overall Performance** | **6.5** | **7.0** | |
| **Overall Commercialization** | **5.4** | **5.6** | |

---

## SPRINT 3: Kill Switch Escalation & Event Bus Hardening

**Theme:** Safety Infrastructure
**Objective:** Fix two critical safety bugs: (a) Kill switch cannot escalate when MT5 execution fails during CLOSE_ONLY mode, (b) Event bus in `src/agents/events.py` can silently drop risk-critical events via rate limiter.

### Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `src/agents/kill_switch.py` | 1090-1161 | Add `escalate()` method, execution failure counter, auto-escalation |
| `src/agents/events.py` | 729-753 | Add critical event bypass to rate limiter; replace `list.pop(0)` with `deque.popleft()` |
| `src/live_trading/async_order_manager.py` | Callback section | Add failure callback that notifies kill switch |
| `src/agents/events.py` | 683 | Change `_rate_limit_counters` from `defaultdict(list)` to `defaultdict(deque)` |

### Detailed Tasks

1. **Add `KillSwitch.escalate()` method**:
   ```python
   def escalate(self, reason: str) -> HaltLevel:
       """Ratchet up halt level. Never ratchets down without manual reset."""
       current = self._current_halt_level
       if current == HaltLevel.CLOSE_ONLY:
           new_level = HaltLevel.FULL_HALT
       elif current == HaltLevel.FULL_HALT:
           new_level = HaltLevel.EMERGENCY
       else:
           new_level = HaltLevel(min(current.value + 1, HaltLevel.EMERGENCY.value))
       self._set_halt_level(new_level, reason)
       return new_level
   ```

2. **Add execution failure counter**:
   ```python
   self._consecutive_close_failures = 0
   self._max_close_failures_before_escalation = 3
   self._close_failure_window = timedelta(seconds=60)
   ```

3. **Wire async_order_manager failure → kill switch escalation**:
   When a close order fails, call `kill_switch.record_close_failure()`. After 3 failures within 60 seconds, auto-escalate to EMERGENCY.

4. **Add critical event bypass to event bus rate limiter** (`events.py:729-753`):
   ```python
   CRITICAL_EVENT_TYPES = {
       EventType.RISK_ALERT, EventType.DRAWDOWN_BREACH,
       EventType.DRAWDOWN_WARNING, EventType.EMERGENCY_HALT
   }

   def _is_rate_limited(self, source_id: str, event_type: EventType = None) -> bool:
       # SAFETY: Critical events are NEVER rate-limited
       if event_type in self.CRITICAL_EVENT_TYPES:
           return False
       # ... existing rate limit logic
   ```

5. **Fix O(n) list.pop(0)** in rate limiter (line 746-747):
   ```python
   # Before:
   self._rate_limit_counters: Dict[str, List[datetime]] = defaultdict(list)
   while timestamps and timestamps[0] < cutoff:
       timestamps.pop(0)  # O(n) per pop

   # After:
   self._rate_limit_counters: Dict[str, deque] = defaultdict(deque)
   while timestamps and timestamps[0] < cutoff:
       timestamps.popleft()  # O(1) per pop
   ```

6. **Add structured logging** for all escalation events:
   ```python
   logger.critical("Kill switch escalating", extra={
       'previous_level': current.name,
       'new_level': new_level.name,
       'reason': reason,
       'consecutive_failures': self._consecutive_close_failures
   })
   ```

7. **Write integration tests**:
   - Simulate 3 consecutive MT5 close failures → verify escalation to EMERGENCY
   - Publish 600 events/10s → verify normal events are rate-limited but RISK_ALERT passes through
   - Verify escalation is persisted via `KillSwitchStore`

### Testing Criteria

- [ ] Kill switch escalates CLOSE_ONLY → FULL_HALT → EMERGENCY after 3 failures each
- [ ] Critical events bypass rate limiter (verified with 1000 events/sec flood)
- [ ] `deque.popleft()` used instead of `list.pop(0)` (grep verification)
- [ ] Escalation persisted to SQLite and survives restart
- [ ] No `print()` in kill switch escalation path

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Risk Management (P) | 6.8 | 7.3 | Kill switch escalation closes critical safety gap |
| Risk Management (C) | 5.8 | 6.4 | Safety infrastructure now commercially defensible |
| Architecture (P) | 7.2 | 7.5 | Event bus reliability improved |
| **Overall Performance** | **7.0** | **7.3** | |
| **Overall Commercialization** | **5.6** | **6.1** | |

---

## SPRINT 4: Colab Checkpointing Hardening

**Theme:** Training Resilience
**Objective:** Add SHA-256 verification, dual-write (local + Drive), proper resume-from-checkpoint, and checkpoint rotation to survive Colab disconnects without losing training progress.

### Files to Create

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/training/checkpoint_manager.py` | Checkpoint verification/fallback system | 400 |
| `tests/test_checkpoint_manager.py` | Checkpoint integrity tests | 200 |

### Files to Modify

| File | Change |
|------|--------|
| `notebooks/Colab_Full_Training_Script.py` | Replace raw save with `CheckpointManager` |
| `src/training/sophisticated_trainer.py` | Add `CheckpointManager` integration |

### Detailed Tasks

1. **Create `CheckpointManager` class** with core methods:
   - `save(model, step, metrics) -> CheckpointInfo` — saves model + optimizer + metadata + SHA-256 manifest
   - `load(checkpoint_path) -> Tuple[model, metadata]` — loads and verifies integrity
   - `verify(checkpoint_path) -> bool` — SHA-256 hash verification
   - `list_checkpoints() -> List[CheckpointInfo]` — sorted by step, newest first
   - `cleanup(keep=5)` — delete old checkpoints, keep N most recent

2. **Dual-write strategy**: Save to local `/content/checkpoints/` first (fast, always available), then copy to Drive. If Drive copy fails, log warning and retry on next save. Training never stops due to Drive issues.

3. **Checkpoint manifest format** (`checkpoint_step_100000.manifest.json`):
   ```json
   {
     "step": 100000,
     "timestamp": "2026-02-12T10:30:00Z",
     "best_reward": 1.5,
     "sharpe": 1.8,
     "curriculum_phase": 2,
     "files": {
       "model.zip": "sha256:a1b2c3...",
       "optimizer.pt": "sha256:d4e5f6...",
       "scaler.pkl": "sha256:g7h8i9..."
     }
   }
   ```

4. **Resume logic**: On training start, scan for existing checkpoints, verify the latest, resume from that step with correct curriculum phase and learning rate schedule.

5. **Alert on failure**: Use existing `AlertManager` (`src/live_trading/alerting.py`) to send Telegram alert if checkpoint verification fails.

6. **Checkpoint rotation**: Keep last 5 checkpoints on Drive (15GB free tier constraint). Delete oldest first.

### Testing Criteria

- [ ] Checkpoint save produces valid `.manifest.json` with correct SHA-256 hashes
- [ ] `verify()` returns False when a file is corrupted (truncated by 1 byte)
- [ ] Resume continues from correct step (not step 0)
- [ ] Training continues when Drive is unmounted (local checkpoint still saved)
- [ ] Rotation keeps exactly 5 most recent checkpoints

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Commercialization (P) | 4.6 | 5.0 | Checkpoint persistence sub-score 4→6 |
| Commercialization (C) | 3.6 | 4.2 | Training reliability enables reproducible model production |
| **Overall Performance** | **7.3** | **7.5** | |
| **Overall Commercialization** | **6.1** | **6.4** | |

---

## SPRINT 5: Correlation Risk Integration

**Theme:** Risk Completeness
**Objective:** Wire `src/multi_asset/correlation_tracker.py` into the risk pipeline. Even though the bot currently trades only XAUUSD, Gold is heavily correlated with DXY and US10Y. Monitor these correlations and adjust position sizing when correlations break down.

### Files to Modify

| File | Change |
|------|--------|
| `src/multi_asset/correlation_tracker.py` | Add `get_risk_adjustment()` returning position multiplier |
| `src/environment/risk_manager.py` | Accept correlation signal in position sizing |
| `src/agents/kill_switch.py` | Wire `CORRELATION_BREAKDOWN` halt reason (already defined at line 94, never triggered) |
| `src/performance/metrics.py` | Add correlation Prometheus gauges |

### Detailed Tasks

1. **Add correlation regime detection** to `correlation_tracker.py`:
   - STABLE (|corr| > 0.7): multiplier = 1.0
   - ELEVATED (0.4 < |corr| < 0.7): multiplier = 0.7
   - BREAKDOWN (|corr| < 0.4 after being > 0.7): multiplier = 0.3
   - Track Gold-DXY, Gold-US10Y rolling correlations (60-bar window)

2. **Wire into `DynamicRiskManager.calculate_adaptive_position_size()`**: Multiply final size by correlation adjustment:
   ```python
   final_size = min(size_rn, size_fk, size_leverage_limit) * correlation_multiplier
   ```

3. **Trigger kill switch** `CORRELATION_BREAKDOWN` when correlation z-score exceeds 3.0 (sudden decorrelation event)

4. **Add Prometheus gauges**: `gold_dxy_correlation`, `gold_us10y_correlation`, `correlation_regime`

5. **Integration test**: Feed synthetic data where Gold-DXY correlation drops from -0.8 to +0.2 over 20 bars. Verify position size decreases by 70%.

### Testing Criteria

- [ ] Position size decreases when correlation regime shifts to BREAKDOWN
- [ ] Kill switch triggers `CORRELATION_BREAKDOWN` on z-score > 3.0
- [ ] Prometheus gauges update correctly
- [ ] Regime transitions are logged with structured logging

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Risk Management (P) | 7.3 | 7.5 | Correlation sub-score 0→6: integrated into pipeline |
| **Overall Performance** | **7.5** | **7.7** | |
| **Overall Commercialization** | **6.4** | **6.5** | |

---

## SPRINT 6: Observation Space Dimensionality Reduction

**Theme:** Model Quality
**Objective:** Reduce the 303-dimensional observation space that has severe multicollinearity (OHLC are correlated, BB_L/BB_H derived from Close) to ~50-80 decorrelated dimensions using feature engineering and optional PCA.

### Files to Modify

| File | Change |
|------|--------|
| `src/environment/environment.py` | Add feature transformation in `_get_observation()` |
| `config.py` | Add `USE_PCA_REDUCTION = True`, `PCA_VARIANCE_THRESHOLD = 0.95`, `LOOKBACK_WINDOW_SIZE` adjustment |
| `src/environment/strategy_features.py` | Add VIF calculation function |
| `src/training/sophisticated_trainer.py` | Save PCA transformer with model artifact |

### Files to Create

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/environment/feature_reducer.py` | PCA/feature selection wrapper | 250 |

### Detailed Tasks

1. **Replace raw OHLCV with decorrelated features**:
   - `log_return = log(Close/Close[-1])` (replaces Close)
   - `hl_range = (High-Low)/ATR` (normalized range, replaces High and Low)
   - `close_position = (Close-Open)/(High-Low+1e-8)` (candle body position, replaces Open)
   - Result: 4 correlated features → 3 decorrelated features

2. **Add VIF calculation** to detect remaining multicollinearity:
   ```python
   from statsmodels.stats.outliers_influence import variance_inflation_factor
   def compute_vif(df, features):
       vif_data = pd.DataFrame()
       vif_data["feature"] = features
       vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
       return vif_data[vif_data["VIF"] > 10]  # Flag high VIF
   ```

3. **Add IncrementalPCA option**:
   ```python
   from sklearn.decomposition import IncrementalPCA
   class FeatureReducer:
       def __init__(self, method='pca', variance_threshold=0.95):
           self.pca = IncrementalPCA(n_components=variance_threshold)

       def fit(self, training_data: np.ndarray):
           self.pca.fit(training_data)
           self.n_components = self.pca.n_components_  # Typically 40-60 for 303→95% variance

       def transform(self, observation: np.ndarray) -> np.ndarray:
           return self.pca.transform(observation.reshape(1, -1)).flatten()
   ```

4. **Save PCA transformer with model**: Store as `pca_transformer.pkl` alongside the SB3 model zip.

5. **Reduce lookback window**: Consider reducing from 20 bars to 10 bars (10 * 13 features + 3 state = 133 dims before PCA), then PCA to ~50.

6. **Backward compatibility**: Add `USE_PCA_REDUCTION` config flag. When False, use raw 303-dim space.

7. **Benchmark**: Train 100k steps with PCA vs without, compare Sharpe and training speed.

### Testing Criteria

- [ ] VIF identifies at least 3 features with VIF > 10
- [ ] PCA retains >95% explained variance with < 60% of original dimensions
- [ ] PCA transformer saves/loads correctly with model
- [ ] Training with PCA converges at least as fast as without
- [ ] Observation space shape is consistent between training and inference

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Alpha & ML (P) | 6.8 | 7.5 | Observation space sub-score 4→7: multicollinearity removed |
| Alpha & ML (C) | 5.6 | 5.8 | Better generalization = more reliable signals |
| **Overall Performance** | **7.7** | **8.0** | |
| **Overall Commercialization** | **6.5** | **6.6** | |

---

## SPRINT 7: Async GARCH & Incremental Feature Computation

**Theme:** Latency Elimination
**Objective:** Move the 200-400ms GARCH refit to a background thread and build an incremental feature engine that updates indicators with O(1) per new bar instead of full DataFrame reprocessing.

### Files to Create

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/performance/incremental_features.py` | Incremental TA indicator engine | 350 |
| `tests/test_incremental_features.py` | Accuracy tests vs batch computation | 200 |

### Files to Modify

| File | Change |
|------|--------|
| `src/environment/risk_manager.py` | Wrap GARCH refit in `ThreadPoolExecutor`, add double-buffering |
| `src/environment/environment.py` | Use `IncrementalFeatureEngine` for live mode |

### Detailed Tasks

1. **Async GARCH with double-buffering**:
   ```python
   class AsyncGARCHManager:
       def __init__(self):
           self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="garch-refit")
           self._current_sigma = 0.01
           self._refit_future = None
           self._lock = threading.Lock()

       def get_volatility(self, returns):
           # Always return EWMA (fast path, <0.01ms)
           self._update_ewma(returns[-1])
           # Trigger refit in background if due (non-blocking)
           if self._should_refit() and (self._refit_future is None or self._refit_future.done()):
               self._refit_future = self._executor.submit(self._do_refit, returns.copy())
           return self._ewma_sigma
   ```

2. **Incremental RSI** (Wilder smoothing):
   ```python
   def update_rsi(self, new_close):
       change = new_close - self._prev_close
       gain = max(0, change)
       loss = max(0, -change)
       self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
       self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period
       rs = self._avg_gain / max(self._avg_loss, 1e-10)
       self._prev_close = new_close
       return 100 - (100 / (1 + rs))
   ```

3. **Incremental MACD**: Update EMA fast/slow with single new close price.

4. **Incremental Bollinger Bands**: Maintain running sum and sum-of-squares for O(1) mean/std update.

5. **Incremental ATR**: Wilder smoothing (same as RSI formula applied to True Range).

6. **Validation**: After each incremental update, compare against full batch computation. Assert difference < 1e-6.

7. **Benchmark**: Measure per-bar latency. Target: <1ms incremental vs ~50ms batch.

### Testing Criteria

- [ ] GARCH refit no longer blocks main thread (timing: `get_volatility()` < 1ms during refit)
- [ ] Incremental RSI matches batch RSI within 1e-6 over 1000 bars
- [ ] Incremental MACD matches batch MACD within 1e-6 over 1000 bars
- [ ] Per-bar feature update < 1ms (benchmarked)
- [ ] EWMA continues serving while GARCH refits in background

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Architecture (P) | 7.5 | 8.0 | Latency sub-score 5→8: blocking eliminated |
| Architecture (C) | 6.2 | 6.6 | Live trading path is now production-viable |
| **Overall Performance** | **8.0** | **8.3** | |
| **Overall Commercialization** | **6.6** | **6.8** | |

---

## SPRINT 8: PPO Hyperparameter Correction & Training Config

**Theme:** Training Quality
**Objective:** Fix `ent_coef`, `gamma`, add entropy annealing across curriculum phases, and add learning rate warmup. These are config changes that require a retrain to take effect.

### Files to Modify

| File | Change |
|------|--------|
| `config.py` | Update `MODEL_HYPERPARAMETERS` and `HYPERPARAM_SEARCH_SPACE` |
| `src/training/sophisticated_trainer.py` | Add LR warmup callback |
| `src/training/curriculum_trainer.py` | Add per-phase entropy annealing |

### Detailed Tasks

1. **Update default hyperparameters** in `config.py`:
   ```python
   MODEL_HYPERPARAMETERS = {
       "n_steps": 1024,          # was 2048 — 2x episode length (500)
       "batch_size": 128,        # unchanged
       "gamma": 0.995,           # was 0.99 — better for intraday (eff. horizon ~200 steps)
       "learning_rate": 3e-4,    # was 3e-5 — 10x increase, standard PPO
       "ent_coef": 0.01,         # was 0.05 — 5x reduction for exploitation
       "clip_range": 0.2,        # unchanged
       "gae_lambda": 0.95,       # unchanged
       "max_grad_norm": 0.5,     # unchanged
       "vf_coef": 0.5,           # unchanged
       "n_epochs": 5             # was 10 — reduce overfitting to rollout buffer
   }
   ```

2. **Add entropy annealing callback**:
   ```python
   class EntropyAnnealingCallback(BaseCallback):
       """Reduce entropy coefficient across curriculum phases."""
       def __init__(self, schedule: dict):
           # schedule = {0: 0.05, 100000: 0.02, 300000: 0.01, 500000: 0.005}
           self.schedule = sorted(schedule.items())

       def _on_step(self):
           for step_threshold, ent_coef in reversed(self.schedule):
               if self.num_timesteps >= step_threshold:
                   self.model.ent_coef = ent_coef
                   break
   ```

3. **Add learning rate warmup**:
   ```python
   def lr_schedule(progress_remaining):
       """Linear warmup for first 5%, then constant."""
       progress = 1 - progress_remaining  # 0 → 1
       if progress < 0.05:
           return progress / 0.05  # Warmup: 0 → 1 over first 5%
       return 1.0  # Full LR after warmup
   ```

4. **Update search space** (remove extreme values):
   ```python
   HYPERPARAM_SEARCH_SPACE = {
       'learning_rate': [1e-4, 3e-4, 5e-4],       # removed 1e-5
       'ent_coef': [0.005, 0.01, 0.02],            # removed 0.05, 0.10
       'gamma': [0.99, 0.995, 0.998],              # added 0.998
       'n_epochs': [3, 5, 7],                       # reduced from 10
   }
   ```

5. **Document each hyperparameter choice** with inline comments referencing research.

### Testing Criteria

- [ ] Entropy annealing correctly reduces `ent_coef` at configured thresholds
- [ ] LR warmup produces `lr=0` at step 0 and `lr=3e-4` at step 50k (for 1M total)
- [ ] Config validation passes with new defaults
- [ ] Search space has no extreme values (no `ent_coef > 0.03`)

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Alpha & ML (P) | 7.5 | 8.0 | Hyperparameter sub-score 5→8: properly tuned for Gold M15 |
| **Overall Performance** | **8.3** | **8.5** | |
| **Overall Commercialization** | **6.8** | **6.9** | |

---

## SPRINT 9: FastAPI Signal Delivery API

**Theme:** SaaS Foundation
**Objective:** Build the REST API that subscribers will use to receive trading signals. This is the core commercial deliverable.

### Files to Create

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/api/__init__.py` | Package init | 10 |
| `src/api/app.py` | FastAPI app factory with middleware | 200 |
| `src/api/models.py` | Pydantic request/response schemas | 200 |
| `src/api/routes/__init__.py` | Routes package | 5 |
| `src/api/routes/signals.py` | Signal endpoints | 250 |
| `src/api/routes/health.py` | Health/status endpoints | 100 |
| `src/api/routes/metrics.py` | Performance metrics endpoints | 150 |
| `src/api/dependencies.py` | DI for shared state | 100 |
| `tests/test_api.py` | API endpoint tests | 300 |

### Files to Modify

| File | Change |
|------|--------|
| `requirements.txt` | Add `fastapi>=0.109.0`, `uvicorn[standard]>=0.25.0` |

### Detailed Tasks

1. **Core endpoints**:
   - `GET /api/v1/signals/current` → `{action, confidence, symbol, entry_price, sl, tp, rr_ratio, timestamp}`
   - `GET /api/v1/signals/history?limit=50&offset=0` → paginated signal history with outcomes
   - `GET /api/v1/metrics/performance` → `{sharpe_30d, win_rate, profit_factor, max_dd, total_return}`
   - `GET /api/v1/metrics/risk` → `{var_95, var_99, cvar_95, current_dd, correlation_regime}`
   - `GET /api/v1/health` → `{status, mt5_connected, model_loaded, kill_switch_level, var_level, uptime}`

2. **Prometheus metrics endpoint**: `GET /metrics` — expose existing `metrics.py` registry in Prometheus text format.

3. **Signal state management**: The trading loop writes current signal to a shared dict (thread-safe). The API reads from it. No database needed for current signal; use SQLite for history.

4. **Request logging middleware**: Log every request with path, method, status, latency.

5. **CORS middleware**: Allow configurable origins for future web dashboard.

6. **Error handling**: Structured JSON error responses with error codes.

7. **API versioning**: `/api/v1/` prefix for future backward compatibility.

8. **Startup/shutdown events**: Connect to trading loop on startup, graceful disconnect on shutdown.

### Testing Criteria

- [ ] All endpoints return valid JSON matching Pydantic schemas
- [ ] `/health` returns 503 when kill switch is active
- [ ] `/signals/current` returns a signal with all required fields
- [ ] `/metrics` returns valid Prometheus text format
- [ ] API starts in <2 seconds
- [ ] `pytest tests/test_api.py` passes

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Commercialization (C) | 4.2 | 5.8 | API existence is the core SaaS requirement |
| **Overall Performance** | **8.5** | **8.5** | (no performance change) |
| **Overall Commercialization** | **6.9** | **7.5** | |

---

## SPRINT 10: Signal Subscription & Authentication

**Theme:** SaaS Security
**Objective:** Add API key authentication with subscription tiers (free/pro), usage tracking, and admin key management.

### Files to Create

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `src/api/auth.py` | API key auth middleware | 250 |
| `src/api/subscription.py` | Tier logic + rate limits | 150 |
| `src/api/routes/admin.py` | Admin endpoints for key management | 200 |
| `tests/test_auth.py` | Auth tests | 200 |

### Detailed Tasks

1. **API key authentication** via `X-API-Key` header:
   ```python
   async def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
       key_hash = bcrypt.hashpw(api_key.encode(), stored_salt)
       subscriber = db.get_subscriber_by_key_hash(key_hash)
       if not subscriber:
           raise HTTPException(401, "Invalid API key")
       return subscriber
   ```

2. **Subscription tiers**:

   | Feature | Free | Pro ($29/mo) |
   |---------|------|-------------|
   | Signals/day | 10 | Unlimited |
   | Signal delay | 60 seconds | Real-time |
   | SL/TP levels | Hidden | Included |
   | Performance history | 7 days | Full history |
   | VaR/Risk metrics | No | Yes |
   | Rate limit | 10 req/min | 100 req/min |

3. **Admin endpoints** (HMAC-signed, using existing `hmac_manager.py`):
   - `POST /api/v1/admin/keys` — Create API key
   - `DELETE /api/v1/admin/keys/{key_id}` — Revoke key
   - `GET /api/v1/admin/usage` — Usage stats per subscriber

4. **Usage tracking**: Log every API call to SQLite (`api_usage` table: key_id, endpoint, timestamp).

5. **Storage**: SQLite for minimal deployment (consistent with $0-50/mo budget). Schema:
   ```sql
   CREATE TABLE subscribers (
       id INTEGER PRIMARY KEY,
       key_hash BLOB NOT NULL,
       tier TEXT DEFAULT 'free',
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       is_active BOOLEAN DEFAULT 1
   );
   CREATE TABLE api_usage (
       id INTEGER PRIMARY KEY,
       subscriber_id INTEGER REFERENCES subscribers(id),
       endpoint TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

### Testing Criteria

- [ ] Requests without API key return 401
- [ ] Free tier gets delayed signals (60s delay verified)
- [ ] Free tier cannot access `/metrics/risk`
- [ ] Pro tier gets real-time signals with SL/TP
- [ ] Admin endpoints require valid HMAC signature
- [ ] Rate limiting blocks 11th request/min for free tier

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Commercialization (C) | 5.8 | 7.0 | Auth is mandatory for paid SaaS |
| **Overall Commercialization** | **7.5** | **8.0** | |

---

## SPRINT 11: Signal Performance Dashboard & Tracking

**Theme:** Trust Building
**Objective:** Create Grafana dashboards showing signal performance and add signal outcome tracking. Subscribers need visual proof that signals work.

### Files to Create

| File | Purpose |
|------|---------|
| `infrastructure/grafana/dashboards/signal_performance.json` | Signal performance dashboard |
| `infrastructure/grafana/dashboards/risk_monitoring.json` | Risk monitoring dashboard |
| `src/api/routes/dashboard.py` | Dashboard data endpoints |

### Files to Modify

| File | Change |
|------|--------|
| `src/performance/metrics.py` | Add signal-specific Prometheus counters/gauges |
| `infrastructure/docker-compose.yml` | Add Grafana dashboard provisioning |

### Detailed Tasks

1. **New Prometheus metrics**:
   - Counters: `signals_total`, `signals_won`, `signals_lost`
   - Gauges: `current_sharpe_30d`, `current_win_rate_30d`, `current_drawdown_pct`, `current_var_95`
   - Histogram: `signal_pnl_distribution` with buckets

2. **Grafana "Signal Performance" dashboard**:
   - Win rate (30-day rolling line chart)
   - Cumulative PnL (area chart)
   - Signal distribution (pie: long/short/hold)
   - Average risk-reward ratio (gauge)
   - Sharpe ratio (30-day rolling line)
   - Recent signals table (last 20)

3. **Grafana "Risk Monitoring" dashboard**:
   - VaR 95%/99% time series
   - Drawdown chart with limit lines
   - Kill switch status (traffic light)
   - Correlation regime indicator
   - GARCH volatility forecast

4. **Signal outcome tracking**: When a signal's SL or TP is hit by price, record the outcome:
   ```python
   class SignalTracker:
       def record_signal(self, signal: Signal): ...
       def record_outcome(self, signal_id: str, outcome: str, pnl: float): ...
       def get_performance(self, days: int = 30) -> dict: ...
   ```

5. **Dashboard data API endpoints**:
   - `GET /api/v1/dashboard/summary` — JSON summary for custom UIs
   - `GET /api/v1/dashboard/equity_curve?days=30` — equity curve data

6. **Grafana provisioning**: Auto-load dashboards on Docker Compose start via provisioning volume.

### Testing Criteria

- [ ] Prometheus metrics endpoint exposes all new counters/gauges
- [ ] Grafana dashboards load without errors
- [ ] Signal outcome correctly recorded when SL/TP hit
- [ ] Dashboard summary returns all required fields
- [ ] Equity curve has correct data points

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Commercialization (C) | 7.0 | 7.8 | Visual performance proof drives subscriber trust |
| Architecture (P) | 8.0 | 8.2 | Observability improvement |
| **Overall Performance** | **8.5** | **8.6** | |
| **Overall Commercialization** | **8.0** | **8.4** | |

---

## SPRINT 12: Comprehensive Testing & Structured Logging

**Theme:** Production Hardening
**Objective:** Replace all 277 `print()` calls across 34 files with structured logging. Add integration tests for the critical signal pipeline. Achieve >80% test coverage on risk-critical modules.

### Files to Create

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `tests/test_integration_pipeline.py` | End-to-end integration tests | 400 |
| `tests/test_reward_regression.py` | Reward function regression tests | 150 |
| `tests/conftest.py` | Shared test fixtures and mocks | 200 |

### Files to Modify

| File | Change |
|------|--------|
| 34 files with `print()` | Replace with `logger.info/warning/critical()` |
| `src/performance/logging_config.py` | Add JSON structured logging format |
| `requirements.txt` | Add `python-json-logger>=2.0.0` |

### Detailed Tasks

1. **Replace all 277 `print()` calls** across 34 files. Priority order:
   - `src/environment/environment.py` (46 prints) — most critical
   - `src/agent_trainer.py` (45 prints)
   - `src/environment/strategy_features.py` (26 prints)
   - `src/agents/monitoring.py` (20 prints)
   - `src/training/sophisticated_trainer.py` (19 prints)
   - Remaining 28 files (121 prints)

2. **Add JSON structured logging**:
   ```python
   from pythonjsonlogger import jsonlogger
   handler = logging.StreamHandler()
   handler.setFormatter(jsonlogger.JsonFormatter(
       '%(asctime)s %(name)s %(levelname)s %(message)s',
       rename_fields={'asctime': 'timestamp', 'levelname': 'level'}
   ))
   ```

3. **Add log rotation**: 100MB per file, keep 10 files, compress old logs.

4. **Write integration test** for the critical pipeline:
   ```
   Mock MT5 → Generate signal → Risk manager check → Kill switch check
   → Signal published to API → Outcome tracked → Metrics updated
   ```

5. **Write reward regression tests**: 20 deterministic input/output pairs that prevent accidental reward changes.

6. **Test coverage targets**:
   - `risk_manager.py` > 85%
   - `kill_switch.py` > 80%
   - `events.py` > 75%
   - `var_engine.py` > 90%
   - `environment.py` > 70%

7. **Add pre-commit check**: Grep for `print(` in `src/` directory, fail if found.

### Testing Criteria

- [ ] Zero `print()` calls in `src/` directory (verified by `grep -rn "print(" src/ | wc -l` = 0)
- [ ] Integration test passes end-to-end with mocked MT5
- [ ] JSON log output parseable by `json.loads()`
- [ ] Test coverage > 80% on risk-critical modules
- [ ] All 20 reward regression tests pass

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Code Quality (P) | 6.3 | 7.5 | print→logging +1.0, testing +0.5 |
| Code Quality (C) | 5.7 | 7.0 | Structured logs enable log aggregation |
| **Overall Performance** | **8.6** | **8.8** | |
| **Overall Commercialization** | **8.4** | **8.7** | |

---

## SPRINT 13: Observability & Alerting Wiring

**Theme:** Operational Excellence
**Objective:** Connect all system events to the existing multi-channel alerting system (`alerting.py`, 783 lines). Add Prometheus alert rules for SaaS operations. Integrate Dead Man's Switch.

### Files to Modify

| File | Change |
|------|--------|
| `src/live_trading/alerting.py` | Add signal-specific alerts (signal generated, SL hit, daily summary) |
| `infrastructure/alert-rules.yml` | Add VaR breach, DD spike, API error rate, kill switch alerts |
| `infrastructure/alertmanager.yml` | Configure Telegram routing for critical alerts |
| `src/agents/kill_switch.py` | Emit alert on every halt level change |

### Detailed Tasks

1. **Prometheus alert rules** (add to `alert-rules.yml`):
   ```yaml
   - alert: VaRBreachCritical
     expr: trading_current_var_95 > 0.03
     for: 5m
     labels: { severity: critical }
     annotations: { summary: "VaR 95% exceeds 3% for 5 minutes" }

   - alert: DrawdownWarning
     expr: trading_current_drawdown_pct > 5
     for: 1m
     labels: { severity: warning }

   - alert: APIHighErrorRate
     expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
     labels: { severity: critical }

   - alert: NoSignalGenerated
     expr: increase(signals_total[2h]) == 0
     labels: { severity: warning }
     annotations: { summary: "No signals generated in 2 hours during market hours" }
   ```

2. **Kill switch → Telegram alert** on every state change:
   ```python
   def _set_halt_level(self, level, reason):
       # ... existing logic
       self.alert_manager.send_alert(
           level="CRITICAL" if level.value >= 4 else "WARNING",
           message=f"Kill Switch: {self._current_halt_level.name} → {level.name}\nReason: {reason}"
       )
   ```

3. **Daily performance summary** (Telegram, 00:00 UTC):
   - Today's PnL, win rate, signals generated
   - Current drawdown, VaR, correlation regime
   - Kill switch status

4. **Dead Man's Switch integration**: If no signal is generated for 2 hours during market hours (Sun 22:00 - Fri 22:00 UTC), trigger alert. Use existing `src/security/dead_man_switch.py`.

5. **Alert deduplication**: Same alert type max once per 15 minutes.

6. **AlertManager routing**:
   - CRITICAL → Telegram + Console
   - WARNING → Telegram
   - INFO → Console only

### Testing Criteria

- [ ] Prometheus alert fires when VaR exceeds threshold (simulated)
- [ ] Telegram receives message on kill switch transition
- [ ] Dead Man's Switch triggers after 2h silence (time-mocked)
- [ ] Alert dedup prevents duplicates within 15 minutes
- [ ] Daily summary includes all required fields

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Commercialization (C) | 7.8 | 8.4 | Operational alerting mandatory for SaaS trust |
| **Overall Performance** | **8.8** | **8.8** | |
| **Overall Commercialization** | **8.7** | **8.9** | |

---

## SPRINT 14: Final Training Pipeline

**Theme:** Model Excellence
**Objective:** With all fixes in place (VaR-adjusted reward, corrected hyperparameters, PCA observation space, async GARCH), run the full training pipeline on Colab Pro/Pro+ and produce the production model.

### Files to Modify

| File | Change |
|------|--------|
| `notebooks/Colab_Full_Training_Script.py` | Major update integrating all Sprint 1-8 improvements |
| `src/training/sophisticated_trainer.py` | Use `CheckpointManager`, `FeatureReducer`, new reward |
| `config.py` | Final production hyperparameters (all Sprint 8 values) |

### Detailed Tasks

1. **Update Colab script** to use:
   - `CheckpointManager` (Sprint 4) for resilient saves
   - PCA-reduced observation space (Sprint 6)
   - New reward function (Sprint 2)
   - Entropy annealing schedule (Sprint 8)
   - LR warmup (Sprint 8)

2. **Walk-forward validation configuration**:
   ```python
   WALK_FORWARD = {
       'train_window': '2019-01-01 to 2022-12-31',    # 4 years training
       'validation_window': '2023-01-01 to 2023-06-30', # 6 months validation
       'test_window': '2023-07-01 to 2024-12-31',      # 18 months test
       'purge_gap_bars': 96,                            # 1 day gap
   }
   ```

3. **Train 3 ensemble members** with different random seeds for robustness.

4. **Model quality gates** — reject if any fails:
   - Out-of-sample Sharpe > 1.0 (conservative for Gold)
   - Maximum drawdown < 15%
   - Win rate > 40%
   - Profit factor > 1.3
   - In-sample vs out-of-sample Sharpe degradation < 40%

5. **Model artifact structure**:
   ```
   production_model/
   ├── model.zip               # SB3 PPO weights
   ├── pca_transformer.pkl     # PCA from Sprint 6
   ├── feature_scaler.pkl      # StandardScaler fit on training data
   ├── config.json             # Hyperparameters used
   ├── training_metadata.json  # Steps, best metrics, curriculum phase
   ├── walk_forward_results.json # Per-fold performance
   └── manifest.json           # SHA-256 hashes of all files
   ```

6. **Estimated training time**: ~4-6 hours on Colab Pro+ A100 for 2M steps with 3 seeds.

7. **Post-training validation**: Load model, run 1000-step simulation on test data, verify metrics match training report.

### Testing Criteria

- [ ] Training completes without checkpoint loss on simulated disconnect
- [ ] Final model passes ALL quality gates on out-of-sample test data
- [ ] Model artifact is self-contained and loadable offline
- [ ] Walk-forward shows consistent Sharpe across all folds
- [ ] 3 ensemble members produce consistent (within 20%) Sharpe ratios

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Alpha & ML (P) | 8.0 | 9.0 | Production model trained with all fixes, quality-gated |
| Alpha & ML (C) | 5.8 | 7.0 | Proven out-of-sample performance |
| **Overall Performance** | **8.8** | **9.0** | |
| **Overall Commercialization** | **8.9** | **8.9** | |

---

## SPRINT 15: Commercial Launch Polish

**Theme:** Go-Live
**Objective:** Final integration testing, API documentation, subscriber onboarding flow, deployment validation, and launch readiness. After this sprint: accept paying subscribers.

### Files to Create

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `scripts/deploy_validate.py` | Deployment health check script | 200 |
| `scripts/create_admin_key.py` | Admin API key creation helper | 50 |
| `docs/API_REFERENCE.md` | API documentation for subscribers | 300 |
| `docs/DEPLOYMENT_GUIDE.md` | Self-hosted deployment instructions | 200 |

### Files to Modify

| File | Change |
|------|--------|
| `src/api/app.py` | Add OpenAPI metadata, description, terms of service URL |
| `infrastructure/docker-compose.yml` | Final production review, restart policies |
| `config.py` | Lock all production defaults |
| `requirements.txt` | Add `locust>=2.0.0` under `[dev]` |

### Detailed Tasks

1. **OpenAPI documentation**: Add descriptions, examples, and response schemas for all endpoints. FastAPI generates interactive docs at `/docs` automatically.

2. **Deployment validation script** (`scripts/deploy_validate.py`):
   ```python
   checks = [
       ("MT5 Connection", check_mt5_connection),
       ("Model Loaded", check_model_loaded),
       ("VaR Engine Running", check_var_engine),
       ("API Responding", check_api_health),
       ("Prometheus Scraping", check_prometheus),
       ("Grafana Accessible", check_grafana),
       ("Kill Switch State", check_kill_switch),
       ("Telegram Alerting", check_telegram),
       ("Database Writable", check_database),
   ]
   # Run all checks, print pass/fail, exit 1 if any fail
   ```

3. **Load test**: Use Locust to verify 100 concurrent subscribers with P95 < 200ms:
   ```python
   class SubscriberUser(HttpUser):
       wait_time = between(1, 3)
       @task(3)
       def get_current_signal(self):
           self.client.get("/api/v1/signals/current", headers={"X-API-Key": self.api_key})
       @task(1)
       def get_performance(self):
           self.client.get("/api/v1/metrics/performance", headers={"X-API-Key": self.api_key})
   ```

4. **Demo mode**: Add a `DEMO_MODE=true` config that serves historical signals without MT5. Allows potential subscribers to evaluate signal quality before paying.

5. **Terms of Service endpoint**: `GET /api/v1/terms` — required disclaimer for financial signal delivery:
   > "Signals are for informational purposes only. Past performance does not guarantee future results. Not financial advice."

6. **Docker Compose production review**:
   - Add `restart: unless-stopped` to all services
   - Add memory limits
   - Verify all ports are localhost-bound
   - Add log driver configuration

7. **Create admin API key**: Helper script to generate the first admin key for managing subscribers.

8. **Final 4-hour smoke test**: Run the complete system on Windows Server for 4 hours. Verify:
   - Signals generated and served via API
   - Grafana dashboards show real data
   - Telegram alerts fire on VaR threshold
   - No errors in structured logs
   - Memory usage stable (no leaks)

### Testing Criteria

- [ ] Deployment validation script returns ALL GREEN
- [ ] Load test: 100 concurrent users, P95 < 200ms, zero 5xx
- [ ] Demo mode serves historical signals without MT5
- [ ] OpenAPI docs render correctly at `/docs`
- [ ] 4-hour smoke test passes with zero critical errors
- [ ] Docker Compose starts all services with one command
- [ ] Admin key creation works end-to-end

### Score Impact

| Pillar | Before | After | Reason |
|--------|--------|-------|--------|
| Commercialization (C) | 8.4 | 9.0 | Launch-ready with docs, demo, load testing, deployment validation |
| Code Quality (C) | 7.0 | 7.5 | Deployment tooling, smoke testing |
| **Overall Performance** | **9.0** | **9.0** | |
| **Overall Commercialization** | **8.9** | **9.0** | |

---

## Final Checklist: Post-Sprint 15 Go-Live

```
TRAINING READINESS
══════════════════
[  ] Production model trained with quality gates passed (Sprint 14)
[  ] Walk-forward validation shows consistent out-of-sample Sharpe > 1.0
[  ] Model artifact is self-contained with SHA-256 verification
[  ] Checkpoint manager tested for Colab disconnect resilience
[  ] PCA transformer saved alongside model weights

SYSTEM READINESS
════════════════
[  ] VaR engine computing and feeding kill switch (Sprint 1)
[  ] Kill switch escalates on execution failure (Sprint 3)
[  ] Event bus does not drop critical events (Sprint 3)
[  ] GARCH refit is non-blocking (Sprint 7)
[  ] Incremental features compute in <1ms per bar (Sprint 7)
[  ] Correlation risk integrated in position sizing (Sprint 5)
[  ] Zero print() calls in src/ (Sprint 12)
[  ] Structured JSON logging active (Sprint 12)
[  ] Test coverage >80% on risk modules (Sprint 12)

API & SAAS READINESS
════════════════════
[  ] FastAPI serving signals at /api/v1/ (Sprint 9)
[  ] API key authentication working (Sprint 10)
[  ] Free/Pro tier differentiation working (Sprint 10)
[  ] Grafana dashboards loading real data (Sprint 11)
[  ] Signal outcome tracking accurate (Sprint 11)
[  ] Prometheus metrics exposed (Sprint 11)

OPERATIONAL READINESS
═════════════════════
[  ] Telegram alerts firing on kill switch changes (Sprint 13)
[  ] Dead Man's Switch monitoring signals (Sprint 13)
[  ] Daily performance summary sending (Sprint 13)
[  ] Deployment validation script passes (Sprint 15)
[  ] Load test: 100 users, P95 < 200ms (Sprint 15)
[  ] 4-hour smoke test passed (Sprint 15)
[  ] Docker Compose starts everything with one command (Sprint 15)
[  ] Demo mode available for prospects (Sprint 15)
[  ] API documentation at /docs (Sprint 15)
[  ] Terms of service endpoint active (Sprint 15)
```

---

## Summary

This roadmap transforms TradingBOT Agentic from a **6.0P / 5.2C research prototype** into a **9.0P / 9.0C commercial-grade Signal-as-a-Service platform** through 15 focused weekly sprints. The plan prioritizes safety-critical fixes first (VaR, kill switch, reward function), then performance hardening, then commercial infrastructure, and finally training + launch polish. Total new dependencies: 5 packages. Total budget impact: $0/mo (all self-hosted). The system will be ready to accept paying subscribers on day 1 after Sprint 15 completion.

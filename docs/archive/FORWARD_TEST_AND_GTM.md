# Forward Test + GTM — Operational Playbook

Generated: 2026-04-29. Pre-requisites: regime_filter (`high_vol`) wired,
state_machine wired (enter=40, exit=25), TP=5×ATR for XAU. PF backtest
1.30 OOS, ~205 sig/yr.

---

## Part 1 — J3 Forward Test (30 days, MT5 demo)

### Goal
Validate that **PF live ≥ 1.20** on real M15 bars before any monetisation.

### Setup (~1h)

1. **MT5 demo account** — IC Markets / Pepperstone / your existing broker
   (reason: MT5 demo data = same ticks as live, so any divergence is
   broker-feed-specific, not strategy-specific).

2. **`.env` overrides** — set in the project root:
   ```
   DATA_SOURCE=mt5
   MT5_LOGIN=<demo login>
   MT5_PASSWORD=<demo password>
   MT5_SERVER=<broker server name>
   SYMBOLS=XAUUSD
   NARRATIVE_MODE=template            # zero LLM cost during forward test
   REGIME_FILTER_ENABLED=1
   REGIME_FILTER_NY_MODE=high_vol     # surgical default
   REGIME_FILTER_VOL_PCTL_MAX=0.75
   STATE_MACHINE_ENABLED=1
   STATE_MACHINE_ENTER_THRESHOLD=40
   STATE_MACHINE_EXIT_THRESHOLD=25
   TELEGRAM_BOT_TOKEN=<from BotFather>
   TELEGRAM_CHAT_ID=<your chat ID>
   ```

3. **Launch the scanner**:
   ```bash
   python -m src.intelligence.main
   ```
   Verify in logs:
   - `Regime filter ON: ny_mode=high_vol, vol_pctl_max=0.75`
   - `State machine ON: enter=40, exit=25`
   - `MT5 connection successful`
   - First scan logs `bars_scanned=1`

4. **Schedule the daily summary** (Windows Task Scheduler or cron):
   ```
   23:55 UTC daily:
     python scripts/forward_test_daily.py --telegram
   Every 6h:
     python scripts/forward_test_pf_monitor.py --alert
   ```

### KPIs to track (auto-computed by `forward_test_daily.py`)

| KPI | Target | Action if missed |
|-----|--------|------------------|
| Signals/day (30d rolling) | ≥ 0.27 (≈100/yr min) | Loosen filter → `vol_pctl_max=0.85` |
| Win rate live (n≥20) | ≥ 45% | Investigate if persistent <40% |
| Rolling 30d PF | ≥ 1.20 | < 1.0 → **pause Telegram emission** |
| Filter drop rate | 50-70% | Sanity check the filter is firing |
| Tier distribution | PREMIUM > 0 | If 0 PREMIUM in 30d, recalibrate |

### When to declare forward test PASSED

- ≥ 30 closed trades observed
- Rolling 30d PF ≥ 1.20
- No catastrophic streak (max consecutive losses ≤ 6)
- Telegram delivery: no missed alerts > 1h late

→ Then proceed to Part 2 (GTM).

### When to ABORT

- 30d PF < 0.9 with n ≥ 20 → strategy broke OOS, NOT a noise event
- 7-day silence (no signals) → filter or data feed issue
- Drawdown > 15% on equity simulation

---

## Part 2 — J4 GTM Checklist

> Sequencing principle: do NOT spend money on acquisition before forward
> test passes. The compute and time costs of these steps are real but
> reversible; ad-spend isn't.

### Phase 0 — Pre-launch artefacts (5-10h, in parallel with forward test)

**Landing page** (decide on Carrd $19/yr or Webflow free tier first):
- Hero: "AI-graded XAU/USD signals — sessions Asia/London hors haute volatilité"
- Honest evidence: PF backtest 1.30 over 7 yrs, ~200 signaux/an,
  forward test in progress (link to live tracker if you build one)
- Call: "Join the closed beta — 25 spots, free 30 days"
- Email capture (Mailerlite free, ConvertKit, or Buttondown $9/mo)
- Disclaimer at the bottom: "Not financial advice. Backtest performance
  does not guarantee future results."
- Domain: a `smartsentinel.ai` / `.io` / `.app` whichever is available.

**ICP interview script** (5 calls, 30 min each, 5h):
- Find candidates on r/Forex, r/Daytrading, FR trading Discord servers,
  Twitter (search "trade XAU FR retail")
- Offer: "I'm building a signals tool, can I show you a screenshot
  of the historical signals and ask 5 questions?"
- Questions:
  1. What signals do you currently use? Pay or free?
  2. What's your typical hold time / R:R?
  3. Show backtest PF chart. Would 1.30 PF be interesting?
  4. What price would feel fair: $19, $29, $49, $99, $149?
  5. What would make you cancel after the trial?
- Goal: 3 of 5 say "would pay $19-49 for this if PF live confirms"

**Disclaimer + minimal CGU** (€500-1500 with avocat fintech FR — SKIP if
solo PoC, but BLOCKING for Stripe):
- Get template from `legalstart.fr` or `Captain Contrat` (~€200-500)
- Or paste a "Not financial advice" boilerplate + reuse standard SaaS CGU
- For Stripe onboarding: NEED clear product description, refund policy,
  KYC info on the founder.

**Stripe account + product**:
- Create Stripe account (15 min)
- One product, one price ($29/mo recurring monthly)
- Test with $1 charge to your own card → confirm webhook reception
- DO NOT launch payment until forward test passes

### Phase 1 — Soft launch (during forward test, week 3-4)

- Open landing page email signup
- Post once on r/Forex and FR Discord servers, no paid ads
- Goal: 25 emails for the closed beta
- DO NOT charge yet — give 30-day free Telegram access to the beta cohort
- Track: signup → activation (clicked Telegram bot) → engagement (👍/👎
  on signals)

### Phase 2 — Decision point (after forward test, ~day 35)

| Condition | Action |
|-----------|--------|
| Forward PF ≥ 1.20 AND ≥ 5 ICP interviews positive | Open paid ($29/mo, no decoy yet) |
| Forward PF 1.0-1.20 OR mixed feedback | Extend beta 30 more days, re-decide |
| Forward PF < 1.0 OR negative ICP feedback | Stop, investigate scoring v3 |

### Phase 3 — Paid (if Phase 2 = GO)

- Stripe live, $29/mo only (no decoy until 50 paying users)
- Continue daily forward_test_daily monitoring
- Add one PREMIUM tier ($79/mo) only when 50+ STANDARD users → decoy effect

### Anti-patterns (don't)

- ❌ Buy Google/Meta ads before Phase 2
- ❌ Add 4 tiers + decoy pricing before 50 paying users (premature)
- ❌ Hire a community manager / write 50 blog posts (premature)
- ❌ Add a 2nd instrument before XAU forward test passes
- ❌ Migrate to Kubernetes / build a dashboard / add a mobile app

---

## Part 3 — J2 Multi-Timeframe (deferred, not blocking)

### Status

Only `data/XAU_15MIN_2019_2025.csv` exists. M5 and H1 data not present.

### What's needed

1. **Download** XAU M5 + H1 from Dukascopy (use `scripts/download_dukascopy_xau.py`):
   ```
   python scripts/download_dukascopy_xau.py --tf M5 --years 2019-2025
   python scripts/download_dukascopy_xau.py --tf H1 --years 2019-2025
   ```
   Cost: ~2-4h download time per timeframe (M5 = 12× more bars than M15;
   H1 = 4× fewer).

2. **Re-run feature edge audit** on M5 and H1 separately. The pattern
   "skip NY × high vol" is M15-specific — it may transfer or invert on
   other timeframes. Run `scripts/audit_subset_edge.py` adapted for each.

3. **Decision**:
   - If M5 has its OWN stable filter combo with PF ≥ 1.30 → add to scanner
     as a separate signal stream (different cadence, different cooldown).
   - If H1 transfers → add as "Swing" tier (low frequency, higher R per
     trade). Marketable as a distinct product feature.
   - If neither transfers → stay M15-only.

### Estimated effort

- Data download: 4-8h (background, no human attention)
- Audit + verdict: 2-3h human attention
- Wiring multi-TF in the scanner if YES: ~6-10h (cooldowns must not
  cross-fire across TFs; add `timeframe` column to SignalStore)

### When to do it

After forward test confirms M15 is real. Multi-TF without a real M15 base
just multiplies the same noise.

---

## Quick reference — env vars introduced this session

| Var | Default | Effect |
|-----|---------|--------|
| `REGIME_FILTER_ENABLED` | 1 | Master switch for the regime filter |
| `REGIME_FILTER_NY_MODE` | high_vol | off / all / high_vol (surgical) |
| `REGIME_FILTER_VOL_PCTL_MAX` | 0.75 | Drop bars whose ATR_PCTL > this |
| `REGIME_FILTER_SKIP_NY` | 1 | Legacy; 0 forces ny_mode=off |
| `STATE_MACHINE_ENABLED` | 1 | Master switch for state machine gating |
| `STATE_MACHINE_ENTER_THRESHOLD` | 40 | Score needed to confirm entry |
| `STATE_MACHINE_EXIT_THRESHOLD` | 25 | Score below which signal exits |
| `STATE_MACHINE_PERSIST_PATH` | data/state_machine.json | State snapshot |

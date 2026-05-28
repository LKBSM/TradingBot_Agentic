# Smart Sentinel AI — Business Plan & Technical Pivot Document

**Version:** 1.0
**Date:** 2026-03-31
**Status:** Strategic Planning Phase

---

## 1. Executive Summary

Smart Sentinel AI is a **pivot** from autonomous trading bot to **AI-powered market intelligence platform**. Instead of managing client capital (high liability, regulatory complexity, unsolved RL problem), we sell **explainable trading signals with institutional-grade reasoning** — a Decision Support System (DSS).

**The Core Insight:** Most retail traders fail not because they can't find patterns, but because they lack the *narrative context* to know if a pattern is high-probability. Smart Sentinel fills this "Context Gap" by explaining the institutional logic behind every market setup.

**Revenue Model:** SaaS subscriptions ($49-$149/month) with 78-98% gross margins depending on scale.

**Competitive Moat:** We are the only platform combining:
1. Numba-optimized SMC detection (sub-millisecond)
2. Multi-agent risk/news/regime context
3. LLM-generated institutional narratives (explainable AI)
4. Full regulatory compliance via transparency

---

## 2. The Problem We Solve

### 2.1 Why Retail Traders Fail
- **80% of retail traders lose money** (AMF Quebec, ESMA, FCA statistics)
- They can identify patterns (FVG, Order Blocks) but lack institutional context
- They trade during high-impact news without knowing it
- They don't adjust for regime changes (trending vs. ranging)
- They use fixed stop-losses instead of volatility-adjusted levels

### 2.2 Why Trading Bots Don't Work (For a Business)
Our own 6-version RL training journey demonstrates the challenge:

| Version | Result | Root Cause |
|---------|--------|------------|
| v1-v3   | Sharpe -32 to -26 | Accounting bugs, degenerate policies |
| v4 (DSR) | Sharpe -30 | Always-hold policy |
| v5 (Masking) | Sharpe -30 | Still won't trade |
| v6 (Penalties) | Sharpe 0.00 | Zero trades |

Even with PhD-level engineering (MaskablePPO, DSR reward, EWC regularization, 4-phase curriculum), autonomous Gold trading on M15 remains unsolved. More critically:
- Managing client money requires AMF/CSA registration
- One bad sequence = lawsuit
- Compliance, insurance, and auditing costs kill margins
- Liability is unbounded

### 2.3 The Market Gap
- **LuxAlgo** ($40-$120/mo): Visual indicators only. No reasoning. No news context.
- **ICT Killzones**: Free YouTube content. No automation.
- **Bloomberg Terminal** ($24K/yr): Institutional-grade but priced for hedge funds.
- **No one** offers: Automated SMC detection + institutional narrative + risk context + regulatory compliance in a $49/mo package.

---

## 3. The Product: Smart Sentinel AI

### 3.1 Core Features

#### A. The Sentinel Scanner (Local, Free)
- Monitors Gold (XAU/USD) on M15 timeframe continuously
- Detects SMC patterns: FVG, BOS, CHOCH, Order Blocks, Fractals
- Tracks market regime (8 states: strong uptrend → high volatility)
- Monitors economic calendar (blocks signals 30min before NFP/FOMC/CPI)
- **Technology:** Python + Numba JIT (50-100x native speed)
- **Cost:** $0 (runs on VPS or user's machine)

#### B. The Confidence Engine (Deterministic Scoring)
Weighted confluence scoring (0-100):

| Factor | Weight | Source |
|--------|--------|--------|
| Break of Structure (BOS) | 25 | strategy_features.py |
| Fair Value Gap (FVG) | 20 | strategy_features.py |
| Regime Alignment | 20 | market_regime_agent.py |
| Clear News Calendar | 15 | news_analysis_agent.py |
| Volume Confirmation | 10 | OHLCV data |
| Order Block Zone | 10 | strategy_features.py |

Thresholds:
- 0-40: No signal (noise)
- 40-60: "Monitoring" (free tier sees this)
- 60-80: "Setup Detected" (Haiku validates)
- 80-100: "High-Conviction Setup" (Sonnet generates narrative)

#### C. The Narrative Engine (LLM-Powered, Premium)
When confidence > 60, the AI generates:

> **Gold Setup — Bullish (Confidence: 84/100)**
>
> "A 15-minute Break of Structure to the upside has formed above the 2,340 level, with an unfilled Fair Value Gap at 2,335-2,338 acting as potential support. The London session regime is trending bullish with ADX at 32, and no high-impact news is scheduled for the next 4 hours. Institutional intent score is elevated: the Order Block at 2,332 showed 2.3x average volume, suggesting smart money accumulation."
>
> **Entry Zone:** 2,338-2,340 | **Stop Loss:** 2,328 (2x ATR) | **Take Profit:** 2,356 (4x ATR)
> **Position Size (1% risk, $10K account):** 0.08 lots
>
> [View Full Logic →]

#### D. Risk Dashboard
- ATR-based dynamic TP/SL levels
- Kelly Criterion position sizing (adjusted by confidence)
- GARCH volatility estimation
- VaR (95/99) and CVaR exposure metrics
- Daily loss limit tracking

#### E. "Chat with the Market" (Premium)
Natural language queries:
- "Why did Gold just dump 20 points?"
- "Is London session bullish or bearish today?"
- "Should I be cautious right now?"

Uses Claude Sonnet with cached market context for instant, informed responses.

### 3.2 What Smart Sentinel is NOT
- NOT a trading bot (does not execute trades)
- NOT financial advice (educational market analysis)
- NOT a black box (every signal shows full reasoning)
- NOT managing money (user controls their own capital)

---

## 4. Technical Architecture

### 4.1 The Logic Split

```
┌─────────────────────────────────────────────────────────────┐
│                    SMART SENTINEL AI                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌────────────────────────────────┐     │
│  │ Data Feed     │───▶│ LOCAL SENTINEL (Python+Numba)  │     │
│  │ (MT5/Broker)  │    │                                │     │
│  └──────────────┘    │ strategy_features.py            │     │
│                      │  → FVG, BOS, CHOCH, OB, Fractals│     │
│                      │ market_regime_agent.py           │     │
│                      │  → 8 regime states (ADX/BB/MA)  │     │
│                      │ news_analysis_agent.py           │     │
│                      │  → Calendar blocks/warnings     │     │
│                      │ risk_manager.py                  │     │
│                      │  → ATR TP/SL, Kelly, GARCH      │     │
│                      │                                │     │
│                      │ ConfluenceDetector              │     │
│                      │  → Score 0-100                  │     │
│                      │  → "Is this a SETUP?" (Y/N)     │     │
│                      └──────────┬─────────────────────┘     │
│                                 │                            │
│                    Score >= 60 (5-15% of bars)               │
│                                 │                            │
│                      ┌──────────▼─────────────────────┐     │
│                      │ LAYER 2: Haiku 4.5 (Validator) │     │
│                      │ $1/MTok in — validates setup    │     │
│                      │ "Is this high-quality?" (Y/N)   │     │
│                      └──────────┬─────────────────────┘     │
│                                 │                            │
│                    Haiku confirms (score > 70)               │
│                                 │                            │
│                      ┌──────────▼─────────────────────┐     │
│                      │ LAYER 3: Sonnet 4.5 (Narrator) │     │
│                      │ $3/$15 MTok — full narrative    │     │
│                      │ 3-sentence institutional thesis  │     │
│                      │ + entry/SL/TP + position size   │     │
│                      └──────────┬─────────────────────┘     │
│                                 │                            │
│                      ┌──────────▼─────────────────────┐     │
│                      │ DELIVERY LAYER                  │     │
│                      │                                │     │
│                      │ • Web Dashboard (React)         │     │
│                      │ • Telegram Bot (alerts)         │     │
│                      │ • REST API (developers)         │     │
│                      │ • TradingView Webhook           │     │
│                      └────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Existing Code Reuse Map

| Component | Current File | Reuse | Adaptation Needed |
|-----------|-------------|-------|-------------------|
| SMC Engine | strategy_features.py | 95% | Add liquidity zones, MSS |
| Risk Scoring | risk_sentinel.py | 80% | Adapt from trade gating → confidence input |
| News Agent | news_analysis_agent.py | 90% | Add real-time feed |
| Regime Agent | market_regime_agent.py | 85% | None |
| Risk Manager | risk_manager.py | 90% | Extract TP/SL calculator as service |
| EventBus | events.py | 100% | None |
| Orchestrator | orchestrator.py | 80% | Adapt for scoring pipeline |
| Spread Model | execution_model.py | 75% | Use for cost estimation |
| Ensemble ML | ensemble_risk_model.py | 70% | Optional confidence boost |
| MT5 Connector | mt5_connector.py | 90% | Read-only (no order execution) |
| FastAPI | api/app.py | 80% | New routes for signals/narratives |
| Auth System | api/auth.py | 100% | Add tier-based rate limiting |
| Signal Store | api/signal_store.py | 85% | Extend schema for narratives |
| Security | security/* | 100% | None |

**Estimated reuse: 60-70% of existing codebase.**

### 4.3 What Gets Deleted

| Component | Why |
|-----------|-----|
| src/training/ (entire directory) | No RL training needed |
| curriculum_trainer.py | No curriculum phases |
| unified_agentic_env.py | No Gymnasium environment |
| colab_training_full.py | No cloud training |
| ewc_regularization.py | No catastrophic forgetting concern |
| TradingEnv reward shaping | No DSR, flat penalty, entry bonus |
| MaskablePPO integration | No RL policy |
| trained_models/ | No model checkpoints |

### 4.4 New Components to Build

| Component | Description | Effort |
|-----------|-------------|--------|
| ConfluenceDetector | Weighted scoring engine | 1 week |
| LLMNarrativeEngine | Claude API integration + caching | 1 week |
| SemanticCache | Hash-based response deduplication | 3 days |
| TelegramAlertBot | Signal delivery via Telegram | 3 days |
| UserTierManager | Subscription tiers + rate limiting | 1 week |
| Web Dashboard (React) | Signal cards + heatmaps + chat | 3 weeks |
| TradingView Webhook Bridge | Pine Script + webhook integration | 1 week |

---

## 5. Claude API Cost Optimization Strategy

### 5.1 The Three Pillars

#### Pillar 1: Local Filtering (99% cost avoidance)
The Sentinel processes every M15 bar locally. Only 5-15% trigger any API call.
- 96 bars/day × 99% filtered = 1-5 API calls/day during quiet markets
- 96 bars/day × 85% filtered = ~15 API calls/day during active sessions

#### Pillar 2: Tiered Inference
| Tier | Model | Cost (in/out per MTok) | Purpose | When |
|------|-------|----------------------|---------|------|
| 1 | Local Python | $0 | Feature detection | Every bar |
| 2 | Haiku 4.5 | $1 / $5 | Setup validation | Score >= 60 |
| 3 | Sonnet 4.5 | $3 / $15 | Full narrative | Haiku confirms |
| 4 | Opus 4.6 | $5 / $25 | Freeform chat | User questions only |

#### Pillar 3: Prompt Caching (90% discount on repeated context)

The SMC analysis rulebook (~2,000 tokens) is cached:
- 5-minute ephemeral cache: $0.0006/read vs $0.006/fresh = 90% savings
- Cache hit rate during London/NY sessions: ~85-90%

### 5.2 Token Engineering

Input format optimized for minimal tokens:
```
# Cached system prompt (2000 tokens, written once per 5-min window)
# Contains: SMC rules, output format, confidence criteria, risk params

# Per-signal dynamic payload (~150 tokens as CSV):
XAU,2345.50,2346.20,2344.80,2345.90,12500,1,1,0,0.78,bull_strong,clear,32.5,0.42,2.1
# price,high,low,close,vol,bos,fvg,choch,ob_str,regime,news,adx,bb_w,atr
```

### 5.3 Semantic Response Cache

```python
# All users on the same asset + same bar = same analysis
cache_key = f"XAU:{bar_timestamp}:{bos}:{fvg}:{regime}:{news_status}"

# 1000 Gold users, same M15 bar → 1 Sonnet call serves everyone
# Cost per user drops from $10.50/mo to $0.50/mo at scale
```

### 5.4 Monthly Cost Projections

| Scale | Users | API Calls/mo | Shared Cache Hits | Net API Cost | Cost/User |
|-------|-------|-------------|-------------------|-------------|-----------|
| Seed | 50 | 27,000 | 40% | ~$300 | $6.00 |
| Growth | 500 | 27,000 | 95% | ~$600 | $1.20 |
| Scale | 5,000 | 27,000 | 99% | ~$900 | $0.18 |

API calls don't scale with users — they scale with **market events**. This is the fundamental cost advantage.

---

## 6. Monetization: SaaS Pricing

### 6.1 Tier Structure

| Tier | Name | Price | Features |
|------|------|-------|----------|
| Free | **Observer** | $0 | SMC visual overlays (TradingView), regime status, basic alerts. No AI narrative. |
| Tier 2 | **Analyst** | $49/mo | 1 asset (Gold), AI narratives, confidence scoring, ATR TP/SL, Telegram alerts. |
| Tier 3 | **Strategist** | $99/mo | Multi-asset (Gold + 3 Forex pairs), unlimited narratives, chat with market, SMS alerts. |
| Tier 4 | **Institutional** | $149/mo | All assets, API access, custom webhooks, priority support, backtesting data. |
| API | **Developer** | $0.05/call | Pay-per-use API for other platforms/developers. |

### 6.2 Why These Prices

- **LuxAlgo charges $40-$120/mo** for visual indicators with ZERO AI reasoning
- We offer MORE (AI narratives + risk context + news filtering) at competitive prices
- $49 is below the "impulse buy" threshold for serious traders
- The free tier creates a funnel (2-5% conversion rate is industry standard)

### 6.3 Revenue Projections (Conservative)

| Month | Free Users | Paid Users | MRR | API Cost | Net Margin |
|-------|-----------|------------|-----|----------|------------|
| 3 | 500 | 15 | $735 | $100 | 86% |
| 6 | 2,000 | 80 | $3,920 | $300 | 92% |
| 12 | 8,000 | 320 | $15,680 | $600 | 96% |
| 24 | 25,000 | 1,000 | $49,000 | $900 | 98% |

Infrastructure costs (VPS, database, monitoring): ~$200-500/mo regardless of scale.

---

## 7. Regulatory & Legal Strategy

### 7.1 Quebec/Canada Compliance

**AMF (Autorite des marches financiers) Requirements:**
1. We are NOT a portfolio manager (no discretionary trading)
2. We are NOT an investment advisor (no personalized advice)
3. We ARE a market analysis tool / Decision Support System
4. Quebec's Law 25 (privacy): All data processing is transparent
5. AMF's 2025 AI guideline: We exceed requirements via "View Logic" explainability

**Key Legal Language:**
- Product is described as "market analysis and educational intelligence"
- Every signal includes: "This is not financial advice. Past analysis is not indicative of future results."
- No personalized recommendations (same analysis for all users on same asset)
- Users make their own trading decisions

### 7.2 Language That Builds Credibility Without Legal Risk

| We SAY | We NEVER say |
|--------|-------------|
| "High-conviction setup detected" | "Buy signal" |
| "Institutional confluence: 87/100" | "Guaranteed profit" |
| "Smart Money is positioned bullish" | "You should buy now" |
| "Risk-adjusted entry zone: 2,338-2,340" | "Buy at 2,340" |
| "Suggested risk parameters" | "Set your stop loss here" |

### 7.3 Compliance Features (Built Into Product)

1. **"View Logic" Button**: Every signal shows the exact parameters (BOS direction, FVG size, regime state, ATR value, news calendar status)
2. **Audit Trail**: All signals stored with full reasoning chain (signal_store.py already does this)
3. **Risk Disclaimers**: Displayed on every signal card
4. **No Execution**: We never connect to user's broker. Zero ability to move their money.

---

## 8. Competitive Positioning

### 8.1 Competitive Matrix

| Feature | LuxAlgo ($40-$120) | ICT YouTube (Free) | Bloomberg ($24K/yr) | **Smart Sentinel ($49-$149)** |
|---------|--------------------|--------------------|--------------------|-----------------------------|
| SMC Detection | Visual overlays | Manual | N/A | **Automated + Numba-optimized** |
| AI Reasoning | None | None | Analyst reports | **Real-time LLM narratives** |
| News Context | None | None | Yes | **Yes (auto calendar + blocks)** |
| Confidence Score | None | Subjective | Analyst opinion | **Quantified 0-100** |
| Position Sizing | None | Manual | Internal | **Kelly + ATR + GARCH** |
| Explainability | Black box | N/A | Analyst writes | **"View Logic" on every signal** |
| Multi-Timeframe | Basic | Manual | Yes | **Automated HTF alignment** |
| Regime Detection | None | Manual | Implied | **8-state automated detection** |
| Compliance | High risk | N/A | Compliant | **Designed for compliance** |

### 8.2 Our Moat

1. **Technical Moat**: 25K+ lines of battle-tested Python (6 months of development). Numba JIT, EventBus architecture, 876 tests. A competitor would need 6+ months to replicate.

2. **AI Moat**: We don't just detect patterns — we EXPLAIN them. This requires the multi-agent architecture (risk sentinel + news + regime + SMC + LLM). No indicator vendor has this.

3. **Cost Moat**: Shared semantic cache means our costs DON'T scale linearly with users. At 1,000 users, our per-user API cost is ~$0.50/mo while competitors without caching pay $10+/user.

4. **Regulatory Moat**: "Explainable AI" positions us favorably with regulators. Competitors offering black-box signals face increasing scrutiny under Quebec's Law 25 and the AMF's 2025 AI guideline.

---

## 9. Go-To-Market Strategy

### Phase 1: MVP (Weeks 1-6) — Gold Only
- Core scanning engine (reuse existing code)
- Confluence scoring (new, 1 week)
- Claude API integration with caching (new, 1 week)
- Telegram bot for signal delivery (new, 3 days)
- Simple web dashboard (new, 2 weeks)
- Beta launch with 20-50 users from trading communities

### Phase 2: Polish (Weeks 7-10)
- TradingView integration (Pine Script overlay + webhook)
- "Chat with the Market" feature
- Payment integration (Stripe)
- Landing page + marketing site

### Phase 3: Growth (Weeks 11-16)
- Add Forex pairs (EUR/USD, GBP/USD, USD/JPY)
- Referral program (1 month free per referral)
- Content marketing (YouTube: "What Smart Money Did Today")
- Discord/Telegram community

### Phase 4: Scale (Months 5-12)
- Add indices (NAS100, SPX)
- Mobile app (or progressive web app)
- Enterprise API
- Advanced features (backtesting, custom alerts)

### Marketing Channels
1. **YouTube**: Daily "Smart Money Recap" videos (builds authority + SEO)
2. **TradingView**: Free indicator drives awareness → upgrade funnel
3. **Telegram/Discord**: Community builds retention
4. **Twitter/X**: Real-time market commentary from the AI
5. **SEO**: "Smart Money Concepts indicator" "Gold analysis AI" "ICT automated"

---

## 10. Technical Requirements (Infrastructure)

### 10.1 Production Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| Backend API | FastAPI (existing) | - |
| Database | PostgreSQL + Redis | $20/mo |
| VPS (Sentinel) | Hetzner CX31 (4 vCPU, 8GB) | $15/mo |
| LLM API | Claude (Haiku + Sonnet) | $300-900/mo |
| Telegram Bot | Telegram API | Free |
| Web Dashboard | React + Vercel | $20/mo |
| Domain + SSL | Cloudflare | $10/mo |
| Monitoring | Prometheus + Grafana | $0 (self-hosted) |
| **Total Infrastructure** | | **$365-965/mo** |

### 10.2 Data Pipeline

```
MT5/Broker API
    │
    ▼
OHLCV Collector (M15 bars, real-time)
    │
    ▼
Feature Pipeline (strategy_features.py + ta-lib)
    │
    ├── SMC Features (FVG, BOS, CHOCH, OB)
    ├── Technical Indicators (RSI, MACD, ATR, BB)
    ├── Multi-Timeframe (1H, 4H trends)
    └── Regime + News Context
    │
    ▼
ConfluenceDetector (scoring)
    │
    ▼ (if score >= 60)
    │
LLM Pipeline (Haiku → Sonnet)
    │
    ▼
Signal Store (PostgreSQL) → Delivery (Telegram, Web, API, TradingView)
```

---

## 11. Risk Assessment

### 11.1 Business Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| AMF regulatory action | High | DSS positioning, "View Logic" transparency, no money management |
| LLM API outage | Medium | Fallback to Haiku-only, or pre-cached narratives |
| LLM cost increase | Medium | Multi-provider support (Claude + Gemini), aggressive caching |
| Signal performance decline | High | Transparent track record, disclaimers, continuous monitoring |
| Competition copies us | Medium | 6-month technical moat, continuous innovation, community lock-in |
| Low conversion rate | Medium | Aggressive free tier, referral program, content marketing |

### 11.2 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data feed interruption | High | Multi-broker fallback, health monitoring (existing) |
| Numba compatibility issues | Low | Pure Python fallback already exists |
| Cache poisoning | Medium | HMAC validation (existing security layer) |
| API key leakage | High | Secrets manager (existing), key rotation |

---

## 12. Team & Resources Needed

### MVP (1 person — the founder)
- Full-stack development (Python backend exists, React frontend needed)
- Claude API integration
- Marketing and community building

### Growth (2-3 people)
- 1 Backend/DevOps engineer
- 1 Frontend developer
- 1 Marketing/Community manager

### Scale (5-7 people)
- Add: 1 data engineer, 1 mobile developer, 1 customer success

---

## 13. Key Metrics to Track

| Metric | Target (Month 6) | Target (Month 12) |
|--------|-------------------|---------------------|
| Free users | 2,000 | 8,000 |
| Paid subscribers | 80 | 320 |
| MRR | $3,920 | $15,680 |
| Conversion rate (free→paid) | 4% | 4% |
| Churn rate (monthly) | < 8% | < 5% |
| Signal accuracy (60+ confidence) | > 55% | > 58% |
| Average confidence at entry | > 70 | > 72 |
| API cost per user | < $5 | < $1 |
| NPS score | > 40 | > 50 |

---

## 14. Summary

Smart Sentinel AI transforms 6 months of battle-tested trading infrastructure into a scalable SaaS business. By pivoting from "trading bot" to "intelligence platform," we:

1. **Eliminate** the unsolved RL training problem
2. **Eliminate** client money management liability
3. **Reuse** 60-70% of existing code (25K+ lines)
4. **Achieve** 78-98% gross margins through shared semantic caching
5. **Comply** with Quebec AMF regulations via explainable AI
6. **Compete** against $40-$120/mo indicators with superior AI reasoning
7. **Scale** because API costs grow with market events, not user count

The product answers the one question every trader asks: **"Why should I take this trade?"**

---

*This document is confidential and proprietary. Smart Sentinel AI, 2026.*

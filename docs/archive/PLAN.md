# Smart Sentinel AI v2 — Claude as Primary Analyst

## What changes and why

The current architecture uses a weighted formula (ConfluenceDetector) as the decision-maker and Claude as a narrator. This rebuild promotes Claude to primary analyst while keeping the formula as a fast pre-filter.

**Before:** Formula decides → Claude writes about it
**After:** Formula pre-filters → Claude decides (TAKE/SKIP/MODIFY) with multi-timeframe context → Claude explains as part of the decision

---

## Architecture Overview

```
M15 bar closes
    │
    ├── MT5 fetches 3 timeframes: H4(50), H1(100), M15(200)
    │
    ├── SmartMoneyEngine runs on each timeframe independently
    │   ├── H4:  BOS, FVG, OB, RSI, MACD, ATR, structure
    │   ├── H1:  BOS, FVG, OB, RSI, MACD, ATR, structure
    │   └── M15: BOS, FVG, OB, RSI, MACD, ATR, structure
    │
    ├── MarketRegimeAgent analyzes H4 prices → regime/bias
    │
    ├── NewsAnalysisAgent → calendar + sentiment
    │
    ├── ConfluenceDetector (M15 only, score ≥ 50) → pre-filter
    │   If no signal passes threshold → stop here, zero cost
    │
    └── If candidate signal exists:
        │
        ├── Build multi-TF context payload for Claude
        │
        ├── SemanticCache check (same bar + same features → serve cached)
        │
        └── Claude Sonnet analyzes full context → structured JSON decision:
            {
              "decision": "TAKE" | "SKIP" | "WAIT",
              "direction": "LONG" | "SHORT",
              "confidence": 0-100,
              "entry": 2400.00,
              "stop_loss": 2388.50,    ← structural (behind OB), not fixed ATR
              "take_profit": 2425.00,  ← next key level, not fixed ATR
              "invalidation": "Close below 2385 order block",
              "reasoning": "3-paragraph institutional analysis"
            }
```

---

## Files to CREATE (4 new files)

### 1. `src/intelligence/multi_tf_collector.py`

Fetches and enriches data across H4, H1, M15.

```python
class MultiTimeframeCollector:
    """Collects and enriches OHLCV across H4/H1/M15."""

    def __init__(self, data_provider, smc_config, regime_agent, news_agent)

    def collect(self, symbol) -> MultiTFContext:
        """
        Fetch all 3 timeframes, run SMC on each, run regime on H4.
        Returns structured context object with everything Claude needs.
        """

@dataclass
class TimeframeData:
    timeframe: str          # "H4", "H1", "M15"
    bars: pd.DataFrame      # Enriched OHLCV with SMC columns
    latest: Dict[str, float]  # Last bar's key features
    key_levels: List[float]   # Support/resistance from OB + fractals
    structure_summary: str    # "bullish BOS above 2395" etc.

@dataclass
class MultiTFContext:
    h4: TimeframeData
    h1: TimeframeData
    m15: TimeframeData
    regime: RegimeAnalysis
    news: NewsAssessment
    current_price: float
    atr_m15: float
    timestamp: str
```

**Why:** The current SentinelScanner only fetches M15. This component fetches all 3 timeframes and runs SmartMoneyEngine on each, giving Claude the full picture.

### 2. `src/intelligence/claude_analyst.py`

The core new component — Claude as decision-maker with structured JSON output.

```python
@dataclass
class TradeDecision:
    decision: str           # "TAKE", "SKIP", "WAIT"
    direction: str          # "LONG", "SHORT"
    confidence: int         # 0-100
    entry: float
    stop_loss: float        # Structural, not ATR-based
    take_profit: float      # Next key level
    rr_ratio: float
    invalidation: str       # What kills the thesis
    reasoning: str          # Full 3-paragraph analysis
    key_confluences: str
    risk_warnings: str
    cost_usd: float
    latency_ms: float
    model_used: str
    cache_hit: bool = False

class ClaudeAnalyst:
    """Claude as primary trading analyst with structured output."""

    def __init__(self, api_key, model, enable_caching=True)

    def analyze(self, context: MultiTFContext, pre_filter_score: float) -> TradeDecision:
        """
        Send full multi-TF context to Claude Sonnet.
        Returns structured TradeDecision with JSON parsing.

        1. Build prompt with H4/H1/M15 context + news + regime
        2. Call Sonnet with cached system prompt
        3. Parse JSON response into TradeDecision
        4. Validate fields (price sanity, SL behind structural level, etc.)
        """

    def generate_update(self, original_decision, current_context) -> Optional[TradeUpdate]:
        """
        Check if an active signal needs a management update:
        - Move SL to breakeven after 1R
        - Tighten SL on bearish structure
        - Early exit on thesis invalidation
        """
```

**System prompt (cached, ~2500 tokens):** Contains the full SMC rulebook, multi-timeframe analysis framework, structural SL/TP rules, and the JSON output schema. Cached via `cache_control: {"type": "ephemeral"}` — 90% cost savings on repeated calls.

**JSON output enforcement:** The prompt explicitly defines the JSON schema and says "Reply with ONLY valid JSON." If Claude returns malformed JSON, fallback to the formula's ATR-based levels.

**Key design decisions:**
- Uses Sonnet directly (no Haiku gate). Haiku's value was filtering bad signals, but Claude Sonnet now makes the full decision — Haiku adds latency without adding value.
- Structural SL/TP: The prompt instructs Claude to place SL behind the nearest order block or swing low (from the multi-TF data), not at fixed ATR.
- Confidence 0-100 replaces the formula's score for tier classification.

### 3. `tests/test_multi_tf_collector.py`

- 3 timeframes collected and enriched independently
- SMC features present in all timeframes
- key_levels extracted from fractals + order blocks
- Regime runs on H4 data
- News assessment included
- Graceful handling when H4/H1 data unavailable

### 4. `tests/test_claude_analyst.py`

- TAKE decision parsed correctly from JSON
- SKIP decision suppresses signal
- WAIT decision → no publish
- Structural SL behind order block (not fixed ATR)
- TP at key level (not fixed ATR)
- Malformed JSON → fallback to ATR-based levels
- Cache hit → no API call
- Cost tracking accurate
- Trade update: move SL to breakeven after 1R
- Trade update: early exit on invalidation

---

## Files to MODIFY (5 existing files)

### 1. `src/intelligence/sentinel_scanner.py`

**Current:** Fetches M15 → SMC → ConfluenceDetector → LLMNarrativeEngine → publish
**New:** Fetches via MultiTimeframeCollector → ConfluenceDetector (pre-filter at ≥50) → ClaudeAnalyst → publish

Changes to `_scan_once()`:
- Replace single-TF fetch with `MultiTimeframeCollector.collect()`
- Lower ConfluenceDetector threshold from 40 to 50 (still a pre-filter, but Claude has final say)
- Replace `LLMNarrativeEngine.generate_narrative()` with `ClaudeAnalyst.analyze()`
- If Claude returns SKIP → don't publish (formula said yes, Claude said no)
- If Claude returns TAKE → publish with Claude's structural SL/TP (not formula's ATR-based)
- If Claude returns WAIT → don't publish, log for next bar

New: Add `_check_active_signals()` method that runs on every bar to check if active (published, not yet closed) signals need management updates.

### 2. `src/intelligence/confluence_detector.py`

**Changes:**
- Lower default `min_score` from 40 to 50 (only candidates worth sending to Claude)
- Add `to_context_string()` method that formats the M15 confluence data as a readable string for Claude's prompt (replaces the CSV serialization — Claude reads better prose than CSV)
- No other changes — the formula stays as-is for pre-filtering

### 3. `src/api/signal_store.py`

**Schema v3:** Add columns for Claude's decision fields:
- `claude_confidence` (INTEGER) — Claude's 0-100 confidence
- `claude_decision` (TEXT) — TAKE/SKIP/WAIT
- `invalidation` (TEXT) — what kills the thesis
- `h4_bias` (TEXT) — H4 regime context summary
- `h1_structure` (TEXT) — H1 structure summary

Update `SignalRecord` dataclass and `publish()` / `_row_to_record()` accordingly.

### 4. `src/delivery/telegram_notifier.py`

**Changes to `format_signal_message()`:**
- Show Claude's confidence instead of formula score
- Show invalidation level ("Invalidates below 2385")
- Show H4 bias context ("H4: Strong uptrend, bullish structure intact")
- Show structural SL reasoning ("SL behind H1 order block at 2388")
- For STRATEGIST+: show full 3-paragraph reasoning from Claude

New: `format_update_message()` for trade management alerts:
- "Move SL to breakeven"
- "Tighten SL to 2408"
- "Close trade — thesis invalidated"

### 5. `src/intelligence/llm_narrative_engine.py`

**Changes:**
- Keep as a lightweight utility for the Institutional chat endpoint (`/api/v1/narratives/chat`)
- Remove the 3-layer cascade (VISUAL/VALIDATOR/NARRATOR) — replaced by ClaudeAnalyst
- Keep `_call_api()` as shared infrastructure (ClaudeAnalyst will use it)
- Move `SMC_SYSTEM_PROMPT` to a shared location (both ClaudeAnalyst and chat use it)

---

## Files NOT changed

- `src/intelligence/semantic_cache.py` — still works (keyed by bar + features)
- `src/api/tier_manager.py` — still works (tier gating unchanged)
- `src/api/routes/narratives.py` — still works (reads from signal_store)
- `src/api/auth.py` — already enriches with tier
- `src/api/app.py` — already registers narratives router
- `src/api/dependencies.py` — already has llm_engine + scanner slots
- `src/api/models.py` — NarrativeResponse already has all needed fields
- All existing agents (read-only usage, no modifications)

---

## Prompt Engineering: The Claude Analyst System Prompt

This is the most critical component. The cached system prompt (~2500 tokens) defines:

1. **Multi-Timeframe SMC Framework** — How to read H4 bias, H1 structure, M15 entry
2. **Structural SL/TP Rules** — SL behind OB or swing low, TP at next key level
3. **Decision Criteria** — When to TAKE, SKIP, or WAIT
4. **JSON Output Schema** — Exact fields Claude must return
5. **Anti-Hallucination Rules** — "Only reference patterns present in the data. If unsure, SKIP."

The user prompt (~300-500 tokens) contains the actual data:
- H4 summary (last 5 bars, structure, key levels)
- H1 summary (last 10 bars, structure, OBs, FVGs)
- M15 full context (last bar features, formula pre-filter score)
- Regime analysis
- News context
- Pre-filter score and component breakdown

Total prompt: ~3000 tokens per call. With caching: ~500 new tokens per call (system prompt cached). Cost: ~$0.005-0.010 per signal.

---

## Execution order

### Step 1: Create `multi_tf_collector.py` + tests
- MultiTimeframeCollector with H4/H1/M15 data collection
- TimeframeData and MultiTFContext dataclasses
- key_levels extraction from fractals + order blocks
- structure_summary generation from BOS/CHOCH signals

### Step 2: Create `claude_analyst.py` + tests
- ClaudeAnalyst with structured JSON decision output
- TradeDecision and TradeUpdate dataclasses
- System prompt with multi-TF SMC rules + JSON schema
- JSON parsing with fallback to ATR-based levels
- generate_update() for trade management

### Step 3: Modify `sentinel_scanner.py`
- Integrate MultiTimeframeCollector
- Replace LLMNarrativeEngine cascade with ClaudeAnalyst
- Add trade management loop (_check_active_signals)
- Lower pre-filter threshold

### Step 4: Modify `confluence_detector.py`
- Lower min_score to 50
- Add to_context_string() method

### Step 5: Modify `signal_store.py`
- Schema v3 with Claude decision fields

### Step 6: Modify `telegram_notifier.py`
- Updated message format with Claude's structural levels + invalidation
- New format_update_message() for trade management

### Step 7: Modify `llm_narrative_engine.py`
- Simplify to chat-only utility
- Extract shared system prompt

### Step 8: Run all tests (new + existing) — zero regressions

---

## Cost analysis

**Per signal (formula pre-filters to ~3-5/day):**
- Sonnet input: ~3000 tokens × $3/1M = $0.009
- With cache: ~500 fresh + 2500 cached × $0.30/1M = $0.0023
- Sonnet output: ~400 tokens × $15/1M = $0.006
- **Total per signal: ~$0.008 with caching**
- **Daily cost (5 signals): ~$0.04/day = ~$1.20/month**

**Trade management updates (~2/signal):**
- Additional $0.005/update × 10/day = $0.05/day
- **Monthly total: ~$2.70/month** for full multi-TF + trade management

Compared to current design: similar cost, massively better signal quality.

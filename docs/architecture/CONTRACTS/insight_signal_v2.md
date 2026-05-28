# Contrat — `InsightSignalV2` (canonique riche)

**Statut** : v2.1.0 — production
**Source de vérité** : `src/intelligence/insight_v2/`
**Importé par** : `src/api/schemas/insight.py` (à créer Phase B), webapp `lib/api/generated/`
**Versioning** : MAJOR.MINOR.PATCH ; breaking → nouvelle MAJOR + maintien v(N-1) en parallèle 6 mois.

## Vue d'ensemble

C'est **le contrat unique** dont dérivent toutes les surfaces. Toute autre payload (Telegram, B2B, FocusCard, etc.) est une **projection** de ce contrat. Aucun renderer ne calcule de valeur métier — il lit `InsightSignalV2` et projette.

## Structure (Pydantic v2, simplifiée)

```python
class InsightSignalV2(BaseModel):
    # A. Identité
    id: str                            # SHA-1(symbol|bar_ts|direction|score:.4f)[:12]
    schema_version: Literal["2.1.0"]
    instrument: InstrumentId           # XAUUSD, EURUSD, ...
    timeframe: Timeframe               # M1, M5, M15, H1, H4, D1, W1
    created_at_utc: datetime
    valid_until_utc: datetime          # created_at_utc + 4 heures

    # B. Direction
    direction: Direction               # BULLISH_SETUP / BEARISH_SETUP / NEUTRAL

    # C. Conviction (calibrated B + fallback raw A)
    conviction_0_100: int              # 0-100, mode B calibré préféré
    conviction_label: ConvictionLabel  # weak / moderate / strong / institutional
    conviction_mode: Literal["calibrated", "raw"]

    # D. Incertitude calibrée
    uncertainty: UncertaintyContext
        conformal_lower: float (0-100)
        conformal_upper: float (0-100)
        coverage_alpha: float           # default 0.10 ⇒ 90% CI
        n_calibration: int
        empirical_coverage: float

    # E. Lecture de structure SMC
    structure_readout: StructureReadout
        bos_level: Optional[float]
        bos_event_age_bars: int
        choch_present: bool
        fvg_zone: Optional[Tuple[float, float]]
        fvg_size_atr: Optional[float]
        ob_zone: Optional[Tuple[float, float]]
        ob_strength: Optional[float]
        retest_state: Literal["idle","awaiting","armed","consumed"]
        structural_invalidation: Optional[float]
        liquidity_zone_upper: Optional[Tuple[float, float]]   # aspirationnel
        liquidity_zone_lower: Optional[Tuple[float, float]]   # aspirationnel

    # F. Régime de marché
    regime_readout: RegimeReadout
        hmm_label: Literal["trend_bullish","trend_bearish","range_low_vol","stress"]
        hmm_posterior: float (0-1)
        bocpd_changepoint_prob: float (0-1)
        expected_run_length: int
        jump_ratio: float (0-1)
        regime_gate_decision: Literal["TRADE","REDUCE","BLOCK"]

    # G. Volatilité prévisionnelle
    volatility_readout: VolatilityReadout
        regime: Literal["low","normal","high"]
        forecast_atr_pips: float
        naive_atr_pips: float
        forecast_vs_naive_pct: float
        confidence_interval_pips: Tuple[float, float]
        is_fallback: bool

    # H. Contexte event-driven
    event_readout: EventReadout
        news_blackout_active: bool
        next_event_label: Optional[str]
        next_event_in_minutes: Optional[int]
        sentiment_score: float (-1, 1)
        sentiment_confidence: float (0-1)
        session: Literal["asian","london","ny_overlap","ny_afternoon","after_hours"]

    # I. Décomposition 8 composantes
    breakdown: List[ComponentBreakdown]
        - name: str
        - contribution: float
        - weight_max: float
        - reasoning: str

    # J. Statistiques historiques
    historical_stats: HistoricalStats
        similar_setups_n: int
        hit_rate_observed: float
        profit_factor: float
        profit_factor_ci95: Tuple[float, float]
        empirical_coverage: float
        backtest_window: str

    # K. Narratif & sources
    narrative_short: str               # ≤ 400 chars
    narrative_long: Optional[str]      # ≤ 2000 chars
    narrative_language: Literal["fr","en","de","es"]
    sources_cited: List[SourceCitation]  # Phase 2B RAG

    # L. Compliance
    compliance: ComplianceMeta
        disclaimer_lang: str
        jurisdiction_blocked: List[str]
        edge_claim: bool               # False par défaut, codifié
        is_paper_demo: bool            # True tant que pas validé live
```

## Champs critiques pour les projections

| Surface | Champs minimaux requis |
|---|---|
| **FocusCard** (≤ 200 chars) | direction · conviction_label · narrative_short[:140] · event_readout.next_event_in_minutes (si ≤ 4h) · historical_stats.profit_factor + ci95 |
| **CopilotCard** (6 sections) | + structure_readout.bos_level + ob_zone + fvg_zone + retest_state · regime_readout.hmm_label + gate_decision · volatility_readout.regime + forecast_vs_naive_pct · event_readout.full · breakdown[].name (count only) |
| **ExpertFull** | tout, incl. uncertainty + breakdown détaillé + sources_cited |
| **TelegramB2C** (≤ 800 chars) | direction + conviction_label + bos_level + fvg_zone + retest_state + structural_invalidation + regime_label + gate + vol_regime + forecast_vs_naive_pct + next_event + session + narrative_short |
| **B2B JSON** | TOUT le modèle (équivalent `signal.model_dump(mode="json")`) |

## Versioning policy

- **Patch (2.1.0 → 2.1.1)** : bug fix, ajout d'un champ optionnel, pas d'impact downstream.
- **Minor (2.1.0 → 2.2.0)** : ajout d'un champ requis non-breaking pour les clients laxistes, ajout d'un sous-modèle entier (ex. `microstructure_readout`).
- **Major (2.1.0 → 3.0.0)** : changement de sémantique d'un champ, suppression d'un champ, refactor structurel. Maintien v2 en parallèle 6 mois.

Header API : `X-Schema-Version: insight_v2_2.1.0`.

## Règles d'évolution

1. Toute modification commence par un PR sur `src/intelligence/insight_v2/v2_X_Y.py` (versionné).
2. Le PR doit générer le nouveau client TypeScript dans `webapp/lib/api/generated/` et passer les tests.
3. Le contrat est figé une fois en prod. Pour changer, on incrémente.
4. `valid_until_utc` reste à `created_at_utc + 4h` jusqu'à preuve empirique d'une autre fenêtre optimale (ne pas changer sans données).

## Tests obligatoires

- Round-trip JSON : `InsightSignalV2 → JSON → InsightSignalV2` reproduit l'objet.
- Signal ID déterministe : même input → même `id`.
- `valid_until_utc > created_at_utc` invariant.
- `0 ≤ conviction_0_100 ≤ 100`.
- `conformal_lower ≤ conviction_0_100 ≤ conformal_upper` (sauf cas conformal saturé bords).
- `compliance.edge_claim == False` tant que les 4 critères du board ne sont pas franchis.

## Lien avec les autres contrats

- `CONTRACTS/focus_card.md` — projection FOCUS
- `CONTRACTS/copilot_card.md` — projection CO-PILOT (à créer)
- `CONTRACTS/expert_full.md` — projection EXPERT (à créer)
- `CONTRACTS/telegram_render.md` — projection Telegram B2C
- `CONTRACTS/tradingview_payload.md` — payload TV Pine (no API)
- `CONTRACTS/b2b_json.md` — payload B2B JSON full

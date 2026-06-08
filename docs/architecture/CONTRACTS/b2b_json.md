# Contrat — B2B JSON full payload

**Statut** : Vague 3 (DEFER MRR B2C > $5k 3 mois — DG-071)
**Renderer** : `src/delivery/renderers/b2b.py` (à créer Phase F)
**Source** : `InsightSignalV2` canonical
**Cible** : partenaires brokers / éducateurs via API REST `/api/v1/insights/{id}` (pull) ou webhook B2B push (DG-030 reactivé)

## Vue d'ensemble

Le payload B2B est **le plus riche** : c'est `InsightSignalV2.model_dump(mode="json")` exposé tel quel, sans masquage, avec ajout d'éléments contextuels destinés aux intégrations machines (scénarios alternatifs, télémétrie régime, hashes pour déduplication).

## Structure (JSON Schema 2020-12 simplifié)

```json
{
  "$schema": "https://mia.markets/schemas/insight_v2/2.1.0.json",
  "type": "object",
  "required": ["id", "schema_version", "instrument", "timeframe",
               "created_at_utc", "valid_until_utc", "direction",
               "conviction_0_100", "conviction_label",
               "structure_readout", "regime_readout", "volatility_readout",
               "event_readout", "breakdown", "historical_stats",
               "narrative_short", "compliance"],
  "properties": {

    "id":               { "type": "string", "pattern": "^[0-9a-f]{12}$" },
    "schema_version":   { "const": "2.1.0" },
    "instrument":       { "type": "string", "enum": ["XAUUSD","EURUSD","BTCUSD","US500","GBPUSD","USDJPY"] },
    "timeframe":        { "type": "string", "enum": ["M1","M5","M15","H1","H4","D1","W1"] },
    "created_at_utc":   { "type": "string", "format": "date-time" },
    "valid_until_utc":  { "type": "string", "format": "date-time" },
    "direction":        { "type": "string", "enum": ["BULLISH_SETUP","BEARISH_SETUP","NEUTRAL"] },

    "conviction_0_100":   { "type": "integer", "minimum": 0, "maximum": 100 },
    "conviction_label":   { "type": "string", "enum": ["weak","moderate","strong","institutional"] },
    "conviction_mode":    { "type": "string", "enum": ["calibrated","raw"] },

    "uncertainty": {
      "type": "object",
      "properties": {
        "conformal_lower":    { "type": "number" },
        "conformal_upper":    { "type": "number" },
        "coverage_alpha":     { "type": "number" },
        "n_calibration":      { "type": "integer" },
        "empirical_coverage": { "type": "number" }
      }
    },

    "structure_readout": { "$ref": "#/definitions/StructureReadout" },
    "regime_readout":    { "$ref": "#/definitions/RegimeReadout" },
    "volatility_readout":{ "$ref": "#/definitions/VolatilityReadout" },
    "event_readout":     { "$ref": "#/definitions/EventReadout" },

    "breakdown": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "contribution", "weight_max", "reasoning"],
        "properties": {
          "name":         { "type": "string" },
          "contribution": { "type": "number" },
          "weight_max":   { "type": "number" },
          "reasoning":    { "type": "string" }
        }
      }
    },

    "historical_stats": {
      "type": "object",
      "properties": {
        "similar_setups_n":    { "type": "integer" },
        "hit_rate_observed":   { "type": "number" },
        "profit_factor":       { "type": "number" },
        "profit_factor_ci95":  { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },
        "empirical_coverage":  { "type": "number" },
        "backtest_window":     { "type": "string" }
      }
    },

    "narrative_short":    { "type": "string", "maxLength": 400 },
    "narrative_long":     { "type": "string", "maxLength": 2000 },
    "narrative_language": { "type": "string", "enum": ["fr","en","de","es"] },
    "sources_cited":      { "type": "array", "items": { "$ref": "#/definitions/SourceCitation" } },

    "compliance": {
      "type": "object",
      "properties": {
        "disclaimer_lang":      { "type": "string" },
        "jurisdiction_blocked": { "type": "array", "items": { "type": "string" } },
        "edge_claim":           { "type": "boolean" },
        "is_paper_demo":        { "type": "boolean" }
      }
    },

    // ===== B2B-only extensions =====

    "scenarios": {
      "description": "Scénarios alternatifs : si direction renverse, si gate passe BLOCK, etc.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "condition":    { "type": "string" },
          "probability":  { "type": "number" },
          "implication":  { "type": "string" }
        }
      }
    },

    "telemetry": {
      "description": "Métriques de débogage et trace amont — utile pour audit broker",
      "type": "object",
      "properties": {
        "pipeline_version":   { "type": "string" },
        "lgbm_model_hash":    { "type": "string" },
        "data_provider":      { "type": "string" },
        "compute_duration_ms":{ "type": "number" },
        "cache_hit":          { "type": "boolean" },
        "tier_routed_llm":    { "type": "string", "enum": ["haiku","sonnet","opus","template"] }
      }
    },

    "signed_at_utc": { "type": "string", "format": "date-time" },
    "signature":     { "type": "string", "description": "HMAC-SHA256 of payload using shared partner secret" }

  }
}
```

## Endpoints exposés

### Pull
- `GET /api/v1/insights/current/{symbol}` → dernier insight publié
- `GET /api/v1/insights/{signal_id}` → insight précis
- `GET /api/v1/insights?since={ts}&symbol={...}&limit=100` → batch query
- `GET /api/v1/track-record/aggregate?window=30d&symbol=XAUUSD` → stats agrégées

### Webhook push (DG-030 reactivé)
- Le partenaire enregistre un webhook URL via `POST /api/v1/webhooks/subscribe`
- MIA pousse `POST {partner_url}` à chaque publication
- Payload signé HMAC-SHA256 (header `X-MIA-Signature: t={ts},v1={hmac}`)
- Retry exponentiel : 1s, 5s, 30s, 5min, 30min (max 5 tentatives)
- DLQ après 5 échecs → notification email partenaire

## Sécurité

| Mécanisme | Détail |
|---|---|
| **Auth** | API key bearer per partner (rotation 90 jours) |
| **Signature** | HMAC-SHA256 sur `t={unix_ts}.{json_body}` |
| **Tolerance window** | ± 5 min sur timestamp pour éviter replay |
| **Nonce** | `signal_id` sert d'idempotency key (le partenaire dedup côté lui) |
| **Rate limit** | 600 req/min per partner (DG-006 tier INSTITUTIONAL) |
| **IP allowlist** | Optionnel, configurable per partner |

## Exemple complet (raccourci)

```json
{
  "id": "0193c7a42f1b",
  "schema_version": "2.1.0",
  "instrument": "XAUUSD",
  "timeframe": "M15",
  "created_at_utc": "2026-05-16T11:47:00Z",
  "valid_until_utc": "2026-05-16T15:47:00Z",
  "direction": "BULLISH_SETUP",
  "conviction_0_100": 72,
  "conviction_label": "strong",
  "conviction_mode": "calibrated",
  "uncertainty": {
    "conformal_lower": 54,
    "conformal_upper": 82,
    "coverage_alpha": 0.10,
    "n_calibration": 2000,
    "empirical_coverage": 0.91
  },
  "structure_readout": {
    "bos_level": 2391.5,
    "bos_event_age_bars": 2,
    "choch_present": false,
    "fvg_zone": [2378.0, 2381.0],
    "fvg_size_atr": 0.42,
    "ob_zone": [2375.0, 2378.0],
    "ob_strength": 0.73,
    "retest_state": "armed",
    "structural_invalidation": 2378.0,
    "liquidity_zone_upper": null,
    "liquidity_zone_lower": null
  },
  "regime_readout": {
    "hmm_label": "trend_bullish",
    "hmm_posterior": 0.71,
    "bocpd_changepoint_prob": 0.03,
    "expected_run_length": 180,
    "jump_ratio": 0.12,
    "regime_gate_decision": "TRADE"
  },
  "volatility_readout": {
    "regime": "normal",
    "forecast_atr_pips": 8.7,
    "naive_atr_pips": 7.9,
    "forecast_vs_naive_pct": 10.13,
    "confidence_interval_pips": [7.2, 10.4],
    "is_fallback": false
  },
  "event_readout": {
    "news_blackout_active": false,
    "next_event_label": "FOMC Minutes",
    "next_event_in_minutes": 1083,
    "sentiment_score": 0.3,
    "sentiment_confidence": 0.7,
    "session": "ny_overlap"
  },
  "breakdown": [
    { "name": "bos",      "contribution": 13.5, "weight_max": 15.0, "reasoning": "BOS retest armé sans CHOCH" },
    { "name": "fvg",      "contribution": 8.91, "weight_max": 15.0, "reasoning": "FVG 0.42 ATR aligné" },
    { "name": "ob",       "contribution": 7.3,  "weight_max": 10.0, "reasoning": "OB strength 0.73 normalized" },
    { "name": "regime",   "contribution": 17.5, "weight_max": 25.0, "reasoning": "trend_bullish aligned, confidence 0.71" },
    { "name": "news",     "contribution": 9.1,  "weight_max": 20.0, "reasoning": "sentiment +0.3, confidence 0.7" },
    { "name": "volume",   "contribution": 6.0,  "weight_max": 10.0, "reasoning": "volume 1.2× MA20" },
    { "name": "momentum", "contribution": 1.8,  "weight_max": 3.0,  "reasoning": "RSI 58, MACD aligned bull" },
    { "name": "rsi_div",  "contribution": 0.0,  "weight_max": 2.0,  "reasoning": "no divergence on CHOCH" }
  ],
  "historical_stats": {
    "similar_setups_n": null,
    "hit_rate_observed": null,
    "profit_factor": null,
    "profit_factor_ci95": null,
    "empirical_coverage": 0.91,
    "backtest_window": "OOS validation pending — Sprint 1 (Blocker #3)"
  },
  "narrative_short": "Lecture haussière XAU M15 : BOS + retest FVG armé. Régime trend bullish, gate TRADE.",
  "narrative_long": null,
  "narrative_language": "fr",
  "sources_cited": [],
  "compliance": {
    "disclaimer_lang": "fr",
    "jurisdiction_blocked": ["US","QC","UK","OFAC"],
    "edge_claim": false,
    "is_paper_demo": true
  },
  "scenarios": [
    {
      "condition": "Si BOCPD cp_prob > 0.10 dans les 2 prochaines barres",
      "probability": 0.03,
      "implication": "regime_gate_decision basculerait vers BLOCK"
    }
  ],
  "telemetry": {
    "pipeline_version": "v2.1.0",
    "lgbm_model_hash": "sha256:abc123...",
    "data_provider": "csv:XAU_15MIN_2019_2026.csv",
    "compute_duration_ms": 47,
    "cache_hit": false,
    "tier_routed_llm": "template"
  },
  "signed_at_utc": "2026-05-16T11:47:00.123Z",
  "signature": "t=1747397220.eyJhbGc..."
}
```

## Tests obligatoires

- Round-trip JSON ↔ Pydantic `InsightSignalV2`.
- HMAC signature vérifiable avec secret partenaire.
- `valid_until_utc` > `created_at_utc`.
- `compliance.edge_claim == false` tant que critères non franchis (gate de test).
- Pas de PII utilisateur dans le payload.
- Idempotency : 2 requêtes pour le même `signal_id` → identique au byte près (hors `signed_at_utc`).

## Lien avec autres contrats

- `insight_signal_v2.md` — source canonique.
- `CONTRACTS/webhook_b2b_subscription.md` (à créer V3) — protocole inscription / désinscription webhook.

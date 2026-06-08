# Contrat — TradingView Pine showcase (V1)

**Statut** : à livrer Phase D · effort 26-36 h
**Source** : aucun (calcul Pine local, pas de payload réseau côté MIA)
**Cible** : utilisateurs TradingView qui chargent le script publié

## Vue d'ensemble

Le **TradingView Pine showcase V1** est un script Pine v5 **autonome** (aucun appel API externe MIA). Il calcule **localement** sur le chart de l'utilisateur les éléments SMC essentiels et affiche un score simplifié, en redirigeant vers `mia.markets` pour la version complète conversationnelle.

**Distinction importante** :
- **V1 (ce contrat)** : script Pine pur, showcase pour acquisition. Pas de payload côté MIA.
- **V2 (`tv_webhook_payload.md` à créer)** : webhook receiver MIA qui reçoit les alerts TV configurées par les users payants TV Pro.

## Structure Pine v5 (organisation script)

```pine
//@version=5
indicator("M.I.A. Markets — SMC Sentinel (V1)",
         shorttitle="MIA-SMC",
         overlay=true,
         max_lines_count=100,
         max_boxes_count=100,
         max_labels_count=100)

// ============ INPUTS ============
i_show_bos      = input.bool(true,  "Afficher BOS / CHOCH")
i_show_fvg      = input.bool(true,  "Afficher FVG")
i_show_ob       = input.bool(true,  "Afficher Order Blocks")
i_show_retest   = input.bool(true,  "Afficher état retest")
i_show_score    = input.bool(true,  "Panel score 5 facteurs")
i_invalidation  = input.bool(true,  "Afficher niveau invalidation")
i_lang          = input.string("fr", "Langue", options=["fr", "en"])

// ============ ATR + Wilder ============
atr_len = 14
atr = ta.atr(atr_len)

// ============ Williams Fractal 2-bar ============
up_fractal   = high[2] > high[4] and high[2] > high[3] and high[2] > high[1] and high[2] > high[0]
down_fractal = low[2]  < low[4]  and low[2]  < low[3]  and low[2]  < low[1]  and low[2]  < low[0]

// ============ BOS / CHOCH ============
// var float last_up_fractal_price, last_down_fractal_price, ...
// ... (logique détection cassure swing high/low)
// Output : bos_signal (-1, 0, +1), bos_break_level, choch_signal

// ============ FVG ============
bullish_fvg = low[0]  > high[2]
bearish_fvg = high[0] < low[2]
fvg_size    = bullish_fvg ? (low[0] - high[2]) : (bearish_fvg ? (low[2] - high[0]) : 0)
fvg_size_norm = fvg_size / atr

// ============ Order Block ============
// (engulfing + last opposite candle before break)

// ============ Retest state machine ============
// (idle / awaiting / armed / consumed)

// ============ Score simplifié 5 facteurs ============
score_bos      = bos_age_bars < 5 ? 25 : (bos_age_bars < 20 ? 15 : 5)
score_fvg      = fvg_present ? math.min(15, fvg_size_norm * 30) : 0
score_ob       = ob_present ? ob_strength_norm * 15 : 0
score_retest   = retest_state == "armed" ? 20 : (retest_state == "awaiting" ? 10 : 0)
score_atr_vol  = atr / sma_atr_50 > 0.8 and atr / sma_atr_50 < 1.5 ? 10 : 5

score_total = score_bos + score_fvg + score_ob + score_retest + score_atr_vol  // /85

// ============ Rendu visuel ============
// line.new() pour bos_break_level
// box.new() pour FVG (couleur direction)
// box.new() pour OB
// label.new() pour annotations "Lecture haussière · Score 65/85"
// label.new() pour "Invalidation 2378"

// ============ Panel info bottom-right ============
// Score + direction + retest_state + ATR normalized

// ============ Disclaimer bottom ============
// label.new(x=last_bar, y=low - 5*atr, text="Lecture éducative · Version complète + chatbot Sentinel : mia.markets")
```

## Format des annotations chart

| Élément | Type Pine | Couleur | Style |
|---|---|---|---|
| Niveau BOS | `line.new()` | `color.green` (bull) / `color.red` (bear) | `line.style_solid`, width=2 |
| Label "BOS 2391.5" | `label.new()` | idem | text size small |
| Zone FVG bullish | `box.new()` | `color.new(green, 80)` | bordure dotted |
| Zone FVG bearish | `box.new()` | `color.new(red, 80)` | idem |
| Zone OB bullish | `box.new()` | `color.new(blue, 70)` | bordure solid |
| Zone OB bearish | `box.new()` | `color.new(orange, 70)` | idem |
| Niveau invalidation | `line.new()` | `color.gray` | `line.style_dashed`, width=1 |
| Label "Invalidation 2378" | `label.new()` | gray | text size tiny |
| Annotation principale | `label.new()` | direction-dependent | "Lecture haussière · Score 65/85" |
| Panel info | `table.new()` (bottom-right) | dark theme | 4 lignes × 2 cols |
| Disclaimer bottom | `label.new()` | gray, italic | "Lecture éducative · Version complète : mia.markets" |

## Compliance — wording autorisé

✅ Autorisés :
- "Lecture haussière" / "Lecture baissière" / "Marché illisible"
- "Cassure structure" / "BOS" (avec disclaimer)
- "Zone de déséquilibre" (FVG)
- "Niveau d'invalidation"
- "Retest armé" / "Retest en attente"
- "Score 65/85"

❌ Interdits :
- "Achetez" / "Vendez" / "BUY" / "SELL"
- "Take profit" / "Stop loss" / "Cible"
- "Trade gagnant garanti"
- "Edge prouvé"
- "Signal de trading"

Garde-fou : la description du script TV (auto-published) inclut le disclaimer compliance complet UE 2024/2811 + lien `/legal/disclaimer/{lang}`.

## Profil TradingView "M.I.A. Markets"

- **Username** : `MIAMarkets` ou `MiaMarkets`
- **Bio** : "M.I.A. Markets — Indicateur de marché conversationnel. Or, FX, multi-actifs. Lectures algorithmiques éducatives. Version complète + chatbot Sentinel : mia.markets"
- **Lien externe** : `https://mia.markets/tv?utm_source=tradingview&utm_medium=profile`
- **Tags script** : SMC, ICT, Smart Money, BOS, FVG, Order Block, Educational
- **Screenshots requis** : 3-5 captures haute résolution montrant le rendu en contexte XAU + EUR.

## Landing page dédiée `/tv`

Le lien dans la description du script et du profil pointe vers `https://mia.markets/{locale}/tv?utm_source=tradingview` qui héberge :

1. Hero "Tu viens de TradingView ?"
2. Comparatif : "Sur TradingView (gratuit) → 5 facteurs locaux" **vs** "Sur mia.markets (à partir de FREE) → 8 facteurs calibrés + chatbot pédagogique + méthodologie publique (12 papers, 7 ans de data, 2 actifs)"

   > ⚠️ Depuis pivot 2026-05-27, **aucun chiffre de performance** (PF, IC, win-rate, setups historiques chiffrés) n'apparaît côté client tant que la Gate de promotion premium n'est pas franchie. Voir `pivot_positioning_2026_05_27`.
3. CTA "Essayer gratuitement"
4. Témoignages utilisateurs TV (V2 après collecte 10+ retours)

## Métriques

| Source | Métrique | Périodicité |
|---|---|---|
| TradingView Creator Dashboard | Script views | quotidien |
| TradingView Creator Dashboard | Favorites | quotidien |
| TradingView Creator Dashboard | Likes/ratings | quotidien |
| TradingView Creator Dashboard | Comments | quotidien |
| Plausible | `visits?utm_source=tradingview` | event |
| Plausible | `signup?utm_source=tradingview` | event |
| Plausible | `paid_conversion?utm_source=tradingview` | event |
| Plausible (custom) | Funnel TV → signup → trial → paid | dashboard mensuel |

## Tests obligatoires

Tests Pine (manuels, hors CI) :
1. Le script charge sans erreur sur XAU M15, EUR M15, BTC H1, ES H1.
2. Les annotations BOS/FVG/OB s'affichent dans les bonnes couleurs selon direction.
3. Le retest state machine transitions correctement (idle → awaiting → armed → consumed).
4. Le score 5 facteurs ne dépasse jamais 85.
5. Le disclaimer en bas est visible sur toutes les résolutions.
6. Le lien dans le label dispose de l'UTM `?utm_source=tradingview&utm_medium=label`.

Tests landing page `/tv` (CI Playwright) :
1. Page accessible en FR et EN.
2. CTA "Essayer gratuitement" redirige vers `/[locale]/signup?utm_source=tradingview`.
3. Disclaimer UE 2024/2811 présent en footer.
4. Page passe Lighthouse score ≥ 90.

## Lien avec autres contrats

- `insight_signal_v2.md` — source canonique (ne s'applique pas au showcase V1, mais s'applique au V2 webhook receiver).
- `CONTRACTS/tv_webhook_payload.md` (à créer V2) — normalisation des alerts TV vers `InsightSignalV2`.

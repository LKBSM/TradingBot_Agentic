# SMC jargon policy — untranslatable terms

**Status:** normative. Established by mission I18N-1 (2026-09-03). Applies to all
three shipped locales (`fr`, `en`, `es`) and any locale added later.

## The rule

Smart-Money-Concepts (SMC) trading terms stay **in English in every language**.
SMC traders use these terms in English whatever their own language, so
translating them would make the product *less* legible, not more.

What is translated is everything *around* the term. « liquidité vente · intacte »
becomes « sell liquidity · intact » (en) and « liquidez venta · intacta » (es) —
but **« Order Block » never becomes « Bloc d'ordres » / « Bloque de órdenes ».**

## The exact list of terms that are NEVER translated

| Term | Abbrev. | Meaning (context only — the term itself stays English) |
|------|---------|--------------------------------------------------------|
| Order Block | OB | last opposite candle before a break of structure |
| Fair Value Gap | FVG | imbalance left by three consecutive candles |
| Break of Structure | BOS | a swing high/low taken in the trend direction |
| Change of Character | CHOCH | the first swing taken against the prior trend |
| Buy-Side Liquidity | BSL | resting liquidity above (buy stops) |
| Sell-Side Liquidity | SSL | resting liquidity below (sell stops) |

Instrument **codes** are never translated either: `XAUUSD`, `EURUSD`, `XAU/USD`,
`BTCUSD`, `US500`, `GBPUSD`, `USDJPY`. The human market *name* IS localized
(« Gold (XAU/USD) » / « Or (XAU/USD) » / « Oro (XAU/USD) ») via the
`reading.labels.instrument_<code>` keys; the code inside the parentheses is not.

Brand names stay verbatim in every language: `M.I.A`, `M.I.A Markets`,
`M.I.A Agent`.

## Jargon must be EXPLAINED where it appears — in each language

Keeping the term in English does not mean leaving it unexplained. Wherever an SMC
term surfaces to the reader, a plain-language gloss accompanies it **in the active
language** — e.g. the on-chart liquidity label pairs the acronym with its meaning
(`SSL · EQL (creux égaux) · intacte`), and the `zones.mia.answer.explain*` strings
define « Order Block » / « Fair Value Gap » in fr, en AND es while keeping the term
itself English.

## How this is enforced

- `lib/i18n/__tests__/i18n-no-leak.test.ts` asserts « Order Block » and
  « Fair Value Gap » appear **verbatim** in every locale bundle, and exempts the
  jargon list from the cross-language witness scan.
- Reviewers reject any PR that renders a translated form of a term above.

## Adding a market (scales to 80+)

A new market is a purely additive step — no term above changes:

1. add one entry to `config/markets.json` (`id`, `symbol`, `type`, `priceDecimals`, `glyph`, `timeframes`) — the `label` there is the FR baseline;
2. run `node scripts/gen_markets.mjs`;
3. add `reading.labels.instrument_<CODE>` to `messages/{fr,en,es}.json` (three
   locale values — the localized human name; the code stays literal).

The sidebar and the header both read that one i18n key, so the market name is
identical across the menu and the header in all three languages.

---
name: market-profiles
description: Market profiles (catalog/markets.yaml), their implementation status and the execution rules each market needs. Use when adding an asset, diagnosing why an experiment refuses to run, or proposing a market other than crypto perpetuals.
---
# Market profiles

An asset belongs to a market (`market` in assets.yaml). The market defines sessions, shorting rules, fees and settlement events. The engine reads these fields; it does not assume them.

## Status
- `implemented`: the engine can run experiments on this market.
- `not_implemented`: the engine refuses to run. This is intentional: a result built on invented rules is worse than no result.

## Adding a market
1. Declare the profile in `catalog/markets.yaml` with `implementation`, sessions, shorting rules and fee model.
2. Create `engine/trading_engine/markets/<implementation>.py` with the `MarketRules` class (is_tradable, can_short, fee, settlement_events).
3. Register the class in `markets/__init__.py`.
4. Switch the status to `implemented` only when the rules have tests and a validated data source exists.
5. Markets with sessions also need a trading calendar: walk-forward buffers counted in calendar days leak on them (ML4T ch06), and `bars_per_year` changes.

## USD-M crypto perpetuals (implemented)
24/7. Shorting allowed. Maker/taker fees in bps, declared in each experiment. Funding (every 8 h) is not modeled in v1: a run that needs it fails with an explicit error, and `check-experiment` warns that results exclude it. The book's perpetual-funding case study is the reference for modeling it.

## Known pending markets
- US equities: regular sessions, open and close auctions, short locates and the uptick rule.
- Dated futures: contract rolling and margins.
- Crypto spot: no funding, no shorting in most cases.

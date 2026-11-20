---
name: execution-model
description: "How the engine simulates orders: fill price against bid/ask, fees, slippage, sizing, stops, time windows and avoiding look-ahead. Use when implementing or reviewing the runner, or when interpreting a result that depends on costs."
---
# Execution model

This is what the engine does or must do. The methodology behind it is ML4T ch16 (backtest protocol) and ch18 (transaction costs) in the `trading-cuantitativo` skill.

## Time windows
- Every period in an experiment is half-open, `[start, end)`: the end is excluded. The runner must use the same convention.
- Walk-forward folds and the sealed holdout are defined in `setup.yaml` (see `experiment-design`). Between training and validation there is a purge equal to the label horizon.
- `fetch` dates are the exception: both are inclusive, because they name whole days of data to download.

## Labels
Labels follow execution: with `next_bar_open`, `fwd_ret_h(t) = open[t + cadence + h] / open[t + cadence] - 1` (see `engine/trading_engine/labels.py`).

## No look-ahead
- A signal computed with the close of bar t executes at the open of bar t+1 (`execution_delay: next_bar_open`) or at the first quote after the close of t (`next_quote`). Same-bar fills are rejected by the validator.
- Sizing only uses information available at decision time.
- Bars are built causally: a bar stamped t covers trades in [t - cadence, t) and is known at t.
- Feature families are checked for look-ahead by recomputing them on truncated histories (`LookAheadError`).

## Fill price
- With quotes: buy at the ask, sell at the bid. This is the main advantage of L1 over OHLC bars.
- With trades only: there is no bid or ask. The setup declares the approximation in `costs.slippage_model` (`estimated_half_spread` or `fixed_bps`), and every result produced with it is labeled as such.

## Fees and cost sensitivity
- In bps of notional: `fee = |notional| × bps / 10,000`.
- Maker or taker depending on the order type. Rates come from the setup, never hard-coded.
- Costs are always on. A result without costs is not reported.
- Every strategy result is re-run across `costs.sensitivity_bps` to find the break-even cost (ch16, ch18). If it survives only in the optimistic case, it is fragile.

## Stops and targets
- If a stop and a target are both touched within the same bar, assume the worst case for the position. Document the rule in the run.
- Adding a stop rule where none existed is a mechanic change: it bumps `setup_version`.

## Out of scope in v1
- Perpetual funding, liquidations and mark price. The engine raises an error if a run needs them, and `check-experiment` warns that results exclude funding (`costs.funding_modeled: false`). The book's perpetual-funding case study (`case_studies/`, see `ml4t-code-scout`) is the reference for implementing it.
- Market impact on the book: the model assumes execution at the reference price, which is optimistic for large orders. Ch18 (square-root impact, Almgren-Chriss) is the reference for implementing it.

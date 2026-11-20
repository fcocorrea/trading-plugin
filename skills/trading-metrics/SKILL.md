---
name: trading-metrics
description: Performance, signal and risk metrics for runs, organized in the three ML4T metric layers (model, signal, strategy), with annualization and multiple-testing corrections. Use when computing or interpreting backtest results, choosing the selection metric of an experiment, or comparing experiments.
---
# Metrics

Three layers, evaluated in this order (ML4T ch06). An experiment selects on one model or signal metric (`metrics.selection`) and reports a separate set (`metrics.report`).

## 1. Model diagnostics (no costs)
Does the model learn the label and generalize across folds? Loss (MSE, MAE, log loss), calibration, AUC or hit rate for direction labels, stability across folds.

## 2. Signal diagnostics
Does the score behave like a tradable signal?
- **Rank IC**: Spearman correlation between the score and the subsequent return. Cross-sectional (per timestamp, across symbols) with at least 10 symbols; otherwise time series (per symbol and period, then averaged across symbols per period). The runner picks the mode with this rule and reports it.
- **ICIR**: mean IC divided by its standard deviation.
- Report the IC with a HAC (autocorrelation-robust) interval. A model climbs the ladder only if that interval excludes zero (ch11, ch12).

## 3. Strategy outcomes (late stage)
- Gross and net return, annualized volatility, net Sharpe and Sortino with a declared risk-free rate, Calmar.
- Maximum drawdown and its duration.
- Number of trades, exposure, turnover, expectancy per trade, hit rate and average loss / average win.
- Fee share of gross result and **break-even cost** from the cost sweep (`costs.sensitivity_bps`). If fees take more than half of the gross result, the result depends on cost assumptions.

## Annualization
Use `bars_per_year` from `catalog/timeframes.yaml`. Values assume a 24/7 calendar (crypto). A market with sessions needs other values.

## Multiple testing
- With N trials, the best in-sample Sharpe is biased upward. Correct it with the **Deflated Sharpe Ratio** using the trial count that `check-experiment` reports (all trials, including failed ones) (ch16).
- For many signals or features at once, control the false discovery rate (Holm, Benjamini-Hochberg) (ch07, ch20).
- With few trades or decision times, the metric is noise. Set a minimum and flag it when it is not met.
- Folds and the sealed holdout are always reported separately.

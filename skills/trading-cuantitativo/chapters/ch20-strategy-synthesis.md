# Chapter 20: Strategy Synthesis

## Core Idea
Looking at the nine case studies (seven asset classes) as a set, prediction quality does not determine strategy performance: signal quality, portfolio translation, cost survival and temporal stability are separate evaluation stages, and attrition concentrates in the holdout (8 strategies go in, 5 come out with a positive Sharpe). It is a deliberately standard first iteration (no retraining in the holdout, mostly price-only), not a measure of the maximum achievable.

## Frameworks Introduced
- **Four-stage evaluation pipeline**: signal quality → portfolio translation (allocator) → cost survival → temporal stability / risk overlays (+ causal lens).
  - When to use: diagnose where a strategy loses value.
  - How: evaluate each stage separately; when a holdout disappoints, identify the mechanism before re-running with other hyperparameters.
- **Three forms of validation → holdout drift**: (1) prediction-quality drift (IC loses strength or flips sign: review features, labels, horizon, model selection); (2) portfolio-translation drift (the prediction holds up but the portfolio loses: a different allocator or regime-conditioned weighting); (3) regime change (relationships become less informative: regime detection, shorter retraining windows, regime-conditional models).
- **Feature triage ledger (FDR)**: Benjamini-Hochberg at 5% on daily IC; PROCEED = FDR-significant + sign stability across folds + monotonic IC profile by decile; classifies PROCEED/REVISE/STOP.
  - Why it works / failure mode: it is a screen, not a predictor: R² 0.14 between PROCEED rate and holdout Sharpe (0.45 among the seven with <50% survival). US Firms (70.2% PROCEED) +1.77 vs Crypto (71.1%) -0.13.
- **Signal-quality diagnostic bundle**: level (IC with HAC CI), stability (ICIR across folds, positive-fold share) and selection sensitivity (Sharpe by iteration/epoch/checkpoint). No single number is enough.
- **Top-cluster / bootstrap uncertainty**: compare configurations with stationary block bootstrap and PSR; if the top cluster falls within the uncertainty band, the best one is a member of a near-tied set.
- **Ensembling top-3 clusters**: cross-sectional z-score per timestamp of the 3 best runs, averaged, same top-K/cadence/costs. It does not raise peak Sharpe; it reduces dispersion; it helps when the top cluster is narrow relative to the standard error.
- **Allocator isolation test**: fix the best signal and pass it through up to six allocators (EW, inverse vol, score weight, MVO-LW, risk parity, HRP); best-worst spread = allocator headroom.
- **Cost survival**: cost grid of 0-50 bps per leg, breakeven, instrument-appropriate model, capacity (1-5% participation over minimum ADV).
- **Risk overlay sweep**: position-level trailing stops and time exits; MAE-calibrated trailing stop (e.g., 25th percentile of the 20-bar MAE distribution).
- **Dual ML causal screen (Ch15)**: DML effect, HAC p-value, refutation p-value (time-shuffled), and bias % (naive vs DML gap). Thresholds p<0.05 and refutation p>0.05 to pass. Bias as a complementary fragility metric.

## Key Concepts
- **Daily cross-sectional IC / ICIR**: mean Spearman IC per session with HAC CI; ICIR = mean IC / standard error across folds.
- **Binding constraint**: the factor that limits each case (capacity, regime, costs, lack of signal).
- **Cross-stage best configuration**: the one with the highest validation Sharpe across the signal, allocation and risk stages.
- **Cost-feasibility screen**: restrict the universe to names whose measured half-spread leaves room for the edge.
- **Win rate vs payoff asymmetry**: ~81% monthly in US firms vs ~48-52% in FX/crypto/NASDAQ-100.
- **Checkpoint sensitivity**: Sharpe changes depending on the chosen boosting epoch/iteration; it is a per-case-study decision.
- **Path asymmetry**: a return distribution that a stop can intercept.
- **Structural short-vol premium**: a premium that survives execution without depending on model selection.
- **Label engineering**: preprocessing the label usually pays off more than changing the architecture.

## Mental Models
- Use GBM as the default in all settings; invest in deep learning only with evidence of structural nonlinearity (linear vs GBM gap) or temporal structure.
- Use IC as one diagnostic among several: IC-Sharpe correlation per variant 0.34; IC vs holdout Sharpe across cases 0.61.
- Use the allocator as a second-order lever: median spread 0.49 Sharpe, range 0.02-0.69; "no allocator rescues a failing signal".
- Use a slower cadence before tuning the model when costs dominate.
- Use overlays where there is path asymmetry; they are nearly neutral when the window matches the natural holding period.
- Use causal bias to size risk: high bias = tighter risk budgets and more frequent reassessment.

## Anti-patterns
- **Assuming a high validation Sharpe guarantees the holdout**: US Equities +2.03 → -0.49 with a 48% drawdown after 16 folds.
- **Picking the top-1 by validation Sharpe when there is a tied cluster**: NASDAQ-100 linear 5-slot +2.11 → holdout -0.21.
- **Using surviving-feature counts as a predictor of profitability**.
- **Optimizing architecture with a fixed label**: neglecting winsorization (1%/99%), horizon and regression vs classification.
- **Locking in family rankings after one iteration with little evidence** (Crypto: 2 folds; DL not trained on US Firms).
- **bps-on-notional sweep for options or fixed per-share on adjusted prices**: wrong cost model.
- **Universe-mean spreads in mixed universes**: they understate the friction of the illiquid tail (US Firms: 100-500 bps in small caps).
- **Reporting only the Sharpe point estimate**: a 12-month annualized Sharpe has SE ~±1; all holdout intervals are wide.
- **Overlays to rescue a signal with no edge**.

## Code Examples
No code in this range; the chapter is based on tables and case-study notebooks.

## Reference Tables
| Case study | Rebalance | Daily IC | Val Sharpe | Holdout Sharpe |
|---|---|---|---|---|
| US Firm Characteristics | Monthly | +0.037** | +2.75 | +1.77 |
| US Equities Panel | Weekly | +0.036** | +2.03 | -0.49 |
| S&P 500 Eq+Options | Weekly | +0.041** | +2.39 | -0.73 |
| CME Futures | Weekly | +0.032* | +1.36 | +1.11 |
| ETFs | Monthly | +0.085** | +1.36 | +1.00 |
| Crypto Perpetuals | 24-hour | +0.026** | +2.57 | -0.13 |
| S&P 500 Options | Weekly | -0.002 | +0.16 | +0.97 |
| FX Pairs | Daily | +0.012 | +0.05 | +0.19 |
| NASDAQ-100 Microstructure | 15-min | +0.004 | +1.13 | +0.53 |

| Case (pattern) | Reading |
|---|---|
| US Firms, CME, ETFs | Hold up close to validation |
| US Equities, Eq+Opt, Crypto | Decay or flip sign |
| S&P 500 Options | Improves due to structural premium |
| FX Pairs | No resolvable edge |
| NASDAQ-100 | Marginal only after cost-feasibility screen |

| Best allocator per case (fixed signal) | Best | Worst | Spread |
|---|---|---|---|
| US firms | EW +2.75 | score +2.51 | 0.25 |
| Crypto | RP +2.57 | SW +1.88 | 0.69 |
| US Equities | SW +2.02 | HRP +1.49 | 0.53 |
| S&P 500 Eq+Opt | EW +1.25 | MVO +0.73 | 0.52 |
| CME Futures | inverse vol +1.19 | SW +0.63 | 0.56 |
| ETFs | HRP +1.12 | SW +0.68 | 0.44 |
| S&P 500 Options | HRP +0.17 | RP +0.15 | 0.02 |
| FX Pairs | HRP +0.04 | MVO -0.42 | 0.46 |

| Cost breakeven (bps per leg) | Value |
|---|---|
| ETFs, US firms, US equities | 50+ (above the grid ceiling) |
| CME futures, S&P 500 Eq+Opt | ~30 |
| Crypto perps | ~15 (net Sharpe +3.3 at 1 bp → +0.66 at 15 bps) |
| NASDAQ-100 (cost-feasible) | ~10 |
| FX pairs | ~7 (spread 1-8 bps) |
| S&P 500 options | requires a premium-scaled model |

| Family winner | Count |
|---|---|
| Highest IC: GBM | 4 (US firms, US equities, S&P500 options, NASDAQ-100) |
| Highest IC: deep learning | 3 (Crypto, FX, ETFs) |
| Highest post-pipeline Sharpe: GBM | 4; the highest-IC family differs from the highest-Sharpe family in 7 of 9 |

Per-family IC ceiling on the primary label: ~+0.085 (ETFs latent factors); US firms GBM +0.080.

| Dual ML (candidate) | DML effect | p HAC | Refutation p |
|---|---|---|---|
| ETFs momentum | -0.058 | 0.000 | 0.00 |
| S&P 500 Options VRP 21d | -0.123 | 0.000 | 0.03 |
| US firms momentum 12-2 | +0.009 | 0.003 | 1.00 (fails) |
| Rest (6) | ~0 | >0.16 | n/a |

PROCEED funnel: Crypto 32/45 (71.1%), US Firms 40/57 (70.2%), ETFs 17/71 (23.9%), CME 13/69 (18.8%), FX 10/61 (16.4%), S&P Opt 6/51 (11.8%), Eq+Opt 4/49 (8.2%), NASDAQ-100 4/88 (4.5%), US Equities 0/73.

## Worked Example
End-to-end ETF cascade: latent-factor SDF signal with daily IC +0.085 (t=4.89) → HRP allocator (+1.12 in validation) → 4.3% trailing stop overlay (25th percentile of the 20-bar MAE) raises Sharpe by +0.24 and cuts drawdown from -28% to -11% → validation +1.36 → holdout +1.00 (indistinguishable by paired bootstrap), but below the cross-asset equal-weight benchmark (+1.33 during the 2024-2025 mega-cap rally). FF5+momentum attribution: 6.7% annualized alpha (HAC t≈7.7), market and CMA dominate; the relative lag comes from residual market exposure. Cost: breakeven >50 bps. Reading: portfolio-translation/regime relative to the benchmark, not signal loss. Next step: ensemble SDF + NLinear + ridge and broaden features (sentiment, fundamentals, cross-asset).

US Firms contrast: GBM equal-weight 50 names, val +2.75 → holdout +1.77, drawdown -8.6% vs a 10-name alternative at -34%: diversification survives, not signal strength; binding constraint = capacity (long leg in small caps with 100-500 bps spreads; filter to the top three quartiles by market cap).

## Key Takeaways
- Evaluate signal, allocator, costs and stability separately; diagnose the drift mechanism before retraining.
- Treat IC as one diagnostic within a bundle (level, stability, checkpoint sensitivity) and use bootstrap to compare configurations.
- Start with GBM, a well-engineered label (continuous, winsorized, longer horizon) and a top-3 ensemble when the cluster is narrow.
- Choose the allocator according to universe structure: RP/inverse-vol for vol-heterogeneous universes, score-weight if magnitude is informative, EW if the signal is thin; MVO was never the best.
- Match the cost model to the instrument and report breakeven and capacity; move to a slower cadence before tuning the model.
- Use trailing stops where there is path asymmetry; do not expect them to rescue signals with no edge.
- Report causal bias as fragility and scale monitoring accordingly; no case is capital-ready.

## Connects To
Ch6 (workflow and case-study design), Ch7 (HAC, IC), Ch8 (FDR triage), Ch11-14 (model families, IPCA/SDF), Ch15 (dual ML and refutation), Ch16 (backtest), Ch17 (allocators, Sec 17.7), Ch18 (options cost cascade, Sec 18.8), Ch19 (overlays and stops), Ch21+ (RL, agents and live trading).

## Code Reference

Official repository: [20_strategy_synthesis/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/20_strategy_synthesis). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

This chapter has no notebooks; only `README.md` in the folder.

Other notebooks in the folder, not linked from the README: `01_aggregate_synthesis.ipynb`, `02_feature_evaluation.ipynb`, `03_signal_quality.ipynb`, `04_signal_to_strategy.ipynb`, `05_portfolio_allocation.ipynb`, `06_cost_survival.ipynb`, `07_regime_risk.ipynb`, `08_recommendations.ipynb`.

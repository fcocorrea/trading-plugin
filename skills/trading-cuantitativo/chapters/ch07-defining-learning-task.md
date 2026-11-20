# Chapter 7: Defining the Learning Task

## Core Idea
A model is only as good as the learning problem it is given: you must pin down split-aware preprocessing, execution-consistent labels, and a fold-aware evaluation framework (with search accounting and mechanism checks) that separates real signal from timing, overlap, or search artifacts. Every diagnostic is computed per walk-forward fold and aggregated across folds, because pooled statistics hide instability.

## Frameworks Introduced
- **Box 7.1 – Five protocol invariants**: Observability, Train-only fitting, Overlap-aware evaluation, Versioned metadata, Auditable masks. *When*: every run that produces labels/features. *How*: (1) every input is observable at decision time (reporting lags included); (2) parameterized transformations are fit only on the train portion of each split; (3) report overlap diagnostics alongside metrics; (4) feature/label = full versioned recipe, and any change creates a new trial family; (5) store the eligibility mask definition and the effective sample size per fold/group.
- **Split-aware preprocessing**: every step that estimates parameters (means, standard deviations, quantiles, imputations) is refit at each fold boundary using only past data. *When*: whenever a population statistic is involved; stateless steps (filtering invalid rows, fixed unit conversion, calendar flags) do not need it. *How*: when in doubt, treat a step as "fitted"; a single pipeline for training and inference. *Why it works / failure mode*: fitting on the full panel leaks the future and inflates performance (demo: test mean 0.110 train-only vs 0.101 with full-panel fit).
- **Execution-consistent labels**: every price in the label formula must be tradeable under the execution convention. *When*: defining any label. *How*: choose next-open-to-open (signal at the close of t, entry at the open of t+1, exit at the open of t+1+H) or close-to-close; document it and keep it identical in the label and in backtest PnL. *Failure mode*: close-to-close vs next-open differ by 50-100 bps per trade in daily equities (a large share of US returns is overnight).
- **Triple-barrier label** (López de Prado): upper barrier π_t>0, lower ℓ_t>0, and vertical t0+H_max; +1 if the cumulative return hits +π first, -1 if it hits -ℓ, 0 if the vertical barrier expires. *When*: strategies with stop/take-profit where price determines when the outcome resolves. *How*: set widths by volatility scaling (asymmetric multipliers if you have conviction); calibrate with MFE/MAE; if both barriers are crossed in the same bar, fix a rule and log it (count as a loss, or exclude).
- **Trend scanning**: for each observation, evaluates candidate horizons (typically 5-60 bars) and picks the one with the highest t-stat from a linear fit. *When*: the natural horizon is uncertain. *How*: mitigate selection bias with Bonferroni (divide by the number of horizons) or restrict the range; report the distribution of chosen horizons and its stability per fold (there is no single IC).
- **Overlap / uniqueness**: concurrency c(u) = labels alive at bar u; uniqueness w_t = average of 1/c(u) over the label's life; N_eff ≈ Σ w_t. *How*: (1) splits with purge ≥ H_max bars; (2) subsampling every H bars; (3) sample weighting by uniqueness; (4) sequential bootstrap. Combine at least two. *Failure mode*: SPY H=21, ~10 years: 248,436 nominal obs → N_eff ≈ 11,830, nominal CIs ~4.6x too narrow; use N_eff for standard errors.
- **Meta-labeling**: a primary model generates signals; secondary label = whether the signal was profitable after costs; calibrated meta-model → size via sigmoid. *When*: a directional model already exists and you want to filter weak signals (the book's case studies do NOT use it: one model, one horizon, sizing via the Ch 17 allocator).
- **4-screen triage (7.3)**: Correctness → Association → Shape → Feasibility; each must pass to advance; the Proceed/Revise/Stop decision is logged in the research ledger. *Why it works*: the output is an auditable decision, not a backtest. *Failure mode*: using univariate triage as model selection (duplicates, conditional value).
- **Search accounting (7.4)**: define the searched set (label variants, feature variants, conditioning templates) and log its size and generation rules; separate the exploration pass (promote by per-fold stability, not by peak) from the confirmation pass (freeze candidates, re-run with a minimal comparison). *Failure mode*: promoting by peak and "confirming" on the same data.
- **Multiplicity correction**: FWER (Holm-Bonferroni: sort p ascending, reject p(i) only if p(j) ≤ α/(m-j+1) for all j ≤ i) when promoting few candidates from a large pool; FDR (Benjamini-Hochberg: reject p(i) ≤ (i/m)·q) for large screens where some false positives are tolerated. Harvey-Liu-Zhu: threshold t > 3.0 for new factors. With correlated candidates, Rademacher complexity (Ch 16).
- **Mechanism plausibility via DAG (7.5)**: draw a 3-5 node DAG with one hypothesis; derive testable implications; 4 checks (timing placebo, shared-driver, regime heterogeneity, event-time alignment) → PROCEED/REVISE/STOP. It can only reject mechanisms, not confirm them; REVISE is the expected and informative outcome. *Failure mode*: conditioning on the wrong role (collider/mediator).

## Key Concepts
- **Rank IC (Spearman)**: Pearson correlation of cross-sectional ranks between the feature at t and the label anchored at t; a learnability diagnostic, not a measure of tradeable performance.
- **Per-fold ICIR / sign consistency**: ratio of mean IC to its dispersion within the fold; sign consistency = fraction of folds with the sign of the global median (key stability gate).
- **Anchor t0 and horizon H**: t0 = point where the position is established; H counts bars (time or volume/information bars) until resolution.
- **MFE/MAE**: maximum favorable/adverse excursion during the holding period, used to calibrate barriers.
- **Effective sample size N_eff**: sample size corrected for overlap (Σ uniqueness).
- **Collider / Mediator / Confounder**: common effect / intermediate channel / common cause; conditioning on a collider creates spurious correlation, conditioning on a mediator removes the signal, not conditioning on a confounder inflates it.
- **Timing placebo**: shift the feature Δ bars back and watch the IC decay.
- **HAC (Newey-West)**: standard errors robust to serial dependence in overlapping IC series.
- **Grinold fundamental law**: IR ≈ IC × √B (B = effective breadth); only an approximation.
- **Adaptive choices are additional trials (Box 7.4)**: any threshold/lookback/rule chosen after seeing results counts as a new trial.

## Mental Models
- Use **the fold as the unit of summary** when evaluating IC/AUC: overlapping observations and universes with common shocks are not IID; report median, IQR, and worst fold.
- Use **rank IC** when the cross-section is large and the prior is monotonic; **Kendall τ** if n ≲ 30; **MI** only if quantile plots suggest a nonlinear/non-monotonic pattern.
- Use **PR AUC** (alongside ROC AUC) when the positive class is rare: ROC can look high with many false positives at operable thresholds.
- Use **FWER** when each promoted candidate is expensive to model/backtest; **FDR** when you are building a pool that model selection will later prune.
- Use **preserve tails** when the label targets extreme moves or barrier events: clipping erases exactly what needs to be learned.

## Anti-patterns
- **Fitting scalers/imputers on the full panel**: leaks the future.
- **Aggressive clipping of valid returns**: discards real information; distinguish invalid (negative volume, crossed quotes) from rare.
- **Silent imputation**: hides informative missingness; impute robustly + a "was missing" indicator.
- **Global label threshold/global percentiles**: leaks the future distribution; estimate within the train fold.
- **Threshold too narrow (50/50, labels noise) or too wide (rare event, infeasible trading)**.
- **Reporting pooled IC**: 21d momentum on ETFs: pooled IC 0.001 vs mean per-fold IC 0.064, ICIR 0.79, 75% of folds positive.
- **Reporting p-values without the searched-set size** (5 horizons × 3 thresholds × 10 variants = 150 bundles → ~7-8 false positives at 5%).
- **Optimizing a parameter after seeing results without counting it as a trial**.
- **Conditioning without deciding the variable's causal role**; if it cannot be decided, defer to the multivariate toolkit (Ch 15).

## Code Examples
The chapter contains no code (only references to notebooks 02-08). Reconstructed uniqueness formula:
```
c(u)  = #{ labels alive at bar u }
w_t   = (1/|L_t|) * sum_{u in L_t} 1 / c(u)
N_eff = sum_t w_t
```
What it demonstrates: overlap is quantified as concurrency; uniqueness 1.0 = no overlap, ≈1/H = maximum overlap.

## Reference Tables
Triage (Table 7.2):

| Decision | Condition | Action |
|---|---|---|
| Proceed | Correctness passes; primary diagnostic consistent across folds; quantile/confusion matrix supports a plausible mapping | Move on to the modeling chapters |
| Revise | Plausible idea, flawed definition (anchor/horizon, unstable base rates) | Change the definition, re-run, log the delta |
| Stop | Sign unstable across folds; spread does not survive costs; signal only in untradeable liquidity | Archive and document the failure mode |

Plausibility scorecard (Table 7.5):

| Check | Pass | Caution | Stop |
|---|---|---|---|
| Timing placebo | Significant IC (HAC) with decay | IC persists without clear decay | No timely signal; IC rises at distant lags |
| Shared-driver | IC ≈ 0 on control outcome | Small non-zero IC | Significant IC on control |
| Regime heterogeneity | IC stable across partitions | Varies; non-significant sign flip | Significant flip (|t_HAC|>2) AND unconditional IC ≈ 0 |
| Event-time | IC concentrated post-event | IC spread out | IC peaks pre-event |

Missingness: noise (impute the cross-sectional median or leave NaN) / observed coverage (impute + indicator) / informative (preserve with an indicator or a "not reported" category). Encodings: one-hot for low cardinality; ordinal if there is an order (rating); hashing/embeddings for high cardinality.

Label thresholds: fixed absolute (base rate drifts with vol) / volatility-scaled τ_t = k·σ̂ (past-only) / time-series percentile (rolling, e.g. P75 of |r| over 252 bars) or cross-sectional (top quintile = +1, bottom = -1; stable proportions). Multi-asset → cross-sectional; single-asset → vol-scaled or time-series.

## Worked Example
12-1 momentum vs 1-day reversal on ETFs, label = 21d forward return (scan of 10 features × 5/21/63d horizons: no short-lookback feature survives BH-FDR at 5d; 126d+ momentum does contribute at the monthly horizon; with 13 ETF features, 0 survive BH/Holm at α=.05, 10d RVol leads with t_HAC 3.36 but drops out after correction).
- **12-1 Momentum → REVISE**: timing placebo IC 0.053 at lag 0, persists at 0.034 at 252d (partly mechanical, 231d lookback); shared-driver: does not predict Treasuries, IC 0.042 with permutation p<0.001; regime: low-VIX IC 0.086 vs high-VIX 0.017 (stable sign, 5x magnitude) → consistent with "momentum crash"; motivates regime-conditional modeling.
- **1-day Reversal → STOP**: IC 0.002 (t_HAC 0.4), no signal at any lag, permutation p=0.535, low-VIX IC -0.019 vs high-VIX +0.028 with a significant flip and unconditional IC ≈ 0 → aggregation artifact.
Other numbers: HAC inflates the SE of the 21d momentum IC by 2.54x and N_eff 4,989 → 773 (85% loss); a simulated collider (conditioning on fund flows) creates ≈ -0.25 correlation between independent variables; 100 noise factors (50 assets, 252 days): best IC 0.020 (theoretical maximum 0.023), naive p<.05 yields 5 false discoveries per simulation, BH-FDR ≈ 0.

## Key Takeaways
- Fit all preprocessing only on the train portion of each fold; version the full label/feature recipe.
- Define labels with tradeable prices and the same convention in label and backtest; audit anchor, horizon, resolution rule, overlap, and base rate.
- Report IC per fold (median, IQR, worst fold, sign consistency) and use N_eff/HAC for inference.
- Apply feasibility early: gross spread ≥ 2× estimated costs (e.g. a 20 bps spread with 15 bps costs = 5 bps net, close to noise); failure in the low-liquidity quantile = stop.
- Log the searched-set size and correct (Holm/BH); explore for stability, confirm in a frozen pass.
- Before modeling, draw a minimal DAG and run the 4 plausibility checks; expect REVISE.

## Connects To
Ch 6 (walk-forward, purge, research ledger), Ch 3 (volume/information bars), Ch 8-10 (candidate features evaluated with this framework), Ch 9 (stationarity), Ch 11-12 (multivariate selection, tree-based with missingness), Ch 15 (multivariate causality, DML), Ch 16 (Rademacher, Deflated Sharpe, PBO, MinTRL), Ch 17 (allocator), Ch 20 (stops/sizing).

## Code Reference

Official repository: [07_defining_the_learning_task/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/07_defining_the_learning_task). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`03_label_methods.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/07_defining_the_learning_task/03_label_methods.ipynb) + `.py` | 7.2 | This notebook demonstrates all major labeling methods for ML-based trading strategies, with practical examples on real ETF data. |
| [`04_maximum_favorable_adverse_excursion.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/07_defining_the_learning_task/04_maximum_favorable_adverse_excursion.ipynb) + `.py` | 7.2 | This notebook provides empirical justification for triple-barrier parameter choices. |
| [`05_signal_evaluation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/07_defining_the_learning_task/05_signal_evaluation.ipynb) + `.py` | 7.3 | This notebook demonstrates single-factor evaluation using Information Coefficient (IC) analysis, quantile returns, and spread metrics. |
| [`06_ic_inference.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/07_defining_the_learning_task/06_ic_inference.ipynb) + `.py` | 7.3 | This notebook addresses statistical inference for IC: how confident should we be that a signal's IC is not just noise? |
| [`07_multiple_testing.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/07_defining_the_learning_task/07_multiple_testing.ipynb) + `.py` | 7.4 | This notebook addresses the factor zoo problem: when testing many signals, even with proper inference, the "best" will be inflated by selection bias. |
| [`08_causal_sanity_checks.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/07_defining_the_learning_task/08_causal_sanity_checks.ipynb) + `.py` | 7.5 | This notebook implements lightweight falsification tests for feature evaluation. |
| [`10_ml4t_library_ecosystem.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/07_defining_the_learning_task/10_ml4t_library_ecosystem.ipynb) + `.py` | 7.1 | This notebook introduces the ml4t library ecosystem used throughout Chapters 7-12: data loaders (ml4t-data), feature computation (ml4t-engineer), and evaluation tools (m… |

Other notebooks in the folder, not linked from the README: `01_data_quality_diagnostics.ipynb`, `02_preprocessing_pipeline.ipynb`.

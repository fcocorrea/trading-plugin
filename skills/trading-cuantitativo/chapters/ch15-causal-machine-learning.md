# Chapter 15: Causal Machine Learning

## Core Idea
Causal inference in trading is a pipeline: DAG -> admissible adjustment set -> estimand -> ML estimation -> refutation. ML makes nuisance estimation flexible but does not create identification; the goal is to distinguish robust claims from those that collapse under placebo, sensitivity and stability checks, not to "prove" causality.

## Frameworks Introduced
- **Causal pipeline**: DAG specification -> adjustment-set identification -> estimand derivation -> ML-based estimation -> refutation testing.
  - When to use: before any estimator; define treatment T, outcome Y, estimand (ATE, ATT, CATE), target population and timing in advance.
  - How: write the DAG before choosing the estimator (DoWhy makes this operational); controls W pre-treatment only.
  - Why it works / failure mode: without this discipline the analysis degenerates into post-hoc rationalization (another form of selection bias). A sophisticated estimator does not rescue failed identification.
- **Backdoor criterion / adjustment set**: W is admissible if it blocks every non-causal T->Y path that starts with an arrow into T and contains no descendants of T. Estimand: E[Y|do(T=t)] = E_W[E[Y|T=t,W]].
  - How: include true confounders and strong pre-treatment predictors of Y (precision) if they are not colliders, mediators, treatment descendants or selection variables.
  - Failure mode: kitchen-sink regression: conditioning on a mediator changes the estimand, on a collider opens a non-causal path, on a descendant introduces post-treatment bias. Timing discipline is necessary but not sufficient (liquidity, analyst coverage or flows can be colliders/selection).
- **Assumptions for causal interpretation**: consistency, overlap, no interference (SUTVA), conditional exchangeability.
- **Alternative identification designs**: IV (relevance, independence, exclusion; local effect; not implemented, use linearmodels); Difference-in-Differences (parallel trends; e.g. short-selling ban; pre-trend checks; beware staggered timing); Regression Discontinuity (continuity, local effect at the threshold; e.g. index inclusion, ratings); Event-study counterfactual (BSTS).
  - How to choose: backdoor if confounders are observed; IV with an external source of variation; DiD with plausible parallel trends; RD with a hard-to-manipulate threshold; BSTS with stable pre-event relationships and uncontaminated controls.
- **Validation & refutation**: (1) placebo/negative-control (fake treatment, date or outcome should give ~0); (2) sensitivity to omitted confounding (synthetic confounders); (3) stability (subsample, specification, outcome triangulation). A claim that passes is not proven; one that fails is downgraded to a research hypothesis.
- **Double/Debiased ML (DML; Chernozhukov 2018)**: (1) predict Y from X: g(X)=E[Y|X], residual Ỹ = Y - g(X); (2) predict T from X: m(X)=E[T|X], residual T̃ = T - m(X); (3) regress Ỹ on T̃, slope = causal effect (partially linear case; ML analog of Frisch-Waugh-Lovell).
  - When to use: continuous treatment with high-dimensional, nonlinear confounders under a backdoor design.
  - How: orthogonal score (nuisance errors have a second-order effect) + cross-fitting that respects temporal order (expanding/rolling windows, not random folds, with a time gap because of overlapping forward returns); compare at least 2 classes of nuisance models (GBM, random forest, linear); controls W vs effect modifiers X (separate arguments in EconML) change the estimand if mixed up.
  - Failure mode: weak/poorly tuned nuisance produces biased estimates; large changes across learners = unstable conclusion (ETF momentum: -0.055 linear, -0.060 shallow GBM, -0.088 deep GBM, spread ~1.6x).
  - DML checklist: T with fixed timing and X strictly pre-treatment; estimand declared; unconfoundedness argued; temporal cross-fitting; sensitivity to omitted variables.
- **CATE / heterogeneous effects**: EconML (metalearners, CausalForestDML); subgroup ATE (re-estimating ATE per stratum) is not CATE (that requires a heterogeneity estimator).
- **Post-double-selection LASSO (BCH / Feng-Giglio-Xiu 2020)**: tests whether candidate factors retain pricing power after controlling for the existing zoo.
  - Result (nb 11): of 4 managed-portfolio factors only MeanRev passes the naive threshold; it collapses to insignificance with the first 10 PCs as controls.
- **BSTS (Brodersen 2015; tfcausalimpact)**: (1) learn the target-control relationship in the pre-period; (2) project the counterfactual trajectory with the post-event controls; (3) observed - counterfactual difference = causal impact (Bayesian credible intervals for the cumulative effect).
  - Assumptions: stable relationships and no spillover (the event does not affect the controls). Checklist: ATT estimand on the treated series; control selection rule and spillover risk; pre-period stability; placebo dates; robustness to alternative control sets.
  - Failure mode: spillover (FOMC moves IEF and SPY; SPY as a control corrupts the counterfactual). Mitigate with controls from weakly linked asset classes (e.g. agricultural commodities for banking regulation); run BSTS using each control as the target and exclude those that show a significant effect.
- **Causal discovery (hypotheses, not causal truth)**: Granger (pairwise predictive screening); PCMCI (Tigramite; PC phase = superset of parents, MCI phase = momentary conditional independence tests; autocorrelation and indirect links); VAR-LiNGAM (VAR + LiNGAM; non-Gaussian errors, contemporaneously acyclic, no hidden common causes); NOTEARS (cross-sectional DAG as continuous optimization with acyclicity constraint h(W) = tr(e^{W∘W}) - d = 0).
  - Assumptions: causal sufficiency, faithfulness, stationarity.
  - How: edges confirmed by several methods carry more weight; look at effect sizes, not just p-values; block bootstrap (50% robustness threshold).
- **Two-gates evaluation**: HAC significance |t| > 1.96 and block-permutation refutation p_ref < 0.05; neither is sufficient alone.
- **Regime-conditional position sizing (CATE)**: w_{i,t} = w_base_{i,t} · s(regime), with s = CATE/std (signal-to-noise) clipped to [0.5, 1.5] (shrinks toward neutral exposure if the estimate is imprecise); w_base gross-normalized Σ|w|=1.

## Key Concepts
- **ATE / ATT / CATE**: average effect in the population / on the treated (typical in event studies) / conditional on covariates or regime.
- **Estimand**: the exact causal quantity; fixes population, timing, outcome horizon and comparison.
- **Confounder vs effect modifier**: the former is adjusted for to remove bias; the latter defines how the effect varies.
- **Collider / mediator / treatment descendant**: types of "poor controls".
- **Orthogonal score**: estimating equation that is (first-order) insensitive to nuisance errors.
- **Cross-fitting**: nuisance trained on one fold and evaluated on another.
- **Block-permutation refutation**: shuffles the treatment in blocks (7 days) to respect autocorrelation; very short blocks (1 day) inflate the statistic.
- **OOS drift / placebo ratio**: change in the effect across splits (lower = more stable) / proportion of placebos that match the original effect.
- **Negative control**: an outcome that should not respond to the treatment.
- **Placebo date**: dates without an event where the counterfactual model should find no effect.
- **Posterior probability vs sample frequency**: the fraction of rows with a positive effect is not a posterior probability; the reliable diagnostic is a cumulative-effect CI away from zero + placebo with a low false-positive rate.

## Mental Models
- Use DML when there is a defensible DAG and adjustment set and a continuous treatment; BSTS when the treatment is a discrete event and uncontaminated controls exist; discovery only to generate hypotheses when the structure is unknown.
- Use an outcome close to the mechanism (premium reversion) rather than a broad outcome (forward returns): a simple estimator on a good outcome gives a more credible claim than a sophisticated one on a weakly linked outcome.
- Use refutation as falsification: a result that survives is "credible under assumptions", not proven.
- Use causal analysis to understand/size effects and prediction (Ch11-14) to rank; a causal effect is not a tradable signal by itself.

## Anti-patterns
- **Kitchen-sink regression**: controlling for mediators/colliders/descendants introduces bias.
- **Controls measured post-event (e.g. variables that react to the FOMC)** or built with future returns: violates timing discipline.
- **Random folds in DML with overlapping returns**: leakage; use temporal windows with a gap.
- **A single nuisance specification**: changes >1.5x across learners signal instability.
- **Evaluating a factor without orthogonalizing**: median |naive - DML| = 59.7%, >50% in 5 of 9 case studies.
- **Global controls for macro events in BSTS**: contaminated (VEA, EFA, DBC responded to the FOMC).
- **Reading significance from credible intervals without placebo**: 7 of 12 placebo dates (58%) came out "significant".
- **Interpreting discovery edges as deployable causality**: 1 to 42 edges depending on the method on the same data.
- **Taking the ADIA Lab Challenge as evidence that discovery is solved**: it is amortized inference on a synthetic simulator with labeled graphs.
- **Letting causal sizing rescue a signal that has flipped**: no sizing fixes a treatment whose sign has changed.

## Code Examples
The extracted text contains no code; only references to notebooks (01_library_overview ... 11_factor_zoo_validation). Described CausalForestDML configuration (minimal reconstruction):

```
CausalForestDML(
    X = [market_volatility, regime_indicators],   # effect modifiers
    W = [volatility_horizon_panel, yield_curve_slope]  # controls
)
```
What it demonstrates: in EconML, effect modifiers go in X and controls in W (together they form the adjustment set); mixing them up changes the estimand.

## Reference Tables
Table 15.1 (method selection):

| Known DAG | Treatment | Method | Library |
|---|---|---|---|
| Yes | Binary/Cont. | Backdoor adj. | DoWhy |
| Yes | Continuous | DML | EconML |
| Yes | Binary (event) | BSTS | tfp-causalimpact |
| No | Discovery | PCMCI | Tigramite |
| No | Discovery | NOTEARS | causal-learn |
| No | Discovery | VAR-LiNGAM | causal-learn |

Crypto funding-rate outcome (treatment: funding z-score > 2):

| Outcome | OOS Drift | Placebo Ratio |
|---|---|---|
| Forward returns | 85% | 12.7% |
| Premium reversion | 51% | 5.7% |

Table 15.4 (7 daily ETFs 2015-2024; SPY, QQQ, IWM, TLT, GLD, EEM, XLF): Granger 26 FDR-significant edges out of 42 pairs; PCMCI 42 lagged links out of 147 possible (7x7x3 lags; median |effect| ~0.06; XLF 1-day on SPY, QQQ, EEM with |partial corr| >0.10); NOTEARS 5 contemporaneous edges (all bootstrap-stable; SPY->QQQ 100%, IWM->SPY and IWM->XLF >80%, XLF->SPY 69%); VAR-LiNGAM 1 edge (causal-learn, XLF->SPY) vs 11 (from scratch). On the 4-asset panel (SPY, IEF, GLD, VIX) PCMCI finds no significant links (most stable edge IEF->VIX in 37% of resamples, below the 50% threshold).

Table 15.5: Granger = screening, simple, pairwise/predictive; PCMCI = multivariate lagged, controls for autocorrelation and indirect links, conservative; VAR-LiNGAM = uses non-Gaussianity, linear and no hidden common causes; NOTEARS = contemporaneous, continuous optimization, originally linear.

Table 15.6 (DML effect primary label; t_HAC; signed bias; p_ref; gates): ETFs 21d momentum -0.058, t -9.84, bias +33.2%, p_ref 0.00, both; US Firms 1m +0.007, t +1.98 (p 0.048), +42.8%, 0.03, both; SP500 Options VRP HTM -0.123, t -4.69, +49.8%, 0.03, both; US Equities 1d momentum -0.001, t -2.16, +68.8%, 0.00, both; NASDAQ-100 15m microstructure ~0.00, t +2.52, -9.9%, 0.11, HAC only; Crypto 8h premium z -0.001, t -1.03, +68.3%, 0.00, refutation only; SP500 Eq+Opt 5d IV-RV -0.002, t -1.50, +86.5%, 0.00, refutation only; FX 1d +0.0004, t +0.42, -213.5%, 0.30, neither; CME 5d carry +0.0003, t +0.75, -59.7%, 0.19, neither.

## Worked Example
DML on ETF momentum (nb 03): T = momentum score (risk-adjusted 6m return); Y = 21d forward return; X = realized volatility, rate-curve regime (controls, not effect modifiers).
- Naive OLS: -0.0385 with 95% HAC CI [-0.0484, -0.0285] (HAC bandwidth = 21 days because of overlapping returns). EconML DML: -0.0537, CI [-0.0613, -0.0462], excludes zero; manual DML with walk-forward cross-fitting: -0.0494. The naive estimate underestimates the magnitude by ~28%.
- Crypto funding (nb 04): same sign naive vs DML but the confounders suppress ~90% of the magnitude; block-permutation with 7-day blocks passes, with 1-day blocks it inflates; neither naive (t≈...44) nor DML (t≈...23) is significant (exact values illegible in the extraction): DML does not guarantee robustness with a weak signal. High vs low vol interaction not significant.
- Regime sizing (nb 05): train 2007-01 to 2022-11, test 2022-12 to 2025-12 (everything fixed before the test). CausalForestDML: CATE +0.019 (low vol), +0.043 (medium), -0.066 (high); ATE -0.0097 [-0.0138, -0.0055] vs OLS -0.0037 (61% bias). Scale factors: 0.50 high vol, 1.09 low, 1.28 medium. Costs 10 bps (sweep 5-20). Train/holdout Sharpe: Naive +0.12/-0.18, Causal +0.23/-0.05, Heuristic (reduce in high vol) +0.20/-0.11. Momentum IC falls from +0.023 to -0.053 (the base signal flips post-2022). Conclusion: causal analysis reduces OOS degradation when the train heterogeneity is informative, but does not guarantee alpha.
- BSTS FOMC on IEF: controls VEA, EFA, DBC; pre 60 days, post 20; 4 announcements with cumulative effects (log-indices): Mar-2023 +0.59 (25bp hike amid banking stress), Jul-2023 -0.28, Nov-2023 -0.09, Mar-2024 -0.29; all with CI away from zero. Placebo: 7 of 12 dates (58%) significant, and all 3 controls respond in Mar-2023, Nov-2023 and Mar-2024 (DBC only in Jul-2023). Decision: the design is rejected; use duration-matched non-US sovereign bonds or a local-level model without cross-asset controls.

## Key Takeaways
- Specify treatment, outcome, estimand, population and timing before fitting any model; write the DAG first.
- Do not control for mediators/colliders/descendants; make sure every control is pre-treatment.
- In DML use temporal cross-fitting, compare >=2 classes of learners and report naive vs DML.
- Require two gates (|t_HAC|>1.96 and p_ref<0.05) and refutation (placebo, sensitivity, subsample) before elevating a claim to causal; results that fail remain hypotheses.
- In BSTS validate spillover (control-as-target) and placebo dates (low false-positive rate) before believing any impact.
- Treat discovery edges as hypotheses and prioritize those confirmed by several methods.
- Choose the method by fit of assumptions, not by sophistication; the horizon and the label change the causal estimate as much as the IC.

## Connects To
Ch7 (DAGs, confounders/mediators/colliders, placebo/plausibility checks, multiple testing, partial-IC), Ch11-14 (predictive signals, factor zoo, PCs as controls), Ch12 (predictive crypto case study), Ch14 (factor returns), Ch16 (consumes predictions, not causal coefficients), Ch17 (weighting factors by causal confidence), Ch19 (effect uncertainty in sizing), Ch23 (GNNs).

## Code Reference

Official repository: [15_causal_estimation/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/15_causal_estimation). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_library_overview.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/01_library_overview.ipynb) + `.py` | 15.1 | Decision guide for choosing among EconML, DoWhy, CausalML, Tigramite, and causal-learn by question type, treatment type, and data structure. |
| [`02_dowhy_causal_graph.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/02_dowhy_causal_graph.ipynb) + `.py` | 15.3, 15.4 | Walks the full DoWhy validation workflow on the crypto funding-rate setting, comparing the same extreme-premium treatment against two outcomes (forward returns vs. |
| [`03_econml_dml.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/03_econml_dml.ipynb) + `.py` | 15.4 | Estimates the causal effect of skip-recent 6-1 momentum on 21-day forward ETF returns. |
| [`04_dml_crypto_regime.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/04_dml_crypto_regime.ipynb) + `.py` | 15.4 | Applies DML to the crypto perpetual-funding premium z-score on 8-hour forward returns, with subgroup effects by volatility regime and a single-model regime-interaction t… |
| [`05_momentum_causal_trading.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/05_momentum_causal_trading.ipynb) + `.py` | 15.4 | Uses CausalForestDML to estimate regime-conditional CATEs and tests whether causal position sizing improves out-of-sample performance against naive and heuristic baselin… |
| [`06_fed_announcement_bsts.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/06_fed_announcement_bsts.ipynb) + `.py` | 15.5 | A BSTS event study of four FOMC announcements on the IEF Treasury ETF. |
| [`07_tigramite_time_series.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/07_tigramite_time_series.ipynb) + `.py` | 15.6 | Demonstrates PCMCI on a four-asset macro panel (GLD, IEF, SPY, VIX), testing lagged links with ParCorr independence and confirming with block-bootstrap stability, contra… |
| [`08_neural_causal_discovery.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/08_neural_causal_discovery.ipynb) + `.py` | 15.6 | Runs NOTEARS, VAR-LiNGAM (from scratch and via causal-learn), PCMCI, and Granger causality on a shared seven-ETF panel, enabling direct comparison on identical data. |
| [`09_adia_causal_benchmark.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/09_adia_causal_benchmark.ipynb) + `.py` | 15.6 | Reproduces the ADIA Lab Causal Discovery Challenge (Olivetti et al., 2026) in a simplified setting: a supervised classifier learns variable roles from engineered causal… |
| [`10_case_study_insights.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/10_case_study_insights.ipynb) + `.py` | 15.7 | Queries the per-case-study causal registries to build a cross-dataset forest of DML effects with HAC confidence intervals, a confounding-bias comparison against naive OL… |
| [`11_factor_zoo_validation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/15_causal_estimation/11_factor_zoo_validation.ipynb) + `.py` | 15.4 | Applies post-double-selection LASSO (Belloni-Chernozhukov-Hansen 2014; Feng, Giglio, and Xiu 2020) to test which factors retain marginal pricing power for a held-out bro… |

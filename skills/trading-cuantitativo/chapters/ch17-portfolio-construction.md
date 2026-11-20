# Chapter 17: Portfolio Construction

## Core Idea
Portfolio construction translates forecasts into positions and is itself a model (inputs, assumptions, hyperparameters, failure modes): a good allocator preserves weak but diversified signals; a naive one concentrates risk and destroys a strong signal. Simple heuristics are a very high bar; differences between allocators tend to be narrower than those between predictive models.

## Frameworks Introduced
- **Allocator term sheet**: a compact document that fixes the objective, inputs (expected returns, covariance, horizons), constraints with numerical values (hard vs soft), rebalancing protocol (frequency, drift thresholds, costs) and evaluation plan (metrics, subperiods, regimes).
  - When to use: before implementing any allocator, after model selection.
  - How: (1) explicit objective (mean-variance, max Sharpe, min-var, risk parity); (2) inputs and their horizon; (3) constraints with values and type; (4) rebalancing and costs; (5) evaluation fixed in advance.
  - Why it works / failure mode: prevents the allocation from becoming an unrecorded search layer; the failure is tuning shrinkage, risk aversion or the turnover penalty while looking at the final test ("fitting to the answer sheet").
- **Fundamental Law of Active Management (FLAM)**: IR ≈ IC × √BR (IC = information coefficient, BR = breadth of independent bets).
  - When to use: order-of-magnitude calibration between signal quality and portfolio performance.
  - How: estimate IC; count effective (not nominal) breadth; IR = IC×√BR; doubling IC is equivalent to quadrupling breadth.
  - Why it works / failure mode: assumes unbiased forecasts, a correct covariance, independent bets and zero frictions; correlation between bets, a poor risk model and costs erode it.
- **Baselines**: equal weight (1/N), inverse volatility (w_i ∝ 1/σ_i), score-weighted (w_i ∝ s_i/Σ|s_j|), conformal position sizing (inverse-width weights with Mondrian split-conformal, 80% coverage), risk parity / ERC, volatility targeting (overlay: scale by σ_target/σ_base).
  - When to use: always as a hurdle; any complex allocator must beat them.
  - How: see Worked Example and tables; for dollar-neutral, normalize the long/short legs separately.
  - Why it works / failure mode: they ask little of the data; score-weighting fails if extreme scores concentrate exposure in correlated assets (mitigate with position limits or vol targeting); inverse vol does not equalize risk contributions when there are correlations.
- **Kelly criterion**: f* = edge/odds (binary); continuous f* ≈ μ/σ²; multi-asset w* = Σ⁻¹μ (same direction as the tangency portfolio, but it also sets the leverage).
  - When to use: as an upper bound for sizing; fractional Kelly (1/2, 1/4) in practice.
  - How: estimate μ, Σ; scale by the fraction; raw Kelly in the ETF case implied typical leverage of 30×+ before shrinkage.
  - Why it works / failure mode: maximizes log growth; overestimating μ produces excessive leverage and severe drawdowns.
- **Mean-variance optimization (MVO) + robust fixes**: max w'μ − (λ/2) w'Σw s.t. 1'w = 1; fixes: Ledoit-Wolf shrinkage (Σ̂ = (1−α)S + αF), factor-model covariance (BΩ_f B' + Ω_ε), position caps, long-only, turnover penalties.
  - When to use: modest universe, tractable covariance, signal strong enough to survive costs.
  - How: shrink the covariance, constrain weights, penalize turnover; treat constraints as part of the estimator (robustness).
  - Why it works / failure mode: Markowitz curse: inverting Σ amplifies noise; the optimizer concentrates in assets with overestimated alpha (error-maximization). The ranking of alpha matters more than its scale calibration.
- **Hierarchical Risk Parity (HRP)** (Lopez de Prado): (1) distance from correlation (typically d = √(½(1−ρ))); (2) hierarchical clustering + quasi-diagonalization following the dendrogram; (3) recursive bisection: split each cluster in two and allocate capital inversely proportional to the cluster variance.
  - When to use: when the priority is diversification stability, large and heterogeneous universes, weak signals.
  - Why it works / failure mode: no matrix inversion, hierarchical regularization, stable structure; but it ignores expected returns, is long-only by construction and is sensitive to the clustering recipe.
- **Nested Clustered Optimization (NCO)**: optimizes within clusters and then across cluster portfolios; allows views, constraints and alternative objectives; more flexible but reintroduces optimizer assumptions. **Schur Complementary Allocation** (Cotton, 2024): a continuum between HRP and min-variance.
- **Shrinkage hedging**: reduce the hedge ratio when the beta estimation error is large relative to the aggregate exposure; a full hedge with a noisy beta can increase variance. **Factor-mimicking portfolio**: minimum-residual-risk portfolio that isolates a factor (uses loadings b and idiosyncratic covariance Ω_ε).
- **End-to-end deep portfolio learning**: 5-stage pipeline: (1) inputs (per-asset lookback window: multi-horizon vol-scaled returns, EWMA vol with 63d half-life, indicators); (2) sequence encoder with shared weights (channel-independence) and ticker embeddings; (3) linear + tanh signal head → s ∈ [-1,1]; (4) vol-targeting position layer (σ* typically 5-15%, rule fixed at 10%, ablations at 15%); (5) aggregation and loss = −annualized Sharpe (A=252, ε stabilizer) with turnover cost.
  - Refinements: pooled Sharpe (not per asset); cost-aware loss; for minibatches use an exact two-pass computation (Sharpe is not separable).
  - DeePM: Directed Delay (causal attention with asynchronous closes), V-VSN temporal backbone (VSN-LSTM-Attention), Macro Graph Prior with GAT, SoftMin loss over sub-windows (L = −SR_pool − λ·SoftMin({SR_b}); equivalent to DRO with a KL ball).
  - Failure mode: training Sharpe >12 with validation collapsing within a few dozen epochs; diagnostic opacity.

## Key Concepts
- **Gross vs net exposure**: Σ|w_i| measures total risk; Σw_i measures directional bias.
- **Effective breadth**: number of truly independent bets (e.g. 20 ETFs with ~5 common factors → 5×12=60, not 240).
- **Tracking error / IR / active share**: active risk and return vs the benchmark, and difference in holdings in weight space.
- **HHI / effective number of bets**: capital concentration (HHI) and its reciprocal.
- **Risk contribution**: each position's contribution to total volatility; equal capital does not imply equal risk.
- **Hard vs soft constraint**: literal limit (5% cap) vs progressive penalty (preference).
- **Markowitz curse**: unregularized MVO yields concentrated portfolios that are unstable out-of-sample.
- **Fractional Kelly**: scaling Kelly by 1/2 or 1/4 to lower leverage and sensitivity to error.
- **Pooled Sharpe / SoftMin loss**: Sharpe over all concatenated returns / penalty on the worst sub-windows.
- **Allocator shopping**: overfitting by testing many allocators against the same test set.

## Mental Models
- Use FLAM when you need to estimate whether a small IC (e.g. 0.03-0.05) can sustain a portfolio: effective breadth, not nominal.
- Use equal weight or inverse volatility when signals are weak or noisy: they ignore μ and resist estimation error.
- Use HRP when the covariance is the bottleneck and there are no reliable return views; use NCO when you need views + stability.
- Use fractional Kelly as a sizing ceiling, never literal full Kelly, because μ is estimated with error.
- Use an end-to-end allocator when the objective does not factor well into forecast + optimization (regime robustness, cost-aware sizing), and always against heuristics on the same test.

## Anti-patterns
- **Unregularized MVO (max-Sharpe without shrinkage/caps)**: collapses to 1-2 assets (see ETF: XLK 78.9%/GLD 21.1%; min-var to SHY ~98%).
- **Counting nominal breadth**: correlated assets are not independent bets; overstates IR.
- **Hedging to zero beta with a noisy beta**: trades against noise and can raise variance.
- **Tuning allocator hyperparameters on the final test**: overfits the allocator even if the model is fixed.
- **Comparing allocators with different inputs**: confounds sizing with forecast quality; keep forecasts, costs, constraints and windows identical.
- **Mixing horizons without a conscious decision** (weekly signal + 10-year covariance).
- **Equal-notional long-short without considering leg volatility** (French, 2024: the short leg tends to be more volatile; scaling by volatility brings market exposure closer to zero).
- **Comparing neural allocators without checking seeds, turnover and costs**: the ranking can reorder with the seed budget.

## Code Examples
There is no source code in the range (only references to notebooks: 02_mean_variance_optimization, 03_robust_optimization with Riskfolio-Lib, 06_hierarchical_risk_parity, 07_conformal_position_sizing, 08_library_comparison with skfolio WalkForward/PyPortfolioOpt, 09_allocator_comparison, 11-13 deep learning).

## Reference Tables
| Allocator (top-5 ETF GBM, 252d cov, 2010-2023) | Ann. Return | Sharpe | Max DD | Calmar |
|---|---|---|---|---|
| Min Variance (LW) | 11.83% | 0.8437 | -28% | 0.4215 |
| HRP | 11.41% | 0.8293 | -28% | 0.4063 |
| Inverse Volatility | 10.92% | 0.8074 | -28% | 0.3893 |
| Equal Weight | 10.38% | 0.7763 | -28% | 0.3698 |

| Allocator (ETF Ridge, 2018-2023) | Ann. Ret | Ann. Vol | Sharpe | Max DD | Avg Turnover |
|---|---|---|---|---|---|
| MVO (Ledoit-Wolf) | 6.7% | 18.0% | 0.450 | -21.7% | 10.5% |
| Inverse Volatility | 6.8% | 19.2% | 0.439 | -23.3% | 8.3% |
| HRP | 6.1% | 19.3% | 0.404 | -25.6% | 10.1% |
| Equal Weight | 4.6% | 19.5% | 0.327 | -19.7% | 6.8% |

| Conformal sizing (Sharpe val / holdout, base → conformal) | Result |
|---|---|
| etfs | holdout +1.00 → +1.02 |
| sp500 options analytics | holdout -0.73 → +0.08 (the only material move) |
| cme_futures | +1.11 → +0.70 |
| us_equities_panel | -0.49 → -0.73 |
| fx_pairs | flat signal: val +0.05 → -0.11 |

| End-to-end (ETF test) | Sharpe | Max DD |
|---|---|---|
| Full DeePM | 0.98 | -7.5% |
| DeePM without SoftMin | 0.74 | -8.9% |
| Equal weight / Inv vol | 0.69 / 0.69 | -18.4% / -14.7% |
| VLSTM (gross → 5bps → 10bps → 20bps) | 0.66 → 0.55 → 0.44 → 0.22 | -17.6% |
| Softmax long-only LSTM (Zhang et al.) | 0.48 | -10.9% |

## Worked Example
FLAM in the ETF case: daily IC ≈ 0.054, universe of ~100 ETFs, monthly rebalancing to 20 positions. Nominal breadth = 20×12 = 240 → IR ≈ 0.054×√240 ≈ 0.84 (before costs). If the 20 positions depend on ~5 common factors, effective breadth = 5×12 = 60 → IR ≈ 0.054×√60 ≈ 0.42. The nominal vs effective gap explains why realized performance falls short of the naive projection. Note: an IC of 0.03 explains only ~0.09% of the cross-sectional variance (IC² = R²); its value comes from repetition, stability and low cost.

Second example: vol targeting with a 15% annual base and a 10% target → scale positions by 0.67. And in the controlled ETF comparison (ridge, 2018-2023) the best-worst Sharpe gap is only 0.123 and between the top two 0.011; the cost-aware simulation of inverse-volatility went from +46.6% (weight-based return) to +1.5% after commission, slippage and fill timing.

## Key Takeaways
- Write the allocator term sheet before optimizing; treat allocator hyperparameters as model hyperparameters (out-of-sample validation).
- Evaluate allocators along multiple dimensions (IR, TE, active share, HHI, risk contribution, turnover, leverage stability), not just Sharpe.
- Start with equal weight, inverse volatility and vol targeting; only add complexity if it beats them after costs.
- If you use MVO: Ledoit-Wolf/factor covariance, position caps, turnover penalty; Kelly only fractional.
- Compare allocators with identical forecasts, costs, constraints and covariance; improve signal and execution before the allocator's geometry.
- For neural allocators: pooled Sharpe with costs, evaluate seeds, the cost breakeven curve, and compare them against heuristics.

## Connects To
Ch7 (IC), Ch11-14 (forecasts, conformal prediction, sequence architectures, eigenvector stability), Ch16 (backtest, regime slicing), Ch18 (costs as regularizer and turnover), Ch19 (tail risk, stress constraints, covariance evaluated via the portfolio), Ch20 (cross-case synthesis of allocators, Sec 20.5-20.7).

## Code Reference

Official repository: [17_portfolio_construction/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/17_portfolio_construction). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_portfolio_metrics.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/01_portfolio_metrics.ipynb) + `.py` | 17.3 | Computes the risk-return, drawdown and benchmark-relative statistics of a registered ETF allocation backtest with `ml4t-diagnostic`, which replaces the unmaintained pyfo… |
| [`02_mean_variance_optimization.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/02_mean_variance_optimization.ipynb) + `.py` | 17.5 | Builds the efficient frontier and the maximum-Sharpe and minimum-variance solutions from scratch, then freezes weights estimated on one window and scores them on a later… |
| [`03_robust_optimization.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/03_robust_optimization.ipynb) + `.py` | 17.5 | Runs the two families of response to the estimation problem: Ledoit-Wolf shrinkage improves the estimate, while risk parity, minimum variance and minimum conditional dra… |
| [`04_kelly_criterion.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/04_kelly_criterion.ipynb) + `.py` | 17.4 | Derives the Kelly criterion from a binary wager, extends it to continuous returns and to a multi-asset portfolio, and shows what the resulting leverage would take to hol… |
| [`05_factor_allocation_evidence.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/05_factor_allocation_evidence.ipynb) + `.py` | 17.4 | Works through the published evidence on which return sources diversify each other, using close to a century of AQR and Fama-French factor returns. |
| [`06_hierarchical_risk_parity.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/06_hierarchical_risk_parity.ipynb) + `.py` | 17.6 | Builds HRP from its three steps (clustering, quasi-diagonalization, recursive bisection), traces where its weight concentration comes from, and runs a walk-forward compa… |
| [`07_conformal_position_sizing.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/07_conformal_position_sizing.ipynb) + `.py` | 17.4 | Sizes positions from the width of a per-entity conformal prediction interval rather than from a covariance matrix, and compares that rule against equal weight and score… |
| [`08_library_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/08_library_comparison.ipynb) + `.py` | 17.7 | Fits comparable allocations through PyPortfolioOpt, Riskfolio-Lib and skfolio from one set of estimated moments, checks that matched objectives agree to solver tolerance… |
| [`09_allocator_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/09_allocator_comparison.ipynb) + `.py` | 17.7 | _Runtime ~6 min_. |
| [`11_dl_portfolio_allocation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/11_dl_portfolio_allocation.ipynb) + `.py` | 17.8 | Trains an LSTM whose softmax head emits long-only weights directly, against a differentiable Sharpe-ratio loss, and compares it with equal weight and inverse volatility. |
| [`12_vlstm_portfolio.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/12_vlstm_portfolio.ipynb) + `.py` | 17.8 | Puts a TFT-style variable-selection network in front of the LSTM encoder, emits a scalar signal that a volatility-targeting layer converts into long-short positions, and… |
| [`13_deepm_regime_robust.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/17_portfolio_construction/13_deepm_regime_robust.ipynb) + `.py` | 17.8 | Implements the DeePM framework (Wood, Roberts & Zohren 2026), whose SoftMin objective penalizes the worst rolling window rather than the average, and ablates that penalt… |

# Chapter 16: Strategy Simulation

## Core Idea
The backtest is an exercise in falsification, not verification: it translates ML signals into an explicit trading protocol (timing, execution, costs, sizing, constraints, benchmark) and subjects it to a non-ML baseline, costs, regimes and search-aware inference (DSR, Reality Check) before believing a selected Sharpe.

## Frameworks Introduced
- **Falsification mindset**: ask what would refute the result; three questions: (1) genuine signal or an artifact of leakage/data/selection?; (2) feasible execution in terms of timing, liquidity and impact?; (3) stable across regimes and small specification variations?
  - When to use: every backtest; default posture of disbelief (base rate: few strategies have a real edge, false positives can match or exceed true ones).
  - How: pre-specify the claim and the failure conditions; record negatives; skepticism proportional to the breadth of the search.
- **Three simulation frameworks (Joubert 2024)**: walk-forward (temporal plausibility), resampling/block-bootstrap (metric uncertainty), Monte Carlo (stress, depends on the generating model). They are complementary.
- **Backtest protocol (6 components)**: signal timing; rebalance and holding; position sizing; fill assumptions; constraints; costs. Record them before interpreting results.
  - Signal timing: close-to-next-open (conservative and realistic) vs close-to-close (only if the signal is computed before the close). Same-bar execution merges past and future and is lookahead. Return of a 1-day hold = entry at the t+1 open and exit at the t+2 open.
  - Score-to-signal: fixed threshold (simple, does not adapt); rolling percentile (e.g. >P90 of the 63-day window; adapts, more turnover); cross-sectional percentile (top N%, always invested, ignores absolute strength).
  - Sizing to isolate signal quality: equal weight (1/N) or inverse volatility. Futures: size = contracts x notional (price x multiplier); omitting the multiplier distorts PnL by orders of magnitude.
  - Fills: market at open/close (optimistic for large positions), limit (requires a fill model), VWAP/TWAP.
  - Constraints: leverage, shorting (borrow, availability), minimums, sector/country limits, cash buffer, corporate actions (adjusted prices and delisting returns). "A constraint checked after computing returns was not simulated."
  - Costs: commissions, bid-ask spread, market impact, slippage.
- **Cost sensitivity**: (1) conservative grid of frictions; (2) run the strategy and find the break-even region where net Sharpe/return turns negative; (3) compare with realistic costs. If it only works in the optimistic corner, the signal did not survive. When in doubt, use conservative estimates.
- **Vectorized vs event-driven**: vectorized = computable path of signals, target exposures, rebalance dates and cost formulas without changing meaning (fast, scales in sweeps); event-driven = loop (ingest event, update state, process open orders, apply costs, call strategy, advance clock) when the action depends on simulated state (fills, cash, margin, rejected orders, inventory, stops, drawdown, pending orders).
  - Why: credibility comes from aligning three objects: economic claim, trading protocol and the simulator's state/timing semantics.
  - Vectorized failure mode: semantic drift (same-bar execution, implicit trading on missing bars, forward-fill that creates artificial tradability, weights that ignore cash, costs without checking financing). Event-driven failure mode: misspecified defaults (execution order, cash release, fractional shares, short financing, missing data).
- **Auditable non-ML baseline**: risk-adjusted momentum + yield-curve regime filter (see Worked Example); serves as a yardstick, an infrastructure sanity check and a protocol template. If the baseline far exceeds expectations, look for bugs (lookahead, survivorship, omitted costs); if it falls far short, check the data and the implementation.
- **Tear sheet / metric set**: principles of completeness, parsimony and comparability; returns (cumulative, CAGR, equity curve, hit rate), risk (annualized volatility, max drawdown, drawdown duration, Calmar), risk-adjusted (Sharpe, Sortino), trading/exposure (turnover = 1/2 Σ|w_target - w_drift|, annualized turnover, gross exposure, net exposure, holding period ~ 1/annualized turnover), costs (gross vs net, cost capacity c* = expected gross excess return / turnover).
  - Cost capacity: author's example: gross excess of 1% per period and turnover of 10% of NAV => c* = 10% per dollar traded (a cost of 10 cents per dollar consumes the entire excess). State the one-way/round-trip convention.
- **Economic diagnostics (3 layers)**: implementation robustness (costs/turnover), incremental model value (vs a simple implementable baseline: equal-weight, inverse-vol, buy-and-hold, risk parity, factor neutral), state dependence (regimes).
- **Regime-sliced diagnostics**: volatility regime (60d realized volatility of the index vs point-in-time expanding median) x trend regime (6m return, ρ = P_t/P_{t-126} - 1, vs expanding median) => 4 states: Risk-on (low vol, uptrend), Caution (high, up), Crisis (high, down), Recovery (low, down). Label lagged when aligning with the next period's return; threshold never computed on the full test sample.
  - Anti-snooping rules: (1) pre-specify regimes before looking at performance; (2) transparent definitions (better than HMM/regime-switching VAR for reporting); (3) report all regimes including unfavorable ones; (4) include exploratory regime searches in the trial accounting.
  - Report: regime-conditional drawdown, drawdown attribution and recovery time per regime; coverage (time %, observations).
- **Sharpe inference**: per-period SE approximately 1/√T for small SR under IID normal returns (annualize the SE on the same scale). Autocorrelation adjustment (Lo 2002): SR(q) = q·SR(1) / sqrt(q + 2 Σ (q-k) ρ_k); positive autocorrelation overstates the naive annualized Sharpe; use HAC or a dependence-aware bootstrap.
  - Minimum track record length: approximation as a function of annualized Sharpe, level α (one-tailed) and power 1-β; author's example with α=0.05, power 0.80 and annualized SR 0.5 (formula and result illegible in the extraction; implemented in 11_sharpe_ratio_inference.py). Message: modest Sharpes can be useful as diversifiers but require long samples.
- **Search adjustment**: after searching N candidates, the null hypothesis becomes H0: max_j E[d_j] <= 0; the winner is not a randomly chosen member. The expected maximum under the null grows mechanically with the breadth of the search N (scaling argument).
  - White's Reality Check (2000): joint bootstrap of the differential process (stationary/block bootstrap) to test whether the best of the family exceeds what the search would produce; Hansen's SPA test (2005) reduces the conservativeness.
  - Deflated Sharpe Ratio (DSR; Bailey-López de Prado 2014b) on top of PSR (2012): PSR = Φ(·) of (SR_hat - SR*) scaled by √T and adjusted in the denominator for skewness γ3 and kurtosis γ4 (T = effective sample size if there is serial dependence); DSR evaluates the same probability with SR* = expected Sharpe of the best of N trials; requires the effective number of independent trials (100 adjacent lookbacks are neither 100 ideas nor 1).
  - FWER (Holm-Bonferroni: a false discovery is costly) vs FDR (Benjamini-Hochberg: large signal libraries with subsequent filters).
  - Rademacher Anti-Serum (Paleologo 2025): matrix X of periods x strategies (x_{t,n} = volatility-scaled return, or the IC of period t); complexity R_hat = maximum signed average with random sign vectors {-1,+1}; subtract that snooping term from each θ_hat for simultaneous lower bounds; if many strategies are nearly identical R_hat is low; with serial dependence use non-overlapping blocks.
  - Search budget: the null threshold of the per-period Sharpe grows with N, so a larger search requires larger T or a higher Sharpe (formula illegible); search consumes evidence: two years of daily data support fewer adaptive choices than twenty years of monthly data.
  - Trial accounting: include features, labels, horizons, lookbacks, model class, hyperparameters, training window, rebalance rule, universe filter, cost model, risk overlay, constraints, regime definition and selection metric. Fixing a bug is not a trial; changing the specification after seeing performance is. Separate exploration from confirmation (frozen candidate, reduced set of comparisons, ideally unused data).

## Key Concepts
- **Lookahead bias**: using information not available at decision time; diagnosis: walk-forward and point-in-time audit.
- **Survivorship bias**: only surviving assets; include delisted ones.
- **Data snooping**: the best reported Sharpe is biased upward when many trials are searched (White 2000).
- **Semantic drift**: arrays that no longer correspond to a feasible protocol.
- **Simulation semantics**: signal-to-fill timing, fill price basis, execution order (sells before buys free up cash), translation of target weights into orders, cash accounting, fractional vs integer shares, behavior under insufficient cash, short rules, stops/limits, missing bars, cost models.
- **Conditional drawdown / drawdown attribution**: maximum loss per regime and decomposition of large drawdowns by regime.
- **Hit rate**: fraction of positive periods; misleading without context (trend-following: low hit rate, positive expectancy).
- **Calmar**: compounded return / max drawdown; sensitive to short samples.
- **Net vs gross return**: r_net = r_gross - c_t (trading, financing, borrow).
- **Break-even cost**: cost per side at which the net return becomes zero (fragility ceiling).
- **Regime snooping**: testing many regime definitions and reporting the most favorable one.

## Mental Models
- Use vectorized when the protocol is weight-based and path-independent; event-driven when the strategy depends on simulated state (stops, drawdown circuit breakers, Kelly with realized capital, multi-leg positions, roll and margin in futures).
- Use a simple baseline as a floor: ML must beat it by a significant margin net of costs, or the complexity is not justified.
- Use break-even cost vs realistic cost as a fragility measure: a thin edge demands more sensitivity analysis; higher frequency = less headroom (10x turnover = 1/10 of the headroom).
- Use the unadjusted Sharpe only as a screening prior (Table 16.7), and require all four levels of evidence to be coherent (measurement, economic diagnostics, fixed-strategy inference, search-adjusted inference).

## Anti-patterns
- **Same-bar execution (signal and fill at the same close)**: lookahead; switching to next-bar can turn an attractive result into a losing one.
- **Checking constraints after computing returns**: they were not simulated.
- **Mixing frequencies (daily signals, monthly rebalance) without modeling it.**
- **A single optimistic cost assumption.**
- **Taking a smooth equity curve or the best Sharpe as confirmation.**
- **Regime threshold computed on the full test sample**: contaminates the diagnostic.
- **Counting every configuration as independent (over-penalizes) or ignoring the search (under-penalizes).**
- **Engine defaults as conveniences**: treat them as assumptions and verify them.
- **Comparing engines as a brand ranking**: they reveal which assumptions drive the result.
- **Using a simple √252-annualized SR with positive autocorrelation and no adjustment.**

## Code Examples
The extracted text includes no code; the notebooks (01_backtest_first_principles.py ... 18_vectorbt_engine_parity.py) contain it. Reconstruction of the described event-driven loop:

```
for event in market_events:
    update_prices_and_portfolio_state(event)
    process_open_orders(fill_model)
    apply_costs(commissions, slippage)
    strategy.on_bar(state)
    advance_clock()
```
What it demonstrates: the order of the event-driven simulation (state -> fills -> costs -> decision) that makes the protocol inspectable.

## Reference Tables
Table 16.1 (failure mode -> diagnostic): lookahead -> walk-forward, point-in-time audit; survivorship -> include delisted, verify universe; data snooping -> Deflated Sharpe, hold-out; unrealistic execution -> slippage model, capacity analysis; underestimated costs -> cost sensitivity; regime fragility -> per-regime diagnostics.

Table 16.2 (simulation semantics that change results): signal-to-fill timing; fill price basis; order sequencing; target-weight translation; cash accounting; fractional vs integer shares; insufficient-cash behavior; short-sale rules; stop/limit semantics; missing bars and calendars; cost and slippage models.

Table 16.3 (ETF baseline 2010-01-04 to 2023-12-29 vs 60/40): Total Return 171.4% vs 282.8%; CAGR 7.4% vs 10.1%; Vol 11.4% vs 12.7%; Sharpe 0.65 vs 0.80; Sortino 0.86 vs 0.96; Max DD -31.9% vs -26.9%.

Table 16.4 (baseline spec): universe of 10 ETFs (SPY, QQQ, IWM, EFA, EEM, AGG, TLT, GLD, VNQ, DBC); signal: 6-month risk-adjusted momentum (return/volatility); positioning: top 3 (risk-on) or 60% AGG/40% TLT (risk-off); regime: 10Y-2Y spread > 0.5% = risk-on; equal weight; monthly rebalance on the first business day (signal at the month-end close); costs 5 bp per trade (10 bp round trip); no leverage; benchmark 60% SPY/40% AGG.

Table 16.5 (regimes): Risk-on = low vol/uptrend; Caution = high/up; Crisis = high/down; Recovery = low/down.

Table 16.6 (baseline by regime; 3,395 active days after a 126d warm-up; Time%, Obs, CAGR, Vol, Sharpe, MDD): Risk-on 36.0%, 1,223, 1.6%, 10.7%, 0.15, -18.3%; Caution 13.4%, 454, 17.1%, 14.0%, 1.23, -13.9%; Crisis 33.8%, 1,146, 6.6%, 11.9%, 0.55, -24.0%; Recovery 16.8%, 572, 16.4%, 9.9%, 1.66, -10.9%; Overall 100%, 3,395, 7.7%, 11.5%, 0.67, -31.9%.

Table 16.7 (unadjusted Sharpe -> interpretation): <0.5 weak standalone evidence (useful only with strong diversification, low costs or high capacity); 0.5-1.0 interesting if costs, drawdowns and diversification are favorable; 1.0-1.5 promising if net of costs, stable across regimes, better than baselines and with a CI; 1.5-2.0 strong for medium frequency but requires checking leakage, costs, sample selection and search bias; >2.0 unusual for diversified and scalable strategies, requires aggressive scrutiny.

Cited rule of thumb: McLean-Pontiff, published anomalies lose ~26% of their magnitude out of sample and an additional ~32% after publication (Ch14 cites 26% and 58% post-publication).

## Worked Example
ETF momentum baseline with regime filter (nb 01, NumPy and Pandas only):
- Steps: (1) universe of 10 liquid ETFs; (2) signal = 6m cumulative return divided by 6m realized volatility; (3) rank; risk-on (10Y-2Y > 0.5%) = top 3 equal-weight; risk-off = 60% AGG / 40% TLT ignoring the ranking; (4) protocol: signal at the month-end close, execution on the next business day, cost of 5 bp per side, no leverage; (5) tear sheet with equity curve, drawdowns, monthly heatmap, metrics and per-regime diagnostics.
- Results: Sharpe 0.65 vs 0.80 for the static 60/40, CAGR 7.4% vs 10.1%, max DD -31.9% vs -26.9%: the baseline does not beat the benchmark, a "healthy" result for a baseline that is hard to romanticize.
- Cost sensitivity (nb 14): annualized two-way turnover ~460%; a sweep from 0 to 200 bp per side takes the net Sharpe from 0.68 to slightly negative; net CAGR crosses zero near ~166 bp per side; the 5 bp assumption is ~30x below the ceiling; a strategy with the same gross CAGR and 10x the turnover would have 1/10 of the headroom.
- Regimes: the aggregate Sharpe of 0.67 (active days) is carried by Recovery (1.66) and Caution (1.23); Risk-on is the weakest (0.15, CAGR 1.6%); Crisis concentrates the worst intra-regime drawdown (-24.0%) because the binary curve filter rotates into bonds late. Against equal-weight across the 10 ETFs: EW wins in Caution (1.50 vs 1.23), ties in Recovery (1.74 vs 1.66), and in Crisis 0.78 vs 0.55 with DD ~-20% vs ~-23%; only Risk-on favors the baseline (0.15 vs ~0).
- Decision: the filter trails continuous diversification where it matters; the complexity has not demonstrated incremental value.

Inference example: annualized Sharpe 1.0 with 5 years of daily data (T≈1,260, A=252) -> annualized SE ≈ 0.45 -> 95% CI ≈ 1.0 ± 1.96 x 0.45 = [0.12, 1.88]. Selection simulation (nb 12): among 100 strategies with no skill, the best observed annualized Sharpe reaches 2.54 purely from the selection effect.

## Key Takeaways
- Fix the complete protocol (timing, fills, sizing, constraints, costs, benchmark) before looking at results; always execute on the next bar.
- Build an auditable non-ML baseline first and require ML to beat it net of costs.
- Report gross and net, turnover, break-even cost and a cost sweep; if it survives only in the optimistic case, it is fragile.
- Segment by pre-specified regimes and also report the unfavorable ones with sample sizes.
- Choose vectorized vs event-driven based on whether the protocol depends on state; verify that the engine defaults match what was declared.
- Treat the Sharpe as a selected estimate: CI, autocorrelation adjustment, DSR/Reality Check/FDR and complete trial accounting.
- Advance a strategy only when measurement, economic diagnostics, fixed inference and search-adjusted inference are coherent.

## Connects To
Ch3 (order types), Ch4 (point-in-time, survivorship), Ch6 (setup.yaml, validation, walk-forward), Ch7 (multiple testing, trial accounting, IC/AUC screening), Ch9 (regime features), Ch11-14 (predictions that feed signals; Ch12 on IC), Ch17 (portfolio construction, constraints, allocator comparison), Ch18 (cost and execution models), Ch19 (risk overlays, stops), Ch20 (cross-case synthesis, search-adjustment results).

## Code Reference

Official repository: [16_strategy_simulation/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/16_strategy_simulation). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_backtest_first_principles.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/01_backtest_first_principles.ipynb) + `.py` | 16.2, 16.4 | Builds a portfolio |
| [`02_futures_backtesting.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/02_futures_backtesting.ipynb) + `.py` | 16.2, 16.3 | Also introduces the protocol questions |
| [`03_single_asset_vectorbt.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/03_single_asset_vectorbt.ipynb) + `.py` | 16.3 | An RSI mean-reversion rule on daily |
| [`04_single_asset_ml4t_backtest.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/04_single_asset_ml4t_backtest.ipynb) + `.py` | 16.3, 16.5 | The same rule under the |
| [`05_stateful_strategies.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/05_stateful_strategies.ipynb) + `.py` | 16.3 | Three strategies whose logic cannot be |
| [`06_framework_parity.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/06_framework_parity.ipynb) + `.py` | 16.3 | One ETF strategy run as array arithmetic and |
| [`07_engine_divergence_anatomy.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/07_engine_divergence_anatomy.ipynb) + `.py` | 16.3 | Changes one configuration |
| [`08_signal_method_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/08_signal_method_comparison.ipynb) + `.py` | 16.2 | Converts registered model |
| [`09_performance_reporting.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/09_performance_reporting.ipynb) + `.py` | 16.5 | Assembles that report from the |
| [`10_regime_backtest_analysis.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/10_regime_backtest_analysis.ipynb) + `.py` | 16.6 | Slices the baseline by |
| [`11_sharpe_ratio_inference.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/11_sharpe_ratio_inference.ipynb) + `.py` | 16.7 | The sampling distribution of the |
| [`12_dsr_validation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/12_dsr_validation.ipynb) + `.py` | 16.7 | The Deflated Sharpe Ratio and the probability of |
| [`13_ras_protocol.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/13_ras_protocol.ipynb) + `.py` | 16.7 | The Rademacher Anti-Serum bound, which charges for |
| [`14_cost_sensitivity.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/14_cost_sensitivity.ipynb) + `.py` | 16.6 | Re-runs the strategy at every cost on a grid, |
| [`16_case_study_lean_parity.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/16_strategy_simulation/16_case_study_lean_parity.ipynb) + `.py` | 16.3 | The whole audit in one place, |

Other notebooks in the folder, not linked from the README: `15_lean_engine_parity.ipynb`, `17_backtrader_zipline_engine_parity.ipynb`, `18_vectorbt_engine_parity.ipynb`.

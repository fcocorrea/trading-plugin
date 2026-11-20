# Chapter 18: Transaction Costs

## Core Idea
Transaction costs explain most of the gap between theoretical alpha and realized return; they must be integrated throughout the workflow (feature evaluation, simulation, portfolio construction, risk, production) with ex ante models and ex post TCA in a circular cycle: estimate, trade, measure, recalibrate.

## Frameworks Introduced
- **Cost taxonomy (explicit / implicit / capacity)**: explicit (commissions, exchange fees, clearing, regulatory, financing: margin interest, borrow, futures roll, taxes/stamp duty); implicit (bid-ask spread, slippage, temporary vs permanent market impact); capacity (participation, crowding, opportunity cost). Costs in bps of notional, one-way unless otherwise stated.
  - When to use: deciding which terms to model based on the strategy type (Table 18.1).
  - How: insert explicit costs exactly per schedule; model implicit costs with regimes; evaluate capacity via participation.
- **Three baseline backtest cost models** (choose by participation as % of ADV):
  - Spread: cost = half-spread (+ fees); a round trip pays the full spread. Conservative defaults: US large-cap 5-10 bps, small-cap 20-50 bps, emerging/illiquid 50-100+ bps.
  - Linear slippage: cost ∝ α · (Q/V); slippage ~2-3x higher in volatile periods; defaults 5 bps large-caps, 15-25 bps mid/small.
  - Square-root impact: C_impact = η·σ·√(Q/V); typical η 0.1-0.5; one-way = C_explicit + half-spread + η·σ·√(Q/V).
  - Why it works / failure mode (sqrt): concave impact with exponents ~0.5 across multiple asset classes (Sato-Kanazawa 2024); it breaks down under stress (impact steepens), for very small trades (fixed costs dominate) and in fragmented/queue-driven microstructure; and for instruments whose price is not proportional to notional (options) a price-scaled model must be used.
- **Regime-aware cost modelling**: parameters per volatility bucket and intraday window; when the regime is ambiguous use stressed parameters; backtests with contemporaneous crisis costs; monitor normalized spread, relative volume, depth imbalance and implied vol jointly.
- **Execution algorithms as controls**: TWAP (equal slices; simple, ignores the volume pattern), VWAP (slices ∝ expected volume; optimizes benchmark tracking), regime-aware participation (dampen participation when spread, vol or depth deteriorate), receding-horizon/MPC (optimize multi-period, execute the first period, re-optimize).
- **Almgren-Chriss**: minimize E[cost(x)] + λ·Var[cost(x)] with λ = urgency; temporary impact η (typ. ∝ (v/V)), permanent impact γ, σ = timing risk. The optimal trajectory is front-loaded (not TWAP): it steepens with σ, λ or a shorter horizon (via κ). Generate schedules under scenarios (base, high vol +50% vol/+30% impact, low liquidity +20%/+50%, benign -20%/-20%).
  - Failure mode: assumes linear permanent impact and instantaneous temporary impact; if trajectories vary widely across scenarios, the trade is fragile: use a blended/conservative approach.
- **TCA (Implementation Shortfall)**: IS = (execution price − decision price)/decision price for buys (negate for sells); decompose into spread, impact, timing/slippage, opportunity cost, fees; first ask market-driven vs execution-driven; tag the regime.
- **Cost guardrails**: break-even turnover, minimum required edge, alpha-to-go (AR(1) signal with persistence φ and speed Γ), capacity analysis (participation-based and impact-adjusted), kill-switch criteria (absolute and relative, precommitted).
  - When to use: before deployment and in continuous monitoring.
  - How: see Worked Example.

## Key Concepts
- **Implementation shortfall (IS)**: difference between the paper portfolio and the executed one; the standard TCA metric.
- **Temporary vs permanent impact**: liquidity pressure that reverts vs an informational footprint that persists.
- **Participation rate**: Q/ADV; determines the appropriate impact model.
- **Alpha-to-go**: the value of alpha discounted by the costs of building and unwinding the position.
- **Capacity**: the AUM at which net Sharpe falls below a minimum threshold (typ. 0.5 or hurdle).
- **Propagator / transient-impact**: models with an impact decay kernel, for multi-period execution.
- **Costs as portfolio regularizer**: rebalancing costs make the optimizer less reactive to small forecast changes.
- **Risk-model-driven turnover**: turnover caused by changes in the covariance, not in the alpha.
- **Kill switch**: a precommitted rule that stops the strategy.

## Mental Models
- Use the spread model when <0.5% ADV; linear 0.5%-2% ADV; square-root >2% ADV.
- Use conservative calibration (quoted spread + slippage that widens with vol + η in the upper half of the range) when there is no execution data; use TCA to recalibrate when there is.
- Use aggressive execution when alpha decays fast; passive when it decays slowly: urgency is tied to alpha decay.
- Use break-even turnover and minimum required edge together: if they disagree, turnover or cost is misestimated.
- Modify execution before abandoning the signal; if after reducing turnover, improving execution, moving to a more liquid universe or lowering capacity there is still no acceptable net Sharpe, abandon it.

## Anti-patterns
- **Assuming zero friction or subtracting a % at the end**: approves strategies whose gross alpha never covered spread, slippage and impact.
- **Constant cost parameters**: spreads, slippage and impact change with the regime (March 2020: multiples, not percentages).
- **Calibrating with TAQ for patient execution**: overstates costs (Frazzini et al. 2018, $1.7T of real executions); the symmetric error is zero-cost, which approves the infeasible.
- **Spread-only for strategies that trade several % of volume**: materially underestimates.
- **Static TWAP/VWAP as the default under stress**: they keep going through regime changes.
- **Fixed per-share cost on historical adjusted prices**: a 5¢ charge is 5 bps at $200 but 50 bps at $20 and 200 bps at $5.
- **Bps-of-notional for options**: premium 3-5% of notional with a spread of 5-15% of premium = 15-75 bps of notional per side.
- **Comparing stressed fills with calm fills in TCA**: noise, not evidence.
- **Defining kill criteria during stress**: they must be set before deployment.

## Code Examples
There is no code in the range (only references to notebooks: 01_cost_taxonomy, 02_spread_estimation, 03_market_impact_calibration, 04_vwap_twap_execution, 05_almgren_chriss_optimal_execution, 10_gross_vs_net_performance, 11_cost_cliff).

## Reference Tables
| Asset class | Retail bps (one-way) | Institutional bps | Dominant component |
|---|---|---|---|
| US large-cap equities | 1-5 | 5-15 | Spread + execution quality |
| US small-cap equities | 10-30 | 20-50 | Spread + impact |
| Liquid ETFs (SPY, QQQ) | 0.5-3 | 1-5 | Spread |
| Equity options | 10-100+ (of premium) | 3-20 | Spread |
| Index futures (ES, NQ) | 0.7-1.5 | 0.5-1.0 | Tick-constrained spread |
| FX spot majors | 5-30 | 0.5-3 | Venue segmentation |
| Crypto perps (CEX) | 5-50 | 2-10 | Fee tier + vol regime |
| IG corporate bonds | 30-80 | 5-20 | Embedded markup |

| Strategy | Dominant costs |
|---|---|
| High-frequency | Spread, exchange fees |
| Daily rebalancing | Spread, slippage |
| Weekly/monthly | Market impact |
| Low-turnover | Explicit fees, opportunity cost |

| Asset type | Typical η |
|---|---|
| Large-cap | 0.1-0.3 |
| Mid-cap | 0.2-0.4 |
| Small-cap | 0.3-0.6 |
| Liquid futures | 0.1-0.2 |
| FX majors | 0.05-0.15 |

| Participation | Model |
|---|---|
| <0.5% ADV | Spread |
| 0.5%-2% ADV | Linear slippage |
| >2% ADV | Square-root |

US hours (ET): open 9:30-10:00 (wide spreads, maximum impact); midday 11:00-14:00 (cheapest); close 15:30-16:00 (concentrated volume, depth is consumed quickly).

TCA example (bps): total IS 2.83 = spread 1.00 (35%) + impact 0.63 (22%) + timing/slippage 0.50 (18%) + fees 0.70 (25%).

## Worked Example
The author's numerical guardrails:
- Break-even turnover: expected gross return 5% annual, round-trip cost 25 bps → sustainable turnover ≈ 0.05/0.0025 = 20x capital/year (order-of-magnitude diagnostic).
- Minimum required edge: 20 bps per round trip with a 5-day holding period → 4 bps/day required; if the signal is worth 2 bps/day it is not viable.
- Participation-based capacity: max participation 5%, universe ADDV $1B → daily capacity $50M; with full daily rebalancing (τ=1) AUM ≈ $50M; with τ=20% → ≈ $250M. Impact-adjusted: invert η·σ·√(Q/V)=I_max for the maximum daily notional, then divide by turnover.
- Options cascade (S&P 500 short straddle with daily delta hedge): Rung 1 naive round trip → strongly negative Sharpe (the spread consumes the VRP). Rung 2 hold-to-expiry (eliminates the exit spread) → negative within a wide CI. Rung 3 + liquidity filter (bottom quintile of relative half-spread, ~120 of ~600 names) → validation Sharpe +0.160 [-0.975, +1.778], holdout +0.974 [-0.836, +3.135]; both CIs cross zero; the model assumes 20.3% of the quoted half-spread (a favorable assumption), which is why 0.203/0.5/0.75/1.0 is swept. Annual Sharpe over 12 months has a standard error of ~±1. Tightening the universe beyond the quintile leaves it too thin.
- US equities panel (~3,200 names): the bps grid vs the per-share grid (0-10¢) diverge: bps positive up to 20 bps (0.86) and ~0 at 30 bps; per-share drops to 0.66 at 2.5¢ and -0.29 at 5¢; the disagreement diagnoses the cost model, not the signal.

## Key Takeaways
- Evaluate features with turnover: a somewhat higher IC is not worth 10x the turnover.
- Choose the cost model by participation (spread/linear/sqrt) and calibrate conservatively until you have TCA data.
- Condition costs on regime and intraday window; when in doubt, use stressed parameters.
- Execution is a control of the impact vs timing-risk trade-off; tie urgency to alpha decay and evaluate schedules under scenarios.
- Run break-even turnover, minimum required edge and capacity before deploying; define absolute and relative kill switches in advance.
- Use bps-of-notional in universes with a wide price range; price-scaled for options; compare both as a sensitivity check.
- Measure turnover induced by the risk model (fix alpha, vary covariance): large turnover = trades with no informational content.

## Connects To
Ch3 (microstructure, FIX), Ch7-8 (IC, factor scoring vs breakeven turnover), Ch14 (eigenvector stability), Ch16 (backtest, simulation), Ch17 (turnover constraints, costs as MVO regularizer), Ch19 (capacity and execution as risk limits), Ch20 (cost survival Sec 20.6, deflated Sharpe 20.5, net-of-cost monitoring).

## Code Reference

Official repository: [18_transaction_costs/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/18_transaction_costs). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_cost_taxonomy.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/01_cost_taxonomy.ipynb) + `.py` | 18.1, 18.2, 18.7 | This notebook maps the transaction cost landscape across our seven asset classes, comparing exchange fee structures, spread regimes, and the resulting breakeven alpha re… |
| [`02_spread_estimation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/02_spread_estimation.ipynb) + `.py` | 18.2, 18.3 | This notebook estimates bid-ask spreads from OHLCV data using two classical estimators, validates them against ground-truth microstructure data, and applies them across… |
| [`03_market_impact_calibration.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/03_market_impact_calibration.ipynb) + `.py` | 18.1, 18.3, 18.4, 18.8 | This notebook calibrates market impact models using real market data, estimates Kyle's lambda from NASDAQ-100 trade classification data, maps intraday volume profiles, a… |
| [`04_vwap_twap_execution.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/04_vwap_twap_execution.ipynb) + `.py` | 18.5 | This notebook implements the two most common execution benchmarks: Uses synthetic data. |
| [`05_almgren_chriss_optimal_execution.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/05_almgren_chriss_optimal_execution.ipynb) + `.py` | 18.6 | This notebook implements the seminal Almgren-Chriss (2001) framework for optimal trade execution. |
| [`06_ml4t_execution_demo.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/06_ml4t_execution_demo.ipynb) + `.py` | 18.4 | This notebook demonstrates the ml4t.backtest.execution module for realistic execution cost modeling. |
| [`07_ml4t_volume_participation.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/07_ml4t_volume_participation.ipynb) + `.py` | 18.5 | This notebook demonstrates VolumeParticipationLimit from ml4t.backtest.execution for realistic institutional order execution: Uses synthetic data. |
| [`08_ml_dynamic_execution.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/08_ml_dynamic_execution.ipynb) + `.py` | 18.5 | This notebook explores how machine learning can be used to dynamically adapt execution strategies based on real-time market conditions. |
| [`09_frequency_tradeoff.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/09_frequency_tradeoff.ipynb) + `.py` | 18.8 | This notebook demonstrates the critical tradeoff between signal quality and transaction costs at different rebalancing frequencies. |
| [`10_gross_vs_net_performance.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/10_gross_vs_net_performance.ipynb) + `.py` | 18.7, 18.8 | This notebook provides a comprehensive framework for analyzing the gap between gross (theoretical) and net (realized) strategy performance. |
| [`11_cost_cliff.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/11_cost_cliff.ipynb) + `.py` | 18.8 | This notebook demonstrates the dramatic impact of transaction costs on intraday trading strategies. |
| [`12_commission_slippage_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/18_transaction_costs/12_commission_slippage_comparison.ipynb) + `.py` | 18.2 | Provides a side-by-side comparison of all 6 commission models and 5 slippage models in ml4t.backtest.models. |

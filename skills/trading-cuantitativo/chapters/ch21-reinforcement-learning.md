# Chapter 21: Reinforcement Learning

## Core Idea
RL learns policies for sequential decisions under feedback and delay; in finance its advantage lies where the action itself is the object of optimization with a measurable reward (execution, market making, derivatives hedging), not in directional alpha discovery. The practical barrier is the simulation-to-reality gap.

## Frameworks Introduced
- **Financial MDP** ⟨S, A, P, R, γ⟩ (state, action, transition, reward, discount): the design of state, reward and action granularity matters more than the algorithm.
  - When to use: formulate execution, market making and hedging as (PO)MDPs.
  - How: (1) state = public market data (LOB: spread, multilevel depth, order flow imbalance) + private info (inventory, unrealized PnL, time remaining) + regime indicators (rolling vol, spread, depth ratios, time-of-day); stack observations and rolling stats to approximate Markov; LSTM/GRU as belief state; (2) action: discrete (buy/sell/hold, spread levels) for DQN, continuous (slice size, limit price, weights) for actor-critic; constraints via clipping, penalty or constrained MDP; hard constraints (kill switches, regulatory limits) are never only learned; (3) reward aligned with the objective; (4) γ 0.99-0.999 for execution/hedging.
  - Why it works / failure mode: the transition is unknown (multi-agent) → model-free; "model-free" does not remove assumptions about state, reward and simulator; under partial observability, poor features push the problem away from the assumed MDP.
- **Reward engineering**: the reward is the key piece; naive PnL is noisy and favors high variance; Sharpe/Sortino/Calmar as the objective; quadratic reward (Kolm-Ritter) ⇒ mean-variance (valid for elliptical distributions) and an analytical Q-function (basis of QLBS); λ controls return vs risk; in market making, asymmetrically dampened rewards (penalizing speculative gains more than spread capture) discourage trend-following.
  - Failure mode: reward hacking (the agent maximizes what is encoded, not the intent).
- **Algorithm selection**: DQN (value-based, off-policy, discrete; experience replay + target networks; Double DQN reduces overestimation; Dueling); PPO (on-policy actor-critic, clipped surrogate; stable default, less sample-efficient); DDPG/TD3 (off-policy continuous; TD3 with clipped double Q, delayed updates, target smoothing); SAC (off-policy continuous with entropy; more efficient, more tuning). Hybrid: pre-train off-policy in a simulator, fine-tune on-policy.
- **Risk-aware RL**: mean-variance RL (E[R] − λ·Var[R]), CVaR-based RL (requires deep networks), distributional RL (QR-DQN, IQN: any risk measure at decision time without retraining).
- **Optimal execution RL**: a policy that maps market state, remaining inventory and time to deadline to the next child order; benchmark: TWAP and Almgren-Chriss (IS = arrival price − average execution price); continuous actions often discretized into bins (0%, 10%, 20%... of inventory) for DQN, or a choice between market vs limit orders.
- **Market making RL vs Avellaneda-Stoikov**: A-S gives closed-form quotes with a reservation price shifted by inventory (long → reservation below the mid) and a spread depending on volatility and γ_AS (distinct from the MDP's γ); RL learns offsets (δ_bid, δ_ask) and sizes; reward = spread-capture PnL − λ·inventory².
- **Deep Hedging (Buehler et al. 2019)**: a network that maps state (price, time to expiration, current position, vol features, path-dependent characteristics) to a hedge ratio; objective = risk measure of terminal PnL (variance, CVaR/expected shortfall, mean-variance) with costs; produces a no-transaction band; model-based (requires retraining if the process changes, e.g., Heston/SABR). pfhedge library (PyTorch). **QLBS** (Halperin): Q-learning with quadratic reward, price emerging from hedging.
- **Inverse RL**: infer the reward from behavior. MaxEnt IRL (Ziebart 2008), GAIL, AIRL (transferable reward), T-REX (ranked demonstrations, can outperform the demonstrator). Behavior cloning: supervised state→action.
- **Sim-to-real toolkit**: multi-agent simulators (ABIDES), domain randomization (impact, latency, spreads, fees), offline RL (penalize out-of-distribution actions; Decision Transformers), offline policy evaluation (importance weighting, doubly robust).
- **Deployment checklist (4 phases)**: (1) pre-flight (OOS across different regimes, simulator with impact, OPE, independent review); (2) staged deployment (paper → limited capital with strict limits and quantitative go/no-go criteria); (3) real-time controls (hard-coded limits, manual kill switch, monitoring of latency and slippage vs simulation, human oversight); (4) governance (scheduled retraining, decay alerts, versioning for audit).

## Key Concepts
- **Temporal credit assignment**: attributing the final outcome to the sequence of actions; solved by the value function V(s).
- **Markov property / partial observability**: the future state depends only on the current one; markets are POMDPs.
- **Bellman optimality / TD learning**: Q*(s,a) = expected immediate reward + discounted value of the best next action.
- **Experience replay and target networks**: stabilize DQN.
- **On-policy vs off-policy**: PPO discards experience after each update; SAC/TD3 reuse it.
- **No-transaction band**: region where no trading occurs because the cost exceeds the benefit of rebalancing.
- **Reservation price**: indifference price shifted from the mid by inventory.
- **Reflexivity**: other participants react to the agent's orders.
- **Distribution shift (BC/offline)**: the agent enters states that the expert or the data never visited.
- **Domain randomization**: training over a distribution of plausible parameters.

## Mental Models
- Use RL when the action is the objective and the reward is measurable (IS, spread capture with inventory penalty, hedge CVaR); use the predict-then-act pipeline (Ch11-19) for alpha.
- Use PPO with scarce data or non-stationarity noise (more robust); SAC/TD3 when a high-fidelity simulator makes data cheap.
- Use DQN only with discrete actions; actor-critic for continuous sizes and hedge ratios.
- Use IRL instead of behavior cloning when you need portability across cost changes or a suboptimal expert.
- Use an analytical benchmark (TWAP, Almgren-Chriss, Avellaneda-Stoikov, delta hedging, Whalley-Wilmott) and identical accounting for every comparison; notebook results are diagnostics, not proofs of dominance.

## Anti-patterns
- **RL for directional alpha**: non-stationarity, impact from exploration, short-window Sharpe rewards that invite overfitting and reward hacking.
- **Training with simple price replay**: the agent exploits instant fills and does not experience impact, queues, partial fills or latency.
- **State information leakage**: e.g., including the day's closing price in the state of an intraday agent.
- **Reward hacking**: an agent penalized for negative returns learns not to trade.
- **Ignoring market impact**: the agent places arbitrarily large orders.
- **Relying on a single seed or a single path**: the character of the policy (uniform, back-loaded) varies by seed.
- **Hard constraints that are only learned**: kill switches and regulatory limits must be deterministic.
- **Conclusions from small benchmarks**: the PPO crypto comparison is too small for production claims.
- **Behavior cloning under shift**: compounding errors outside the expert's distribution.

## Code Examples
No code in this range (references to notebooks: 02_optimal_execution_ppo, 03_market_making_ppo, 04_crypto_execution_rl, 05_deep_hedging_pfhedge, inverse_reinforcement_learning, backtest_with_impact, algorithms_comparison; Gymnasium environment; pfhedge library).

## Reference Tables
| Algorithm | Type | Policy | Action | Sample efficiency | Uses |
|---|---|---|---|---|---|
| DQN | Value-based | Off-policy | Discrete | Moderate | Simple trading, discretized execution |
| PPO | Actor-critic | On-policy | Both | Lower | Allocation, execution, general trading |
| DDPG/TD3 | Actor-critic | Off-policy | Continuous | Higher | Execution, hedging, continuous allocation |
| SAC | Actor-critic | Off-policy | Continuous | Higher | Market making, dynamic hedging, microstructure |

| Task | Objective | Reward |
|---|---|---|
| Optimal execution | Minimize implementation shortfall | −(P_arrival − P_exec) with penalties for residual inventory/impact |
| Market making | Spread capture vs inventory risk | PnL − λ·inventory² |
| Deep hedging | Minimize terminal risk | −CVaR(terminal PnL) |
| Portfolio allocation | Risk-adjusted return | Sharpe or adjusted return |

| Aspect | Deep Hedging | QLBS |
|---|---|---|
| Method | Direct policy optimization | Value-based (Q-learning) |
| Learns | Hedge policy | Q-function → policy |
| Price | External input | Emerges from hedging cost |
| Reward | General (CVaR, variance) | Quadratic (mean-variance) |
| Tractability | Requires networks | Analytical in simple cases |

Extraction note: Table 21.1 came out with broken formulas in the source; the reconstruction above follows the chapter text.

## Worked Example
Execution in the teaching environment (Gymnasium, PPO with a bounded action space and a soft penalty for deviating from a reference schedule): mean IS of ~44 bps for PPO vs ~53 TWAP and ~52 Almgren-Chriss; PPO's schedule sits between uniform TWAP and the back-loaded Almgren-Chriss path. Author's reading: modest evidence that state-dependent pacing improves on simple schedules in a compact simulator; sensitive to simulator design and seed.

Crypto perps extension (Binance, hourly data, premium index as a feature): mean shortfall ≈ −123 bps vs −90 (TWAP) and −89 (premium-aware heuristic); negative = sales above the arrival price in a trending market. Mean hourly volume ~4.4 shares in high-premium vs 3.7 in low-premium; ~3.6 shares/h within 2h of funding vs 4.4 outside. Stylized demonstration, not production-ready.

Impact vs liquidity (backtest_with_impact): same long-only momentum strategy and same $5M order on US stocks 2010-2016. Mega-cap: order ≈1.5% of daily volume, profitable even with strong impact. Micro-cap: the order is multiples of daily volume, strong impact turns a paper gain into an almost total loss. Erosion grows monotonically with participation (square-root law); almost all illiquid winners (order > daily volume) become losers under strong impact.

Deep hedging (European call, proportional costs, Heston scenarios, expected shortfall objective): in the notebook, discrete delta hedging is the best on dispersion and 5% CVaR; deep hedging improves on the tabular Q-learning baseline and lands between delta and Whalley-Wilmott. Market making (PPO, 300K steps): liquidated wealth comparable to reservation-price baselines but with much more dispersion and more terminal inventory; weak quote skew (~±2 bps, inventory ≈ -24 to +10).

## Key Takeaways
- Apply RL where the action is the objective and a measurable reward exists; for alpha use the predictive pipeline.
- Design state, action and reward carefully (inventory, time, LOB and regime features); the reward matters more than the algorithm.
- Default: PPO for stability with little data; SAC/TD3 with a cheap simulator; DQN only with discrete actions.
- Always compare against TWAP/Almgren-Chriss, Avellaneda-Stoikov or delta/Whalley-Wilmott with identical accounting and multiple seeds.
- Train in simulation with impact, queues, partial fills and latency; use domain randomization and offline RL with care for distribution shift.
- Keep risk limits and deterministic kill switches outside the agent; deploy in phases with go/no-go criteria.
- Use IRL when you need to infer the objective; it is expensive and depends on demonstration quality and parameterization.

## Connects To
Ch3 (microstructure, inventory and adverse selection), Ch13 (LSTM/GRU, sequences), Ch16-17 (backtest, DeePM and end-to-end allocation), Ch18 (Almgren-Chriss, IS, square-root impact), Ch19 (CVaR, kill switches, deep hedging, Whalley-Wilmott), Ch22 (RAG), Ch24 (agent explainability); regulation: MiFID II Arts. 17 and 48.

## Code Reference

Official repository: [21_rl_execution_hedging/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/21_rl_execution_hedging). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

| Notebook | Section | What it does |
|---|---|---|
| [`01_algorithms_comparison.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/21_rl_execution_hedging/01_algorithms_comparison.ipynb) + `.py` | 21.3 | Trains DQN, PPO and A2C on one trading environment with the same interaction budget and scores them on the same episodes. |
| [`02_optimal_execution_ppo.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/21_rl_execution_hedging/02_optimal_execution_ppo.ipynb) + `.py` | 21.4 | Trains a PPO agent to pace a liquidation and compares it against TWAP and an Almgren-Chriss schedule on identical simulated paths, with the paired differences and their… |
| [`03_market_making_ppo.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/21_rl_execution_hedging/03_market_making_ppo.ipynb) + `.py` | 21.5 | Trains a PPO agent to choose a quote skew and a spread width against three fixed rules. |
| [`04_crypto_execution_rl.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/21_rl_execution_hedging/04_crypto_execution_rl.ipynb) + `.py` | 21.4 | Replays recorded hourly perpetual-futures bars instead of a simulated path, so the agent's state can carry the perpetual-spot premium and the hours to the next funding s… |
| [`05_deep_hedging_pfhedge.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/21_rl_execution_hedging/05_deep_hedging_pfhedge.ipynb) + `.py` | 21.6 | Hedges a short call under discrete rebalancing and proportional costs, where exact replication is impossible and the question becomes which P&L distribution to accept. |
| [`06_inverse_reinforcement_learning.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/21_rl_execution_hedging/06_inverse_reinforcement_learning.ipynb) + `.py` | 21.7 | Runs the arrow backwards: given a record of what a trader did, what objective would make those actions sensible? |
| [`07_backtest_with_impact.ipynb`](https://github.com/stefan-jansen/machine-learning-for-trading/blob/HEAD/21_rl_execution_hedging/07_backtest_with_impact.ipynb) + `.py` | 21.8 | Runs one momentum strategy and one dollar order across real US equity names spanning the liquidity range, under four strengths of a square-root impact model. |

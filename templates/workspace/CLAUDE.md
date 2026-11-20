# Trading workspace rules

Copy this file to the root of your workspace repository (not the plugin's).

## Scope
- Research and simulation only. No orders are sent to any exchange or broker. If a request implies executing real orders, stop and explain the limit.
- Data from paid sources is not added without the user's explicit authorization.
- No exchange credentials are stored in this repository.

## Method (Machine Learning for Trading, Stefan Jansen)
- Strategies are discovered from data inside a versioned setup, not hunted and then tuned. Follow the `experiment-design` skill: `setup.yaml`, `training/<label>.yaml`, `research.md`.
- Before proposing code or a research design (labels, features, walk-forward or CV, models, backtest, costs, portfolio, risk), find the reference implementation in the book's repository with the `ml4t-code-scout` agent, or with `/trading:ml4t`.
- Base the proposal on that code and cite the file and its permalink (`ml4t-ref url`). Point out every deviation and its reason. If the repository has no code for the problem, say so before proposing a solution of your own.
- Adapt the code to the plugin's engine and workspace; do not import `utils.*` or paths from the book's repository.

## Personal risk management
- `risk_profile.yaml` at the workspace root holds the user's personal risk rules; follow the `risk-management` skill and validate it with `trading-engine check-risk`.
- At most 1% of equity at risk per trade and 3% across all open trades, one position at most 25% of equity; RSI gate (no buys above 70, no sells below 30); a 2-hour pause after 3 consecutive losing trades across the portfolio, plus a graduated drawdown kill switch; reward/risk between 1.5 and 3; every distance in ATR multiples, never in price units.
- Where a personal rule and the book differ, the book wins; the `risk-management` skill records each decision.
- The simulated account is Exness Standard Cent (10,000 USC = 100 USD, MT5). Broker conditions come from the user's MT5 terminal, never from memory.
- Never loosen a value under `limits` or `kill_switch`: only the user changes them.

## Data
- An asset is only used on the market declared in `catalog/assets.yaml`.
- No run uses data that has not passed the engine's validation, including the volume requirement.
- Timestamps are normalized to UTC microseconds. Units are never mixed.
- Features and labels are point-in-time: built only from data available at their timestamp.

## Experiments
- Write the hypothesis, economic mechanism, pre-registration, success and kill criteria in `research.md` before running anything. `check-experiment` refuses the experiment otherwise.
- Changing a mechanic (universe, cadence, execution delay, state space, entry logic, cost components, stop rules) bumps `setup_version` and needs a new baseline. Record every change in the decision log.
- Diagnostics in order: model, then signal (rank IC), then strategy outcomes. Never select on strategy outcomes.
- Costs are always on (fees, slippage model, cost sweep). A result without costs is not reported as a result.
- The sealed holdout is not evaluated until the user explicitly asks, and only once, after the pipeline is frozen.
- Every run is logged, including failed and discarded ones. Every model config is a trial; the Deflated Sharpe uses the full count.
- Before concluding that a strategy works, review it with the `bias-reviewer` agent.

## Working style
- Numbers always come with their period (fold or holdout), their costs and their trial count.
- When in doubt about data, a market or a requirement, ask before assuming.
- Talk to the user in their language; files in the repository are written in English.

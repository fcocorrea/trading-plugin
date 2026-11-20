---
name: risk-management
description: The user's personal risk management, aligned with Machine Learning for Trading (ch17, ch19) - fixed fractional sizing at 1% of equity per trade with 3% total open risk and a 25% concentration cap, an RSI state gate, a losing-streak pause plus a graduated drawdown kill switch, ATR-based stop-loss, take-profit, trailing stop, break-even and partial closes with reward/risk between 1.5 and 3, and the simulated Exness Standard Cent account. Use when designing or reviewing sizing, exits or portfolio limits, when simulating trades, when editing risk_profile.yaml, or whenever a proposal involves position size, stops or targets.
---
# Personal risk management

Risk tolerance is personal, so the rules live in the workspace's `risk_profile.yaml` (template: `${CLAUDE_PLUGIN_ROOT}/templates/workspace/risk_profile.yaml`), validated with `trading-engine check-risk`. **The book wins**: where a personal rule and Machine Learning for Trading differ, the profile follows the book, and the table below records why. Never loosen a value under `limits` or `kill_switch` on your own; only the user changes them.

## Rules and their source

| Rule | Profile | Book source | Decision |
|---|---|---|---|
| Risk per trade ≤ 1% | `limits.max_risk_per_trade_pct` | `19_risk_management/03_position_sizing_mae_mfe.py`, fixed fractional sizing (`risk_per_trade = 0.01`) | Kept. The base is **equity** (the book's portfolio value V), not free margin: free margin shrinks with open positions and depends on the broker's margin model. |
| Size from the stop | `size = equity × risk / stop distance` | Same notebook: shares = V·r / \|entry − stop\|, integer floor | Size is rounded **down** to the lot step; below the minimum lot the trade is skipped, never rounded up. Fees and slippage at the stop count as risk. |
| Concentration cap | `limits.max_position_pct: 25` | Same notebook, `max_position_pct = 0.25` | **Added.** Tight ATR stops in calm markets would otherwise produce oversized notional. When the cap binds, the trade risks less than 1%. |
| Total open risk ≤ 3% | `limits.max_total_risk_pct` | ch19 control matrix (concentration, leverage limits) | User rule, consistent with the book. Open risk is measured to each position's current stop; a new trade opens only if it fits. |
| RSI filter 70/30 | `limits.rsi_filter` | ch08 Signal × State gating; ch19 anti-pattern "thresholds mined in-sample" | Kept as a **pre-registered state gate** (thresholds fixed ex ante, closed bars only). Its effect is measured against no gate on the calibration window and reported, because a gate changes which signals trade (ch08 three-step filter). |
| 3 losses → 2 h pause | `kill_switch.consecutive_losses` | ch19: every trigger needs metric, threshold, action, escalation and reinstatement | Kept as a behavior trigger across the whole portfolio, with explicit reinstatement (entries resume when the pause ends; the counter resets). |
| Drawdown kill switch | `kill_switch.drawdown_ladder` | ch19 Table 19.7 (5/10/15/20/30% drawdown); "graduated escalation instead of a single binary cutoff"; `ml4t.backtest.risk.MaxDrawdownLimit` | **Added.** The losing-streak pause alone is a single binary cutoff; the ladder is the escalation path. Levels clear when drawdown recovers below the previous threshold; unwind and terminate need the user's review. |
| SL, TP, trailing, break-even, partial closes | `trade_management` | `19_risk_management/02_exit_strategies.py`, `03` (scale-out), `11_systematic_risk_sweep.py` | Kept, configurable. Values are **pre-registered priors** (stop 2 ATR, target 3 ATR: the book's `ATRStopConfig`), calibrated only on a calibration window with MAE/MFE and frozen before evaluation. Each configuration compared is a trial. |
| RR between 1.5 and 3 | `limits.reward_risk` | The book's ATR prior gives 1.5 | Kept as an ex-ante bound on the calibration search. |
| Distances in ATR | `volatility` | `02_exit_strategies.py` takeaway 3: ATR-scaled stops are portable across assets and periods | Kept. ATR is taken **at entry** from the last closed bar and fixed for the trade (the book lists using current ATR as a limitation). |
| Same-bar stop and target | simulation contract | `02_exit_strategies.py` takeaway 4 | The stop settles first. Assuming the target first manufactures winners. |

## Simulated account

Exness Standard Cent, 10,000 USC (= 100 USD), MT5 (check that the platform offers each symbol). Leverage, margin call and stop-out are `null`, and `account.symbols` is empty, until the user reads them in MT5 (Market Watch > Specification, and the account page) for each symbol an experiment simulates. `check-risk` warns while they are missing; no trade simulation runs without them. Never fill them from memory or third-party pages.

Consequences of a 100 USD account:
- 1% risk is 1 USD (100 USC) per trade. With the minimum lot, many stop distances will require more risk than that; those trades are **skipped**, and the share of skipped signals must be reported, because it changes the strategy.
- Costs are the broker's: spread (and swaps when holding past rollover), not the research venue's fees.

## Research data versus execution venue

Signals are researched on the data source declared for each asset (it must have real traded volume). When execution is simulated on a different venue, such as Exness CFDs, which quote their own prices and have no exchange volume, treat it explicitly:
- The research venue's price is a proxy for the execution venue's price; the gap between them is a cost and a risk, not zero.
- The spread model must come from the execution venue's own quotes (for example, recorded from MT5), not from the research venue.
- Any result says which venue its costs assume.

## Evaluation protocol for the rules (ch16, ch19)

- Calibrate exit rules and the RSI gate on the calibration window only, freeze them, then evaluate once (`11_systematic_risk_sweep.py`: "calibrate before evaluating").
- Test exit rules on entries that carry no edge too: a signal's timing can flatter or damn a rule (`02_exit_strategies.py` takeaway 2).
- Combining rules inherits the tightest one: adding an overlay only shortens trades (takeaway 6).
- Labels must match exits (ch07): with ATR stops and targets, use triple-barrier labels with the same barriers and a time limit.
- Changing an overlay is a mechanic change: bump `setup_version` and rebuild the baseline.
- Stops, trailing, break-even, partial closes and both kill switches are path-dependent, so they are simulated event by event (`ml4t-backtest`: StopLoss, TrailingStop, ScaledExit, MaxDrawdownLimit).
- The pause is clock time: two bars at a 1h cadence, less than one bar at 4h. Say so when it happens.
- Same profile in backtest, paper and any future live system (ch25).

## Scope

Research and simulation only. Nothing in this plugin places real orders.

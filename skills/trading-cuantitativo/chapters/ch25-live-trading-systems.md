# Chapter 25: Live Trading Systems

## Core Idea
Most algorithmic projects fail in the move from backtest to live because of (self-inflicted) technical divergence, not a lack of edge. The solution is a unified framework where the same strategy code runs in backtest, paper and live (ml4t-backtest + ml4t-live), plus broker integration, an order state machine, pipeline parity verification and operational discipline.

## Frameworks Introduced
- **Unified research-to-production framework**: the same source files in both modes; only the data source and execution destination change.
  - When to use: whenever a validated strategy goes to production (ETFs, crypto perps, futures, FX, equities, options from the 9 case studies).
  - How: 4 principles: (1) abstract data and execution interfaces; (2) deterministic signal generation (no system time or divergent seeds); (3) shared event-driven core (on_data()); (4) identical position/order/strategy state, recoverable after a restart.
  - Why it works / failure mode: it eliminates divergence by construction. Sources of divergence: features (adjusted vs unadjusted prices, lookbacks), timing (signal at the close/execution at the open vs asynchronous), data edge cases (missing values, corporate actions, tickers), translation into orders (lot sizes, liquidity). Example: backtest 15% annual vs live 3%.
- **Ch16-19 continuity into live**: MLSignalStrategy/WeightFollowerStrategy (Ch17) move to LiveEngine; cost parameters (Ch18) become the live cost budget; kill-switch levels and drawdown thresholds (Ch19) become controls enforced by the broker wrapper.
- **Order state machine**: primary states Created, Submitted, Acknowledged, Partially Filled, Filled (terminal), Canceled (terminal), Rejected (terminal); the full implementation has 10 states and 19 transitions (including PENDING_CANCEL).
  - How: accept valid transitions regardless of order (fill before ack); on concurrent cancel/fill go through PENDING_CANCEL, discard the in-flight cancel and audit PENDING_CANCEL to FILLED; on timeout query the broker's state before retrying; partial fills: cumulative quantity, VWAP, remainder.
- **Idempotent crash recovery (client order IDs)**: generate and persist a unique ID before sending; after recovery: (1) check whether it exists among pending orders; (2) query the broker by client ID; (3) if it exists, align state; (4) if not, retry with the same ID. IDs: UUID or deterministic by strategy+symbol+signal timestamp.
- **End-of-day reconciliation**: positions, orders (orphans at the broker are dangerous), cash. On a discrepancy, stop automated trading; do not auto-correct.
- **Pipeline verification (4 stages)**: raw data to features, features to predictions, predictions to signals, signals to orders; same inputs to both systems, compare outputs stage by stage (binary search for the divergence).
  - How: feature parity with point-in-time data on multiple dates; prediction consistency with the same vector; sizing verification with the same optimizer; automated regression tests (every morning compare the previous day's live signals vs backtest; mismatch = halt); periodic replay of the full pipeline.
  - Why it works / failure mode: it separates technical failures (bugs) from statistical ones (market changed/overfit); without parity, diagnosis is impossible.
- **Divergence diagnosis**: verify identical inputs first; same features + different predictions = model layer; same predictions + different signals = optimization layer; check dependency versions; log inputs/outputs at every boundary.
- **Preflight checklist (startup gate)**: (1) healthy environment (network, up-to-date feeds, services); (2) authentication (correct account/environment/permissions; watch out for partial success); (3) coherent state (positions = broker, cash reconciles, no orphan orders); (4) market context (venue open, no circuit breakers, fresh prices); (5) configuration (risk limits, caps, kill switch access). Automated, with a readable report and a block if it fails.
- **4-level kill switch**: (1) pause new signals; (2) cancel working orders without touching positions; (3) flatten to cash; (4) full shutdown. A single tested control, independent of the component it controls.
- **Graduated paper-to-live transition**: extended paper phase (correct routing, daily reconciliation without drift), first live run with a fraction of capital (verifies fills/fees, not profitability), scale gradually, parallel operation/shadow portfolio.

## Key Concepts
- **Technical divergence vs statistical failure**: the same input gives a different output (bug) vs the market changed or the backtest was overfit.
- **SafeBroker**: single point that enforces a position-value cap, per-order cap, daily-loss kill switch and max data-staleness on every order intent; raises an exception if violated.
- **Startup reconciliation**: SafeBroker.connect() compares the persisted snapshot with the broker's state; do not start until the report is clean.
- **Shadow mode**: the same risk surfaces with virtual fills; health states: ok, waiting_for_data, feed_silent, idle_market_closed, broker_disconnected.
- **Training-cutoff guard**: LABEL_AVAILABLE_AS_OF = LIVE_WINDOW_START - forward_horizon_days.
- **Basket disposition**: intended_basket, attempted_basket, accepted_basket, failed_basket persisted separately.
- **PFOF**: payment for order flow; routing matters more than commission (Schwarz et al. 2022: round-trip costs from 0.07% to 0.46%, PFOF explains only 3.4% of the variance).
- **TWS vs IB Gateway**: GUI vs headless; Gateway preferred for production.
- **LEAN / QuantConnect**: open source engine (Apache 2.0) with cloud, local or self-hosted deployment.
- **CCXT**: unified Python interface for crypto exchanges.

## Mental Models
- Use a unified framework when you want the backtest to be evidence of what live will do; every rewrite is an opportunity for a bug.
- Use IBKR for multi-asset (170 markets) and Alpaca for US equities/ETF/crypto with a simple API; direct exchange APIs for crypto derivatives.
- Use a managed platform for prototypes, teams with little engineering capacity or simple strategies; self-hosted when custom data, control or latency matter; calculate the cost crossover.
- Use "parity" and "regime stability" as separate questions: Ch25 asks whether live computes the same thing; Ch26 asks whether it should keep trading.
- Use position polling for daily rebalancing; event-driven reconciliation (execDetailsEvent) as the production-grade alternative.

## Anti-patterns
- **Rewriting the strategy for production in another language/team**: two implementations, two sources of bugs.
- **Mixing training data from one exchange with live inference on another**: distribution shift that no code test detects (Binance vs OKX/Coinbase: spreads, fees, liquidity).
- **Blindly retrying rejections**: investigate the cause (buying power, symbol, market closed).
- **Auto-correcting reconciliation discrepancies**: turns record-keeping errors into trading errors.
- **Using paper trading to measure slippage or impact**: it simulates fills.
- **Kill switch without persistence**: a crash resets the counter and re-enables trading at the worst moment (fails open); likewise a staleness check without a per-asset timestamp.
- **Using research loaders for live warmup**: stale prices and sizing on historical levels.
- **Multi-step emergency procedures**: they fail under stress.
- **Checking jurisdiction after building the strategy**.
- **Suspecting a "bug" when the holdout fails with verified parity**: it may be regime (see Worked Example).

## Code Examples
Structure of idempotent recovery (reconstructed from the description):

```python
client_order_id = persist_new_order_id(strategy, symbol, signal_ts)
try:
    broker.submit(order, client_order_id=client_order_id)
except TimeoutError:
    existing = broker.get_order_by_client_id(client_order_id)
    if existing is None:
        broker.submit(order, client_order_id=client_order_id)
    else:
        sync_internal_state(existing)
```
What it demonstrates: the client_order_id as an idempotency key prevents duplicates after timeouts or crashes.

## Reference Tables
**IBKR error codes (Table 25.1)**

| Code | Meaning | Action |
|---|---|---|
| 502 | Cannot connect | Gateway/TWS not running or not accepting connections |
| 504 | Not connected | reconnect and reconcile |
| 201 | Order rejected | read the reason (margin, symbol, hours) |
| 202 | Order cancelled | client or broker (e.g., GTC end of day) |
| 162 | Historical data pacing | request queue with delays |

**Alpaca HTTP errors**: 403 auth (paper vs live keys vs endpoint); 422 order validation; 429 rate limit (exponential backoff); 500/503 retry with backoff.

**IBKR vs Alpaca**

| Aspect | Alpaca | IBKR |
|---|---|---|
| Assets | US equities/ETF, listed options, crypto | 170 markets, all types |
| Orders | market, limit, stop, stop-limit | adaptive, pegged, conditional, etc. |
| Rate limit | 200 req/min (unlimited on paid plan) | 50 messages/s on TWS API |
| Auth | non-expiring API key/secret | session with periodic renewal |
| Routing | PFOF (standard); smart routing (paid) | SmartRouting, no PFOF (Pro) |

Ports: TWS 7496 live / 7497 paper; IB Gateway 4001 live / 4002 paper; unique client ID per process. IBKR Pro: commissions from ~$0.0035/share; portfolio margin with equity from $100,000. Heartbeat with reqCurrentTime(); reconnection with exponential backoff and reconciliation (open/completed orders, executions, positions, account values). Callbacks: position(), updateAccountValue(), execDetails(), error().

**Regulation**: US PDT rule $25,000 (4+ day trades in 5 business days), SEC 15c3-5, FINRA 3110; EU MiFID II Art. 17 (price collars, max order values, kill buttons; applies to firms, not retail); India approvals via broker and exchange identifiers (check current guidance); taxes (UK stamp duty, FTT).

## Worked Example
Funding-rate crypto perps (OKX live data, Alpaca paper for execution). 3-class LightGBM model (stop_loss, timeout, profit_take) with 13 features (3 funding: premium z-score, 8h and 24h momentum; 10 price). Pipeline every minute: fetch hourly bars and funding (8 h cycle), compute 13 features, predict, decide entry/hold/exit; exits: prediction-flip (P(profit) below the threshold) and stop-loss (adverse move > ~1%). Universe of 19 perps; Alpaca covers 11 (ADA, APT, ATOM, BNB, COMP, INJ, NEAR, SUI are not tradable there; they are logged signal-only). Trained on 2020-2023; validation positive but holdout (which includes the late-2025 crypto drawdown) not. With verified parity (same features, predictions, signals) a bug is ruled out: the signal loads the long leg on small alts (COMP, DOT, SUI, AVAX, ~10% each) and the short on majors (BTC, ETH); the broad drawdown hit the long alt leg harder; the top 3 loss contributors are only ~8% of the total; filtering to majors would leave an unhedged short. Decision: treat the demo as a parity artifact, not a profitable strategy; whether to keep trading is a monitoring question (Ch26). Contrast: daily FX (Ridge, 20 pairs, IDEALPRO) with a holdout consistent with validation. 08_pipeline_verification harness: 5 parity tests pass (features, predictions, signals, order count, order details). ETF guard: live window 2025-01-01 with a 21-day forward horizon forces the training cutoff to 2024-11-29.

## Key Takeaways
- Run the same Strategy in backtest/paper/live; prove parity before risking capital.
- Verify stage by stage (data, features, predictions, sizing, orders) with automated tests per commit and a daily signal check; on mismatch, halt.
- Model orders as a state machine with a persisted client_order_id; reconcile positions, orders and cash daily and at startup.
- Enforce limits in the broker wrapper (caps, persisted daily-loss kill switch, staleness) and an independent 4-level kill switch.
- Scale gradually: extended paper, a fraction of capital, shadow portfolio in parallel.
- Evaluate routing, not just commission; verify legality, automation support, taxes and leverage in the jurisdiction before building.
- Train and run inference on the same venue; if that is not possible, measure and document the mismatch.

## Connects To
Ch 16-19 (backtest, allocation, costs, risk: Strategy, WeightFollowerStrategy, kill-switch levels); Ch 3 (order types, ITCH); Ch 6 and 12 (funding-rate, LightGBM, FX Ridge); Ch 24 (read-only agents that feed this layer); Ch 26 (monitoring, drift, circuit breakers, rollout); backtest overfitting (Bailey et al. 2015), Almgren-Chriss.

## Code Reference

Official repository: [25_live_trading/](https://github.com/stefan-jansen/machine-learning-for-trading/tree/HEAD/25_live_trading). Each `.ipynb` notebook has a `.py` pair (jupytext) with the same name. Descriptions come from the repo README.

This chapter has no notebooks; only `README.md` in the folder.

Other notebooks in the folder, not linked from the README: `01_unified_framework_demo.ipynb`, `02_etfs_deployment_loop.ipynb`, `03_ib_paper_trading_demo.ipynb`, `04_alpaca_paper_trading_demo.ipynb`, `05_alpaca_crypto_live_demo.ipynb`, `06_quantconnect_case_study.ipynb`, `07_order_state_machine.ipynb`, `08_pipeline_verification.ipynb`, `09_crypto_funding_deployment_loop.ipynb`, `10_safety_risk_demo.ipynb`, `11_fx_deployment_loop.ipynb`, `12_ib_basket_rebalance_demo.ipynb`, `13_runtime_safety_showcase.ipynb`.

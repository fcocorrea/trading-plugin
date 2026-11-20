---
name: experiment-design
description: How a research experiment is laid out and validated in this plugin, following the ML4T research framework - versioned setup.yaml, strategy map, labels with purge, feature register, walk-forward folds stepped back from a sealed holdout, model menus per label and search accounting. Use when creating, editing or reviewing an experiment, interpreting check-experiment errors, or deciding whether a change is a new setup version.
---
# Experiment design

The unit of research is not a strategy with parameters to optimize; it is a **versioned trading setup** inside which models are trained, evaluated walk-forward and counted (ML4T ch06). Strategies are discovered from data inside fixed invariants, never tuned until a backtest looks good.

The layout mirrors a case study of the book's repository (`case_studies/<id>/config/` and `run_log/`). Reference: any case study's `config/setup.yaml` and `config/training/<label>.yaml` (read them with `ml4t-ref path`).

```
experiments/<strategy_id>/
├── setup.yaml              invariants of the research program
├── training/<label>.yaml   model menu per label; every entry is a trial
├── features.py             one function per feature family declared in setup.yaml
├── research.md             strategy map narrative, pre-registration, decision log
└── run_log/                every run, including failed ones: registry.db + predictions/
```

Templates live in `${CLAUDE_PLUGIN_ROOT}/templates/experiment/`. Validate with `trading-engine check-experiment <folder>`: it refuses any `CHANGE_ME`, prints the fold plan and the trial count.

## setup.yaml

| Section | What it fixes | Rules the validator enforces |
|---|---|---|
| `strategy_id`, `setup_version` | identity | version `v1`, `v2`, ... |
| `strategy_map` | family, edge source, PnL decomposition, failure modes (ch06) | family and edge source from `catalog/strategy_map.yaml`; at least 2 failure modes |
| `universe` | symbols, market, data requirements, point-in-time eligibility | every symbol resolves on that market; `requires` includes `volume` |
| `decision` | cadence, bar sampling, snapshot, execution delay | cadence in `timeframes.yaml`; no same-bar fills |
| `mapping` | score-to-trade: state space, entry logic, sizing | shorting only where the market allows it; sizing and exits follow the personal risk profile (`risk-management` skill) |
| `constraints` | leverage and other hard limits | — |
| `costs` | fees, slippage model, cost sweep | fees ≥ 0; quoted spread needs quotes; sweep starts at 0 |
| `labels` | targets and horizons | horizon is a multiple of the cadence; exactly one primary |
| `features.families` | feature register (ch08) | each family has role, hypothesis, inputs, lookback, lag, failure mode; at least one `signal` family |
| `metrics` | selection metric and report set | selection is a model or signal metric, never a strategy outcome |
| `evaluation` | walk-forward protocol and sealed holdout | enough history before the first fold, feature warm-up included |

## features.py

One function per family in `features.families`, with the family's name. It receives the bars of one symbol, sorted by `ts` (bar close time, UTC), with `open, high, low, close, vwap, volume, quote_volume, buy_volume, sell_volume, n_trades`, and returns a polars DataFrame with the same number of rows. The runner prefixes columns with `<family>__`, applies the family's `lag` and drops warm-up rows.

Before using a family, the runner recomputes it on truncated histories; if any past value changes when the future is removed, it raises `LookAheadError`. Use rolling windows and `shift(+n)`; never `shift(-n)`, centered windows or full-sample statistics. Editing `features.py` changes the identity of every run, so all trials are retrained.

## Labels and execution

With `execution_delay: next_bar_open`, a decision at bar t enters at the next bar's open and exits h later: `fwd_ret_h(t) = open[t + cadence + h] / open[t + cadence] - 1`. Direction labels are 1 when that return is positive. `next_quote` and non-time bars are not implemented yet.

## Versioning: parameter or mechanic

Explore freely inside the invariants. A **parameter** change keeps the version: a threshold, top-k, a lookback, a cost level. A **mechanic** change bumps `setup_version` and needs a new baseline: universe, cadence, execution delay, long-only to long/short, rank to threshold entry, a new stop rule, a new cost component. Record every change in the decision log of `research.md`.

## Temporal protocol

- Folds match `utils/cv_splits.py` in the book's repository (`WalkForwardCV`, `fold_direction="backward"`, rolling): validation windows of `val_size` tile backward from the holdout start and are identical for every label; each training window spans the `train_size` before its validation window. A test checks parity with the library.
- The **purge equals the label horizon**: training rows whose label would reach into validation are dropped, and so are the last validation rows whose label would reach into the holdout. A sample at t is admissible only if its label is resolved before the window it is used against.
- Embargo (feature buffer) only matters when training data comes after validation (k-fold, CPCV), which the validator does not support yet.
- The **sealed holdout** `[start, end)` is evaluated once, after the pipeline is frozen and only when the user asks. Looking at it earlier spends it.
- Calendar: the engine counts durations in calendar time, which equals trading time only on 24/7 markets.

## Model ladder and search accounting

- Each label has a menu in `training/<label>.yaml`. A `linear` family is mandatory and must use a linear estimator: it is the baseline every other model must beat (ch11). Climb to GBM or deep learning only if the baseline's IC is credibly above zero.
- Estimators: `ols`, `ridge`, `lasso`, `elastic_net`, `lightgbm` for return labels; `logistic`, `lightgbm_classifier` for direction labels. Other keys are passed to the estimator. Linear models are standardized inside each training window.
- Every named config is a trial. Trials × folds = fits. All of them, including failed and discarded ones, count for the Deflated Sharpe Ratio (ch16). Remove configs before running, never after seeing results.
- Pre-register confirmatory hypotheses in `research.md`; anything added later is exploratory and is reported as such.

## Running

`trading-engine run <folder>` (phase 1) loads data strictly before the holdout, builds bars, features and labels, trains every config on every fold, and logs each run in `run_log/` with a 12-character id hashed from its configuration (setup sections that affect training, label, config, `features.py`, data fingerprint, engine version). Identical runs are reused. Failed runs are logged too.

Signal is the rank IC with Newey-West (HAC) inference from `ml4t-diagnostic`: cross-sectional per timestamp with at least 10 symbols, otherwise time series per symbol and period (the book's `min_cross_section` rule). Strategy simulation, costs and the Deflated Sharpe are phase 2.

## Diagnostics order (ch06)

Model diagnostics (does it learn and generalize across folds) → signal diagnostics (rank IC, ICIR) → strategy outcomes (net Sharpe, drawdown, turnover, break-even cost). Strategy outcomes come last; optimizing them at every micro-decision overfits the simulator.

## Preflight checks before the first run

Timing (features at t only use data available at t), coverage (every symbol and fold has data; `check-data` warnings understood) and trading intensity (turnover plausible for cadence and costs). They are checklist items in `research.md`.

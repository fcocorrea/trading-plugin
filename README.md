# trading — quant research plugin

A multi-market research lab for Claude Code, built on the workflow of *Machine Learning for Trading* (Stefan Jansen, 3rd edition). It is for designing, validating and logging research on historical data. **Research and simulation only**: it sends no orders to any exchange or broker.

## Principles

1. **Discover, don't hunt.** The unit of research is a versioned trading setup, not a strategy whose parameters get tuned until the backtest looks good. Models are trained inside fixed invariants, evaluated walk-forward and counted.
2. **Data before strategies.** An experiment declares what it needs (`requires: [trades, volume]`). The engine checks that the source delivers it before loading anything.
3. **Volume is mandatory.** No real traded size, no experiment. Counting updates is not volume.
4. **Explicit errors.** A market or source that is not implemented refuses to run. It never returns invented numbers.
5. **Costs always on.** Fees, slippage and a cost sweep are part of the experiment, not options.
6. **Sealed holdout and search accounting.** The holdout is evaluated once, after the pipeline is frozen. Every trial is logged and counts for the Deflated Sharpe.
7. **Reference code first.** Before proposing a solution, Claude looks for the book's implementation and builds on it.

## Status

| Component | Status |
|---|---|
| Catalog (sources, markets, assets, timeframes, strategy map) | implemented |
| Binance USD-M aggTrades (download, checksum, normalization) | implemented, tested with synthetic data |
| Own CSV (`generic_csv`) | implemented, tested with synthetic data |
| Validation of volume, time, day bounds, side, duplicates and gaps | implemented |
| Experiment validation (`check-experiment`): setup, folds, holdout, model menus, pre-registration | implemented |
| Live bookTicker recorder (`record`) | implemented, not tested against the network |
| Crypto perpetual market (fee rules) | implemented; funding not modeled |
| Book reference code (`ml4t-ref`, `ml4t-code-scout`) | implemented |
| Personal risk profile (`risk_profile.yaml`, `check-risk`) | implemented; applied by the simulation in phase 2 |
| Runner phase 1: time bars, features with look-ahead check, labels, walk-forward models, rank IC with HAC (`run`) | implemented, tested with synthetic data |
| Run log (`run_log/`: SQLite + predictions, content-addressed) | implemented |
| Runner phase 2: strategy simulation, costs, cost sweep, Deflated Sharpe | **pending** |
| Tick, volume and dollar bars; `next_quote` execution | **pending** |
| `compare` | **pending** |
| Historical Binance bookTicker | **not verified**: use `record` |
| Equities, futures, crypto spot, Polygon, OANDA | skeletons that fail with a clear message |

> The Binance adapter has not been tested against data.binance.vision yet. The first real test should be a one-day `fetch`.

## Layout

```
trading-plugin/
├── .claude-plugin/plugin.json   manifest (name "trading" -> commands /trading:*)
├── commands/                    one instruction file per command
├── skills/                      knowledge Claude loads on demand (trading-cuantitativo = the book)
├── agents/                      results-analyst, bias-reviewer, ml4t-code-scout
├── hooks/                       blocks calls to order endpoints
├── bin/trading-engine           engine launcher (Claude Code puts bin/ on the PATH)
├── bin/ml4t-ref                 local copy of the book's code (sync, status, path, url)
├── catalog/                     reference definitions (YAML)
├── templates/                   experiment/ and workspace/ templates (CLAUDE.md, risk_profile.yaml)
├── requirements.txt             pinned research environment (Python 3.12+)
└── engine/                      Python package (run through bin/trading-engine)
    ├── trading_engine/core/     interfaces, schema, errors, paths, time
    ├── trading_engine/sources/  data adapters
    ├── trading_engine/markets/  market rules
    ├── trading_engine/data/     ingestion, validation, storage, recorder
    ├── trading_engine/experiment.py  experiment loading and validation
    ├── trading_engine/runner.py      research runner (bars, features, labels, folds, diagnostics)
    └── tests/
```

## Workspace (a separate repository of yours)

Data and experiments do not live in the plugin. Create a workspace repository:

```
trading-lab/
├── CLAUDE.md                    copy of templates/workspace/CLAUDE.md
├── risk_profile.yaml            your personal risk rules (copy of templates/workspace/risk_profile.yaml)
├── catalog/*.yaml               optional: adds or overrides plugin entries by id
├── data/
│   ├── raw/{source}/{symbol}/{stream}/...            original downloads and recordings
│   ├── raw/binance_um_futures/recorder_events.jsonl  recorder connections/disconnections
│   └── {source}/{market}/{symbol}/{stream}/{YYYY-MM-DD}.parquet
└── experiments/<strategy_id>/
    ├── setup.yaml               versioned trading setup (invariants)
    ├── training/<label>.yaml    model menu per label; every entry is a trial
    ├── features.py              one function per feature family
    ├── research.md              strategy map, pre-registration, decision log
    └── run_log/                 every run, including failed ones (registry.db + predictions/)
```

Set `TRADING_LAB_HOME` to that repository. Otherwise the workspace is the current directory.

Keep the workspace **outside OneDrive, Dropbox or iCloud**: the data takes gigabytes and the recorder writes nonstop, which saturates syncing and can lock files.

Each workspace catalog file (`sources`, `markets`, `assets`, `timeframes`, `strategy_map`) is merged with the plugin's by id: what you declare adds or overrides entries without removing the rest.

## Experiments

An experiment mirrors a case study of the book's repository (`case_studies/<id>/config/setup.yaml`, `config/training/<label>.yaml`, `run_log/`). See the `experiment-design` skill. In short:

- `setup.yaml` fixes the invariants: strategy map (family, edge source, failure modes), universe, decision cadence and execution delay, score-to-trade mapping, costs with a cost sweep, labels, the feature register, the selection metric, and the evaluation protocol.
- Changing a parameter keeps the setup version; changing a mechanic bumps it and needs a new baseline.
- Walk-forward folds step backward from the sealed holdout, with a purge equal to the label horizon. All periods are `[start, end)`; `fetch` dates are inclusive.
- Each label has a model menu with a mandatory linear baseline. `check-experiment` prints the fold plan and the trial count used for the Deflated Sharpe.
- `research.md` holds the hypothesis, economic mechanism, pre-registration, success and kill criteria; the engine refuses the experiment while they are empty.

`/trading:new-experiment <strategy_id>` creates one from `templates/experiment/` and walks through it. `trading-engine run <folder>` trains the model menus walk-forward and reports model and signal diagnostics (rank IC with HAC inference from `ml4t-diagnostic`); the sealed holdout is never read.

## Getting started

Requires Python 3.12 or newer. Inside Claude Code, `/trading:setup` prepares the environment. By hand:

```bash
# 1. Engine environment with pinned versions (bin/trading-engine picks it up automatically)
python -m venv trading-plugin/engine/.venv
trading-plugin/engine/.venv/Scripts/python -m pip install -r trading-plugin/requirements.txt   # Windows
# trading-plugin/engine/.venv/bin/python -m pip install -r trading-plugin/requirements.txt     # macOS/Linux
# Alternative: your own interpreter in TRADING_PYTHON=/path/to/python

# 2. Check
trading-plugin/bin/trading-engine sources
trading-plugin/bin/trading-engine markets
trading-plugin/bin/trading-engine assets

# 3. First test: declare an asset in the workspace (/trading:add-asset), then one day of it
export TRADING_LAB_HOME=/path/to/trading-lab
trading-plugin/bin/trading-engine fetch <SYMBOL> --start <YYYY-MM-DD> --end <YYYY-MM-DD>
trading-plugin/bin/trading-engine check-data <SYMBOL>

# 4. Record live quotes, in a terminal of its own (they cannot be recovered later)
trading-plugin/bin/trading-engine record --symbols <symbol>,<symbol>
```

On Windows, `bin/trading-engine` runs from Git Bash, which is also the shell of Claude Code's Bash tool. Inside Claude Code, `bin/` is on the PATH and `trading-engine` is enough.

To load Claude Code with the plugin under development, use `claude --plugin-dir /path/to/trading-plugin`.

## Personal risk management

Risk tolerance is personal, so the rules live in the workspace's `risk_profile.yaml` (template in `templates/workspace/`), described by the `risk-management` skill and validated with `trading-engine check-risk` or `/trading:risk-profile`:

- Fixed fractional sizing: at most 1% of equity at risk per trade, 3% across all open trades, one position at most 25% of equity.
- RSI state gate: no buys with RSI > 70, no sells with RSI < 30.
- Kill switch: a 2-hour pause after 3 consecutive losing trades across the portfolio, plus a graduated drawdown ladder (5/10/15/20/30%).
- Configurable stop-loss, take-profit, trailing stop, break-even and partial closes, calibrated before evaluation.
- Reward/risk between 1.5 and 3; every distance in ATR multiples, never in price units.
- Simulated account: Exness Standard Cent, 10,000 USC (100 USD), MT5. Broker conditions are confirmed from the terminal.

Where a personal rule and the book differ, the book wins; the skill records each decision and its source.

## Reference code: Machine Learning for Trading

The `trading-cuantitativo` skill summarizes the book. Before proposing a solution, Claude finds the reference implementation in the [book's repository](https://github.com/stefan-jansen/machine-learning-for-trading) and builds on it:

- `ml4t-ref sync [ref]` keeps a read-only local copy (partial clone: the `.py` of every notebook, READMEs and configs; ~40 MB). The default path is `~/.cache/ml4t/machine-learning-for-trading`, configurable with `ML4T_REPO_DIR`. `ref` pins a commit when you want reproducibility.
- The `ml4t-code-scout` agent searches that copy and returns files, lines, dependencies and permalinks.
- `/trading:ml4t <problem>` chains everything: it finds the code and proposes the solution adapted to the plugin.
- The rule lives in `templates/workspace/CLAUDE.md`, so it always applies in the workspace, not only when the skill triggers.

## Tests

```bash
python -m pip install -r requirements.txt
cd engine && python -m pytest -q
```

## Adding a source or a market

Read `skills/add-source` and `skills/market-profiles`. Rule: a source or market becomes `implemented` only after validating one real day, with tests of its rules.

## Known limits

- `hooks/block_orders.sh` is a guard against accidents, not a security boundary: anyone with shell access can get around it. The real protection is having no trading credentials in the environment.
- `bars_per_year` and walk-forward durations assume a 24/7 calendar. Markets with sessions need trading calendars.
- With trades only (no quotes), the spread is not observable. Results with that approximation must be labeled as such.
- Perpetual funding is not modeled: `check-experiment` warns, and results must say they exclude it.

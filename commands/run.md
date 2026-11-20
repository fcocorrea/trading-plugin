---
description: Run an experiment's model menus walk-forward and report model and signal diagnostics (phase 1 of the runner)
argument-hint: <experiment folder> [--label NAME] [--config NAME] [--rerun]
allowed-tools: Bash(trading-engine:*)
---
Arguments: $ARGUMENTS

1. Run `trading-engine check-experiment "<path>"`. If it fails, stop.
2. Run `trading-engine check-data <SYMBOL>` for every symbol in the universe and confirm the data covers `evaluation.data_start` up to the holdout start. The runner refuses missing days.
3. Walk through the preflight checks in `research.md` (timing, coverage, trading intensity) with the user before the first run of a setup version.
4. Run `trading-engine run "<path>"` with the options the user asked for (`--label`, `--config`, `--rerun`). Identical runs are reused from `run_log/`; `--rerun` trains them again.
5. Report the table as is, then interpret it with the `results-analyst` agent's rules (see the `trading-metrics` skill):
   - Model and signal layers only. Phase 1 does not simulate the strategy: no Sharpe, drawdown or costs yet, so do not infer them.
   - The linear baseline comes first. If its HAC t-stat does not clear about 2 (its interval includes zero), say that climbing the model ladder is not justified yet.
   - State the trial count and that every trial counts for the Deflated Sharpe later.
   - Say whether the IC is time-series (`ts`, small universe) or cross-sectional (`xs`, at least 10 symbols).
6. If the run stops with a `LookAheadError`, explain which feature reads the future and how to make it causal. Never bypass the check.

Never evaluate the sealed holdout. The runner does not read it.

---
description: Compare runs or experiments (pending)
argument-hint: [experiment ids]
---
Pending implementation. When it exists, it will read each experiment's `run_log/` and build a table in `reports/`, always separating the walk-forward folds from the sealed holdout, and reporting the trial count next to every Sharpe.

Until then, do not compare results from memory. Ask the user for the result files of each experiment, and use the `results-analyst` agent.

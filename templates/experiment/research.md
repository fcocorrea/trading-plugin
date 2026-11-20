# <strategy_id>

Write the first five sections before running anything. check-experiment refuses the experiment
while any of them is empty or still a `<placeholder>`.

## Hypothesis
<what market behavior is exploited, stated so the data can contradict it>

## Economic mechanism
<why the edge exists and why it should persist: who is on the other side and why they accept it (strategy_map.edge_source)>

## Pre-registration
<confirmatory hypotheses, fixed before looking at validation results; anything added later is exploratory and is reported as such>

## Success criteria
<metric, threshold and layer: e.g. mean rank IC across folds with a HAC interval that excludes zero, then net Sharpe after costs>

## Kill criteria
<results that end this setup version: e.g. baseline IC indistinguishable from zero, or break-even cost below the fee tier>

## Reference code (ML4T)
<files from the book's repository this design follows, with permalinks from `ml4t-ref url`>

## Preflight checks (ch06)
- [ ] Timing: features at t use only data available at t; execution happens after the decision.
- [ ] Coverage: every symbol and fold has data; gaps and warnings from check-data are understood.
- [ ] Trading intensity: expected turnover is plausible for the cadence and the costs.

## Decision log
| date | setup_version | change | type (parameter / mechanic) | reason |
|------|---------------|--------|-----------------------------|--------|

## Results
<only after the runner exists; in-sample folds and the sealed holdout always reported separately>

---
description: Search accounting for an experiment - how many trials and fits it implies, and what that does to the Deflated Sharpe
argument-hint: <experiment folder>
allowed-tools: Bash(trading-engine:*)
---
Path: $ARGUMENTS

Run `trading-engine check-experiment "<path>"` and read its "search accounting" line and the trial count per label.

Explain to the user:
- Every named config in `training/<label>.yaml` is a trial; trials × folds = fits.
- Every trial counts for the Deflated Sharpe Ratio, including failed and discarded ones (ML4T ch16). Adding configs after seeing results inflates the search without showing it.
- If the menu is wide, suggest narrowing it before running: one linear baseline first, then climb the model ladder only if the baseline's rank IC is credibly above zero (ch11).

Do not edit the menus without confirmation.

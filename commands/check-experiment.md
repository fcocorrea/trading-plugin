---
description: Validate an experiment before running it (setup invariants, data requirements, folds, holdout, model menus, pre-registration)
argument-hint: <experiment folder>
allowed-tools: Bash(trading-engine:*)
---
Path: $ARGUMENTS

Run `trading-engine check-experiment "<path>"`. Quote the path: it may contain spaces.

If it passes, show the fold plan, the sealed holdout and the trial count, and repeat any warning (for example, that funding is not modeled).

If it fails, explain the reason and propose the concrete fix, referring to the `experiment-design` skill. Do not edit the experiment until the user confirms. An experiment that does not pass this validation does not run.

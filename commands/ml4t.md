---
description: Solve a quantitative research problem starting from the ML4T book's reference code
argument-hint: <problem to solve>
allowed-tools: Bash(ml4t-ref status), Bash(ml4t-ref path), Bash(ml4t-ref url:*)
---
Problem: $ARGUMENTS

1. Run `ml4t-ref status`. If there is no local copy, ask the user for confirmation and run `ml4t-ref sync` (downloads ~40 MB of code from the book's public repository, no notebooks or data).
2. Delegate the search to the `ml4t-code-scout` agent with the problem as stated and the relevant experiment context (market, cadence, available data, `setup.yaml` if there is one).
3. With its report, propose the solution:
   - Based on the book's code: say which file and function you follow, with its permalink.
   - Adapted to the plugin: schema from `skills/data-schema`, experiment layout from `skills/experiment-design`, rules from `skills/execution-model`, workspace from `TRADING_LAB_HOME`. Do not import `utils.*` or paths from the book's repository.
   - Point out every deviation from the book's code and its reason.
   - If it depends on an `ml4t-*` library, say so and ask before adding the dependency.
4. If the agent found no reference code, say so before proposing anything of your own.

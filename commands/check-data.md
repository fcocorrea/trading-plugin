---
description: Summarize coverage, missing days and warnings of an asset's data
argument-hint: <SYMBOL> [trades|quotes]
allowed-tools: Bash(trading-engine:*)
---
Arguments: $ARGUMENTS

Run `trading-engine check-data <SYMBOL> --stream <stream>`. The stream is the second argument if present; otherwise `trades`.

Summarize the covered range, number of days, rows, missing days and warnings. If there are gaps, suggest the `fetch` that would cover them, but do not run it without confirmation.

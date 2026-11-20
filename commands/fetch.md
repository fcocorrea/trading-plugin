---
description: Download, validate and store an asset's data for a date range
argument-hint: <SYMBOL> <YYYY-MM-DD> <YYYY-MM-DD> [trades|quotes]
allowed-tools: Bash(trading-engine:*)
---
Arguments: $ARGUMENTS

Run `trading-engine fetch <SYMBOL> --start <start> --end <end> --stream <stream>`. The default stream is `trades`. Both dates are included in the download.

Rules:
- Before downloading, confirm the asset, the source and the range with the user, unless they already gave them explicitly.
- If the command fails with `MissingCapabilityError` or `UnimplementedError`, explain the reason as is. Do not try to bypass the validation or change the catalog to make it pass.
- When done, summarize successful, cached and failed days, and show the relevant warnings (trade_id gaps).

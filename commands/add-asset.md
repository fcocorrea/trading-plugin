---
description: Register a new asset in the workspace catalog
argument-hint: <SYMBOL> <market_id> <source_id>
allowed-tools: Bash(trading-engine:*)
---
Add the asset to the **workspace** `catalog/assets.yaml`, not the plugin's. Arguments: $ARGUMENTS

1. Check with `trading-engine markets` that the market exists and is `implemented`. Check with `trading-engine sources` that the source exists and is `implemented`. If either fails, stop and explain what is missing.
2. Add the entry with `market`, `source`, `base`, `quote`, `tick_size`, `lot_size` and `streams`.
3. Declare in `streams` only what the source delivers. Never declare `quotes` if the source has `quotes: false`.
4. Confirm with `trading-engine assets`.

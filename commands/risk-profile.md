---
description: Show and validate the personal risk profile (risk per trade, RSI filter, circuit breaker, ATR-based trade management)
argument-hint: [path to risk_profile.yaml]
allowed-tools: Bash(trading-engine:*)
---
Arguments: $ARGUMENTS

1. If the workspace has no `risk_profile.yaml`, offer to copy `${CLAUDE_PLUGIN_ROOT}/templates/workspace/risk_profile.yaml` to the workspace root, then walk through it with the user.
2. Run `trading-engine check-risk` (with the path if one was given) and show the summary.
3. If it fails, explain which personal rule is broken and propose the fix (see the `risk-management` skill). Never loosen a value in `limits` on your own: those are the user's personal rules, and only the user changes them.
4. If the user changes the profile after experiments exist, remind them that changing an overlay is a mechanic change: affected experiments need a new `setup_version` and a new baseline.

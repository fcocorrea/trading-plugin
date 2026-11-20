---
description: Create a new research experiment (versioned setup, model menus, pre-registration) from the templates
argument-hint: <strategy_id> [SYMBOLS comma-separated] [cadence]
allowed-tools: Bash(trading-engine:*), Bash(ml4t-ref status), Bash(ml4t-ref path), Bash(ml4t-ref url:*)
---
Arguments: $ARGUMENTS

Follow the `experiment-design` skill. The goal is a setup where strategies are discovered from data inside fixed invariants, not a strategy whose parameters get tuned.

1. Create `experiments/<strategy_id>/` and copy into it everything in `${CLAUDE_PLUGIN_ROOT}/templates/experiment/` (`setup.yaml`, `training/`, `features.py`, `research.md`). Refuse if the folder already exists.
2. Fill `setup.yaml` with the user, section by section. Ask; do not invent:
   - `strategy_map`: family, edge source, and the top 2-3 failure modes (ML4T ch06; options in `${CLAUDE_PLUGIN_ROOT}/catalog/strategy_map.yaml`).
   - `universe`, `decision` (cadence from `timeframes.yaml`), `mapping`, `costs`.
   - `labels`: one primary label; horizons that are multiples of the cadence.
   - `features.families`: one entry per economic hypothesis, using only inputs the source delivers (`trading-engine sources`).
   - `evaluation`: folds and the sealed holdout. Check the data range with `trading-engine check-data`.
3. Write one function per feature family in `features.py`, following the contract at the top of the file (causal only: the runner refuses look-ahead).
4. Rename or add `training/<label>.yaml` so there is exactly one menu per label, each with a `linear` baseline. Keep the menu narrow: every config is a trial.
5. Before proposing labels, features or folds, find the reference implementation in the book's code with the `ml4t-code-scout` agent, and record the permalinks in `research.md` under "Reference code (ML4T)".
6. Have the user write the Hypothesis, Economic mechanism, Pre-registration, Success criteria and Kill criteria sections of `research.md`. You may draft them, but the user confirms them before anything runs.
7. Check that `mapping` and any exit design are consistent with the personal risk profile (`trading-engine check-risk`, `risk-management` skill): ATR-based stops and targets, reward/risk within limits, risk-based sizing.
8. Run `trading-engine check-experiment "experiments/<strategy_id>"` and show the fold plan and trial count. The experiment exists only once this passes.

---
description: Create the engine's Python environment and check that it responds
allowed-tools: Bash(trading-engine:*), Bash(ml4t-ref status), Bash(ml4t-ref path)
---
Prepare the interpreter used by `trading-engine`, the launcher in `${CLAUDE_PLUGIN_ROOT}/bin/` that Claude Code puts on the PATH. The engine needs Python 3.12 or newer (the book's `ml4t-diagnostic` library requires it) and installs scikit-learn, LightGBM and ml4t-diagnostic; check `python --version` first.

1. Check whether it already works: `trading-engine --version`. If it responds, skip to step 4.
2. Ask the user which they prefer:
   - **Plugin venv** (default): create `"${CLAUDE_PLUGIN_ROOT}/engine/.venv"` with `python -m venv` and install the pinned dependencies with that venv's python: `-m pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"`. The launcher finds it on its own. If the plugin was installed from a marketplace, every update changes the folder and this step must be repeated.
   - **Their own environment**: install the pinned dependencies there with `pip install -r "${CLAUDE_PLUGIN_ROOT}/requirements.txt"` and set `TRADING_PYTHON` to the absolute path of that python in the `env` block of the workspace's `.claude/settings.local.json`.
   Do not use `--break-system-packages` or install into the system Python without asking.
3. Check with `trading-engine sources`. It must list the sources without errors.
4. The book's reference code: check `ml4t-ref status`. If there is no copy, offer `ml4t-ref sync` (downloads ~40 MB of code, no notebooks or data; stored in `~/.cache/ml4t/` or `ML4T_REPO_DIR`). So that the `ml4t-code-scout` agent can read that folder without asking permission for every file, suggest adding the path from `ml4t-ref path` to `permissions.additionalDirectories` in the workspace's `.claude/settings.local.json`.
5. Personal risk profile: if the workspace has no `risk_profile.yaml`, offer to copy `${CLAUDE_PLUGIN_ROOT}/templates/workspace/risk_profile.yaml` there, then run `trading-engine check-risk` (see `/trading:risk-profile`).
6. Confirm the workspace: if `TRADING_LAB_HOME` is not set, the workspace is the current directory. Tell the user which one is in use. If the path is inside OneDrive, Dropbox or iCloud, warn them: the data takes gigabytes and the recorder writes nonstop, which saturates syncing and can lock files. Recommend a local folder outside sync.

---
description: Explain how to start the live bookTicker (L1 quotes) recorder
argument-hint: <comma-separated symbols>
---
Arguments: $ARGUMENTS

Do not start the recorder from this session: a Claude Code background process dies when the session closes, and every day not recorded is a day lost for good.

Give the user the command to run in a terminal of their own that stays open (or in `tmux`, `nohup` or as a service):

`"${CLAUDE_PLUGIN_ROOT}/bin/trading-engine" record --symbols <symbols>`

`--symbols` is required: ask the user which declared assets to record if none were given. Symbols are comma-separated, without spaces. On Windows, the command runs from Git Bash.

Point out the output paths inside the workspace:
- `data/raw/binance_um_futures/<SYMBOL>/quotes/<YYYY-MM-DD>.jsonl`: one message per line, with `ts_recv_us`.
- `data/raw/binance_um_futures/recorder_events.jsonl`: connections and disconnections. It is what tells a recording gap apart from a quiet market.

Conversion of those JSONL files to Parquet does not exist yet. Do not present it as available.

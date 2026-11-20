#!/usr/bin/env bash
# Blocks Bash commands that target exchange order endpoints.
# Claude Code passes the event as JSON on stdin. Exit 2 blocks the call.
# It is a guard against accidents, not a security boundary: anyone with shell
# access can get around it. The real protection is having no trading credentials.
input="$(cat)"
if printf '%s' "$input" | grep -Eqi '(fapi|api)/v[0-9]+/(order|batchOrders|leverage|marginType|positionSide)|(new_?order|place_?order|create_?order)'; then
  echo "Blocked: this project does not send orders to exchanges. Public data and simulation only." >&2
  exit 2
fi
exit 0

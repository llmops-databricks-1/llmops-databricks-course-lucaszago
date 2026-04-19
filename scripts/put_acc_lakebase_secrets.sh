#!/usr/bin/env bash

set -euo pipefail

PROFILE="${DATABRICKS_CONFIG_PROFILE:-acc}"
SCOPE="${DATABRICKS_SECRET_SCOPE:-semantic-agent-scope}"
CLI_BIN="${DATABRICKS_CLI_BIN:-/Users/lzago/.vscode/extensions/databricks.databricks-2.10.6-darwin-arm64/bin/databricks}"

if [[ ! -x "$CLI_BIN" ]]; then
  echo "Databricks CLI not found at: $CLI_BIN" >&2
  exit 1
fi

CLIENT_ID="${1:-}"
CLIENT_SECRET="${2:-}"

if [[ -z "$CLIENT_ID" ]]; then
  read -r -p "ACC service principal client_id: " CLIENT_ID
fi

if [[ -z "$CLIENT_SECRET" ]]; then
  read -r -s -p "ACC service principal client_secret: " CLIENT_SECRET
  echo
fi

"$CLI_BIN" secrets put-secret --profile "$PROFILE" --json "{
  \"scope\": \"$SCOPE\",
  \"key\": \"client_id\",
  \"string_value\": \"$CLIENT_ID\"
}"

"$CLI_BIN" secrets put-secret --profile "$PROFILE" --json "{
  \"scope\": \"$SCOPE\",
  \"key\": \"client_secret\",
  \"string_value\": \"$CLIENT_SECRET\"
}"

echo "Secrets written to scope '$SCOPE' using profile '$PROFILE'."
echo "Existing keys:"
"$CLI_BIN" secrets list-secrets --profile "$PROFILE" "$SCOPE"

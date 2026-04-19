#!/usr/bin/env bash

set -euo pipefail

CLI_BIN="${DATABRICKS_CLI_BIN:-/Users/lzago/.vscode/extensions/databricks.databricks-2.10.6-darwin-arm64/bin/databricks}"

if [[ ! -x "$CLI_BIN" ]]; then
  echo "Databricks CLI not found at: $CLI_BIN" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_env_pipeline.sh <env> [--profile <profile>] [--skip-deploy] [--from <step>] [--only <step>]

Environments:
  dev | acc | prd

Steps:
  ingestion
  process
  vector-search
  permissions
  deploy

Examples:
  ./scripts/run_env_pipeline.sh dev
  ./scripts/run_env_pipeline.sh dev --profile dev
  ./scripts/run_env_pipeline.sh dev --skip-deploy
  ./scripts/run_env_pipeline.sh acc --from vector-search
  ./scripts/run_env_pipeline.sh prd --only deploy
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ENVIRONMENT="$1"
shift

case "$ENVIRONMENT" in
  dev|acc|prd) ;;
  *)
    echo "Invalid environment: $ENVIRONMENT" >&2
    usage
    exit 1
    ;;
esac

FROM_STEP=""
ONLY_STEP=""
PROFILE="$ENVIRONMENT"
SKIP_DEPLOY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --from)
      FROM_STEP="${2:-}"
      shift 2
      ;;
    --skip-deploy)
      SKIP_DEPLOY=true
      shift
      ;;
    --only)
      ONLY_STEP="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$FROM_STEP" && -n "$ONLY_STEP" ]]; then
  echo "Use either --from or --only, not both." >&2
  exit 1
fi

STEP_NAMES=("ingestion" "process" "vector-search" "permissions" "deploy")
JOB_NAMES=(
  "semantic_scholar_ingestion_job"
  "process_data_job"
  "embeddings_vector_search_job"
  "spn_permissions_job"
  "register_deploy_agent"
)

is_valid_step() {
  local step="$1"
  for valid in "${STEP_NAMES[@]}"; do
    if [[ "$valid" == "$step" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -n "$FROM_STEP" ]] && ! is_valid_step "$FROM_STEP"; then
  echo "Invalid --from step: $FROM_STEP" >&2
  exit 1
fi

if [[ -n "$ONLY_STEP" ]] && ! is_valid_step "$ONLY_STEP"; then
  echo "Invalid --only step: $ONLY_STEP" >&2
  exit 1
fi

should_run=false

if [[ "$SKIP_DEPLOY" != true ]]; then
  echo "==> Deploying bundle for target '$ENVIRONMENT' using profile '$PROFILE'"
  "$CLI_BIN" bundle deploy -t "$ENVIRONMENT" --profile "$PROFILE"
fi

for i in "${!STEP_NAMES[@]}"; do
  step="${STEP_NAMES[$i]}"
  job="${JOB_NAMES[$i]}"

  if [[ -n "$ONLY_STEP" ]]; then
    if [[ "$step" != "$ONLY_STEP" ]]; then
      continue
    fi
  elif [[ -n "$FROM_STEP" ]]; then
    if [[ "$step" == "$FROM_STEP" ]]; then
      should_run=true
    fi
    if [[ "$should_run" != true ]]; then
      continue
    fi
  fi

  echo
  echo "==> Running step '$step' with job '$job' on target '$ENVIRONMENT' using profile '$PROFILE'"
  "$CLI_BIN" bundle run -t "$ENVIRONMENT" --profile "$PROFILE" "$job"
done

echo
echo "Pipeline completed for target '$ENVIRONMENT'."

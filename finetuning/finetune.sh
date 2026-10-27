#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The bootstrap interpreter only parses configuration, so system Python 3.14
# can still load a .ENV that selects Python 3.10-3.12 via PYTHON.
if ! ENV_EXPORTS="$("${PYTHON:-python3}" "$ROOT/ftkit/env.py" --shell)"; then
  exit 2
fi
eval "$ENV_EXPORTS"
unset ENV_EXPORTS
export FINETUNE_HOME="${FINETUNE_HOME:-$ROOT}"
export HF_HOME="${HF_HOME:-$FINETUNE_HOME/.cache/huggingface}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export USE_TF=0
COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$COMMAND" == setup ]]; then
  exec "${PYTHON:-python3}" "$ROOT/setup_env.py" "$@"
fi
case "$COMMAND" in
  train-asr|eval-asr|export-asr|transcribe) ENVIRONMENT=asr ;;
  train-llm|eval-llm|export-llm|predict) ENVIRONMENT=llm ;;
  *) ENVIRONMENT="" ;;
esac
PY="${PYTHON:-python3}"
if [[ -n "$ENVIRONMENT" ]]; then
  PY="$FINETUNE_HOME/.venvs/$ENVIRONMENT/bin/python"
  if [[ ! -x "$PY" ]]; then
    echo "Missing isolated environment. Run: bash finetune.sh setup $ENVIRONMENT" >&2
    exit 2
  fi
fi
exec "$PY" "$ROOT/cli.py" "$COMMAND" "$@"

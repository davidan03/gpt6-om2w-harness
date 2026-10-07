#!/bin/bash
# Run GPT-6 (Responses API, native computer use) on Online-Mind2Web through the OpenWebRL eval harness.
# Usage: scripts/run_om2w.sh OUTPUT_DIR [extra run_evaluate.py args, e.g. --task-indices 0,1,2]
set -euo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (agent and GPT-4.1 judge)}"
: "${OPENWEBRL_UPSTREAM:?set OPENWEBRL_UPSTREAM to an OpenWebRL clone at commit 05f8ed4 (provides slime)}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:?usage: run_om2w.sh OUTPUT_DIR [args]}" && cd "$1" && pwd); shift
[ -z "$(ls -A "$OUT")" ] || { echo "OUTPUT_DIR must be empty: finished tasks are skipped by filename" >&2; exit 1; }

# Node-local scratch for browser profiles and port locks (shared filesystems stall Chromium).
RUNTIME=${RUNTIME:-${TMPDIR:-/tmp}/gpt6_om2w_$$}
mkdir -p "$RUNTIME/profiles" "$RUNTIME/home" "$RUNTIME/ports"
export TMPDIR=$RUNTIME/profiles BROWSER_HOME=$RUNTIME/home BROWSER_PYTHON=${PYTHON:-python}

# This repo's openwebrl/ must come before the upstream clone's.
export PYTHONPATH="$ROOT:$OPENWEBRL_UPSTREAM:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export SLIME_BROWSER_LOCAL_PROCESS_PYTHON=$ROOT/scripts/browser_python.sh
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=$RUNTIME/ports
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=$OUT/env_server_logs
export SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS=1200
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=4
export SLIME_BROWSER_STEP_HANG_GUARD=1
export JUDGE_API_BASE=${JUDGE_API_BASE:-https://api.openai.com/v1} JUDGE_API_KEY=${JUDGE_API_KEY:-$OPENAI_API_KEY}

cd "$ROOT"
exec "$BROWSER_PYTHON" -u astra_eval.py \
  --task-file openwebrl/data/online-mind2web.jsonl --hf-checkpoint gpt-6-astra \
  --output "$OUT" --turn-level --browser-response-format-mode browser_env \
  --judge-api-mode served --judge-model gpt-4.1 --judge-timeout-secs 120 \
  --max-steps 30 --task-timeout-secs 9000 --n-parallel 4 --context-num-screenshots 1 \
  --judge-max-attached-imgs 3 "$@"

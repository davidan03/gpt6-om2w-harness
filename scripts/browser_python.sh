#!/bin/bash
# Started by the harness in place of python for each browser env server ("-m openwebrl.docker.env_server ARGS").
# Installs the native computer actions (native_actions.py), then runs the env server.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
shift 2
exec env HOME="${BROWSER_HOME:-$HOME}" "${BROWSER_PYTHON:-python}" -u -c \
  'import runpy; from native_actions import install; install(); runpy.run_module("openwebrl.docker.env_server", run_name="__main__")' "$@"

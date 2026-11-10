#!/usr/bin/env bash
# Entry point required by the organizers: reads the test set from /data, writes /result/submission.jsonl
# (+ per-question logs in /result/logs/). Fully offline.
set -uo pipefail
cd "$(dirname "$0")"

mkdir -p /result/logs /tmp/dm_work
export DM_OUT=/result
export DM_WORK=/tmp/dm_work
export DM_MODEL_PATH="${DM_MODEL_PATH:-/models/Qwen3-8B}"
export DM_BACKEND="${DM_BACKEND:-deepagents}"
export CUBEJS_TELEMETRY=false
export NODE_ENV=production

# shellcheck disable=SC1091
source scripts/serve_vllm.sh

python run.py predict --split /data --out /result
rc=$?

# make sure every question has a line even if the run died (missing = abstain anyway)
if [ -f "${DM_LOGDIR:-/tmp/vllm_logs}/pids" ]; then
  xargs -r kill < "${DM_LOGDIR:-/tmp/vllm_logs}/pids" 2>/dev/null || true
fi
exit $rc

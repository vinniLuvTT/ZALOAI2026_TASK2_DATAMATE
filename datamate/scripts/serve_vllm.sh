#!/usr/bin/env bash
# Start one vLLM replica per GPU (8B bf16 ~16GB fits a 32GB 5090 with room for KV cache).
# Client round-robins between replicas deterministically (DM_LLM_URLS).
set -euo pipefail

MODEL="${DM_MODEL_PATH:-Qwen/Qwen3-8B}"          # in Docker: /models/Qwen3-8B (offline)
DEFAULT_NGPU=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
if [ "$DEFAULT_NGPU" -lt 1 ]; then DEFAULT_NGPU=1; fi
NGPU="${DM_NGPU:-$DEFAULT_NGPU}"
BASE_PORT="${DM_BASE_PORT:-8000}"
LOGDIR="${DM_LOGDIR:-/tmp/vllm_logs}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
mkdir -p "$LOGDIR"

URLS=()
for ((i = 0; i < NGPU; i++)); do
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve "$MODEL" \
    --served-model-name "${DM_LLM_MODEL:-Qwen/Qwen3-8B}" \
    --port "$port" \
    --max-model-len 20000 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --seed 0 \
    > "$LOGDIR/vllm_$i.log" 2>&1 &
  echo $! >> "$LOGDIR/pids"
  URLS+=("http://127.0.0.1:${port}/v1")
done

# wait until every replica answers
for u in "${URLS[@]}"; do
  for _ in $(seq 1 360); do
    if curl -sf "$u/models" > /dev/null; then break; fi
    sleep 5
  done
done

export DM_LLM_URLS="$(IFS=,; echo "${URLS[*]}")"
echo "vLLM ready: $DM_LLM_URLS"

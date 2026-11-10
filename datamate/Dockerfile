# SKELETON - replace the base image with the one named in the organizers' Docker guideline
# (needs CUDA >= 12.8 + a Blackwell-capable PyTorch/vLLM build).
ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
USER root

# Node 20 for Cube Core
RUN apt-get update && apt-get install -y curl ca-certificates gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev      # at BUILD time: the image runs with networking disabled

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Model weights baked into the image (declare Qwen3-8B, 8B params < 15B limit, in the README)
#   huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B   (do this before docker build)
COPY models /models

COPY dm ./dm
COPY scripts ./scripts
COPY run.py cube_runner.js predict.sh README.md ./
COPY fewshot_bank.jsonl ./fewshot_bank.jsonl
RUN chmod +x predict.sh scripts/*.sh

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DM_MODEL_PATH=/models/Qwen3-8B
ENTRYPOINT []
CMD ["bash", "/app/predict.sh"]

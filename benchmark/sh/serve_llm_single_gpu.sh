#!/bin/bash

MODEL=${MODEL:-"Qwen/Qwen3-8B"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
PORT=${PORT:-9000}
BACKEND=${BACKEND:-"FLASH_ATTN"}
DEVICE=${DEVICE:-0}

export CUDA_VISIBLE_DEVICES=$DEVICE
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=$BACKEND

vllm serve $MODEL \
  --served-model-name       LLM   \
  --max-model-len           $MAX_MODEL_LEN \
  --max-num-seqs            1000  \
  --max-num-batched-tokens  32768 \
  --block-size              32    \
  --tensor-parallel-size    1     \
  --gpu-memory-utilization  0.95  \
  --num-scheduler-steps     8     \
  --no-enable-chunked-prefill     \
  --enable-prefix-caching         \
  --trust-remote-code             \
  --enforce-eager \
  --seed  42      \
  --host  0.0.0.0 \
  --port  $PORT

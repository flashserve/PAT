export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"

vllm serve Qwen/Qwen3-8B \
  --served-model-name       LLM   \
  --max-model-len           32768 \
  --max-num-seqs            1000  \
  --max-num-batched-tokens  32768 \
  --block-size              32    \
  --tensor-parallel-size    1     \
  --gpu-memory-utilization  0.95  \
  --num-scheduler-steps     1     \
  --no-enable-chunked-prefill     \
  --enable-prefix-caching         \
  --trust-remote-code             \
  --enforce-eager \
  --seed  42      \
  --host  0.0.0.0 \
  --port  6001

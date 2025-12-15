#!/bin/bash

NUM_PROMPTS=5000
REQ_RATE=7
MODELS=("meta-llama/Meta-Llama-3-8B" "Qwen/Qwen3-8B")
BACKENDS=("FLASH_ATTN" "FLASHINFER" "PREFIX_ATTN" "Relay_ATTN")
TRACES=("toolagent" "burst")

PROGRESS_LOG="e2e_perf.jsonl"
LOCK_FILE="/tmp/exp_progress.lock"

if [ ! -f "$PROGRESS_LOG" ]; then
    touch "$PROGRESS_LOG"
fi

run_worker() {
    local MY_GPU_ID=$1
    local TOTAL_GPUS=$2
    local MY_PORT=$((9000 + MY_GPU_ID))

    echo "[GPU $MY_GPU_ID] Started. Port: $MY_PORT"

    local TASK_COUNTER=0

    for model in "${MODELS[@]}"; do

        if [[ "$model" == "meta-llama/Meta-Llama-3-8B" ]]; then
            MAX_MODEL_LEN=8192
        elif [[ "$model" == "Qwen/Qwen3-8B" ]]; then
            MAX_MODEL_LEN=32768
        fi

        for trace in "${TRACES[@]}"; do

            if [[ "$trace" == "toolagent" ]]; then
                BLOCK_SIZE=512
            elif [[ "$trace" == "burst" ]]; then
                BLOCK_SIZE=16
            fi

            for backend in "${BACKENDS[@]}"; do

                if [[ "$backend" == "Relay_ATTN" && "$trace" == "toolagent" ]]; then
                    continue
                fi

                if (( TASK_COUNTER % TOTAL_GPUS != MY_GPU_ID )); then
                    ((TASK_COUNTER++))
                    continue
                fi

                ((TASK_COUNTER++))

                CURRENT_TASK_ID="Model:${model}|Backend:${backend}|Trace:${trace}|Rate:${REQ_RATE}"

                if grep -Fxq "$CURRENT_TASK_ID" "$PROGRESS_LOG"; then
                    echo "[GPU $MY_GPU_ID] [SKIP] $CURRENT_TASK_ID"
                    continue
                fi

                echo "[GPU $MY_GPU_ID] Running: $CURRENT_TASK_ID"

                local TEMP_LOG=$(mktemp)

                # python run_e2e_experiments.py \
                #     --model "$model" \
                #     --max-model-len $MAX_MODEL_LEN \
                #     --port $MY_PORT \
                #     --gpu-id $MY_GPU_ID \
                #     --backend "$backend" \
                #     --request-rate $rate \
                #     --trace "$trace" \
                #     --block-size $BLOCK_SIZE \
                #     --num-prompts $NUM_PROMPTS > "$TEMP_LOG" 2>&1

                echo "python run_e2e_experiments.py \
                    --model \"$model\" \
                    --max-model-len $MAX_MODEL_LEN \
                    --port $MY_PORT \
                    --gpu-id $MY_GPU_ID \
                    --backend \"$backend\" \
                    --request-rate $rate \
                    --trace \"$trace\" \
                    --block-size $BLOCK_SIZE \
                    --num-prompts $NUM_PROMPTS" ">" "$TEMP_LOG" "2>&1"

                local EXIT_CODE=$?
                local SUCCESS=false

                if [ $EXIT_CODE -eq 0 ]; then
                    SUCCESS=true
                elif grep -q "Result appended" "$TEMP_LOG"; then
                    SUCCESS=true
                else
                    SUCCESS=false
                fi

                if [ "$SUCCESS" = true ]; then
                    echo "[GPU $MY_GPU_ID] => Success: $CURRENT_TASK_ID"
                    (
                        flock -x 200
                        echo "$CURRENT_TASK_ID" >> "$PROGRESS_LOG"
                    ) 200>"$LOCK_FILE"
                else
                    echo "[GPU $MY_GPU_ID] => Failed (Exit: $EXIT_CODE)"
                    tail -n 10 "$TEMP_LOG"
                fi

                rm "$TEMP_LOG"
            done
        done
    done
}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES// /}"
    NUM_GPUS=0
    for dev in "${CUDA_DEVICES[@]}"; do
        [ -n "$dev" ] && ((NUM_GPUS++))
    done
fi

if [ -z "${NUM_GPUS:-}" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi

echo "Detected GPUs: $NUM_GPUS"

for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
    run_worker $gpu_id $NUM_GPUS &
done

wait

echo "All experiments completed."
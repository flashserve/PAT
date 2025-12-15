#!/bin/bash

export TORCH_CUDA_ARCH_LIST="8.0 9.0"

TREES=(
    "1,10_4096,416"
    "1,256_256,32"
    "1,1024_2048,32"
    "1,2,64_1024,256,256"
    "1,4,256_32,256,32"
    "1,8,512_32,512,256"
    "1,8,512_32,2048,256"
    "1,8,512_512,512,256"
    "1,4,8,256_32,256,256,32"
    "1,4,16,512_1024,256,128,32"
    "1,4,16,64,256,1024_256,32,256,64,32,256"
    "1,16,32,64,128,1024_256,128,64,32,32,32"
    "1,16,32,64,256,1024_256,128,64,32,32,32"
    "1,8,16,32,64,128,1024_256,128,64,32,32,32,32"
    "1,8,16,32,64,256,1024_256,128,64,32,32,32,32"
    "2,8,16,256_256,256,32,32"
    "4,16,256,512_512,32,128,32"
    "8,16,32,256_512,512,256,32"
    "256_1024"
    "256_4096"
)

HEAD_CONFIGS=(
    "32 32"
    "16 8"
    "32 8"
    "64 8"
)

OUTPUT_FILE="kernel_perf.json"

rm -f $OUTPUT_FILE

# --- progress bar helpers ---
progress_bar () {
  local cur=$1 total=$2 extra=$3
  local width=40
  local percent=$(( cur * 100 / total ))
  local filled=$(( cur * width / total ))
  local empty=$(( width - filled ))

  printf "\r(%d/%d) %3d%% [%.*s%*s]: %s \033[K" \
    "$cur" "$total" "$percent" \
    "$filled" "########################################" \
    "$empty" "" \
    "$extra"
}

# --- compute total tasks ---
TOTAL=0
for tree in "${TREES[@]}"; do
  for config in "${HEAD_CONFIGS[@]}"; do
    TOTAL=$((TOTAL+1))
  done
done

CUR=0
for tree in "${TREES[@]}"; do
  for config in "${HEAD_CONFIGS[@]}"; do
    CUR=$((CUR+1))

    read -r hq hkv <<< "$config"
    extra="tree=${tree} config=(nh_q:${hq},nh_kv:${hkv}) (1~2min/task)"
    progress_bar "$CUR" "$TOTAL" " $extra"

    rm -rf ~/.cache/flashinfer
    read -r hq hkv <<< "$config"
    python benchmark_kernel.py --tree "$tree" --nheads_q "$hq" --nheads_kv "$hkv" --output_file "$OUTPUT_FILE" > kernel.log 2>&1
  done
done

echo -e "\nDone."
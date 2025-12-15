# PAT: Prefix Aware Attention

This repository provides **PAT** (Prefix-Aware Attention), a high-performance CUTLASS-based implementation designed to optimize the decoding attention phase in transformer models.

PAT automatically identifies complex shared prefix patterns within batched sequences and adaptively schedules shared prefixes into separate CTA computations. This approach significantly reduces KV cache reads, which is the primary bottleneck in attention computation during decoding.

# Artifact Evaluation Instructions

Below are the instructions to reproduce the experimental results presented in our paper "PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel". The repository is organized as follows:
- `benchmark/`: Contains scripts for kernel performance experiments and end-to-end serving performance experiments.
- `csrc/`: Contains the core implementation of the PAT.
- `plot/`: Contains scripts for generating plots from experimental results.
- `plugin/`: Contains vLLM plugins to integrate PAT with vLLM.
- `prefix_attn/`: Contains the main Python package for PAT.

## Required Hardware and Software

To run these experiments, you will need:
- An x86-64 Linux host with at least 64GB RAM.
- 200GB of free disk space.
- An NVIDIA A100 GPU with 80GB of memory.
- NVIDIA driver >= 550 and CUDA >= 12.4.

We have tested the experiments on Google Cloud `a2-ultragpu-1g` instance (200GB disk) with the `Deep Learning VM with CUDA 12.4 M129` system image. We recommend using a similar setup to ensure convenience and consistent performance.

Hint: You can use multiple GPUs (e.g., `a2-ultragpu-8g` instance) to speed up the end-to-end performance experiments. The scripts will automatically detect the available GPUs and distribute the experiments across them.

## Installation

### Step 1: Install Docker if Needed

If Docker has not been installed, run the following commands:
```shell
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
dockerd &
```

### Step 2: Prepare the Code and Docker Image
```shell
git clone https://github.com/flashserve/PAT.git
docker pull flashserve/pat:ae  # (~50 GB, including model weights)
```

### Step 3: Start the Docker Container
```shell
docker run -it --gpus all -v PAT:/workspace/PAT -w /workspace \
    --shm-size=64g flashserve/pat:ae /bin/bash
```

## Run Experiments

### Step 1: Kernel Performance Experiments

This script evaluates the attention kernel execution latency under synthetic workloads for different methods, as shown in **Section 8.3** and **Figure 10** of the paper.

```shell
cd /workspace/PAT/benchmark
# This experiment takes about 1.5 hours to complete
bash ./run_kernel_bench.sh
```

The results will be saved in `kernel_perf.json`.

### Step 2: End-to-End Performance Experiments

This script evaluates end-to-end inference latency across different methods under real-world workloads, corresponding to **Section 8.4** and **Figure 11** in the paper. Note that completing all experiments requires over 60 GPU-hours, so we provide two scripts: (1) `run_e2e_bench_part.sh`: runs a subset of experiments (QPS=7, all workloads, all baselines) for quick verification of results; (2) `run_e2e_bench_full.sh`: runs all experiments to reproduce the results in the paper.

Hint: To run the full experiments (`run_e2e_bench_full.sh`), you can use multiple GPUs (e.g., `a2-ultragpu-8g` instance) to speed up the experiments. The scripts will automatically detect the available GPUs and distribute the experiments across them.

```shell
cd /workspace/PAT/benchmark
# Quick verification (takes 4-5 GPU-hours to complete)
bash ./run_e2e_bench_part.sh
# Full experiments (takes over 60 GPU-hours to complete)
# bash ./run_e2e_bench_full.sh
```
The results will be saved in `e2e_perf.jsonl`.

## Parsing Results and Plotting

### Step 1: Plot Kernel Performance Results

```shell
cd /workspace/PAT/plot
python eval_kernel_perf.py --log-file ../benchmark/kernel_perf.json
```

This will generate a plot `fig/kernel_performance_overall.pdf`, showing the kernel performance comparison among different methods, corresponding to **Figure 10** in the paper.

### Step 2: Parsing End-to-End Serving Results

```shell
cd /workspace/PAT/plot
python eval_e2e_from_jsonl.py --log-file ../benchmark/e2e_perf.jsonl
```

This will generate a plot `fig/eval_e2e_overall_p99.pdf`, showing the end-to-end serving performance comparison among different methods, corresponding to **Figure 11** in the paper.



## Alternative: Install PAT from Source

You can also set up the environment from source without using Docker. Note that this method may take about 1-3 hours to complete, depending on hardware and network conditions.

1. Requirements: A100 / H100 GPU, CUDA>=12.4

2. Clone PAT, vLLM, and CUTLASS repositories
```shell
mkdir ~/workspace && cd ~/workspace
git clone https://github.com/flashserve/PAT.git
git clone https://github.com/NVIDIA/cutlass.git
git clone https://github.com/vllm-project/vllm.git
```

3. Build vLLM with PAT plugin (1-2 hours)
```shell
cd ~/workspace/vllm
git checkout v0.9.0
# Add PAT plugin to vLLM
rsync -av --progress ../PAT/plugin/vllm/ ./vllm/
TORCH_CUDA_ARCH_LIST="8.0" pip install .
```

4. Install FlashInfer and other dependencies
```shell
pip install flashinfer-python==0.2.5 transformers==4.53.0 numpy==1.24.0
```

4. Build PAT from source (~10 minutes)
```shell
# PyTorch (2.7.0) is required if vLLM is not installed
cd ~/workspace/PAT
# Replace <abs_path_to_cutlass> with the absolute path to CUTLASS repo
CUTLASS_ROOT=<abs_path_to_cutlass> pip install . --no-build-isolation
```



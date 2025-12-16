import json
import subprocess
import time
import requests
import os
import signal
import sys
import argparse
import re
from pathlib import Path

# REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent  # .../prefix-attn/benchmark
SERVE_SCRIPT = BASE_DIR / "sh" / "serve_llm_single_gpu.sh"
BENCHMARK_SCRIPT = BASE_DIR / "benchmark_serving.py"
DEFAULT_OUTPUT_LOG = BASE_DIR / "e2e_perf.jsonl"


def wait_for_server_ready(url, timeout=300):
    """Poll the server until it reports healthy."""
    start_time = time.time()
    print(f"Waiting for vLLM to be ready at {url}...")
    while time.time() - start_time < timeout:
        try:
            # vLLM generally exposes /health or /v1/models endpoints.
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    print("Error: Server timed out waiting for readiness.")
    return False


def parse_benchmark_metrics(log_output):
    """Extract key performance metrics from benchmark output."""
    metrics = {}

    # Define the metrics to extract and their regex patterns.
    patterns = {
        # Basic statistics
        "successful_requests": r"Successful requests:\s+(\d+)",
        "benchmark_duration": r"Benchmark duration \(s\):\s+([\d\.]+)",
        "total_input_tokens": r"Total input tokens:\s+(\d+)",
        "total_generated_tokens": r"Total generated tokens:\s+(\d+)",
        # Throughput metrics
        "request_throughput": r"Request throughput \(req/s\):\s+([\d\.]+)",
        "output_token_throughput": r"Output token throughput \(tok/s\):\s+([\d\.]+)",
        "total_token_throughput": r"Total Token throughput \(tok/s\):\s+([\d\.]+)",
        # Time to first token (TTFT)
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d\.]+)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s+([\d\.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d\.]+)",
        # Tokens-per-output-token (TPOT)
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d\.]+)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s+([\d\.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d\.]+)",
        # Inter-token latency (ITL)
        "mean_itl_ms": r"Mean ITL \(ms\):\s+([\d\.]+)",
        "median_itl_ms": r"Median ITL \(ms\):\s+([\d\.]+)",
        "p99_itl_ms": r"P99 ITL \(ms\):\s+([\d\.]+)",
        # End-to-end latency (E2EL)
        "mean_e2el_ms": r"Mean E2EL \(ms\):\s+([\d\.]+)",
        "median_e2el_ms": r"Median E2EL \(ms\):\s+([\d\.]+)",
        "p99_e2el_ms": r"P99 E2EL \(ms\):\s+([\d\.]+)",
    }

    # Extract the benchmark result block to avoid matching unrelated log content.
    start_marker = "============ Serving Benchmark Result ============"
    end_marker = "=================================================="

    try:
        start_idx = log_output.find(start_marker)
        end_idx = log_output.find(end_marker, start_idx)
        if start_idx == -1 or end_idx == -1:
            print("[Parser] Warning: Could not find benchmark result block.")
            return None

        result_block = log_output[start_idx:end_idx]

        for key, pattern in patterns.items():
            match = re.search(pattern, result_block)
            if match:
                # Attempt to convert to float or int, fallback to string on failure.
                val_str = match.group(1)
                try:
                    if "." in val_str:
                        metrics[key] = float(val_str)
                    else:
                        metrics[key] = int(val_str)
                except ValueError:
                    metrics[key] = val_str
            else:
                metrics[key] = None  # or 0, depending on requirements

    except Exception as e:
        print(f"[Parser] Error parsing output: {e}")
        return None

    return metrics


def run_experiment(config, OUTPUT_LOG_FILE):
    env = os.environ.copy()
    env["DEVICE"] = config["gpu_id"]
    env["BACKEND"] = config["ATTENTION_BACKEND"]
    env["MODEL"] = config["model"]
    env["MAX_MODEL_LEN"] = config["max_model_len"]
    env["PORT"] = config["port"]

    print(
        f"\n[Experiment] Starting: {config['model']} | Backend: {config['ATTENTION_BACKEND']}  |  Port: {config['port']}"
    )
    experiment_record = {}
    experiment_record["config"] = config

    # 1. Start the vLLM server (runs in the background).
    # Use preexec_fn=os.setsid so we can later kill the entire process group (server + children).
    server_process = subprocess.Popen(
        ["bash", SERVE_SCRIPT],
        env=env,
        stdout=subprocess.DEVNULL,  # Or redirect to a file server.log
        stderr=subprocess.PIPE,  # It's recommended to capture stderr for debugging
        preexec_fn=os.setsid,
    )

    base_url = f"http://0.0.0.0:{config['port']}"

    try:
        # 2. Wait for the server to become ready
        if not wait_for_server_ready(base_url):
            raise Exception("Server failed to start.")

        # 3. Run the benchmark client.
        print("[Benchmark] Running client script...")
        bench_cmd = [
            "python",
            str(BENCHMARK_SCRIPT),
            "--port",
            config["port"],
            "--block-size",
            config["block_size"],
            "--num-prompts",
            config["num_prompts"],
            "--request-rate",
            config["request_rate"],
            "--dataset-type",
            config["trace"],
            # Add any additional benchmark-specific arguments below.
        ]
        if config["trace"] == "burst":
            bench_cmd.extend(["--model-path", config["model"]])
        print(f"[Benchmark] Running command: {' '.join(bench_cmd)}")

        # Capture output instead of writing directly to a file, so we can inspect it in Python.
        process_result = subprocess.run(bench_cmd, capture_output=True, text=True)

        if process_result.returncode == 0:
            print("[Benchmark] Finished successfully.")
            # Use .stdout for parsing
            metrics = parse_benchmark_metrics(process_result.stdout)

            experiment_record["status"] = "success"
            experiment_record["metrics"] = metrics
            # Debug output
            if metrics:
                print(f"   -> Measured Throughput: {metrics.get('request_throughput')}")
        else:
            print(f"[Benchmark] Failed! Return code: {process_result.returncode}")
            experiment_record["status"] = "failed"
            experiment_record["error_log"] = process_result.stderr
            # Optionally print stderr here for debugging
            print(process_result.stderr)

    except Exception as e:
        print(f"[Experiment] Error: {e}")
        experiment_record["status"] = "error"
        experiment_record["error_log"] = str(e)

    finally:
        # 4. Append results in JSON Lines format.
        # This format is ideal for append-only logs because each record is self-contained.
        try:
            with open(OUTPUT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(experiment_record, ensure_ascii=False) + "\n")
            print(f"[Data] Result appended to {OUTPUT_LOG_FILE}")
        except Exception as e:
            print(f"[Data] Failed to write result: {e}")

        # 5. Clean up the server process.
        # Allow a brief pause so GPU memory can be reclaimed.
        print("[Cleanup] Killing vLLM server...")
        try:
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            server_process.wait(timeout=10)
        except:
            print("Force killing...")
            os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)

        time.sleep(5)
        print("[Cleanup] Done. Ready for next run.")


# === Main Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E2E Benchmark Single Experiment")

    # Define all required parameters
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--max-model-len", type=int, default=32768, help="Context length"
    )
    parser.add_argument("--port", type=int, default=9000, help="Service port")
    parser.add_argument(
        "--gpu-id", type=str, default="0", help="GPU ID (e.g., 0 or 0,1)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="FLASH_ATTN",
        dest="ATTENTION_BACKEND",
        help="Attention backend",
    )

    # Benchmark-related parameters.
    parser.add_argument(
        "--request-rate", type=float, required=True, help="Request rate (qps)"
    )
    parser.add_argument("--trace", type=str, default="toolagent", help="Dataset type")
    parser.add_argument("--block-size", type=int, default=512, help="Block size")
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts"
    )

    # Output log parameters.
    parser.add_argument(
        "--output-file", type=str, default=DEFAULT_OUTPUT_LOG, help="Log file path"
    )

    args = parser.parse_args()

    config = vars(args)
    config = {
        k: str(v) for k, v in vars(args).items()
    }  # Ensure values stay string-friendly for env injection.

    # Run the experiment
    run_experiment(config, args.output_file)

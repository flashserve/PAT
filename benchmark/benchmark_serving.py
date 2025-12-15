# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput with Mooncake dataset.

This script is a modification of benchmark_serving.py to use a custom
dataset loader (MooncakeLoader) that generates requests based on
pre-recorded traces, including block IDs for prefix caching simulation.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

On the client side, run:
    python benchmark_serving_mooncake.py \
        --backend vllm \
        --model <your_model> \
        --dataset-type <'conversation' or 'toolagent'> \
        --request-rate <request_rate> \
        --num-prompts <num_prompts> \
        --block-size <block_size>
"""
import argparse
import asyncio
import gc
import json
import os
import random
import time
import numpy as np
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint
from typing import Any, Optional, List, Union, Dict
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from backend_request_func import (ASYNC_REQUEST_FUNCS,
                                  OPENAI_COMPATIBLE_BACKENDS, RequestFuncInput,
                                  RequestFuncOutput)

from benchmark_utils import check_goodput_args
from benchmark_utils import calculate_metrics
from loader import MooncakeLoader, QwenLoader, generate_token_ids, BurstGPTLoader

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser
    

PORT_TO_BACKEND = {
    9000: "PAT",
    9001: "FlashAttntion",
    9002: "FlashInfer",
    9003: "RelayAttntion++",
}

# --- Custom Dataset and Request Definitions ---
# Based on the structure from benchmark_dataset.py and your loader.py

@dataclass
class SampleRequest:
    """
    A sample request to the serving backend.
    This is adapted to allow the prompt to be a list of token IDs.
    """
    # The prompt can be a string or a list of token IDs.
    prompt: Union[str, List[int]]
    prompt_len: int
    expected_output_len: int
    timestemp: int
    multi_modal_data: Optional[Dict[str, Any]] = None



class CustomDataset:
    """ A dataset class that wraps MooncakeLoader | QwenLoader to generate requests with token IDs as prompts. """
    def __init__(
            self,
            dataset_type: str,
            block_size: int,
            max_token_id: int,
            max_context_len: int,
            max_output_len: int,
            extra_prefix_len: int,
            text_only: bool,
            filter_zero: bool,
            input_len: int = None,
            sys_path: str = None,
            model_path: str = None,
    ):
        """
        Args:
            dataset_type: The type of dataset, ['conversation', 'toolagent', 'traceA', 'traceB'].
            block_size: The size of each block for token generation.
            max_token_id: The maximum token ID for dummy token generation.
            text_only: If True, only request with "type"=="text" will be loaded (for QwenLoader only).
        """
        if dataset_type in ['conversation', 'toolagent']:
            self.loader = MooncakeLoader(
                dataset=dataset_type,
                max_context_len=max_context_len - extra_prefix_len,
                filter_zero=filter_zero,
                input_len=input_len,
                block_size=block_size,
            )
        elif dataset_type in ['traceA', 'traceB']:
            # TODO(zhixin): fix the max_context_len filtering in QwenLoader
            self.loader = QwenLoader(dataset=dataset_type, max_context_len=max_context_len, text_only=text_only, input_len=input_len)
        elif dataset_type in ['burst']:
            # TODO(zhixin): fix the max_context_len filtering in BurstGPTLoader
            self.loader = BurstGPTLoader(max_context_len=max_context_len, sys_path=sys_path, model_path=model_path)
        self.block_size = block_size
        self.extra_prefix_len = extra_prefix_len
        self.max_token_id = max_token_id
        self.max_output_len = max_output_len
        print(f"[INFO] dataset metrics {dataset_type} -------------------------")
        pprint(self.loader.get_metrics())
        print("---------------------------------------------------------")

    def sample(self, num_requests: int) -> List[SampleRequest]:
        """Generates a list of sample requests."""
        input_lens, output_lens, block_ids_list, timestamp = self.loader.get_dataset(num_requests)
        requests = []

        # We need to cap the number of requests to the smaller of num_requests and available data
        actual_num_requests = min(num_requests, len(input_lens))

        for i in range(actual_num_requests):
            # Generate dummy token IDs from block IDs.
            # These are not real tokens, but integers that the vLLM OpenAI
            # endpoint can accept as a prompt.
            token_ids = generate_token_ids(block_ids_list[i], self.block_size, self.max_token_id, self.extra_prefix_len)
            prompt_len = len(token_ids)
            requests.append(SampleRequest(
                prompt=token_ids,
                prompt_len=prompt_len,
                expected_output_len=min(output_lens[i], self.max_output_len),
                timestemp=timestamp[i]
            ))

        if len(requests) < num_requests:
            print(f'Warning: only {len(requests)} requests generated, but {num_requests} requested.')

        return requests


# --- Core Benchmark Logic from benchmark_serving.py ---


async def get_request(
        input_requests: list[SampleRequest],
        request_rate: float,
        burstiness: float = 1.0,
        use_realtime: bool = False,
) -> AsyncGenerator[SampleRequest, None]:
    """ Generates requests at a specified rate with burstiness. """
    input_requests_iterator: Iterable[SampleRequest] = iter(input_requests)
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)

    d = []
    if use_realtime is True:
        tim = 0
        for request in input_requests:
            d.append(request.timestemp - tim)
            tim = request.timestemp

    i = 0
    for request in input_requests_iterator:
        yield request
        if request_rate == float("inf"):
            continue
        if use_realtime is True:
            print(d[i] / 1000 * 0.55)
            await asyncio.sleep(d[i] / 1000 * 0.55)
            i += 1
        else:
            interval = np.random.gamma(shape=burstiness, scale=theta)
            # print(interval)
            await asyncio.sleep(interval)


async def benchmark(
        backend: str,
        api_url: str,
        host: str,
        port: int,
        model_id: str,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        input_requests: list[SampleRequest],
        dataset_type: str,
        logprobs: Optional[int],
        request_rate: float,
        burstiness: float,
        disable_tqdm: bool,
        profile: bool,
        selected_percentiles: list[float],
        ignore_eos: bool,
        goodput_config_dict: dict[str, float],
        max_concurrency: Optional[int],
        extra_body: Optional[dict],
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Test the request function with the first input request
    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len = \
        input_requests[0].prompt, input_requests[0].prompt_len, \
            input_requests[0].expected_output_len

    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        ignore_eos=ignore_eos,
        extra_body=extra_body,
    )

    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if profile:
        print("[WARN] Starting profiler...")
        profile_input = RequestFuncInput(model=model_id,
                                         model_name=model_name,
                                         prompt=test_prompt,
                                         api_url=f"http://{host}:{port}/start_profile",
                                         prompt_len=test_prompt_len,
                                         output_len=test_output_len,
                                         logprobs=logprobs,
                                         ignore_eos=ignore_eos,
                                         extra_body=extra_body)
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("[INFO] Profiler started")
        else:
            print("[WARN] Profiler start failed")

    # --------------------------------------------------------------------------------
    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"
    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, burstiness, use_realtime=False):
        prompt, prompt_len, output_len = request.prompt, request.prompt_len, request.expected_output_len
        request_func_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=256,
            logprobs=logprobs,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    # --------------------------------------------------------------------------------

    if profile:
        print("[WARN] Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=f"http://{host}:{port}/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("[INFO] Profiler stopped")
        else:
            print("[WARN] Profiler stop failed")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print(f"{'Backend:':<35} {PORT_TO_BACKEND[port]:<10}")
    print(f"{'Dataset type:':<35} {dataset_type:<10}")
    print(f"{'Request rate (req/s):':<35} {request_rate:<10.2f}")
    print(f"{'Successful requests:':<35} {metrics.completed:<10}")
    print("{s:{c}^{n}}".format(s=' Overall Result ', n=50, c='-'))
    print(f"{'Benchmark duration (s):':<40} {benchmark_duration:<10.2f}")
    print(f"{'Total input tokens:':<40} {metrics.total_input:<10}")
    print(f"{'Total generated tokens:':<40} {metrics.total_output:<10}")
    print(f"{'Request throughput (req/s):':<40} {metrics.request_throughput:<10.2f}")
    if goodput_config_dict:
        print(f"{'Request goodput (req/s):':<40} {metrics.request_goodput:<10.2f}")
    print(f"{'Output token throughput (tok/s):':<40} {metrics.output_throughput:<10.2f}")
    print(f"{'Total Token throughput (tok/s):':<40} {metrics.total_token_throughput:<10.2f}")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    metric_definitions = {
        "ttft": ("TTFT", "Time to First Token"),
        "tpot": ("TPOT", "Time per Output Token (excl. 1st token)"),
        # "itl": ("ITL", "Inter-token Latency"),
        # "e2el": ("E2EL", "End-to-end Latency"),
    }

    for attr, (name, header) in metric_definitions.items():
        print("{s:{c}^{n}}".format(s=header, n=50, c='-'))
        mean_val = getattr(metrics, f"mean_{attr}_ms")
        median_val = getattr(metrics, f"median_{attr}_ms")
        std_val = getattr(metrics, f"std_{attr}_ms")
        percentiles = getattr(metrics, f"percentiles_{attr}_ms")

        print(f"{f'Mean {name} (ms):':<40} {mean_val:<10.2f}")
        print(f"{f'Median {name} (ms):':<40} {median_val:<10.2f}")

        result[f"mean_{attr}_ms"] = mean_val
        result[f"median_{attr}_ms"] = median_val
        result[f"std_{attr}_ms"] = std_val

        for p, value in percentiles:
            p_word = str(int(p)) if int(p) == p else str(p)
            print(f"{f'P{p_word} {name} (ms):':<40} {value:<10.2f}")
            result[f"p{p_word}_{attr}_ms"] = value

    print("=" * 50)
    return result


def main(args: argparse.Namespace):
    # print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name if args.served_model_name is not None else model_id

    api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    print(f"filter zero: {args.dataset_filter_zero}")

    # Load dataset
    dataset = CustomDataset(
        dataset_type=args.dataset_type,
        block_size=args.block_size,
        max_token_id=args.max_token_id,
        max_context_len=args.max_context_len,
        max_output_len=args.max_output_len,
        extra_prefix_len=args.extra_prefix_len,
        text_only=args.dataset_text_only,
        filter_zero=args.dataset_filter_zero,
        input_len=args.input_len,
        sys_path=args.sys_path,
        model_path=args.model_path,
    )
    input_requests = dataset.sample(num_requests=args.num_prompts)

    goodput_config_dict = check_goodput_args(args)

    sampling_params = {
        k: v for k, v in {
            "top_p": args.top_p, "top_k": args.top_k,
            "min_p": args.min_p, "temperature": args.temperature
        }.items() if v is not None
    }
    if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
        raise ValueError("Sampling parameters are only supported by openai-compatible backends.")
    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0

    gc.collect()
    gc.freeze()

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            host=args.host,
            port=args.port,
            model_id=model_id,
            model_name=model_name,
            tokenizer=None,  # [zhixin]: tokenizer is not used in this benchmark (random token IDs instead)
            input_requests=input_requests,
            dataset_type=args.dataset_type,
            logprobs=args.logprobs,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            extra_body=sampling_params,
        )
    )

    if args.save_result:
        result_json: dict[str, Any] = {"date": datetime.now().strftime("%Y%m%d-%H%M%S"), **vars(args),
                                       **benchmark_result}

        # Remove fields with too many data points for summary
        for field in ["input_lens", "output_lens", "ttfts", "itls", "generated_texts", "errors"]:
            if field in result_json:
                del result_json[field]

        file_name = args.result_filename
        if not file_name:
            base_model_id = model_id.split("/")[-1]
            file_name = f"{backend}-{args.request_rate}qps-{base_model_id}.json"

        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)

        with open(file_name, "w", encoding='utf-8') as outfile:
            json.dump(result_json, outfile, indent=4)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark online serving throughput with Mooncake dataset.")

    # Server and Model Arguments ---------------------------------------------------------------------------------------
    parser.add_argument("--backend", type=str, default="vllm", choices=list(ASYNC_REQUEST_FUNCS.keys()))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--endpoint", type=str, default="/v1/completions")
    parser.add_argument("--model", type=str, default="LLM",         # [zhixin]: universal name for eval
                        help="Name of the model.")
    parser.add_argument("--served-model-name", type=str, default=None, help="The model name used in the API.")
    parser.add_argument("--tokenizer", type=str, help="Name or path of the tokenizer.")
    parser.add_argument('--tokenizer-mode', type=str, default="auto", choices=['auto', 'slow'])
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code from huggingface")

    # Sampling and Request Arguments -----------------------------------------------------------------------------------
    parser.add_argument("--logprobs", type=int, default=None, help="Number of logprobs-per-token to compute.")
    parser.add_argument("--ignore-eos", type=bool, default=True,                # [zhixin]: default to True
                        help="Set ignore_eos flag when sending the benchmark request.")
    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=0.0)       # [zhixin]: reproducibility

    # Benchmark Configuration ------------------------------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-tqdm", action="store_true", help="Specify to disable tqdm progress bar.")
    parser.add_argument("--metric-percentiles", type=str, default="99",
                        help="Comma-separated list of percentiles for selected metrics.")
    parser.add_argument("--goodput", nargs="+", required=False, help="Specify service level objectives for goodput.")
    parser.add_argument("--profile", type=bool, default=False,
                        help="Enable profiling of the benchmark run. "
                             "Require launch server with `VLLM_TORCH_PROFILER_DIR=...`")

    # ############################################## FIXED ABOVE #######################################################

    # Output Arguments (benchmark results) -----------------------------------------------------------------------------
    parser.add_argument("--save-result", action="store_true", help="Specify to save benchmark results to a json file")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="Specify directory to save benchmark json results.")
    parser.add_argument("--result-filename", type=str, default=None,
                        help="Specify the filename to save benchmark json results.")

    # Dataset Arguments ================================================================================================
    parser.add_argument("--dataset-type", type=str, default="burst",
                        choices=["conversation", "toolagent", "traceA", "traceB", "burst"],
                        help="Type of the Mooncake/Qwen dataset to benchmark on.")
    parser.add_argument("--block-size", type=int, default=16,  # mooncake: 512 | qwen: 16
                        help="Tokens per block for input generation. (does not affect serving engine)")
    parser.add_argument("--num-prompts", type=int, default=500,
                        help="Number of prompts to process. (start from the first prompt in the dataset)")

    parser.add_argument('--extra-prefix-len', type=int, default=0,
                        help="Add extra prefix length for each request.")

    parser.add_argument("--max-token-id", type=int, default=1000000,
                        help="Maximum token ID for dummy token generation.")
    parser.add_argument("--max-context-len", type=int, default=7800,
                        help="Maximum context length to filter the dataset. (input + output)")
    parser.add_argument("--max-output-len", type=int, default=32768,
                        help="Maximum output length to filter the dataset.")
    parser.add_argument("--dataset-text-only", type=bool, default=True,
                        help="Only load text type request (for Qwen traceA/traceB only).")
    parser.add_argument("--dataset-filter-zero", type=str2bool, default=True,
                        help="Filter requests with non-zero first block (for Mooncake trace only).")
    parser.add_argument("--input-len", type=int, default=None,
                        help="select dataset with input length <= input_len.")

    # Traffic Arguments ================================================================================================
    parser.add_argument("--request-rate", type=float, default=10,     # [zhixin]: QPS float("inf")
                        help="Number of requests per second.")
    parser.add_argument("--burstiness", type=float, default=1.0, help="Burstiness factor of the request generation.")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Maximum number of concurrent requests.")

    parser.add_argument("--sys-path", type=str, default="system_prompt.template", help="add sys prompt to burstGPT trace.")
    parser.add_argument("--model-path", type=str, default="/root/share/models/Qwen3-8B", help="select tokenizers for sys")


    args = parser.parse_args()
    pprint('#'*75 + f"\nRunning benchmark with arguments: {args}\n" + '#'*75)

    if args.port in PORT_TO_BACKEND.keys():
        print(f'[INFO] benchmark serving based on {PORT_TO_BACKEND[args.port]} (port {args.port})')
    else:
        print(f"[ERROR] unknown port {args.port}. Use 9000-9003 for PAT, FA, FI, RA.")

    main(args)

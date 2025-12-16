import os
from copy import deepcopy

import nvtx
import time
import math
import pytest
import torch
import numpy as np
from einops import repeat
import triton

try:
    from vllm.vllm_flash_attn import (
        flash_attn_with_kvcache as vllm_flash_attn_with_kvcache,
    )
except ImportError:
    print("[WARN] vllm_flash_attn_with_kvcache not found, skip related tests.")
    vllm_flash_attn_with_kvcache = None

from prefix_attn import (
    prefix_attn_with_kvcache,
    PackedBox,
    Sequence,
    SeqGroup,
    pack_prefix_blocks,
    pack_without_prefix,
    KernelInfo,
)
from prefix_attn import (
    generate_random_prefix_seqs,
    generate_random_kv_cache,
    generate_tree_seqs,
)
from prefix_attn.calc_theoretical import calc_theoretical
from prefix_attn.data_class import create_seq_group
from prefix_attn.block_scheduler import schedule, schedule_naive
from prefix_attn.utils import make_tensor_with_pad
from prefix_attn.prefix_tree import PrefixTree
from utils import CPUTimer, attention_ref, GPUTimer, Timer
from prefix_attn import PrefixTreeCPP


def tree_benchmark(
    seq_group: SeqGroup,
    num_blocks: int,
    nheads_q,  # [attn]
    nheads_kv,  # [attn]
    head_dim,  # [attn]
    block_size,  # [page]
    n_repeats,
    dtype,
    device,
    seed,
    baselines: list = None,
    enable_timing: bool = False,
):
    torch.cuda.set_device(device)
    torch.random.manual_seed(seed)
    assert head_dim in [64, 128], "head_dim should be 64 or 128 currently"
    assert nheads_q % nheads_kv == 0
    assert enable_timing is False, "timing is deprecated"
    scale = 1.0 / math.sqrt(head_dim)
    max_error = 1e-2

    # ----- step-1: generate random q k_cache v_cache tensors -------------------------------------------------
    q = torch.randn(len(seq_group), 1, nheads_q, head_dim, device=device, dtype=dtype)
    (
        seqlens,
        k_cache,
        v_cache,
        kv_padding_mask,
        block_table,
        k_cache_paged,
        v_cache_paged,
    ) = generate_random_kv_cache(
        seq_group=seq_group,
        num_blocks=num_blocks,
        block_size=block_size,
        nheads_kv=nheads_kv,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        seed=int(time.time()),
    )
    # print(f"seqlens: {seqlens}, k_cache: {k_cache.shape}, v_cache: {v_cache.shape}")
    # print(f"table: {block_table.shape}, k_cache_paged: {k_cache_paged.shape}, v_cache_paged: {v_cache_paged.shape}")

    # check table is unpad
    for i, row in enumerate(block_table.cpu().tolist()):
        if row and len(row) > 1 and row[-1] == 0:
            raise ValueError(
                f"Table is expected to be unpadded, but row {i} has padding 0 at the end."
            )

    # ----- step-2: schedule and transfer results to GPU (count to overhead) ----------------------------------
    theoretical_time = calc_theoretical(block_table, nheads_q, nheads_kv, head_dim)
    # print(f"[INFO] (pytest) (attn) approximate theoretical time: {theoretical_time:.4f}ms")

    _start = time.perf_counter()
    MNWs = None
    table = block_table.cpu()

    _start_build = time.perf_counter()
    seq_lens = seq_group.seqlens
    tree = PrefixTreeCPP(block_size)

    tree.build_radix_tree(seq_lens, table)
    tree.pack_schedule(MNWs, nheads_q // nheads_kv, nheads_kv)
    tree.kernel_info.to_gpu(torch.device(device))

    # ----- step-3: run and compare results -------------------------------------------------------------------
    # prefix-attention
    latencies = ""
    out = torch.empty_like(q, device=device, dtype=dtype)

    def pa_func():
        # print(q.shape, k_cache_paged.shape, v_cache_paged.shape, out.shape)
        prefix_attn_with_kvcache(
            q=q,
            k_cache_paged=k_cache_paged,
            v_cache_paged=v_cache_paged,
            tree=tree,
            softmax_scale=scale,
            out=out,
        )

    latencies += f"[INFO] kernel latency: PAT={Timer(pa_func, n_repeats):.3f}ms"

    # vllm-flash-attention
    if (
        vllm_flash_attn_with_kvcache is not None
        and baselines is not None
        and ("all" in baselines or "vllm-fa" in baselines)
    ):

        def flash_func():
            _ = vllm_flash_attn_with_kvcache(
                q,
                k_cache_paged,
                v_cache_paged,
                cache_seqlens=seqlens,
                block_table=block_table,
                causal=True,
                num_splits=0,  # 0: best split;  1: do not split kv
            )

        out_flash = vllm_flash_attn_with_kvcache(
            q,
            k_cache_paged,
            v_cache_paged,
            cache_seqlens=seqlens,
            block_table=block_table,
            causal=True,
            num_splits=0,  # 0: best split;  1: do not split kv
        )

        print(
            f"[DEBUG] Max diff (PAT - FlashAttention): {(out - out_flash).abs().max().item()}"
        )
        print(
            f"[DEBUG] Mean diff (PAT - FlashAttention): {(out - out_flash).abs().mean().item()}"
        )
        latencies += f"  |  FlashAttention={Timer(flash_func, n_repeats):.3f}ms"
        assert (out - out_flash).abs().max().item() <= max_error, "out_flash error"

    print(latencies)
    print("[INFO] successfully pass the test!")


@pytest.mark.parametrize(
    "tree",
    [
        "1,2,4,16,32,64,128,256,512_3072,1024,32,32,32,32,32,32,32",
        "1,2,4,16,32,64,256_4096,32,128,128,32,256,512",
        "1,2,4,16,32,64,128,256,512_3072,1024,512,256,128,32,256,512,32",
        "1,2,4,16,32,64,128,256,1024_512,32,128,128,32,256,512,32,32",
        "1,2,4,16,32,64,128,512_512,32,128,128,32,256,512,256",
        "1,2,4,16,32,64,128_512,32,128,128,32,256,512",
        "4_4096",
        "4,16_4096,256",
        "1,8_640,32",
        "1,4,16_32,256,32",
        "1,2,4,8,16,32,64_32,512,32,256,32,32,32",
        "1,4,256_32,256,32",
        "1,16,32_2048,1024,256",
        "1,10_4096,416",
    ],
)
@pytest.mark.parametrize("nheads_q,nheads_kv", ([64, 8],))
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("n_repeats", [20])
def test_tree_attn_kvcache(
    nheads_q,  # [attn]
    nheads_kv,  # [attn]
    head_dim,  # [attn]
    block_size,  # [page]
    tree,
    n_repeats,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda:0",
    seed: int = 0,
    baselines=["all"],  # ['all', 'fa', 'vllm-fa', 'ta'], None for none
):
    # os.environ["DISABLE_STREAM"] = "1"
    print(f"[DEBUG] (pytest) (tree): {tree}")

    seq_group, num_blocks = generate_tree_seqs(tree, block_size)

    tree_benchmark(
        seq_group,
        num_blocks,
        nheads_q,
        nheads_kv,
        head_dim,
        block_size,
        n_repeats,
        dtype,
        device,
        seed,
        baselines=baselines,
        enable_timing=False,
    )  # ['pa-naive', 'fa', 'ta']


def test_tree_attn_manual():
    """run a single test with manual sequences (debug)"""
    # seq_lens = [17, 36, 44, 44, 32, 36, 44, 44]
    # block_table = [[1, 2], [1, 2, 3], [4, 5, 6], [4, 5, 7],
    #                [4, 8], [1, 9, 10], [4, 5, 11], [4, 5, 12]]

    seq_lens = [33, 33, 33]
    block_table = [[0, 1, 5], [0, 2, 6], [3, 4, 7]]

    # seq_lens = [38, 38, 15, 15]
    # block_table = [[1, 2, 6], [1, 5, 8], [3, 0, 0], [7, 0, 0]]
    block_size = 16
    num_blocks = 13

    seq_group = create_seq_group(block_table, seq_lens, block_size)

    nheads = 32
    nheads_kv = 8
    head_dim = 128  # 128

    tree_benchmark(
        seq_group,
        num_blocks,
        nheads,
        nheads_kv,
        head_dim,
        block_size,
        n_repeats=20,
        dtype=torch.float16,
        seed=0,
        device="cuda:0",
        baselines=["vllm-fa", "ta", "fa"],
        enable_timing=False,
    )


if __name__ == "__main__":
    # test_tree_attn_manual()
    test_tree_attn_kvcache(
        nheads_q=32,
        nheads_kv=8,
        head_dim=128,
        block_size=32,
        tree="1,256_256,32",
        n_repeats=1,
        dtype=torch.float16,
        device="cuda:0",
        seed=0,
        baselines=["all"],
    )

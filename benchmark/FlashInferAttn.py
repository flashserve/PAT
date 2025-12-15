# version: V2.5
import time
from math import ceil
from typing import List
import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

def get_decode_wrapper(
        seq_lens: List[int],
        block_table: torch.Tensor,
        block_size: int,
        nheads_q: int,
        nheads_kv: int,
        head_dim: int,
        scale: float,
        device: str = "cuda:0",
):
    flashinfer_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )
    # use_tensor_cores (bool) – Whether to use tensor cores for the computation.
    # Will be faster for large group size in grouped query attention. Defaults to False.
    # from its documentation, we only use tensor cores when GQA
    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        flashinfer_workspace_buffer, "NHD", use_tensor_cores=nheads_q > nheads_kv
    )
    kv_indptr = [0]
    kv_last_page_lens = []
    kv_page_indices = []
    table = block_table.cpu().tolist()
    for i, kv_len in enumerate(seq_lens):
        num_block = ceil(kv_len / block_size)
        kv_indptr.append(kv_indptr[i] + num_block)
        kv_last_page_lens.append((kv_len - 1) % block_size + 1)
        kv_page_indices.extend(table[i][:num_block])

    indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=device)
    indices = torch.tensor(kv_page_indices, dtype=torch.int32, device=device)
    last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32, device=device)

    decode_wrapper.plan(indptr,
                        indices,
                        last_page_lens,
                        nheads_q,
                        nheads_kv,
                        head_dim,
                        block_size,
                        sm_scale=scale,
    )

    return decode_wrapper
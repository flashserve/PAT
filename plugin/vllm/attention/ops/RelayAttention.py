from typing import List, Tuple, Union
from vllm.vllm_flash_attn import flash_attn_with_kvcache as vllm_flash_attn_with_kvcache
import numpy as np
import triton
import triton.language as tl
import torch
from torch import Tensor

def relay_fusion(out_sys:Tensor,
                 lse_sys:Tensor,
                 out_usr:Tensor,
                 lse_usr:Tensor) -> Tensor:
    """fusion operation for relay attention

        Args:
            out_sys, out_usr: shape = [num_tokens, num_heads, head_size]
            lse_sys, lse_usr: shape = [num_tokens, num_heads], or [num_heads, num_tokens] if trans_lse_x=True
        Returns:
            shape = [num_tokens, num_heads, head_size]
    """
    assert out_sys.size() == out_usr.size()
    assert out_sys.ndim == 3

    out = _relay_fuse_triton(out_sys, lse_sys, out_usr, lse_usr)
    return out


def _relay_fuse_triton(out_sys: Tensor, lse_sys: Tensor, out_usr: Tensor, lse_usr: Tensor):
    # it will be more effeicient to let the final dim contiguous
    assert out_sys.stride(-1) == 1
    assert out_usr.stride(-1) == 1
    out = torch.empty_like(out_sys)

    num_tokens, num_heads, head_size = out_sys.size()
    lse_sys_stride_t, lse_sys_stride_h = lse_sys.stride()
    lse_usr_stride_t, lse_usr_stride_h = lse_usr.stride()

    BLOCK_SIZE = triton.next_power_of_2(head_size)
    num_warps = 4

    _relay_fuse_kernel[(num_tokens, num_heads)](
        out_fused_ptr=out,
        out_sys_ptr=out_sys,
        lse_sys_ptr=lse_sys,
        out_usr_ptr=out_usr,
        lse_usr_ptr=lse_usr,
        head_size=head_size,
        lse_sys_stride_t=lse_sys_stride_t,
        lse_sys_stride_h=lse_sys_stride_h,
        lse_usr_stride_t=lse_usr_stride_t,
        lse_usr_stride_h=lse_usr_stride_h,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


@triton.jit
def _relay_fuse_kernel(
        out_fused_ptr,  # final output
        out_sys_ptr, lse_sys_ptr,
        out_usr_ptr, lse_usr_ptr,
        head_size,
        lse_sys_stride_t, lse_sys_stride_h,
        lse_usr_stride_t, lse_usr_stride_h,
        BLOCK_SIZE: tl.constexpr):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    lse_sys = tl.load(lse_sys_ptr +
                      token_id * lse_sys_stride_t +
                      head_id * lse_sys_stride_h).to(tl.float32)
    lse_usr = tl.load(lse_usr_ptr +
                      token_id * lse_usr_stride_t +
                      head_id * lse_usr_stride_h).to(tl.float32)
    rescale_sys = 1. / (1 + tl.exp(lse_usr - lse_sys))
    rescale_usr = 1. - rescale_sys
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    all_head_id = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    head_offs = tl.arange(0, BLOCK_SIZE)
    io_mask = head_offs < head_size
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    out_sys = tl.load(out_sys_ptr + all_head_id * head_size + head_offs,
                      mask=io_mask, other=0.)
    out_usr = tl.load(out_usr_ptr + all_head_id * head_size + head_offs,
                      mask=io_mask, other=0.)
    out_fused = rescale_sys * out_sys + rescale_usr * out_usr
    # save to output tensor
    tl.store(out_fused_ptr + all_head_id * head_size + head_offs,
             out_fused, mask=io_mask)


def RelayAttentionPlus(
        q: Tensor,
        k_cache_paged: Tensor,
        v_cache_paged: Tensor,
        seqlens: List[Tensor],
        block_tables: List[Tensor], # [table_usr, table_sys]
) -> Tensor:
    out_usr, lse_usr = vllm_flash_attn_with_kvcache(
        q,
        k_cache_paged,
        v_cache_paged,
        cache_seqlens=seqlens[0],
        block_table=block_tables[0],
        causal=True,
        num_splits=0,
        return_softmax_lse=True,
    )
    if len(seqlens) == 1:
        return out_usr.squeeze(1)
    out_sys, lse_sys = vllm_flash_attn_with_kvcache(
        q.view(1, -1, q.size(2), q.size(3)),
        k_cache_paged,
        v_cache_paged,
        cache_seqlens=seqlens[1],
        block_table=block_tables[1],
        num_splits=0,
        return_softmax_lse=True,
    )
    out = relay_fusion(
        out_sys.squeeze(0), lse_sys.squeeze(0).transpose(0, 1),
        out_usr.squeeze(1), lse_usr.squeeze(2),
    )
    return out.squeeze(1)

def RelayAttention(
        q: Tensor,
        k_cache: List[Tensor],
        v_cache: List[Tensor],
        seqlens: List[Tensor], # [table_usr, table_sys]
) -> Tensor:
    out_usr, lse_usr = vllm_flash_attn_with_kvcache(
        q,
        k_cache[0],
        v_cache[0],
        cache_seqlens=seqlens[0],
        causal=True,
        num_splits=0,
        return_softmax_lse=True,
    )
    if len(seqlens) == 1:
        return out_usr
    out_sys, lse_sys = vllm_flash_attn_with_kvcache(
        q.view(1, -1, q.size(2), q.size(3)),
        k_cache[1],
        v_cache[1],
        cache_seqlens=seqlens[1],
        num_splits=0,
        return_softmax_lse=True,
    )
    out = relay_fusion(
        out_sys.squeeze(0), lse_sys.squeeze(0).transpose(0, 1),
        out_usr.squeeze(1), lse_usr.squeeze(2),
    )
    return out

def schedule_single_sys(
        block_table: Tensor,
        seq_lens: List[int],
        block_size: int,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.int32,
        only_sys: bool = False,
) -> Union[int, Tuple[List[Tensor], List[Tensor]]]:
    t = block_table.cpu().numpy()
    seq_lens = np.array(seq_lens)

    if t.shape[0] < 2 or t[0, 0] != t[1, 0]:
        sys_blocks = 0
    else:
        if not np.all((t[:, 0] == t[0, 0])):
            raise ValueError(f"Table {t.tolist()} is not a valid table, this benchmark can only handler one sys")

        first_row = t[0:1]  # [1, M]
        mask = (t == first_row).all(axis=0)

        if mask.all():
            sys_blocks = mask.shape[0]
        else:
            first_false = np.where(~mask)[0]
            sys_blocks = first_false[0].item()

    # it's the main overhead
    seqlen4ra = [torch.tensor(seq_lens - sys_blocks * block_size, device=device, dtype=dtype)]
    if sys_blocks > 0:
        seqlen4ra.append(torch.tensor([sys_blocks * block_size], device=device, dtype=dtype))

    if only_sys:
        return sys_blocks * block_size, seqlen4ra

    block_table4ra = [block_table[:, sys_blocks:]]
    if sys_blocks > 0:
        block_table4ra.append(block_table[:1, :sys_blocks])

    return seqlen4ra, block_table4ra
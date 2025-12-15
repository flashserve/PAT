# prefix_attn/_prefix_attn.pyi

from torch import Tensor
import torch
from typing import List, Union, Optional, overload
from prefix_attn import PrefixTree
from numpy.typing import NDArray
import numpy as np

@overload
def prefix_attn_with_kvcache(
    q: Tensor,
    k_cache_paged: Tensor,
    v_cache_paged: Tensor,
    num_split_per_seq: Tensor,
    query_tables: List[Tensor],
    block_tables: List[Tensor],
    num_seqs_per_CTAs: List[Tensor],
    CTA_ranks: List[Tensor],
    kv_in_CTAs: List[Tensor],
    MNW: List[List[int]],
    max_split_per_seq: int,
    max_seqs_in_CTA: int,
    max_blocks_in_CTA: int,
    softmax_scale: float,
    out: Tensor,
    timing_: Optional[Tensor] = None
): ...

@overload
def prefix_attn_with_kvcache(
    q: torch.Tensor,
    k_cache_paged: torch.Tensor,
    v_cache_paged: torch.Tensor,
    tree: PrefixTreeCPP,
    softmax_scale: float,
    out: torch.Tensor,
    timing_: Optional[torch.Tensor] = None
) -> None: ...

def build_radix_tree_cpp(
    self: PrefixTree,
    seq_lens_np: NDArray[np.int32],
    flat_blocks_np: NDArray[np.int32],
    offsets_np: NDArray[np.int32]
) -> None: ...


class KernelInfo:
    def to_gpu(self, device: Union[torch.device, str]) -> None: ...

class PrefixTreeCPP:
    kernel_info: KernelInfo

    def __init__(self, block_size: int) -> None: ...

    def build_radix_tree(
        self,
        seq_lens: Union[List[int], Tensor],
        block_table: Tensor
    ) -> None: ...

    def pack_schedule(
        self,
        MNWs: Optional[List[List[int]]] = None,
        HRatio: int = 1,
        kvHead: int = 8,
        use_compute_model: bool = False
    ) -> None: ...
from typing import Tuple, TypeVar, Optional, Union
import random
import math
import numpy as np
import torch
from einops import rearrange
from .data_class import SeqGroup


def generate_random_kv_cache(
        seq_group: SeqGroup,
        num_blocks: int,
        block_size: int,
        nheads_kv: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate k_cache and v_cache for each sequence in seq_group.
    Return:
        seqlens: (batch_size,)
        k_cache: (batch_size, max_seqlen, nheads_kv, head_dim)  (rightpad)
        v_cache: (batch_size, max_seqlen, nheads_kv, head_dim)  (rightpad)
        kv_padding_mask: (batch_size, max_seqlen)
        block_table: (batch_size, num_blocks)
        k_cache_paged: (num_blocks, block_size, nheads_kv, head_dim)
        v_cache_paged: (num_blocks, block_size, nheads_kv, head_dim)
    """
    torch.random.manual_seed(seed)
    k_cache_paged = torch.randn(num_blocks, block_size, nheads_kv, head_dim, dtype=dtype, device=device)
    v_cache_paged = torch.randn(num_blocks, block_size, nheads_kv, head_dim, dtype=dtype, device=device)
    block_table = torch.tensor(seq_group.get_padded_block_ids(padding_value=0), dtype=torch.int32, device=device)
    k_cache = k_cache_paged[block_table]  # (batch_size, num_blocks, block_size, nheads_kv, head_dim)
    v_cache = v_cache_paged[block_table]  # (batch_size, num_blocks, block_size, nheads_kv, head_dim)
    k_cache = rearrange(k_cache, "b num_blocks block_size n d -> b (num_blocks block_size) n d")
    v_cache = rearrange(v_cache, "b num_blocks block_size n d -> b (num_blocks block_size) n d")

    seqlens = torch.tensor(seq_group.seqlens, dtype=torch.int32, device=device)
    # do not use seq_group.max_seq_len here, since it may not be divided by block_size
    max_seqlen = seq_group.max_num_blocks * block_size
    arange = rearrange(torch.arange(max_seqlen, device=device), "s -> 1 s")
    kv_padding_mask = arange < rearrange(seqlens, "b -> b 1")

    return seqlens, k_cache, v_cache, kv_padding_mask, block_table, k_cache_paged, v_cache_paged


def generate_tree_seqs(
        tree: str,
        block_size: int,
) -> Tuple[SeqGroup, int]:
    """
    Generate sequences from a tree structure. Return SeqGroup and total_blocks.
    The tree is like “1,2,4,8_64,256,512,32”.
    The left numbers show nodes per level: 1 node at level 1, 2 nodes at level 2, ..., the last number is batch_size.
    The right numbers show tokens per node: 64 at level 1, 256 for each at level 2, 512 for each at level 3



    """
    p1, p2 = tree.split('_')
    p1 = list(map(int, p1.split(',')))
    p2 = list(map(int, p2.split(',')))

    assert min(p2) >= block_size, "to simplify, we assume all tokens in each level are larger than block_size"
    assert all(x % block_size == 0 for x in p2), "all tokens in each level should be multiples of block_size"

    batch = p1[-1]
    length = sum(p2)
    seq_group = SeqGroup(block_size, min_seqlen=length, max_seqlen=length, num_seqs=batch)

    block_per_level = []
    total_blocks = 0
    for level in range(len(p1)):
        nodes = []
        for branch in range(p1[level]):
            ids = []
            blocks = p2[level] // block_size
            for block in range(blocks):
                ids.append(total_blocks)
                total_blocks += 1
            nodes.append(ids)
        block_per_level.append(nodes)

    # print(block_per_level)

    rand_seq = list(range(batch))
    random.shuffle(rand_seq)
    for i in range(batch):
        blocks = []
        for level in range(len(p1)):
            block_ids = block_per_level[level]
            ids = block_ids[i // (batch // p1[level])]
            blocks.extend(ids)
        seq_group.sequences[rand_seq[i]].append_blocks(blocks, 0)

    # print(seq_group)

    return seq_group, total_blocks




def generate_random_prefix_seqs(
        min_seqlen: int,
        max_seqlen: int,
        batch_size: int,
        num_prefix: int,  # number of random prefixes
        prefix_ratio: float,  # ratio of common+random prefixes to max_blocks
        block_size: int,
        seed: int = 0
) -> Tuple[SeqGroup, int]:


    """
    Generate sequences with random common prefix. Return SeqGroup and total_blocks.
    note that total number of random prefix may less than num_prefix

    seq: [common_prefix, random_prefix, individual_suffix]
    SeqGroup: [
      [A, B, C,  D, E, F,  N, O, P],
      [A, B, C,  D, E, F,  Q],
      [A, B, C,  H, I, J,  R, S, T],
      [A, B, C,  H, I, J,  U],
      [A, B, C,  H, I, J,  V, W, X, Y],
      [A, B, C,  K, L, M,  Z]
    ]

    max_blocks = ceil(max_seqlen / block_size)
    min_blocks = max{ ceil(max_blocks * prefix_ratio), ceil(min_seqlen / block_size) }
    [A-C]:  common prefix           length = ceil(max_blocks * prefix_ratio * 0.5)
    [D-M]:  random prefixes         length = floor(max_blocks * prefix_ratio * 0.5), num_types = num_prefix
    [N-Z]:  individual suffixes     length = randint(min_blocks, max_blocks) - len(common_prefix) - len(random_prefix)
    """
    # TODO(Zhixin): support seqlen % block_size != 0 (arbitrary seqlen)
    assert num_prefix <= batch_size
    random.seed(seed)
    total_blocks = 0
    max_blocks = math.ceil(max_seqlen / block_size)
    min_blocks = max(math.ceil(max_blocks * prefix_ratio), math.ceil(min_seqlen / block_size))
    common_prefix_len = math.ceil(max_blocks * prefix_ratio)

    seq_group = SeqGroup(block_size, min_seqlen, max_seqlen, batch_size)

    # step-1: generate common prefix blocks
    common_blocks = random.sample(range(0, common_prefix_len), common_prefix_len)
    seq_group.set_common_prefix_blocks(common_blocks)
    total_blocks += len(common_blocks)

    for i in range(batch_size):
        # generate random suffix blocks
        suffix_len = 0
        if seq_group.sequences[i].seqlen < block_size * common_prefix_len:
            suffix_len = seq_group.sequences[i].seqlen
        else:
            suffix_len = seq_group.sequences[i].seqlen - common_prefix_len * block_size
        if suffix_len > 0:
            suffix_blocks = math.ceil(suffix_len / block_size)
            suffix_blocks = random.sample(range(total_blocks, total_blocks + suffix_blocks), suffix_blocks)
            seq_group.sequences[i].append_blocks(suffix_blocks, 0)
            total_blocks += len(suffix_blocks)

    return seq_group, total_blocks


# from vllm/vllm/utils
T = TypeVar("T")

TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}


def make_ndarray_with_pad(
        x: list[list[T]],
        pad: T, dtype,
        max_len:
        Optional[int] = None
) -> np.ndarray:
    if max_len is None:
        # map is faster than a genexpr over `len`
        max_len = max(map(len, x), default=0)

    padded_x = np.full((len(x), max_len), pad, dtype=dtype)
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len
        padded_x[ind, :len(blocktb)] = blocktb

    return padded_x


def make_tensor_with_pad(
    x: list[list],
    pad: T,
    dtype: torch.dtype,
    max_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor
import torch
import numpy as np
from typing import List, Tuple, Dict
from .data_class import PackedBox, Sequence, SeqGroup, KernelInfo
from .prefix_tree import PrefixTree
from .utils import make_ndarray_with_pad


def pack_prefix_blocks(seq_group: SeqGroup, block_size: int, k: int = 64) -> PackedBox:
    # Fixme(zhixin): remove default value of k, since GQA requires k=64//(nheads//nheads_kv)
    """
    identify multi-level prefix blocks & pack the queries and kv into CTA with common prefix blocks

    :param seq_group: SeqGroup
    :param block_size: block size
    :param k: max number of sequences in a CTA

    ## example seq_group (block_size=b):
    [
      [A, B, C,  D, E, F,  N, O, P],
      [A, B, C,  D, E, F,  Q],
      [A, B, C,  H, I, J,  R, S, T],
      [A, B, C,  H, I, J,  U],
      [A, B, C,  H, I, J,  V, W, X, Y],
      [A, B, C,  K, L, M,  Z]
    ]

    ## example output (10 CTAs):
    PackedBox {
        q_table=[0, 0, 2, 5, 0, 1, 2, 3, 4, 5],             (CTAs,)
        block_table=[                                       (CTSs, ...)
            [A, B, C], [D, E, F], [H, I, J],
            [K, L, M], [N, O, P], [Q], [R, S, T],
            [U], [V, W, X, Y], [Z]],
        num_seqs_per_CTA=[6, 2, 3, 1, 1, 1, 1, 1, 1, 1],    (CTAs,)
        num_split_per_seq=[3, 3, 3, 3, 3, 3],               (num_sequences,)
        CTA_rank=[0, 1, 1, 1, 2, 2, 2, 2, 2, 2],            (CTAs,)
        kv_in_CTA=[3b, 3b, 3b, 3b, 3b, b, 3b, b, 4b, b],    (CTAs,)
        max_split_per_seq=3,                                (num_sequences,)
        max_seqs_in_CTA=6,                                  (num_sequences,)
        max_blocks_in_CTA=3,                                (num_sequences,)
    }
    """
    assert seq_group.is_sorted, "SeqGroup must be sorted"

    # Fixme(Jinjun): Currently, the maximum number of sequences in a CTA is hard-coded to 64 (M64N256).
    #  However, the optimal configuration may vary across different hardware.
    assert k <= 64, "max number of sequences in a CTA should be less than 64"

    data = seq_group.sequences
    n: int = len(data)
    # Pointers into each sequence's block_ids
    pointers: List[int] = [0] * n
    # Track how many query boxes have been created per sequence
    split_per_seq: List[int] = [0] * n

    # Initialize PackedBox
    box = PackedBox(max_seqs_in_CTA=1, max_blocks_in_CTA=0, max_split_per_seq=0)

    def unfinished() -> bool:
        """Check if any sequence still has blocks left to process."""
        return any(pointers[i] < len(data[i].block_ids) for i in range(n))

    # Main packing loop
    while unfinished():
        # Group sequences by their current block ID, splitting into runs of consecutive seq IDs
        groups: Dict[int, List[List[int]]] = {}
        for i in range(n):
            if pointers[i] < len(data[i].block_ids):
                bid = data[i].block_ids[pointers[i]]
                # Ensure we have a list of runs for this bid
                runs = groups.setdefault(bid, [[]])
                last_run = runs[-1]
                # If this is the first in the run or directly follows the previous ID, append;
                # otherwise start a new run
                if not last_run or i == last_run[-1] + 1:
                    last_run.append(i)
                else:
                    runs.append([i])

        for bid, runs in groups.items():
            # Process each block-ID group
            for seqs in runs:
                # Single-sequence group: pack all remaining blocks at once
                if len(seqs) == 1:
                    seq_id = seqs[0]

                    box.q_table.append(seq_id)
                    box.CTA_rank.append(split_per_seq[seq_id])
                    box.kv_in_CTA.append(data[seq_id].seqlen - pointers[seq_id] * block_size)

                    blocks = data[seq_id].block_ids[pointers[seq_id]:]  # shallow copy (slice)
                    box.block_table.append(blocks)

                    box.num_seqs_per_CTA.append(1)
                    box.max_blocks_in_CTA = max(box.max_blocks_in_CTA, len(blocks))
                    pointers[seq_id] = len(data[seq_id].block_ids)
                    split_per_seq[seq_id] += 1

                # Multi-sequence group: chunk into CTAs of up to k sequences
                else:
                    for start in range(0, len(seqs), k):
                        chosen = seqs[start:start + k]
                        common_blocks: List[int] = []
                        to_end = False
                        # Extract longest common prefix of blocks across chosen sequences
                        while True:
                            # If any sequence is done, break
                            if any(pointers[p] >= len(data[p].block_ids) for p in chosen):
                                to_end = True
                                break
                            curr = data[chosen[0]].block_ids[pointers[chosen[0]]]
                            # Check all sequences have same current block
                            if not all(
                                    pointers[p] < len(data[p].block_ids) and
                                    data[p].block_ids[pointers[p]] == curr
                                    for p in chosen
                            ):
                                break
                            common_blocks.append(curr)
                            for p in chosen:
                                pointers[p] += 1

                        if common_blocks:
                            box.max_seqs_in_CTA = max(box.max_seqs_in_CTA, len(chosen))
                            box.q_table.append(chosen[0])
                            box.block_table.append(common_blocks.copy())
                            box.num_seqs_per_CTA.append(len(chosen))
                            box.CTA_rank.append(split_per_seq[chosen[0]])
                            # Compute kv entries in this CTA
                            if to_end:
                                last_len = len(common_blocks)
                                total_blocks = len(data[chosen[0]].block_ids)
                                kv_val = (
                                        block_size * (last_len - 1)
                                        + data[chosen[0]].seqlen
                                        - (total_blocks - 1) * block_size
                                )
                            else:
                                kv_val = block_size * len(common_blocks)
                            box.kv_in_CTA.append(kv_val)
                            box.max_blocks_in_CTA = max(box.max_blocks_in_CTA, len(common_blocks))
                            # Increment split count for each sequence
                            for p in chosen:
                                split_per_seq[p] += 1

    # Finalize per-sequence split counts
    box.num_split_per_seq = split_per_seq.copy()
    box.max_split_per_seq = max(split_per_seq)
    box.sort_index = np.arange(n)
    box.reverse_index = np.arange(n)
    return box


def pack_without_prefix(seq_group: SeqGroup, block_size: int) -> PackedBox:
    """
    pack each sequence into a CTA with no common prefix blocks

    :param seq_group: SeqGroup
    :param block_size: block size

    ## example seq_group (block_size=b):
    [
      [A, B, C,  D, E, F,  N, O, P],
      [A, B, C,  D, E, F,  Q],
      [A, B, C,  H, I, J,  R, S, T],
      [A, B, C,  H, I, J,  U],
      [A, B, C,  H, I, J,  V, W, X, Y],
      [A, B, C,  K, L, M,  Z]
    ]

    ## example output (6 CTAs):
    PackedBox {
        q_table=[0, 1, 2, 3, 4, 5],                         (CTAs,)
        block_table=[                                       (CTSs, ...)
            [A, B, C,  D, E, F,  N, O, P],
            [A, B, C,  D, E, F,  Q],
            [A, B, C,  H, I, J,  R, S, T],
            [A, B, C,  H, I, J,  U],
            [A, B, C,  H, I, J,  V, W, X, Y],
            [A, B, C,  K, L, M,  Z]],
        num_seqs_per_CTA=[1, 1, 1, 1, 1, 1],                (CTAs,)
        num_split_per_seq=[1, 1, 1, 1, 1, 1],               (num_sequences,)
        CTA_rank=[0, 0, 0, 0, 0, 0],                        (CTAs,)
        kv_in_CTA=[9b, 7b, 9b, 7b, 11b, 7b],                (CTAs,)
        max_split_per_seq=1,                                (num_sequences,)
        max_seqs_in_CTA=1,                                  (num_sequences,)
        max_blocks_in_CTA=11,                               (num_sequences,)
    }
    """
    n = len(seq_group)
    return PackedBox(
        q_table=list(range(n)),
        block_table=seq_group.get_block_ids(),
        num_seqs_per_CTA=[1] * n,
        num_split_per_seq=[1] * n,
        CTA_rank=[0] * n,
        kv_in_CTA=[seq.seqlen for seq in seq_group.sequences],
        max_split_per_seq=1,
        max_seqs_in_CTA=1,
        max_blocks_in_CTA=seq_group.max_num_blocks,
    )


def distribute_naive(
        kernel_information: KernelInfo,
        Box: PackedBox,
        h_q_kv_ratio: int = 1,
        qkv_tile_types: List[List[int]] = None,  # Fixme: 实现一个简单的方案，可以做复杂。目前这个简单的方案要求qkv_tile按照从大到小排序
        d: int = 128                             # Fixme: 简单实现，按理来说，qkv_tile_types的设置应该考虑到d的值，所以这个参数应该去掉
):
    if qkv_tile_types is None:
        qkv_tile_types = [[64, 128, 4]] if d == 128 else [[64, 256, 4]]
    CTAs = len(Box.q_table)

    temp_q_tables = [[] for _ in range(len(qkv_tile_types))]
    temp_block_tables = [[] for _ in range(len(qkv_tile_types))]
    temp_num_seqs_per_CTAs = [[] for _ in range(len(qkv_tile_types))]
    temp_CTA_ranks = [[] for _ in range(len(qkv_tile_types))]
    temp_kv_in_CTAs = [[] for _ in range(len(qkv_tile_types))]

    for i in range(CTAs):
        seq_in_CTA = Box.num_seqs_per_CTA[i]
        for idx, qkv_tile in enumerate(qkv_tile_types):
            tile_q, tile_kv, warps = qkv_tile
            if seq_in_CTA * h_q_kv_ratio <= tile_q:
                temp_q_tables[idx].append(Box.q_table[i])
                temp_block_tables[idx].append(Box.block_table[i])
                temp_num_seqs_per_CTAs[idx].append(Box.num_seqs_per_CTA[i])
                temp_CTA_ranks[idx].append(Box.CTA_rank[i])
                temp_kv_in_CTAs[idx].append(Box.kv_in_CTA[i])
                break

    for idx in range(len(qkv_tile_types)):
        if len(temp_q_tables[idx]) == 0:
            continue
        kernel_information.q_tables.append(temp_q_tables[idx])
        kernel_information.block_tables.append(temp_block_tables[idx])
        kernel_information.num_seqs_per_CTAs.append(temp_num_seqs_per_CTAs[idx])
        kernel_information.CTA_ranks.append(temp_CTA_ranks[idx])
        kernel_information.kv_in_CTAs.append(temp_kv_in_CTAs[idx])
        kernel_information.MNWs.append(qkv_tile_types[idx])

    return

def distribute_multi_stream(
        kernel_information: KernelInfo,
        Box: PackedBox,
        h_q_kv_ratio: int = 1,
        qkv_tile_types: List[List[int]] = None,  # Fixme: 实现一个简单的方案，可以做复杂。目前这个简单的方案要求qkv_tile按照从大到小排序
        d: int = 128                             # Fixme: 简单实现，按理来说，qkv_tile_types的设置应该考虑到d的值，所以这个参数应该去掉
):
    if qkv_tile_types is None:
        qkv_tile_types = [[64, 128, 4]] if d == 128 else [[64, 256, 4]]
    CTAs = len(Box.q_table)

    temp_q_tables = [[] for _ in range(len(qkv_tile_types))]
    temp_block_tables = [[] for _ in range(len(qkv_tile_types))]
    temp_num_seqs_per_CTAs = [[] for _ in range(len(qkv_tile_types))]
    temp_CTA_ranks = [[] for _ in range(len(qkv_tile_types))]
    temp_kv_in_CTAs = [[] for _ in range(len(qkv_tile_types))]

    def add_CTA(id: int, boxId: int):
        temp_q_tables[id].append(Box.q_table[boxId])
        temp_block_tables[id].append(Box.block_table[boxId])
        temp_num_seqs_per_CTAs[id].append(Box.num_seqs_per_CTA[boxId])
        temp_CTA_ranks[id].append(Box.CTA_rank[boxId])
        temp_kv_in_CTAs[id].append(Box.kv_in_CTA[boxId])

    # for d128, the qkv_tile_types is
    # [[128, 16, 4], [64, 128, 4], [64, 64, 4], [64, 32, 4], [64, 16, 4], [32, 32, 2], [32, 16, 2], [16, 16, 1]]
    def distribute_box(st: int, ed: int):
        for i in range(st, ed):
            seq_in_CTA = Box.num_seqs_per_CTA[i]
            kv_in_CTA = Box.kv_in_CTA[i]
            seq_in_CTA = seq_in_CTA * h_q_kv_ratio
            if seq_in_CTA <= 16:
                add_CTA(7, i)
            elif seq_in_CTA <= 32:
                match kv_in_CTA:
                    case _ if kv_in_CTA <= 16:
                        add_CTA(6, i)
                    case _:
                        add_CTA(5, i)
            elif seq_in_CTA <= 64:
                match kv_in_CTA:
                    case _ if kv_in_CTA <= 16:
                        add_CTA(4, i)
                    case _ if kv_in_CTA <= 32:
                        add_CTA(3, i)
                    case _ if kv_in_CTA <= 64:
                        add_CTA(2, i)
                    case _:
                        add_CTA(1, i)
            elif seq_in_CTA <= 128:
                if kv_in_CTA <= 16:
                    add_CTA(0, i)
                else:
                    # 将分裂的两块添加到末尾
                    # 第一块
                    Box.q_table.append(Box.q_table[i])
                    Box.block_table.append(Box.block_table[i])
                    Box.num_seqs_per_CTA.append(64 // h_q_kv_ratio)
                    Box.CTA_rank.append(Box.CTA_rank[i])
                    Box.kv_in_CTA.append(kv_in_CTA)
                    # 第二块
                    res_seqs = (seq_in_CTA - 64) // h_q_kv_ratio
                    Box.q_table.append(Box.q_table[i] + 64 // h_q_kv_ratio)
                    Box.block_table.append(Box.block_table[i])
                    Box.num_seqs_per_CTA.append(res_seqs)
                    Box.CTA_rank.append(Box.CTA_rank[i])
                    Box.kv_in_CTA.append(kv_in_CTA)

    CTAs = len(Box.q_table)
    distribute_box(0, CTAs)
    distribute_box(CTAs, len(Box.q_table))

    for idx in range(len(qkv_tile_types)):
        if len(temp_q_tables[idx]) == 0:
            continue
        kernel_information.q_tables.append(temp_q_tables[idx])
        kernel_information.block_tables.append(temp_block_tables[idx])
        kernel_information.num_seqs_per_CTAs.append(temp_num_seqs_per_CTAs[idx])
        kernel_information.CTA_ranks.append(temp_CTA_ranks[idx])
        kernel_information.kv_in_CTAs.append(temp_kv_in_CTAs[idx])
        kernel_information.MNWs.append(qkv_tile_types[idx])

    return


def pack_schedule(
        seq_lens: SeqGroup,
        block_table: torch.Tensor,
        block_size: int,
        MNWs: List[List[int]] = None,
        h_q_kv_ratio: int = 1,
) -> KernelInfo:
    table = block_table.cpu().numpy()
    batch = len(seq_lens)
    table = [table[i][:(seq_lens[i] + block_size - 1) // block_size].tolist() for i in range(len(seq_lens))]
    # print(table)

    tree = PrefixTree(block_size)
    seq_idxs = list(range(batch))
    tree.build_tree(seq_idxs, table)


def schedule(
        seq_lens: List[int],
        block_table: List[List[int]],
        block_size: int,
        k: int,
        qkv_tile_types: List[List[int]],
        h_q_kv_ratio: int = 1,
) -> KernelInfo:
    # TODO(Zhixin): remove the `pack_prefix_blocks` and `pack_without_prefix` function
    """
    Identify multi-level prefix blocks & pack the queries and kv into CTA with common prefix blocks

    This function implement a sort-free prefix packing algorithm, the block_table does not need to be
    sorted before calling this function. It's useful for vLLM integration.

    Returned reverse_index and sort_index are used to reorder the input/output in vLLM.
    """

    # Fixme(Jinjun): Currently, the maximum number of sequences in a CTA is hard-coded to 64 (M64N256).
    #  However, the optimal configuration may vary across different hardware.
    # assert k <= 64, "max number of sequences in a CTA should be less than 64"
    if qkv_tile_types is None:
        qkv_tile_types = [[64, 128, 4]]

    n = len(seq_lens)

    # sort the block table by block ids to group sequences with common blocks.
    # note that change sequence order affects 'q_table' and 'num_split_per_seq'.
    # table = block_table.cpu().numpy()
    table = make_ndarray_with_pad(block_table, 0, np.int32)
    sort_idx_cpu = np.lexsort(table[:, ::-1].T)

    seq_lens = [seq_lens[_] for _ in sort_idx_cpu]
    table = table[sort_idx_cpu]
    # Unpad the block table
    table = [table[i][:(seq_lens[i] + block_size - 1) // block_size].tolist() for i in range(len(seq_lens))]
    # print(table)

    # Pointers into each sequence's block_ids
    pointers: List[int] = [0] * n
    # Track how many query boxes have been created per sequence
    split_per_seq: List[int] = [0] * n

    max_seqs_in_CTA = 1
    max_blocks_in_CTA = 0
    # Initialize PackedBox
    box = PackedBox()

    def unfinished() -> bool:
        """Check if any sequence still has blocks left to process."""
        return any(pointers[i] < len(table[i]) for i in range(n))

    # Main packing loop
    while unfinished():
        # Group sequences by their current block ID, splitting into runs of consecutive seq IDs
        groups: Dict[int, List[List[int]]] = {}
        for i in range(n):
            if pointers[i] < len(table[i]):
                bid = table[i][pointers[i]]
                # Ensure we have a list of runs for this bid
                runs = groups.setdefault(bid, [[]])
                last_run = runs[-1]
                # If this is the first in the run or directly follows the previous ID, append;
                # otherwise start a new run
                if not last_run or i == last_run[-1] + 1:
                    last_run.append(i)
                else:
                    runs.append([i])

        for bid, runs in groups.items():
            # Process each block-ID group
            for seqs in runs:
                # Single-sequence group: pack all remaining blocks at once
                if len(seqs) == 1:
                    seq_id = seqs[0]

                    box.q_table.append([seq_id])
                    box.CTA_rank.append(split_per_seq[seq_id])
                    box.kv_in_CTA.append(seq_lens[seq_id] - pointers[seq_id] * block_size)

                    blocks = table[seq_id][pointers[seq_id]:]  # shallow copy (slice)
                    box.block_table.append(blocks)

                    box.num_seqs_per_CTA.append(1)
                    max_blocks_in_CTA = max(max_blocks_in_CTA, len(blocks))
                    pointers[seq_id] = len(table[seq_id])
                    split_per_seq[seq_id] += 1

                # Multi-sequence group: chunk into CTAs of up to k sequences
                else:
                    for start in range(0, len(seqs), k):
                        chosen = seqs[start:start + k]
                        common_blocks: List[int] = []
                        to_end = False
                        # Extract longest common prefix of blocks across chosen sequences
                        while True:
                            # If any sequence is done, break
                            if any(pointers[p] >= len(table[p]) for p in chosen):
                                to_end = True
                                break
                            curr = table[chosen[0]][pointers[chosen[0]]]
                            # Check all sequences have same current block
                            if not all(
                                    pointers[p] < len(table[p]) and
                                    table[p][pointers[p]] == curr
                                    for p in chosen
                            ):
                                break
                            common_blocks.append(curr)
                            for p in chosen:
                                pointers[p] += 1

                        if common_blocks:
                            max_seqs_in_CTA = max(max_seqs_in_CTA, len(chosen))
                            box.q_table.append(chosen)
                            box.block_table.append(common_blocks.copy())
                            box.num_seqs_per_CTA.append(len(chosen))
                            box.CTA_rank.append(split_per_seq[chosen[0]])
                            # Compute kv entries in this CTA
                            if to_end:
                                last_len = len(common_blocks)
                                total_blocks = len(table[chosen[0]])
                                kv_val = (
                                        block_size * (last_len - 1)
                                        + seq_lens[chosen[0]]
                                        - (total_blocks - 1) * block_size
                                )
                            else:
                                kv_val = block_size * len(common_blocks)
                            box.kv_in_CTA.append(kv_val)
                            max_blocks_in_CTA = max(max_blocks_in_CTA, len(common_blocks))
                            # Increment split count for each sequence
                            for p in chosen:
                                split_per_seq[p] += 1

    # do not use this!
    # reverse the order of q_table, num_split_per_seq
    # box.q_table = [sort_idx_cpu[_] for _ in box.q_table]
    # reverse_idx = np.argsort(sort_idx_cpu)
    # box.num_split_per_seq = [split_per_seq[_] for _ in reverse_idx]

    kernel_information = KernelInfo(max_split_per_seq=max(split_per_seq),
                                    max_seqs_in_CTA=max_seqs_in_CTA,
                                    max_blocks_in_CTA=max_blocks_in_CTA,
                                    num_split_per_seq=split_per_seq.copy(),
                                    reverse_index=np.argsort(sort_idx_cpu).tolist(),
                                    sort_index=sort_idx_cpu.tolist())

    # Fixme: naive function
    distribute_naive(kernel_information, box, h_q_kv_ratio, qkv_tile_types)
    # distribute_multi_stream(kernel_information, box, h_q_kv_ratio, qkv_tile_types)

    # box.num_split_per_seq = split_per_seq.copy()
    # box.max_split_per_seq = max(split_per_seq)
    # box.reverse_index = np.argsort(sort_idx_cpu).tolist()
    # box.sort_index = sort_idx_cpu.tolist()

    return kernel_information


def schedule_naive(seq_lens: List[int], block_table: List[List[int]], device: torch.device) -> KernelInfo:
    """
    params:
        seq_lens: List[int], sequencelengths
        block_table: List[List[int]], block table

    return:
        ...
        indices: List[int]

    """
    assert isinstance(seq_lens, list)
    assert isinstance(block_table, list)

    block_table = make_ndarray_with_pad(block_table, 0, np.int32)

    # sorted_idx = np.lexsort(block_table[:, ::-1].T)   # sort by block ids
    sorted_idx = np.arange(block_table.shape[0])        # no need to sort, just use the original order

    reverse_idx = np.argsort(sorted_idx)
    sorted_idx = torch.from_numpy(sorted_idx).to(device)
    reverse_idx = torch.from_numpy(reverse_idx).to(device)

    n = len(seq_lens)

    q_table = torch.arange(n, dtype=torch.int32, device=device)
    num_seqs_per_CTA = torch.ones(n, dtype=torch.int32, device=device)
    num_split_per_seq = torch.ones(n, dtype=torch.int32, device=device)
    CTA_rank = torch.zeros(n, dtype=torch.int32, device=device)
    kv_in_CTA = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    block_table_gpu = torch.from_numpy(block_table).to(device)

    return KernelInfo(
        q_tables=[q_table[sorted_idx]],
        block_tables=[block_table_gpu[sorted_idx]],
        num_seqs_per_CTAs=[num_seqs_per_CTA[sorted_idx]],
        num_split_per_seq=num_split_per_seq[sorted_idx],
        CTA_ranks=[CTA_rank[sorted_idx]],
        kv_in_CTAs=[kv_in_CTA[sorted_idx]],
        MNWs=[[64, 128, 4]],  # [tile_q, tile_kv, Warps], [64,128,4] for hdim 128 and [64, 256, 4] for hdim 64
        max_split_per_seq=1,
        max_seqs_in_CTA=1,
        max_blocks_in_CTA=max(len(row) for row in block_table),
        reverse_index=reverse_idx,
        sort_index=sorted_idx,
    )

import time
import torch
import random
from typing import List, Union
from dataclasses import dataclass, field


def make_tensor_merge_split(
        x: list[list],
        device: Union[str, torch.device],
        dtype: torch.dtype = torch.int32,
        pad=0,
        pin_memory: bool = True  # speedup
):
    lens = [len(x) for x in x]
    max_len = max(lens)
    _x = [_ + [pad] * (max_len - len(_)) if len(_) < max_len else _ for _ in x]
    merged = torch.tensor(_x, device=device, dtype=dtype, pin_memory=pin_memory)
    return (merged[i][:lens[i]] for i in range(len(x)))


class Sequence:
    def __init__(self, block_size, seqlen=0):
        self.block_ids = []
        self.block_size = block_size
        self.seqlen = seqlen

    def append_blocks(self, blocks: List[int], tokens=None):
        # append blocks to the end of sequence
        self.block_ids.extend(blocks)
        if tokens is None:
            self.seqlen += len(blocks) * self.block_size
        else:
            self.seqlen += tokens


class SeqGroup:
    def __init__(self, block_size, min_seqlen=0, max_seqlen=0, num_seqs=0):
        self.sequences = [Sequence(block_size, random.randint(min_seqlen, max_seqlen)) for _ in range(num_seqs)]
        self.is_sorted = False

    def __len__(self):
        return len(self.sequences)

    def __str__(self):
        seq_str = []
        for i in range(len(self.sequences)):
            seq_str.append(f"\n[{', '.join(map(str, self.sequences[i].block_ids))}], seqlen={self.sequences[i].seqlen}")
        return f"SeqGroup: {', '.join(seq_str)}"

    @property
    def seqlens(self):
        return [s.seqlen for s in self.sequences]

    @property
    def max_seqlen(self):
        return max([s.seqlen for s in self.sequences])

    @property
    def max_num_blocks(self):
        return max([len(s.block_ids) for s in self.sequences])

    def set_common_prefix_blocks(self, prefix_blocks: List[int]):
        # set common prefix blocks for all sequences
        # for i in range(len(self.sequences)):
        #     if self.sequences[i].seqlen != 0:
        #         print(f"[Warning] sequence {i} already has blocks, overwriting")
        #     self.sequences[i].append_blocks(prefix_blocks)
        for i in range(len(self.sequences)):
            if self.sequences[i].seqlen < self.sequences[i].block_size * len(prefix_blocks):
                continue
            self.sequences[i].append_blocks(prefix_blocks, 0)

    def add_random_prefix_blocks(self, random_blocks: List[List[int]]):
        # add random prefix blocks to the end of each sequence
        for i in range(len(self.sequences)):
            choice = random.randint(0, len(random_blocks) - 1)
            self.sequences[i].append_blocks(random_blocks[choice])
        self.is_sorted = False

    def get_padded_block_ids(self, padding_value=-1) -> List[List[int]]:
        # get padding block id tensor for all sequence, shape: (batch_size, max_num_blocks)
        padded_block_ids = []
        max_num_blocks = self.max_num_blocks
        for seq in self.sequences:
            padded_block_ids.append(seq.block_ids + [padding_value] * (max_num_blocks - len(seq.block_ids)))
        return padded_block_ids

    def get_block_ids(self) -> List[List[int]]:
        # get block id list for all sequence, shape: (batch_size, ...)
        block_ids = []
        for seq in self.sequences:
            block_ids.append(seq.block_ids.copy())
        return block_ids

    def sort(self, reverse: bool = False) -> None:
        self.sequences.sort(key=lambda x: x.block_ids, reverse=reverse)
        self.is_sorted = True


@dataclass
class PackedBox:
    """
    A class to hold packed box information for prefix attention.

    q_table:                List(CTAs, max_seqs_in_CTA)             Fixme: 当前版本的max_seqs_in_CTA似乎不会在算子中用上，且不同算子之间的max_seqs_in_CTA不同
    block_table:            List(CTAs, max_blocks_in_CTA)           Fixme: 当前版本的max_blocks_in_CTA似乎不会在算子中用上，且不同算子之间的max_blocks_in_CTA不同
    num_seqs_per_CTA:       List(CTAs,)
    CTA_rank:               List(CTAs,)
    kv_in_CTA:              List(CTAs,)

    """
    q_table:            List[List[int]] or torch.Tensor = field(default_factory=lambda: [])
    block_table:        List[List[int]] or torch.Tensor = field(default_factory=lambda: [])
    num_seqs_per_CTA:   List[int] or torch.Tensor = field(default_factory=list)
    CTA_rank:           List[int] or torch.Tensor = field(default_factory=list)
    kv_in_CTA:          List[int] or torch.Tensor = field(default_factory=list)

    def __str__(self):
        return (f"PackedBox("
                f"\n\tq_table={self.q_table},"
                f"\n\tblock_table={self.block_table},"  # replace with self.block_table.shape if tensor
                f"\n\tnum_seqs_per_CTA={self.num_seqs_per_CTA},"
                f"\n\tCTA_rank={self.CTA_rank},"
                f"\n\tkv_in_CTA={self.kv_in_CTA},"
                f")")

    def __lt__(self, other):
        return self.kv_in_CTA > other.kv_in_CTA

    def sort_by_kv_desc(self):
        """
        Sort the packed box by kv_in_CTA in descending order.

        Let the CTA with lager kv_in_CTA execute first to reduce bubbles.
        While this has not been shown to be effective in tests.
        """

        indices = sorted(range(len(self.kv_in_CTA)), key=lambda i: self.kv_in_CTA[i], reverse=True)

        self.q_table = [self.q_table[i] for i in indices]
        self.block_table = [self.block_table[i] for i in indices]
        self.num_seqs_per_CTA = [self.num_seqs_per_CTA[i] for i in indices]
        self.CTA_rank = [self.CTA_rank[i] for i in indices]
        self.kv_in_CTA = [self.kv_in_CTA[i] for i in indices]

    def to_tensor(self, device):
        # TODO(zhixin): use make_tensor_merge_split to speedup conversion
        if not isinstance(self.num_seqs_per_CTA, torch.Tensor):
            # self.q_table = torch.tensor(self.q_table, device=device, dtype=torch.int32)
            self.num_seqs_per_CTA = torch.tensor(self.num_seqs_per_CTA, device=device, dtype=torch.int32)
            self.CTA_rank = torch.tensor(self.CTA_rank, device=device, dtype=torch.int32)
            self.kv_in_CTA = torch.tensor(self.kv_in_CTA, device=device, dtype=torch.int32)

        if not isinstance(self.q_table, torch.Tensor):
            max_seqs_in_CTA = max(len(table) for table in self.q_table)

            self.q_table = [
                table + [0] * (max_seqs_in_CTA - len(table))
                for table in self.q_table
            ]
            self.q_table = torch.tensor(self.q_table, device=device, dtype=torch.int32)

        if not isinstance(self.block_table, torch.Tensor):
            max_blocks_in_CTA = max(len(table) for table in self.block_table)

            self.block_table = [
                table + [0] * (max_blocks_in_CTA - len(table))
                for table in self.block_table
            ]
            self.block_table = torch.tensor(self.block_table, device=device, dtype=torch.int32)

@dataclass
class KernelInfo:
    """
    A class to record the bounding box of each kernel and maintain shared information between kernels.

    q_tables:                List(List(CTAs, max_seqs_in_CTA))
    block_tables:            List(List(CTAs, max_blocks_in_CTA))
    num_seqs_per_CTAs:       List(List(CTAs,))
    CTA_ranks:               List(List(CTAs,))
    kv_in_CTAs:              List(List(CTAs,))
    MNWs:                    List(List(3))

    // shared information
    num_split_per_seq:      List(batch_size,)
    max_split_per_seq:      int
    max_seqs_in_CTA:        int                                             Fixme： 当前版本的max_seqs_in_CTA似乎不会在算子中用上
    max_blocks_in_CTA:      int                                             Fixme: 当前版本的max_blocks_in_CTA似乎不会在算子中用上，且不同算子之间的max_blocks_in_CTA不同
    reverse_index:          List(batch_size,)  # used for reordering output
    sort_index:             List(batch_size,)  # used for sort input queries

    """
    q_tables: List[List[List[int]]] or List[torch.Tensor] = field(default_factory=lambda: [])
    block_tables: List[List[List[int]]] or List[torch.Tensor] = field(default_factory=lambda: [])
    num_seqs_per_CTAs: List[List[int]] or List[torch.Tensor] = field(default_factory=list)
    CTA_ranks: List[List[int]] or List[torch.Tensor] = field(default_factory=list)
    kv_in_CTAs: List[List[int]] or List[torch.Tensor] = field(default_factory=list)
    MNWs: List[List[int]] = field(default_factory=list)  # [tile_q, tile_kv, Warps]

    # shared information
    num_split_per_seq: List[int] or torch.Tensor = field(default_factory=list)
    max_split_per_seq: int = 0
    max_seqs_in_CTA: int = 0
    max_blocks_in_CTA: int = 0
    reverse_index: List[int] or torch.Tensor = None
    sort_index: List[int] or torch.Tensor = None

    def to_tensor(self, device):
        # TODO(zhixin): use make_tensor_merge_split to speedup conversion
        if not isinstance(self.num_split_per_seq, torch.Tensor):
            self.num_split_per_seq = torch.tensor(self.num_split_per_seq, device=device, dtype=torch.int32)

        if self.reverse_index is not None and not isinstance(self.reverse_index, torch.Tensor):
            self.reverse_index = torch.tensor(self.reverse_index, device=device, dtype=torch.int32)

        if self.sort_index is not None and not isinstance(self.sort_index, torch.Tensor):
            self.sort_index = torch.tensor(self.sort_index, device=device, dtype=torch.int32)

        # 将所有元素都转换到device上
        def convert_nested_list(lst: List[List[int]]) -> List[torch.Tensor]:
            return [torch.tensor(inner, device=device, dtype=torch.int32) for inner in lst]

        # if not isinstance(self.q_tables, torch.Tensor):
        #     self.q_tables = convert_nested_list(self.q_tables)

        if not isinstance(self.num_seqs_per_CTAs, torch.Tensor):
            self.num_seqs_per_CTAs = convert_nested_list(self.num_seqs_per_CTAs)

        if not all(isinstance(qt, torch.Tensor) for qt in self.q_tables):
            padded_q_tables = []
            for qt in self.q_tables:
                max_inner_len = max(len(inner) for inner in qt)
                padded_qt = [
                    inner + [0] * (max_inner_len - len(inner)) for inner in qt
                ]
                padded_q_tables.append(torch.tensor(padded_qt, device=device, dtype=torch.int32))
            self.q_tables = padded_q_tables

        if not isinstance(self.CTA_ranks, torch.Tensor):
            self.CTA_ranks = convert_nested_list(self.CTA_ranks)

        if not isinstance(self.kv_in_CTAs, torch.Tensor):
            self.kv_in_CTAs = convert_nested_list(self.kv_in_CTAs)

        if not all(isinstance(bt, torch.Tensor) for bt in self.block_tables):
            padded_block_tables = []
            for bt in self.block_tables:
                max_inner_len = max(len(inner) for inner in bt)
                padded_bt = [
                    inner + [0] * (max_inner_len - len(inner)) for inner in bt
                ]
                padded_block_tables.append(torch.tensor(padded_bt, device=device, dtype=torch.int32))
            self.block_tables = padded_block_tables


def create_seq_group(block_tables: List[List[int]], seq_lens: List[int], block_size: int):
    # load block tables and sequence lengths from the given lists
    n = len(block_tables)
    seq_group = SeqGroup(block_size=block_size, num_seqs=n)
    for i in range(n):
        seq_group.sequences[i].block_ids = block_tables[i]
        seq_group.sequences[i].seqlen = seq_lens[i]
    return seq_group

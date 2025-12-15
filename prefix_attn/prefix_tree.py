import heapq
import time
from collections import defaultdict, Counter
from copy import deepcopy, copy
from threading import Thread

import torch
from math import ceil
from torch import Tensor
from concurrent.futures import Future
import numpy as np
from .data_class import PackedBox, KernelInfo
from typing import List, Optional, DefaultDict, Dict
# from ._prefix_attn import build_radix_tree_cpp

def _CpadQ(TS, N):
    return TS - ((N - 1) % TS + 1)


def _Cmm(nQ, nK, HRatio):
    if nQ * HRatio >= 64:
        TS = 64
    else:
        TS = 16
    part1 = _CpadQ(TS, nQ) * nK * HRatio

    return part1


def _Cred(nQl, HRatio):
    return 0.1 * nQl * HRatio


def _SplitQCost(nQcurr, nQl, lenv, lenl, HRatio):
    part1 = _Cmm(nQcurr - nQl, lenv, HRatio)
    part2 = _Cmm(nQl, lenv + lenl, HRatio)

    return part1 + part2


def _SplitKCost(nQcurr, nQl, lenl, lenv, HRatio):
    part1 = _Cmm(nQcurr, lenv, HRatio)
    part2 = _Cmm(nQl, lenl, HRatio)
    part3 = _Cred(nQl, HRatio)
    return part1 + part2 + part3

class PrefixTree:
    def __init__(self, block_size):
        self.block_size = block_size

        self.parents: List[int] = []
        self.s_values: List[int] = []
        self.lens: List[int] = []
        self.seq_indices: List[List[int]] = []
        self.block_ids: List[List[int]] = []
        self.tree: DefaultDict[int, Dict[int, int]] = defaultdict(dict)
        self.num_nodes = 0
        self.root = self.add_node(-1)
        self.split_per_seq = []

    def add_node(self,
                 parent: int = -1,
                 s_value: int = 1,
                 length: int = 0,
                 seq_indices: List[int] = None,
                 block_ids: List[int] = None) -> int:
        if seq_indices is None:
            seq_indices = []
        if block_ids is None:
            block_ids = []
        node_id = self.num_nodes
        self.num_nodes += 1

        self.parents.append(parent)
        self.s_values.append(s_value)
        self.lens.append(length)
        self.seq_indices.append(seq_indices)
        self.block_ids.append(block_ids)
        return node_id

    def get_common_block_length(self, blocks1, blocks2):
        l, r = -1, min(len(blocks1), len(blocks2))
        while r - l > 1:
            mid = (l + r) >> 1
            if blocks1[mid] == blocks2[mid]:
                l = mid
            else:
                r = mid
        return l
        # for i in range(min(len(blocks1), len(blocks2))):
        #     if blocks1[i] != blocks2[i]:
        #         return i - 1
        # return min(len(blocks1), len(blocks2)) - 1

    def insert(self, sId, seq_len, blocks):
        node = self.root
        res_block = len(blocks)
        while res_block:
            if blocks[0] in self.tree[node]:
                child_node_id = self.tree[node][blocks[0]]
                child_node_block_ids = self.block_ids[child_node_id]
                common_block_len = self.get_common_block_length(blocks, child_node_block_ids) + 1
                if common_block_len == len(child_node_block_ids):
                    self.s_values[child_node_id] += 1
                    self.seq_indices[child_node_id].append(sId)
                    node = child_node_id
                    res_block -= common_block_len
                    seq_len -= common_block_len * self.block_size
                    blocks = blocks[common_block_len:]
                else:
                    mid = self.add_node(node,
                                        self.s_values[child_node_id] + 1,
                                        common_block_len * self.block_size,
                                        self.seq_indices[child_node_id].copy(),
                                        child_node_block_ids[:common_block_len]
                                        )
                    self.tree[mid][child_node_block_ids[common_block_len]] = child_node_id
                    self.tree[node][child_node_block_ids[0]] = mid
                    self.parents[child_node_id] = mid
                    self.block_ids[child_node_id] = child_node_block_ids[common_block_len:]
                    self.lens[child_node_id] -= common_block_len * self.block_size
                    if common_block_len == res_block:
                        self.seq_indices[mid].append(sId)
                        break
                    new_node = self.add_node(
                        mid,
                        1,
                        min(res_block * self.block_size, seq_len) - common_block_len * self.block_size,
                        [sId],
                        blocks[common_block_len:]
                    )
                    self.seq_indices[mid].append(sId)
                    self.tree[mid][blocks[common_block_len]] = new_node
                    break
            else:
                new_node = self.add_node(
                    node,
                    1,
                    seq_len,
                    [sId],
                    blocks
                )
                self.tree[node][blocks[0]] = new_node
                break

    def build_radix_tree(self, seq_lens: List[int], block_table: List[List[int]]):
        for seq_idx, blocks in enumerate(block_table):
            self.insert(seq_idx, seq_lens[seq_idx], blocks)
        self.split_per_seq = [0] * len(block_table)

    def build_from_batch(self, seq_lens: List[int], block_table: List[List[int]]):
        if not block_table:
            return

        indexed_data = list(zip(range(len(block_table)), seq_lens, block_table))
        sorted_data = sorted(indexed_data, key=lambda x: x[2])

        last_node_id = -1
        previous_blocks = []

        for sId, seq_len, current_blocks in sorted_data:
            if last_node_id == -1:
                node_id = self.root
                if current_blocks:
                    new_node_id = self.add_node(
                        parent=node_id, length=seq_len, seq_indices=[sId], block_ids=current_blocks)
                    self.tree[node_id][current_blocks[0]] = new_node_id
                    last_node_id = new_node_id
                else:
                    self.s_values[self.root] += 1
                    self.seq_indices[self.root].append(sId)
                    last_node_id = self.root
                previous_blocks = current_blocks
                continue

            lcp_len = self.get_common_block_length(previous_blocks, current_blocks) + 1 if \
                previous_blocks and current_blocks and previous_blocks[0] == current_blocks[0] else 0

            path_len = len(previous_blocks)
            node_to_ascend_id = last_node_id
            while path_len > lcp_len:
                parent_id = self.parents[node_to_ascend_id]
                edge_len = len(self.block_ids[node_to_ascend_id])
                if path_len - edge_len < lcp_len:
                    break
                path_len -= edge_len
                node_to_ascend_id = parent_id

            parent_of_split_id = self.parents[node_to_ascend_id]

            if path_len > lcp_len:
                split_point = len(self.block_ids[node_to_ascend_id]) - (path_len - lcp_len)

                mid_node_blocks = self.block_ids[node_to_ascend_id][:split_point]
                mid_node_len = split_point * self.block_size
                mid_node_id = self.add_node(
                    parent=parent_of_split_id,
                    s_value=self.s_values[node_to_ascend_id] + 1,
                    length=mid_node_len,
                    seq_indices=self.seq_indices[node_to_ascend_id].copy(),
                    block_ids=mid_node_blocks
                )

                original_child_remaining_blocks = self.block_ids[node_to_ascend_id][split_point:]
                self.tree[mid_node_id][original_child_remaining_blocks[0]] = node_to_ascend_id

                original_first_block = self.block_ids[node_to_ascend_id][0]
                self.tree[parent_of_split_id][original_first_block] = mid_node_id

                self.parents[node_to_ascend_id] = mid_node_id
                self.block_ids[node_to_ascend_id] = original_child_remaining_blocks
                self.lens[node_to_ascend_id] -= mid_node_len

                remaining_blocks_for_new_seq = current_blocks[lcp_len:]
                new_leaf_len = seq_len - (lcp_len * self.block_size)
                new_leaf_id = self.add_node(
                    parent=mid_node_id,
                    length=new_leaf_len,
                    seq_indices=[sId],
                    block_ids=remaining_blocks_for_new_seq
                )
                self.tree[mid_node_id][remaining_blocks_for_new_seq[0]] = new_leaf_id

                self.seq_indices[mid_node_id].append(sId)

                last_node_id = new_leaf_id
                divergence_node_id = mid_node_id

            else:
                divergence_node_id = node_to_ascend_id
                remaining_blocks = current_blocks[lcp_len:]
                if remaining_blocks:
                    remaining_len = seq_len - (lcp_len * self.block_size)
                    new_leaf_id = self.add_node(
                        parent=divergence_node_id,
                        length=remaining_len,
                        seq_indices=[sId],
                        block_ids=remaining_blocks
                    )
                    self.tree[divergence_node_id][remaining_blocks[0]] = new_leaf_id
                    last_node_id = new_leaf_id
                else:
                    last_node_id = divergence_node_id

            curr_update_id = divergence_node_id
            while curr_update_id != -1:
                if sId not in self.seq_indices[curr_update_id]:
                    self.s_values[curr_update_id] += 1
                    self.seq_indices[curr_update_id].append(sId)

                if curr_update_id == self.root:
                    break
                curr_update_id = self.parents[curr_update_id]

            previous_blocks = current_blocks

        self.split_per_seq = [0] * len(block_table)

    def print_tree(self):
        def _print_node(node_id, depth):
            node = self.tree[node_id]
            print(' ' * depth + f'Node {node_id}: S={self.s_values[node_id]}, len={self.lens[node_id]}, seqs={self.seq_indices[node_id]}, blocks={self.block_ids[node_id]}')
            for child_id in self.tree[node_id]:
                _print_node(child_id, depth + 2)

        _print_node(self.root, 0)


    def _tree_heuristics(self,
                         node,
                         mm,
                         inherited_block_id: List[int],
                         ) -> List[PackedBox]:
        res = []
        if len(self.tree[node]) == 0:
            S = self.s_values[node]
            for s in range(0, S, mm):
                box = PackedBox()
                box.q_table = self.seq_indices[node][s:min(s + mm, S)]
                box.block_table = inherited_block_id + self.block_ids[node]
                box.num_seqs_per_CTA = min(mm, S - s)
                box.kv_in_CTA = self.block_size * len(inherited_block_id) + self.lens[node]
                box.CTA_rank = self.split_per_seq[box.q_table[0]]
                for seq_idx in box.q_table:
                    self.split_per_seq[seq_idx] += 1
                res.append(box)
        else:
            children = self.tree[node].values()
            ops = [0] * len(children)
            S = self.s_values[node]
            kv_len = self.block_size * len(inherited_block_id) + self.lens[node]
            split_child = []
            merge_child = []
            for idx, child_id in enumerate(children):
                if S == self.s_values[child_id] or (ceil(S / mm) - ceil((S - self.s_values[child_id]) / mm) - ceil(
                        self.s_values[child_id] / mm)) * kv_len + 4 * self.s_values[child_id] >= 0:
                    ops[idx] = 1
                    S -= self.s_values[child_id]
                    merge_child.append(child_id)
                else:
                    split_child.append(child_id)
            if S != 0:
                seqs = []
                for idx, child_id in enumerate(children):
                    if ops[idx] == 0:
                        seqs.extend(self.seq_indices[child_id])
                for s in range(0, S, mm):
                    box = PackedBox()
                    box.q_table = seqs[s:min(s + mm, S)]
                    box.block_table = inherited_block_id + self.block_ids[node]
                    box.num_seqs_per_CTA = min(mm, S - s)
                    box.kv_in_CTA = self.block_size * len(inherited_block_id) + self.lens[node]
                    box.CTA_rank = self.split_per_seq[box.q_table[0]]
                    for seq_idx in box.q_table:
                        self.split_per_seq[seq_idx] += 1
                    res.append(box)

            for child_id in split_child:
                res.extend(self._tree_heuristics(child_id, mm, []))
            for child_id in merge_child:
                res.extend(self._tree_heuristics(child_id, mm, inherited_block_id + self.block_ids[node]))

        return res

    def _compute_model_tree_heuristics(self,
                                       node,
                                       mm,
                                       HRatio,
                                       inherited_block_id: List[int],
                                       ) -> List[PackedBox]:
        res = []
        if len(self.tree[node]) == 0:
            S = self.s_values[node]
            for s in range(0, S, mm):
                box = PackedBox()
                box.q_table = self.seq_indices[node][s:min(s + mm, S)]
                box.block_table = inherited_block_id + self.block_ids[node]
                box.num_seqs_per_CTA = min(mm, S - s)
                box.kv_in_CTA = self.block_size * len(inherited_block_id) + self.lens[node]
                box.CTA_rank = self.split_per_seq[box.q_table[0]]
                for seq_idx in box.q_table:
                    self.split_per_seq[seq_idx] += 1
                res.append(box)
        else:
            children = self.tree[node].values()
            ops = [0] * len(children)
            S = self.s_values[node]
            kv_len = self.block_size * len(inherited_block_id) + self.lens[node]
            split_child = []
            merge_child = []
            for idx, child_id in enumerate(children):
                C0 = _SplitKCost(S, self.s_values[child_id], kv_len, self.lens[child_id], HRatio)
                C1 = _SplitQCost(S, self.s_values[child_id], kv_len, self.lens[child_id], HRatio)
                if C0 > C1:
                    ops[idx] = 1
                    S -= self.s_values[child_id]
                    merge_child.append(child_id)
                else:
                    split_child.append(child_id)
            if S != 0:
                seqs = []
                for idx, child_id in enumerate(children):
                    if ops[idx] == 0:
                        seqs.extend(self.seq_indices[child_id])
                for s in range(0, S, mm):
                    box = PackedBox()
                    box.q_table = seqs[s:min(s + mm, S)]
                    box.block_table = inherited_block_id + self.block_ids[node]
                    box.num_seqs_per_CTA = min(mm, S - s)
                    box.kv_in_CTA = self.block_size * len(inherited_block_id) + self.lens[node]
                    box.CTA_rank = self.split_per_seq[box.q_table[0]]
                    for seq_idx in box.q_table:
                        self.split_per_seq[seq_idx] += 1
                    res.append(box)

            for child_id in split_child:
                res.extend(self._compute_model_tree_heuristics(child_id, mm, HRatio, []))
            for child_id in merge_child:
                res.extend(self._compute_model_tree_heuristics(child_id, mm, HRatio, inherited_block_id + self.block_ids[node]))

        return res

    def balancePack(self, PackedBoxs: List[PackedBox], kvHead) -> List[PackedBox]:
        if len(PackedBoxs) == 0:
            return []
        cropPack = []
        total_blocks = 0
        total_Packs = 0
        for box in PackedBoxs:
            if box.kv_in_CTA > 256:
                total_blocks += len(box.block_table)
                total_Packs += 1

        if total_Packs != 0:
            average_blocks_per_pack = total_blocks / total_Packs
            threshold = average_blocks_per_pack * 5

            for pack in PackedBoxs:
                if len(pack.block_table) > threshold:
                    num_splits = ceil(len(pack.block_table) / average_blocks_per_pack)
                    split_size = ceil(len(pack.block_table) / num_splits)
                    for i in range(num_splits):
                        new_pack = PackedBox()
                        new_pack.q_table = pack.q_table
                        new_pack.block_table = pack.block_table[i * split_size:min((i + 1) * split_size, len(pack.block_table))]
                        new_pack.num_seqs_per_CTA = pack.num_seqs_per_CTA
                        new_pack.kv_in_CTA = min(pack.kv_in_CTA - i * split_size * self.block_size, split_size * self.block_size)
                        new_pack.CTA_rank = pack.CTA_rank if i == 0 else self.split_per_seq[new_pack.q_table[0]]
                        cropPack.append(new_pack)
                        if i != 0:
                            for qid in new_pack.q_table:
                                self.split_per_seq[qid] += 1
                else:
                    cropPack.append(pack)
        else:
            cropPack = PackedBoxs

        if kvHead <= 4:
            threshold = 54
        elif kvHead <= 8:
            threshold = 27
        elif kvHead <= 16:
            threshold = 13
        elif kvHead <= 32:
            threshold = 6
        else:
            threshold = 4

        if len(cropPack) >= threshold:
            return cropPack

        while len(cropPack) < threshold and max(self.split_per_seq) < 32:
            longest_idx = max(range(len(cropPack)), key=lambda i: len(cropPack[i].block_table))
            pack = cropPack.pop(longest_idx)

            total_blocks = len(pack.block_table)
            if total_blocks <= 8:
                cropPack.append(pack)
                break
            half = ceil(total_blocks / 2)

            pack1 = PackedBox()
            pack1.q_table = pack.q_table
            pack1.block_table = pack.block_table[:half]
            pack1.num_seqs_per_CTA = pack.num_seqs_per_CTA
            pack1.kv_in_CTA = min(pack.kv_in_CTA, half * self.block_size)
            pack1.CTA_rank = pack.CTA_rank

            pack2 = PackedBox()
            pack2.q_table = pack.q_table
            pack2.block_table = pack.block_table[half:]
            pack2.num_seqs_per_CTA = pack.num_seqs_per_CTA
            pack2.CTA_rank = self.split_per_seq[pack2.q_table[0]]
            pack2.kv_in_CTA = min(pack.kv_in_CTA - half * self.block_size, (total_blocks - half) * self.block_size)
            for qid in pack1.q_table:
                self.split_per_seq[qid] += 1

            cropPack.extend([pack1, pack2])

        return cropPack

    def getMNW(self,
               m_val: int,
               kv_in_CTA: int):
        m, w = 16, 1
        if m_val > 32:
            m, w = 64, 4
        elif m_val > 16:
            m, w = 32, 2
        n = 128
        if kv_in_CTA < 32:
            n = 16
        elif kv_in_CTA < 64:
            n = 32
        elif kv_in_CTA <= 128:
            n = 64
        else:
            n = 128
        if m == 64:
            n = max(32, n)

        return (m, n, w)

    def naive_tree(self,
                   node,
                   mm,
                   HRatio,
                   ):
        res = []
        S = self.s_values[node]
        for s in range(0, S, mm):
            box = PackedBox()
            box.q_table = self.seq_indices[node][s:min(s + mm, S)]
            box.block_table = self.block_ids[node]
            box.num_seqs_per_CTA = min(mm, S - s)
            box.kv_in_CTA = self.lens[node]
            box.CTA_rank = self.split_per_seq[box.q_table[0]]
            for seq_idx in box.q_table:
                self.split_per_seq[seq_idx] += 1
            res.append(box)

        for child_id in self.tree[node].values():
            res.extend(self.naive_tree(child_id, mm, HRatio))

        return res

    def naive_schedule(self,
                       HRatio: int = 1,
                       ):
        default_MNWs = [
            [64, 128, 4], [64, 64, 4], [64, 32, 4],
            [32, 128, 2], [32, 64, 2], [32, 32, 2], [32, 16, 2],
            [16, 128, 1], [16, 64, 1], [16, 32, 1], [16, 16, 1]
        ]
        buckets = default_MNWs
        mm = 64 // HRatio
        PackedBoxs: List[PackedBox] = []
        for node in self.tree[self.root].values():
            PackedBoxs.extend(self.naive_tree(node, mm, HRatio))

        grouped = {tuple(mnw): [] for mnw in buckets}

        for box in PackedBoxs:
            m_val = len(box.q_table) * HRatio
            grouped[self.getMNW(m_val, box.kv_in_CTA)].append(box)

        # Populate KernelInfo
        ki = KernelInfo()
        for mnw in buckets:
            group = grouped.get(tuple(mnw), [])
            if not group:
                continue
            ki.MNWs.append(mnw)
            ki.q_tables.append([box.q_table for box in group])
            ki.block_tables.append([box.block_table for box in group])
            ki.num_seqs_per_CTAs.append([box.num_seqs_per_CTA for box in group])
            ki.CTA_ranks.append([box.CTA_rank for box in group])
            ki.kv_in_CTAs.append([box.kv_in_CTA for box in group])

        ki.num_split_per_seq = self.split_per_seq
        ki.max_split_per_seq = max(ki.num_split_per_seq)
        ki.max_seqs_in_CTA = max((len(box.q_table) for box in PackedBoxs), default=0)
        ki.max_blocks_in_CTA = max((len(box.block_table) for box in PackedBoxs), default=0)

        return ki


    def pack_schedule(self,
                      MNWs: Optional[List[List[int]]] = None,
                      HRatio: int = 1,
                      kvHead: int = 8,
                      use_compute_model: bool = False):
        default_MNWs = [
            [64, 128, 4], [64, 64, 4], [64, 32, 4],
            [32, 128, 2], [32, 64, 2], [32, 32, 2], [32, 16, 2],
            [16, 128, 1], [16, 64, 1], [16, 32, 1], [16, 16, 1]
        ]
        buckets = MNWs if MNWs is not None else default_MNWs
        mm = max(bucket[0] for bucket in buckets)
        mm = mm // HRatio
        PackedBoxs: List[PackedBox] = []

        if use_compute_model:
            for node in self.tree[self.root].values():
                PackedBoxs.extend(self._compute_model_tree_heuristics(node, mm, HRatio, []))
        else:
            for node in self.tree[self.root].values():
                PackedBoxs.extend(self._tree_heuristics(node, mm, []))

        PackedBoxs = self.balancePack(PackedBoxs, kvHead)

        # Define priority MNWs list

        # Prepare grouping dict
        grouped = {tuple(mnw): [] for mnw in buckets}

        if len(buckets) == 1:
            for box in PackedBoxs:
                grouped[tuple(buckets[0])].append(box)
        else:
            # Assign each box to a bucket
            for box in PackedBoxs:
                # Compute ideal M and N
                m_val = len(box.q_table) * HRatio
                # Match bucket with same M and N
                check = False
                grouped[self.getMNW(m_val, box.kv_in_CTA)].append(box)
                # for mnw in buckets:
                #     if mnw[0] >= m_val and mnw[1] <= box.kv_in_CTA:
                #         grouped[tuple(mnw)].append(box)
                #         check = True
                #         break
                # if check == False:
                #     # If no bucket matches, assign to the first bucket
                #     grouped[tuple(buckets[0])].append(box)

        # Populate KernelInfo
        ki = KernelInfo()
        for mnw in buckets:
            group = grouped.get(tuple(mnw), [])
            if not group:
                continue
            ki.MNWs.append(mnw)
            ki.q_tables.append([box.q_table for box in group])
            ki.block_tables.append([box.block_table for box in group])
            ki.num_seqs_per_CTAs.append([box.num_seqs_per_CTA for box in group])
            ki.CTA_ranks.append([box.CTA_rank for box in group])
            ki.kv_in_CTAs.append([box.kv_in_CTA for box in group])


        ki.num_split_per_seq = self.split_per_seq
        ki.max_split_per_seq = max(ki.num_split_per_seq)
        ki.max_seqs_in_CTA = max((len(box.q_table) for box in PackedBoxs), default=0)
        ki.max_blocks_in_CTA = max((len(box.block_table) for box in PackedBoxs), default=0)

        return ki




def unpad(lst):
    i = len(lst) - 1
    while i > 0 and lst[i] == 0:
        i -= 1
    return lst[:i+1]


def unpad_2d(table):
    return [unpad(row) for row in table]


class AsyncPrefixTreeScheduler(Thread):
    """
    Helper class to run box scheduling asynchronously in a thread.
    Start a thread to run the scheduling and set the result in the future.
    """

    def __init__(
            self,
            future: Future,  # Future to set the result
            block_tables: List[List[int]] or torch.Tensor,
            seq_lens: List[int],
            num_decode_tokens: int,
            block_size: int,
            nheads: int,
            nheads_kv: int,
            MNWs: List[List[int]],
            device,
    ):
        super().__init__()
        self.future = future
        self.block_tables = block_tables
        self.seq_lens = seq_lens
        self.num_decode_tokens = num_decode_tokens
        self.block_size = block_size
        self.nheads = nheads
        self.nheads_kv = nheads_kv
        self.MNWs = MNWs
        self.device = device

    def run(self):
        try:
            if isinstance(self.block_tables, torch.Tensor):
                self.block_tables = self.block_tables.cpu().tolist()
            tree = PrefixTree(self.block_size)
            tree.build_radix_tree(self.seq_lens, self.block_tables)
            box: KernelInfo = tree.pack_schedule(
                self.MNWs,
                self.nheads // self.nheads_kv if self.nheads_kv is not None else 64)
            box.to_tensor(device=self.device)
            self.future.set_result(box)
        except Exception as e:
            self.future.set_exception(e)
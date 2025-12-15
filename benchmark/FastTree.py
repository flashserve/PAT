# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code originally from the FastTree-Artifact project by PanZaifeng,
# available at: https://github.com/PanZaifeng/FastTree-Artifact
# Licensed under the Apache License, Version 2.0.
#
# Modifications:
# - We have only modified the `generate_tree` function's logic.
#   It now generates the entire tree information in one pass,
#   without changing the input or the final structure of `tree_info`,
#   which remains identical to the result in the original repository.
# - ！Fix: FastTree could not produce correct results when there were multiple prefixes at level 0. 2025/8/7

import queue
import triton
import triton.language as tl
import torch

class FastTreeParams:
    alpha = 0.66
    beta = 0.33
    gamma = 0.1
    TSQs = [64, 16]
    TSKs = [32, 128]
    kv_group_num = 1

    def set_values(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def set_kv_group_num(self, kv_group_num):
        self.kv_group_num = kv_group_num

    def set_q_tile_sizes(self, TSQs):
        self.TSQs = TSQs

    def set_kv_tile_sizes(self, TSKs):
        self.TSKs = TSKs


class KVTreeNode:
    def __init__(self):
        self.parent = -1
        self.id = -1
        self.seqlen = 0
        self.num_children = 0
        self.requests = []

def generate_tree(layer_sizes, layer_lengths):
    if len(layer_sizes) != len(layer_lengths):
        raise ValueError("The length of layer_sizes must be equal to the length of layer_lengths.")

    total_nodes = sum(layer_sizes)
    # Initialize a list of nodes: [parent, node_id, seq_length, out_degree]
    nodes = [[-1, -1, -1, 0] for _ in range(total_nodes)]

    current_id = 0  # This will be used for assigning node IDs
    prev_layer_ids = []  # This will store the node IDs of the previous layer

    for layer_idx, (size, seq_len) in enumerate(zip(layer_sizes, layer_lengths)):
        current_layer_ids = []

        for j in range(size):
            node_id = current_id
            current_id += 1

            # Determine parent
            if layer_idx == 0:
                # For the first layer (root layer), the parent is -1
                parent_id = -1
            else:
                # Distribute children among the previous layer's nodes
                # This formula ensures a roughly even distribution of children
                p_index = (j * len(prev_layer_ids)) // size
                parent_id = prev_layer_ids[p_index]

                # Update the parent's out_degree
                nodes[parent_id][3] += 1

            # Fill information for the current node
            nodes[node_id][0] = parent_id  # parent_id
            nodes[node_id][1] = node_id  # node_id
            nodes[node_id][2] = seq_len  # sequence length
            # out_degree is already 0; updated for parent if needed

            current_layer_ids.append(node_id)

        # Prepare for the next iteration
        prev_layer_ids = current_layer_ids

    tree_info = []
    for node in nodes:
        tmp = KVTreeNode()
        tmp.parent = node[0]
        tmp.id = node[1]
        tmp.seqlen = node[2]
        tmp.num_children = node[3]
        tree_info.append(tmp)

    return tree_info


def _CpadQ(TS, N):
    return TS - ((N - 1) % TS + 1)


def _CpadK(TS, N):
    return max(0, TS - N)


def _Cmm(nQ, nK, params: FastTreeParams):
    phase = 0 if nQ > params.TSQs[1] else 1
    TSQ = params.TSQs[phase]
    TSK = params.TSKs[phase]
    return (
        params.alpha * _CpadQ(TSQ, nQ) * params.kv_group_num * nK
        + params.beta * _CpadK(TSK, nK) * nQ * params.kv_group_num
    )


def _Cred(nQl, params: FastTreeParams, lenl, lenv):
    return params.gamma * nQl


def _SplitQCost(nQcurr, nQl, lenv, lenl, params: FastTreeParams):
    return _Cmm(nQcurr - nQl, lenv, params) + _Cmm(nQl, lenl + lenv, params)


def _SplitKCost(nQcurr, nQl, lenl, lenv, params: FastTreeParams):
    return (
        _Cmm(nQcurr, lenv, params)
        + _Cmm(nQl, lenl, params)
        + _Cred(nQl, params, lenl, lenv)
    )


def _tree_heuristic(tree_info, params: FastTreeParams):
    node_num = len(tree_info)
    edges = [[] for _ in range(node_num)]
    L = [tree_info[i].seqlen for i in range(node_num)]

    for i in range(node_num):
        if i != 0:
            edges[tree_info[i].parent].append(i)

    node_assignments = [0 for _ in range(node_num)]

    que = queue.Queue()
    que.put(0)

    while not que.empty():
        node = que.get()
        nQcurr = len(tree_info[node].requests)
        lenv = L[node]

        for l in edges[node]:
            nQl = len(tree_info[l].requests)
            lenl = L[l]
            # print(nQcurr, nQl, lenl, lenv)
            C0 = _SplitKCost(nQcurr, nQl, lenl, lenv, params)
            C1 = _SplitQCost(nQcurr, nQl, lenv, lenl, params)
            if C0 > C1:
                node_assignments[l] = 1
                nQcurr -= nQl
                L[l] = lenl + lenv
            else:
                node_assignments[l] = 0
            que.put(l)
    return node_assignments


def fasttree_preparation(
        tree_info,
        KV_ptrs,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        KV_SPLIT_SIZES,
        para_threshs1,
        para_threshs2,
        params: FastTreeParams,
        device="cuda:0"
):
    node_num = len(tree_info)
    Q_TILE_SIZE_PER_PHASE = params.TSQs

    vnode_to_kv_offs = []
    vnode_to_kv_lens = []
    vnode_to_q_entries = []
    vnode_to_q_offs = []
    vnode_to_q_lens = []
    req_to_vnode_entries = [[] for _ in range(batch_size)]
    req_to_vnode_offs = []
    req_to_vnode_lens = []

    KV_SPLIT_SIZE_PER_PHASE = [KV_SPLIT_SIZES[0], KV_SPLIT_SIZES[0]]
    node_assignments = None

    def compute_parallelism():
        node_to_reqs = [[] for _ in range(node_num)]
        que = queue.Queue()
        for i in range(node_num):
            if tree_info[i].num_children == 0:
                que.put(i)
                node_to_reqs[i] = tree_info[i].requests.copy()

        virtual_chidren = [tree_info[n].num_children for n in range(node_num)]
        while not que.empty():
            node = que.get()
            if node_assignments[node] == 0 and node != 0:
                node_to_reqs[tree_info[node].parent] += tree_info[node].requests
            virtual_chidren[tree_info[node].parent] -= 1
            if virtual_chidren[tree_info[node].parent] == 0:
                que.put(tree_info[node].parent)

        parallelisms = [0, 0]
        for i in range(node_num):
            req_num = len(node_to_reqs[i])
            if req_num == 0:
                continue

            node = i
            kv_len = tree_info[i].seqlen
            while node_assignments[node] == 1:
                node = tree_info[node].parent
                kv_len += tree_info[node].seqlen

            phase = 0 if req_num > Q_TILE_SIZE_PER_PHASE[1] else 1
            q_vnode_count = (req_num - 1) // Q_TILE_SIZE_PER_PHASE[phase] + 1
            kv_vnode_count = (kv_len - 1) // KV_SPLIT_SIZE_PER_PHASE[phase] + 1
            parallelisms[phase] += kv_vnode_count * q_vnode_count
        parallelisms = [p * num_kv_heads for p in parallelisms]

        return parallelisms, node_to_reqs

    for i in range(3):
        node_assignments = _tree_heuristic(tree_info, params)
        parallelisms, node_to_reqs = compute_parallelism()
        # print("parallelisms:", parallelisms)
        if i == 0:
            break_flag = True
            for phase in range(2):
                if (
                    parallelisms[phase] > 0
                    and parallelisms[phase] < para_threshs1[phase]
                ):
                    KV_SPLIT_SIZE_PER_PHASE[phase] = KV_SPLIT_SIZES[1]
                    break_flag = False
            if break_flag:
                break
        elif i == 1:
            if parallelisms[0] > 0 and parallelisms[0] < para_threshs2[0]:
                Q_TILE_SIZE_PER_PHASE = [Q_TILE_SIZE_PER_PHASE[1] for _ in range(2)]
                params.set_q_tile_sizes(Q_TILE_SIZE_PER_PHASE)
                params.set_kv_tile_sizes([params.TSKs[1] for _ in range(2)])
            elif parallelisms[1] > 0 and parallelisms[1] < para_threshs2[1]:
                Q_TILE_SIZE_PER_PHASE = [Q_TILE_SIZE_PER_PHASE[0] for _ in range(2)]
                params.set_q_tile_sizes(Q_TILE_SIZE_PER_PHASE)
                params.set_kv_tile_sizes([params.TSKs[0] for _ in range(2)])
            else:
                break

    # print(Q_TILE_SIZE_PER_PHASE)
    # print(KV_SPLIT_SIZE_PER_PHASE)

    vnode_to_kv_entries = []
    Q_TILE_SIZE = Q_TILE_SIZE_PER_PHASE[0]
    for i in range(node_num):
        req_num = len(node_to_reqs[i])
        if req_num == 0:
            continue
        phase = 0 if req_num > Q_TILE_SIZE_PER_PHASE[1] else 1
        KV_SPLIT_SIZE = KV_SPLIT_SIZE_PER_PHASE[phase]

        node = i
        kv_len = tree_info[i].seqlen
        curr_KV_indices = list(range(KV_ptrs[node], KV_ptrs[node + 1]))
        while node_assignments[node] == 1:
            node = tree_info[node].parent
            kv_len += tree_info[node].seqlen
            curr_KV_indices = (
                list(range(KV_ptrs[node], KV_ptrs[node + 1])) + curr_KV_indices
            )
        acc_kv_len = len(vnode_to_kv_entries)
        vnode_to_kv_entries.extend(curr_KV_indices)

        kv_vnode_count = (kv_len - 1) // KV_SPLIT_SIZE + 1
        q_vnode_count = (req_num - 1) // Q_TILE_SIZE + 1
        for kv_vnode_id in range(kv_vnode_count):
            acc_q_len = len(vnode_to_q_entries)
            for req in node_to_reqs[i]:
                vnode_to_q_entries.append(req)

            split_kv_off = kv_vnode_id * KV_SPLIT_SIZE
            vnode_kv_len = min(split_kv_off + KV_SPLIT_SIZE, kv_len) - split_kv_off

            for q_vnode_id in range(q_vnode_count):
                split_q_off = q_vnode_id * Q_TILE_SIZE
                vnode_q_len = min(split_q_off + Q_TILE_SIZE, req_num) - split_q_off

                vnode_to_kv_offs.append(acc_kv_len + split_kv_off)
                vnode_to_kv_lens.append(vnode_kv_len)
                vnode_to_q_offs.append(acc_q_len + split_q_off)
                vnode_to_q_lens.append(vnode_q_len)

    for i, q in enumerate(vnode_to_q_entries):
        req_to_vnode_entries[q].append(i)

    offset = 0
    for i in range(batch_size):
        req_to_vnode_offs.append(offset)
        offset = offset + len(req_to_vnode_entries[i])
        req_to_vnode_lens.append(len(req_to_vnode_entries[i]))

    req_to_vnode_entries = [
        item for sublist in req_to_vnode_entries for item in sublist
    ]

    threshold = Q_TILE_SIZE_PER_PHASE[1]
    above_indices = [i for i, val in enumerate(vnode_to_q_lens) if val > threshold]
    below_indices = [i for i, val in enumerate(vnode_to_q_lens) if val <= threshold]
    # print("Above threshold: ", len(above_indices))
    # print("Below threshold: ", len(below_indices))

    new_order = above_indices + below_indices
    phase_node_nums = [len(above_indices), len(below_indices)]
    phase_node_offsets = [0, len(above_indices)]

    vnode_to_q_lens = [vnode_to_q_lens[i] for i in new_order]
    vnode_to_q_offs = [vnode_to_q_offs[i] for i in new_order]
    vnode_to_kv_lens = [vnode_to_kv_lens[i] for i in new_order]
    vnode_to_kv_offs = [vnode_to_kv_offs[i] for i in new_order]

    vnode_to_kv_entries = vnode_to_kv_entries + [-1] * 32
    req_to_vnode_entries = req_to_vnode_entries + [-1] * 32

    # In practice, we should pre-allocate the buffers
    vnode_to_kv_entries = torch.tensor(vnode_to_kv_entries, dtype=torch.int32, device=device)
    vnode_to_kv_offs = torch.tensor(vnode_to_kv_offs, dtype=torch.int32, device=device)
    vnode_to_kv_lens = torch.tensor(vnode_to_kv_lens, dtype=torch.int32, device=device)
    vnode_to_q_entries = torch.tensor(vnode_to_q_entries, dtype=torch.int32, device=device)
    vnode_to_q_offs = torch.tensor(vnode_to_q_offs, dtype=torch.int32, device=device)
    vnode_to_q_lens = torch.tensor(vnode_to_q_lens, dtype=torch.int32, device=device)
    req_to_vnode_entries = torch.tensor(req_to_vnode_entries, dtype=torch.int32, device=device)
    req_to_vnode_offs = torch.tensor(req_to_vnode_offs, dtype=torch.int32, device=device)
    req_to_vnode_lens = torch.tensor(req_to_vnode_lens, dtype=torch.int32, device=device)
    mid_o = torch.empty(
        (vnode_to_q_entries.numel(), num_qo_heads, head_dim), dtype=torch.float32, device=device
    )
    mid_lse = torch.empty(
        (vnode_to_q_entries.numel(), num_qo_heads), dtype=torch.float32, device=device
    )

    return (
        vnode_to_kv_entries,
        vnode_to_kv_offs,
        vnode_to_kv_lens,
        vnode_to_q_entries,
        vnode_to_q_offs,
        vnode_to_q_lens,
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        mid_o,
        mid_lse,
        phase_node_nums,
        phase_node_offsets,
    ), node_assignments


def qkv_preparation(
        tree_info,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16
):
    node_num = len(tree_info)
    num_requests = 0
    max_seqlen = 0
    K_tree_tensor = []
    V_tree_tensor = []
    KV_ptrs = [0]
    for n in range(node_num):
        seqlen = tree_info[n].seqlen
        K_tree_tensor.append(
            torch.randn(1, seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
        )
        V_tree_tensor.append(
            torch.randn(1, seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
        )
        KV_ptrs.append(KV_ptrs[-1] + seqlen)

        if tree_info[n].num_children == 0:
            node = n
            sum_seqlen = 0
            while node != -1:
                tree_info[node].requests.append(num_requests)
                sum_seqlen += tree_info[node].seqlen
                node = tree_info[node].parent
            if sum_seqlen > max_seqlen:
                max_seqlen = sum_seqlen
            num_requests += 1

    batch_size = num_requests
    K_cache = torch.empty(
        batch_size, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    V_cache = torch.empty(
        batch_size, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    cache_seqlens = [0 for _ in range(num_requests)]
    for n in range(node_num):
        if tree_info[n].num_children == 0:
            node = n
            chain_nodes = []
            while node != -1:
                chain_nodes.append(node)
                node = tree_info[node].parent

            K_temp_tensor = torch.empty(
                1, 0, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            V_temp_tensor = torch.empty(
                1, 0, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            for n in reversed(chain_nodes):
                K_temp_tensor = torch.cat((K_temp_tensor, K_tree_tensor[n]), dim=1)
                V_temp_tensor = torch.cat((V_temp_tensor, V_tree_tensor[n]), dim=1)

            cache_seqlens[tree_info[n].requests[0]] = K_temp_tensor.shape[1]
            K_cache[tree_info[n].requests[0], 0 : K_temp_tensor.shape[1]] = (
                K_temp_tensor
            )
            V_cache[tree_info[n].requests[0], 0 : V_temp_tensor.shape[1]] = (
                V_temp_tensor
            )

    Q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
    cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)

    K_tree_tensor = torch.cat(K_tree_tensor, dim=1).reshape(
        [-1, num_kv_heads, head_dim]
    )
    V_tree_tensor = torch.cat(V_tree_tensor, dim=1).reshape(
        [-1, num_kv_heads, head_dim]
    )

    return Q, K_cache, V_cache, cache_seqlens, K_tree_tensor, V_tree_tensor, KV_ptrs


@triton.jit
def _fwd_fasttree_decode_stage1(
    Q,
    K,
    V,
    stride_q0,
    stride_q1,
    stride_k0,
    stride_k1,
    stride_v0,
    stride_v1,
    stride_o0,
    stride_o1,
    stride_lse0,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    node_offset,
    sm_scale,
    MidO,
    MidLSE,
    head_dim: tl.constexpr,
    kv_group_num: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    KV_TILE_SIZE: tl.constexpr,
):
    cur_vnode = tl.program_id(0) + node_offset
    cur_kv_head = tl.program_id(1)
    Q_BLOCK_SIZE: tl.constexpr = Q_TILE_SIZE * kv_group_num

    offs_d = tl.arange(0, head_dim)
    offs_g = tl.arange(0, kv_group_num)

    cur_q_entry_start = tl.load(vnode_to_q_offs + cur_vnode)
    cur_q_len = tl.load(vnode_to_q_lens + cur_vnode)

    cur_kv_entry_start = tl.load(vnode_to_kv_offs + cur_vnode)
    cur_kv_len = tl.load(vnode_to_kv_lens + cur_vnode)
    cur_qo_head = cur_kv_head * kv_group_num

    offs_q_tile = tl.arange(0, Q_TILE_SIZE)
    q_entries = tl.load(
        vnode_to_q_entries + cur_q_entry_start + offs_q_tile,
        mask=offs_q_tile < cur_q_len,
        other=-1,
    )
    off_q = (
        q_entries[:, None, None] * stride_q0
        + (cur_qo_head + offs_g[None, :, None]) * stride_q1
        + offs_d[None, None, :]
    )
    q = tl.load(Q + off_q, mask=q_entries[:, None, None] >= 0, other=0.0)
    q = q.reshape([Q_BLOCK_SIZE, head_dim])

    acc = tl.zeros([Q_BLOCK_SIZE, head_dim], dtype=tl.float32)
    sum_exp = tl.zeros([Q_BLOCK_SIZE, 1], dtype=tl.float32)
    max_logics = tl.full([Q_BLOCK_SIZE, 1], float("-inf"), dtype=tl.float32)
    offs_kv_tile = tl.arange(0, KV_TILE_SIZE)
    for kv_tile_start in range(0, cur_kv_len, KV_TILE_SIZE):
        offs_kv_tile_new = offs_kv_tile + kv_tile_start
        # kv_entries = tl.load(
        #     vnode_to_kv_entries + cur_kv_entry_start + offs_x,
        #     mask=offs_x < cur_kv_len,
        #     other=-1,
        # )
        kv_entries = tl.load(
            vnode_to_kv_entries + cur_kv_entry_start + offs_kv_tile_new
        )
        kv_entries = tl.where(offs_kv_tile_new < cur_kv_len, kv_entries, -1)
        off_k = (
            kv_entries[None, :] * stride_k0 + cur_kv_head * stride_k1 + offs_d[:, None]
        )
        k = tl.load(K + off_k, mask=kv_entries[None, :] >= 0, other=0.0)
        qk = tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_kv_tile_new[None, :] < cur_kv_len, qk, -1.0e6)

        off_v = (
            kv_entries[:, None] * stride_v0 + cur_kv_head * stride_v1 + offs_d[None, :]
        )
        v = tl.load(V + off_v, mask=kv_entries[:, None] >= 0, other=0.0)

        cur_max_logics = tl.max(qk, axis=1)[:, None]
        new_max_logics = tl.maximum(cur_max_logics, max_logics)
        exp_logics = tl.exp(qk - new_max_logics)
        logic_scale = tl.exp(max_logics - new_max_logics)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logics, axis=1)[:, None]
        acc = acc * logic_scale
        acc = tl.dot(exp_logics.to(v.type), v, acc)
        max_logics = new_max_logics

    off_mid_o = (
        (cur_q_entry_start + offs_q_tile[:, None, None]) * stride_o0
        + (cur_qo_head + offs_g[None, :, None]) * stride_o1
        + offs_d[None, None, :]
    )
    off_mid_lse = (
        (cur_q_entry_start + offs_q_tile[:, None]) * stride_lse0
        + cur_qo_head
        + offs_g[None, :]
    )

    mid_o = (acc / sum_exp).reshape([Q_TILE_SIZE, kv_group_num, head_dim])
    tl.store(
        MidO + off_mid_o,
        mid_o,
        mask=offs_q_tile[:, None, None] < cur_q_len,
    )

    mid_lse = (tl.log(sum_exp) + max_logics).reshape([Q_TILE_SIZE, kv_group_num])
    tl.store(MidLSE + off_mid_lse, mid_lse, mask=offs_q_tile[:, None] < cur_q_len)


def _fasttree_decode_stage1(
    Q,
    K,
    V,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    sm_scale,
    MidO,
    MidLSE,
    phase_node_nums,
    phase_node_offsets,
    head_num,
    head_dim,
    kv_head_num,
    phase_q_tile_sizes,
    phase_kv_tile_sizes,
):
    Lq, Lk = Q.shape[-1], K.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256}

    kv_group_num = head_num // kv_head_num

    for node_num, node_offset, Q_TILE_SIZE, KV_TILE_SIZE in zip(
        phase_node_nums, phase_node_offsets, phase_q_tile_sizes, phase_kv_tile_sizes
    ):
        if node_num == 0:
            continue
        grid = (
            node_num,
            kv_head_num,
        )
        _fwd_fasttree_decode_stage1[grid](
            Q,
            K,
            V,
            Q.stride(0),
            Q.stride(1),
            K.stride(0),
            K.stride(1),
            V.stride(0),
            V.stride(1),
            MidO.stride(0),
            MidO.stride(1),
            MidLSE.stride(0),
            vnode_to_kv_entries,
            vnode_to_kv_offs,
            vnode_to_kv_lens,
            vnode_to_q_entries,
            vnode_to_q_offs,
            vnode_to_q_lens,
            node_offset,
            sm_scale,
            MidO,
            MidLSE,
            head_dim,
            kv_group_num,
            Q_TILE_SIZE,
            KV_TILE_SIZE,
        )


@triton.jit
def _fwd_fasttree_decoding_stage2(
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    MidO,
    MidLSE,
    O,
    stride_mid_o0,
    stride_mid_o1,
    stride_mid_lse0,
    stride_o0,
    stride_o1,
    head_dim: tl.constexpr,
    NODE_TILE_SIZE: tl.constexpr,
):
    cur_req = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_vnode_entry_start = tl.load(req_to_vnode_offs + cur_req)
    cur_vnode_len = tl.load(req_to_vnode_lens + cur_req)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    max_lse = tl.full([1], float("-inf"), dtype=tl.float32)
    acc = tl.zeros([1, head_dim], dtype=tl.float32)

    offs_d = tl.arange(0, head_dim)
    offs_n = tl.arange(0, NODE_TILE_SIZE)
    for tile_start in range(0, cur_vnode_len, NODE_TILE_SIZE):
        offs_n_new = offs_n + tile_start
        # vnode_entries = tl.load(
        #     req_to_vnode_entries + cur_vnode_entry_start + offs_n_new,
        #     mask=offs_n_new < cur_vnode_len,
        #     other=-1,
        # )
        vnode_entries = tl.load(
            req_to_vnode_entries + cur_vnode_entry_start + offs_n_new
        )
        vnode_entries = tl.where(offs_n_new < cur_vnode_len, vnode_entries, -1)

        off_mid_o = (
            vnode_entries[:, None] * stride_mid_o0
            + cur_head * stride_mid_o1
            + offs_d[None, :]
        )
        mid_o = tl.load(MidO + off_mid_o, mask=vnode_entries[:, None] >= 0, other=0.0)

        off_mid_lse = vnode_entries * stride_mid_lse0 + cur_head
        mid_lse = tl.load(
            MidLSE + off_mid_lse, mask=vnode_entries >= 0, other=float("-inf")
        )[None, :]

        tile_max_lse = tl.max(mid_lse, axis=1)
        new_max_lse = tl.maximum(tile_max_lse, max_lse)

        exp_lse = tl.exp(mid_lse - new_max_lse)
        old_scale = tl.exp(max_lse - new_max_lse)
        sum_exp = sum_exp * old_scale + tl.sum(exp_lse, axis=1)
        acc = acc * old_scale + tl.sum(exp_lse[:, :, None] * mid_o[None, :, :], axis=1)
        max_lse = new_max_lse

    tl.store(
        O + cur_req * stride_o0 + cur_head * stride_o1 + offs_d,
        tl.ravel(acc / sum_exp).to(O.type.element_ty),
    )


def _fasttree_decode_stage2(
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    MidO,
    MidLSE,
    O,
    batch_size,
    head_num,
    head_dim,
):
    NODE_TILE_SIZE = 4
    grid = (
        batch_size,
        head_num,
    )
    _fwd_fasttree_decoding_stage2[grid](
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        MidO,
        MidLSE,
        O,
        MidO.stride(0),
        MidO.stride(1),
        MidLSE.stride(0),
        O.stride(0),
        O.stride(1),
        head_dim,
        NODE_TILE_SIZE,
    )


def fasttree_decode(
    q,
    k_buffer,
    v_buffer,
    o,
    vnode_to_kv_entries,
    vnode_to_kv_offs,
    vnode_to_kv_lens,
    vnode_to_q_entries,
    vnode_to_q_offs,
    vnode_to_q_lens,
    req_to_vnode_entries,
    req_to_vnode_offs,
    req_to_vnode_lens,
    mid_o,
    mid_lse,
    phase_node_nums,
    phase_node_offsets,
    phase_q_tile_sizes,
    phase_kv_tile_sizes,
    sm_scale,
    logit_cap=-1,
):
    head_num = q.shape[1]
    head_dim = q.shape[-1]
    kv_head_num = k_buffer.shape[1]

    _fasttree_decode_stage1(
        q,
        k_buffer,
        v_buffer,
        vnode_to_kv_entries,
        vnode_to_kv_offs,
        vnode_to_kv_lens,
        vnode_to_q_entries,
        vnode_to_q_offs,
        vnode_to_q_lens,
        sm_scale,
        mid_o,
        mid_lse,
        phase_node_nums,
        phase_node_offsets,
        head_num,
        head_dim,
        kv_head_num,
        phase_q_tile_sizes,
        phase_kv_tile_sizes,
    )

    batch_size = q.shape[0]
    _fasttree_decode_stage2(
        req_to_vnode_entries,
        req_to_vnode_offs,
        req_to_vnode_lens,
        mid_o,
        mid_lse,
        o,
        batch_size,
        head_num,
        head_dim,
    )

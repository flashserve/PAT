#pragma once

#include "namespace_config.h"
#include <cuda.h>
#include <vector>

#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

namespace PAT_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct base_params {
    // qkv params
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    int q_batch_stride;
    int q_row_stride;
    int q_head_stride;
    // MHA and GQA have keys and values with the same dimensions, they can be merged.
    index_t kv_batch_stride;
    int kv_row_stride;
    int kv_head_stride;

    // The number of heads.
    int h, h_k;

    // The dimensions.
    int b, seqlen_q, d;
    int page_block_size;

    bool is_bf16;

    float scale_softmax;
    float scale_softmax_log2;

    // fwd params
    int max_split_per_seq;
    int max_seqs_in_CTA;
    int max_blocks_in_CTA;

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    int o_batch_stride;
    int o_row_stride;
    int o_head_stride;
    int oaccum_batch_stride;
    int oaccum_head_stride;
    int oaccum_rank_stride;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;
    int softmax_lse_batch_stride;
    int softmax_lseaccum_batch_stride;
    int softmax_lseaccum_head_stride;

    void * __restrict__ num_split_per_seq_ptr;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct pat_fwd_params : public base_params {
    void * __restrict__ query_tables_ptr;
    void * __restrict__ block_tables_ptr;
    void * __restrict__ num_seqs_per_CTA_ptr;
    void * __restrict__ CTA_rank_ptr;
    void * __restrict__ kv_in_CTA_ptr;
    void * __restrict__ timing_ptr;

    index_t block_tables_row_stride;
    index_t query_tables_row_stride;

    int CTAs, Warps, tile_q, tile_kv;

    pat_fwd_params(const base_params& base)
        : base_params(base),
        query_tables_ptr(nullptr),
        block_tables_ptr(nullptr),
        num_seqs_per_CTA_ptr(nullptr),
        CTA_rank_ptr(nullptr),
        kv_in_CTA_ptr(nullptr),
        timing_ptr(nullptr),
        block_tables_row_stride(0),
        CTAs(0),
        Warps(0),
        tile_q(0),
        tile_kv(0) {}

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename elem_type, int Headdim> void pat_run_mha_fwd_splitkv_dispatch(std::vector<pat_fwd_params> &params);

}  // namespace PAT_NAMESPACE
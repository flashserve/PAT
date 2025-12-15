#pragma once

#include "namespace_config.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>
#include <thrust/pair.h>
using namespace cute;
namespace PAT_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int HRatio>
struct Block {

    // The bidh is bidh * HRatio
    template<typename Params>
    __device__ Block(const Params &params, const int bidb)
        : sum_s_q(reinterpret_cast<int*>(params.num_seqs_per_CTA_ptr)[bidb] * HRatio)
        , kv_in_CTA(reinterpret_cast<int*>(params.kv_in_CTA_ptr)[bidb])
    {}

    __forceinline__ __device__ int q_offset(const int bidh, const int head_stride) const {
        return bidh * HRatio * head_stride;
    }

    template<typename Params>
    __forceinline__ __device__ thrust::pair<int, int> o_lse_offset(const Params &params, const int bidb, const int bidh) const {
        const int CTA_rank = reinterpret_cast<int*>(params.CTA_rank_ptr)[bidb];
        int o_offset = bidh * params.oaccum_head_stride
                           + (CTA_rank * params.oaccum_rank_stride);
        int lse_offset = bidh * params.softmax_lseaccum_head_stride + CTA_rank;
        return {o_offset, lse_offset};
    }

    const int sum_s_q;
    const int kv_in_CTA;
};

}  // namespace PAT_NAMESPACE

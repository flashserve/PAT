#pragma once

#include "namespace_config.h"
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <thrust/system/cuda/detail/par.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "static_switch.h"
#include "utils.h"

namespace PAT_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename Kernel_traits, typename Params>
inline __device__ void pat_kernel(const Params &params, const int bidb, const int bidh) {
    using Element = typename Kernel_traits::Element;
    using ElementO = typename Kernel_traits::ElementAccum;

    extern __shared__ char smem_[];

    // The thread index.
    const int tid = threadIdx.x;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    // constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int HRatio = Kernel_traits::HRatio;

    const Block<HRatio> binfo(params, bidb);

    const int *block_table = reinterpret_cast<int*>(params.block_tables_ptr) + bidb * params.block_tables_row_stride;
    const int *query_table = reinterpret_cast<int*>(params.query_tables_ptr) + bidb * params.query_tables_row_stride;
    // row_offset_k and row_offset_v will not exceed the limit of int.
    const int row_offset_kv = bidh * params.kv_head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(bidh, params.q_head_stride)),
                            Shape<Shape<Int<HRatio>, Int<kBlockM/HRatio>>, Int<kHeadDim>>{},
                            make_stride(make_stride(params.q_head_stride, params.q_batch_stride), Int<1>{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_kv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.kv_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_kv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.kv_row_stride, _1{}));


    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});

    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tid);
    typename Kernel_traits::GmemTiledCopyKVPaged gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tid);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    tQgQ.data() = gQ.data() + flash::resolve_thread_qo_slice_offset<HRatio, Kernel_traits::QGmemThreadsPerRow,
            Kernel_traits::QRowsPerThread, Kernel_traits::QGmemElemsPerLoad>(tid, params.q_batch_stride,
            params.q_head_stride, query_table);


    int n_block = cute::ceil_div(binfo.kv_in_CTA, kBlockN) - 1;
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    auto tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    flash::copy<false>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, binfo.sum_s_q);

    Tensor tKgK_ = gmem_thr_copy_KV.partition_S(gK);
    Tensor tKsK_ = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV_ = gmem_thr_copy_KV.partition_S(gV);
    Tensor tVsV_ = gmem_thr_copy_KV.partition_D(sV);

    Tensor tKgK = make_tensor(tKgK_.data(), flash::reshape_thread_tile(tKgK_.layout()));
    Tensor tKsK = make_tensor(tKsK_.data(), flash::reshape_thread_tile(tKsK_.layout()));
    Tensor tVgV = make_tensor(tVgV_.data(), flash::reshape_thread_tile(tVgV_.layout()));
    Tensor tVsV = make_tensor(tVsV_.data(), flash::reshape_thread_tile(tVsV_.layout()));

    // const int final_block_size = binfo.kv_in_CTA - (n_block_max - 1) * kBlockN;
    tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tid, n_block, params.page_block_size,
        block_table, params.kv_batch_stride, params.kv_row_stride);
    tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tid, n_block, params.page_block_size,
        block_table, params.kv_batch_stride, params.kv_row_stride);


    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    auto tKVcKV_ = gmem_thr_copy_KV.partition_S(cKV);
    Tensor tKVcKV = make_tensor(tKVcKV_.data(), flash::reshape_thread_tile(tKVcKV_.layout()));
    flash::copy<false>(gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV,
                                binfo.kv_in_CTA - n_block * kBlockN);
    cute::cp_async_fence();


    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);

    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);


    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);


    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);
    flash::Softmax<2 * size<1>(acc_o)> softmax;
    int iter = 0;
    do {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});

        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();


        flash::copy<false, true>(gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV, binfo.kv_in_CTA - n_block * kBlockN);
        cute::cp_async_fence();

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                             smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if (binfo.kv_in_CTA % kBlockN != 0) {
            Tensor acc_s_t = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int warp_offset = tid / 32 * 16;
            const int lane_offset = tid % 32 / 4;
            const int row_stride = kBlockM * Kernel_traits::kNWarps / 16;
            const int col_offset = tid % 4 * 2;
            #pragma unroll
            for (int mi1 = 0; mi1 < size<0, 1>(acc_s_t); ++mi1) {
                #pragma unroll
                for (int mi0 = 0; mi0 < size<0, 0>(acc_s_t); ++mi0) {
                    int mi = mi1 * 2 + mi0;
                    int row = mi1 * row_stride + warp_offset + lane_offset + mi0 * 8;
                    if (row < binfo.sum_s_q) {
                        #pragma unroll
                        for (int ni1 = 0; ni1 < size<1, 1>(acc_s_t); ++ni1) {
                            #pragma unroll
                            for (int ni0 = 0; ni0 < size<1, 0>(acc_s_t); ++ni0) {
                                int ni = ni1 * 2 + ni0;
                                int col = ni1 * 8 + col_offset + ni0;
                                acc_s_t(mi, ni) = col >= binfo.kv_in_CTA - n_block * kBlockN ? -INFINITY : acc_s_t(mi, ni);
                            }
                        }
                    }
                }
            }
        }

        flash::cp_async_wait<0>();
        __syncthreads();


        if (n_block > 0) {
           tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tid, n_block - 1, params.page_block_size,
               block_table, params.kv_batch_stride, params.kv_row_stride);
           flash::copy(gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV);
           cute::cp_async_fence();
        }
        softmax.template softmax_rescale_o</*Is_first=*/true,
                                          /*Check_inf=Is_local*/false>
                                         (acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = flash::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(),
                                 flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        --n_block;
    }while (false);

    #pragma unroll
    for (; n_block >= 0; --n_block) {

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // advance V
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tid, n_block, params.page_block_size,
           block_table, params.kv_batch_stride, params.kv_row_stride);

        flash::copy<true, false>(gmem_tiled_copy_KV, tVgV, tVsV, tKVcKV);
        cute::cp_async_fence();

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                         smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);


        flash::cp_async_wait<0>();
        __syncthreads();

        if (n_block > 0) {
            tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tid, n_block - 1, params.page_block_size,
                    block_table, params.kv_batch_stride, params.kv_row_stride);
            flash::copy<true, false>(gmem_tiled_copy_KV, tKgK, tKsK, tKVcKV);
            cute::cp_async_fence();
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,
                                      /*Check_inf=*/false>
                                     (acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(),
                                 flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }


    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(acc_o, params.scale_softmax);

    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(smem_)), typename Kernel_traits::SmemLayoutO{});

    using SmemTiledCopyO = typename Kernel_traits::SmemCopyAtomO;
    auto smem_tiled_copyO = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copyO = smem_tiled_copyO.get_thread_slice(tid);

    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor taccOrO = smem_thr_copyO.retile_S(rO);
    Tensor taccOsO = smem_thr_copyO.partition_D(sO);

    __syncthreads();
    cute::copy(smem_tiled_copyO, taccOrO, taccOsO);

    auto [o_offset, lse_offset] = binfo.o_lse_offset(params, bidb, bidh * HRatio);

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(params.oaccum_ptr) + o_offset),
                                 Shape<Shape<Int<HRatio>, Int<kBlockM/HRatio>>, Int<kHeadDim>>{},
                                 make_stride(make_stride(params.oaccum_head_stride, params.oaccum_batch_stride), _1{}));

    using GmemTiledCopyO = typename Kernel_traits::GmemTiledCopyO;
    GmemTiledCopyO gmem_tiled_copyO;
    auto gmem_thr_copyO = gmem_tiled_copyO.get_thread_slice(tid);
    Tensor tOsO = gmem_thr_copyO.partition_S(sO);
    Tensor tOgO = gmem_thr_copyO.partition_D(gOaccum);
    __syncthreads();

    Tensor tOrO = make_tensor<ElementO>(shape(tOgO));

    cute::copy(gmem_tiled_copyO, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);


    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.sum_s_q) {
                float* glse = reinterpret_cast<float*>(params.softmax_lseaccum_ptr) +
                    (query_table[row/HRatio] * params.softmax_lseaccum_batch_stride + lse_offset +
                        row % HRatio * params.softmax_lseaccum_head_stride);
                *glse = lse[mi];
            }
        }
    }
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copyO.partition_D(cO);
    tOgO.data() = gOaccum.data() + flash::resolve_thread_qo_slice_offset<HRatio, Kernel_traits::OGmemThreadsPerRow,
            Kernel_traits::ORowsPerThread, Kernel_traits::OGmemElemsPerLoad>(tid, params.oaccum_batch_stride,
            params.oaccum_head_stride, query_table);
    if (binfo.sum_s_q == kBlockM) {
        flash::copy<true, false>(gmem_tiled_copyO, tOrO, tOgO, tOcO);
    } else {
        flash::copy<false, false>(gmem_tiled_copyO, tOrO, tOgO, tOcO, binfo.sum_s_q);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void forward(const Params &params) {
    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;

    PAT_NAMESPACE::pat_kernel<Kernel_traits>(params, bidb, bidh);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementO_, int HEADDIM, int BLOCKM, int WARPS, typename Params>
inline __global__ void gather_kernel(Params params) {
    using Element = float;
    using ElementO = ElementO_;
    constexpr int kHeadDim = HEADDIM;

    const int tid = threadIdx.x;
    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int lseAccum_offset = bidb * params.softmax_lseaccum_batch_stride + bidh * params.softmax_lseaccum_head_stride;
    const int oAccum_offset = bidb * params.oaccum_batch_stride + bidh * params.oaccum_head_stride;
    const int o_offset = bidb * params.o_batch_stride + bidh * params.o_head_stride;
    const int lse_offset = bidb * params.softmax_lse_batch_stride + bidh;
    const int split_in_seq = reinterpret_cast<int*>(params.num_split_per_seq_ptr)[bidb];

    Element* softmax_lseaccum_ptr = reinterpret_cast<Element*>(params.softmax_lseaccum_ptr) + lseAccum_offset;
    Element* oaccum_ptr = reinterpret_cast<Element*>(params.oaccum_ptr) + oAccum_offset;
    float* softmax_lse_ptr = reinterpret_cast<float*>(params.softmax_lse_ptr) + lse_offset;
    ElementO* acco_ptr = reinterpret_cast<ElementO*>(params.o_ptr) + o_offset;

    if (split_in_seq == 1) {
        softmax_lse_ptr[0] = softmax_lseaccum_ptr[0];
        if (tid < kHeadDim) {
            acco_ptr[tid] = static_cast<ElementO>(oaccum_ptr[tid]);
        }
        return;
    }

    constexpr int kNThreads = WARPS * 32;
    constexpr int kBlockM = BLOCKM;
    constexpr int GmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    constexpr int GmemThreadsPerRow = kHeadDim / GmemElemsPerLoad;
    using GmemLayoutAtom = Layout<Shape<Int<kNThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>,
                               Stride<Int<GmemThreadsPerRow>, _1>>; // (4,32):(32,1)
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1,Int<GmemElemsPerLoad>>>{}
        ));
    using SmemLayoutAtomOaccum = decltype(composition(Swizzle<2, 2, 3>{},
                                                      Layout<Shape<_4, Int<32>>,
                                                             Stride<Int<32>, _1>>{}));
    using SmemLayoutOaccum = decltype(tile_to_shape(
        SmemLayoutAtomOaccum{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    Tensor gOaccum = make_tensor(make_gmem_ptr(oaccum_ptr),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 Stride<Int<kHeadDim>, Int<1>>{});

    __shared__ Element smem_[kBlockM * kHeadDim];
    Tensor sOaccum = make_tensor(make_smem_ptr(smem_),
                                 SmemLayoutOaccum{});
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tid);
    auto tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    auto tOsOaccum = gmem_thr_copy_Oaccum.partition_D(sOaccum);

    // copy OAccum from gmem to smem, not block
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));
    auto tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cO);
    flash::copy<false>(gmem_tiled_copy_Oaccum, tOgOaccum, tOsOaccum, tOcOaccum, split_in_seq);
    cute::cp_async_fence();

    // store sLse from gmem to smem
    __shared__ Element sLse[kNThreads], swarp_max[WARPS], swarp_sum[WARPS];
    Tensor gLseAccum = make_tensor(make_gmem_ptr(softmax_lseaccum_ptr),
                                   Shape<Int<kNThreads>>{},
                                   Stride<Int<1>>{});
    sLse[tid] = tid < split_in_seq ? gLseAccum[tid] : -INFINITY;
    __syncthreads();

    // reduce max in warp
    flash::MaxOp<float> max_op;
    float rLse = sLse[tid];
    float warp_max = flash::Allreduce<32>::run(rLse, max_op);

    // each warp store its own warp_max
    if (lane_id == 0) {
        swarp_max[warp_id] = warp_max;
    }
    __syncthreads();

    float lse_max = lane_id < WARPS ? swarp_max[lane_id] : -INFINITY;
    lse_max = flash::WarpReduce<WARPS>(lse_max, max_op);
    // broadcast lse_max to all threads in a warp
    lse_max = __shfl_sync(0xffffffffu, lse_max, 0);


    // reduce sum in warp
    flash::SumOp<float> sum_op;
    float y = exp2f((rLse - lse_max) * M_LOG2E);
    float warp_sum = flash::Allreduce<32>::run(y, sum_op);

    if (lane_id == 0) {
        swarp_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    float lse_sum = lane_id < WARPS ? swarp_sum[lane_id] : 0;
    lse_sum = flash::WarpReduce<WARPS>(lse_sum, sum_op);
    // broadcast sum_all to all threads in a warp
    lse_sum = __shfl_sync(0xffffffffu, lse_sum, 0);

    Element lse_global = lse_max + log(lse_sum);

    // store the lse result
    if (tid == 0) {
        softmax_lse_ptr[0] = lse_global;
    }
    // scale
    sLse[tid] = exp2f((rLse - lse_global) * M_LOG2E);
    cp_async_wait<0>();
    __syncthreads();

    // copy Oaccum from smem to register
    using SmemLayoutCopyAtomOaccum = Layout<Shape<_1, Int<kNThreads>>,
                                            Stride<Int<kNThreads>, Int<1>>>;
    using SmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        SmemLayoutCopyAtomOaccum{},
                        Layout<Shape < _4, _1>>{}));  // Val layout, 4 vals per store
    SmemTiledCopyOaccum smem_tiled_copy_Oaccum;
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tid);
    auto tOsOAccum = smem_thr_copy_Oaccum.partition_S(sOaccum);
    auto tOgO = smem_thr_copy_Oaccum.partition_D(gOaccum);


    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    clear(tOrO);
    if (tid < kHeadDim) {
        cute::copy(smem_tiled_copy_Oaccum, tOsOAccum(_, 0, 0), tOrO(_, 0, 0));
        #pragma unroll
        for (int m = 0; m < size<1>(tOrO); ++m) {
            const int row = m * GmemElemsPerLoad;
            if (row + GmemElemsPerLoad < split_in_seq) {
                cute::copy(smem_tiled_copy_Oaccum, tOsOAccum(_, m+1, 0), tOrO(_, m+1, 0));
            }
            tOrO(0, m, 0) *= sLse[row];
            #pragma unroll
            for (int i = 1; i < size<0>(tOrO); i++) {
                tOrO(0, m, 0) += row + i < split_in_seq ? tOrO(i, m, 0) * sLse[row+i] : 0;
            }
        }
        #pragma unroll
        for (int m = 1; m < size<1>(tOrO); ++m) {
            tOrO(0, 0, 0) += tOrO(0, m, 0);
        }
        acco_ptr[tid] = static_cast<ElementO>(tOrO(0, 0, 0));
    }
}

} // namespace PAT_NAMESPACE
#pragma once

#include "namespace_config.h"
#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Pat_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

template<typename Element_, typename ElementAccum_, int kBlockM_, int kBlockN_, int kHeadDim_, int Warps, int HRatio_>
struct fwd_kernel_traits {
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kSwizzle = 3;
    static constexpr int kBlockKSmem = 64;
    static constexpr int kNWarps = Warps;
    static constexpr int HRatio = HRatio_;

    using SmemLayoutAtom = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int SmemSize = kSmemQSize + kSmemKVSize;
    static constexpr int KVGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int KVGmemThreadsPerRow = kBlockKSmem / KVGmemElemsPerLoad; // 8

    using GmemLayoutAtom = Layout<Shape <Int<kNWarps * 32 / KVGmemThreadsPerRow>, Int<KVGmemThreadsPerRow>>,
                                  Stride<Int<KVGmemThreadsPerRow>, _1>>; /*(16,8):(8,1)*/
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    // static constexpr int QRowsPerThread = kBlockM / (kNWarps * 32 / QKVGmemThreadsPerRow); // (kBlockM / (128/8))
    static constexpr int QRowsPerThread = HRatio;
    static constexpr int QGmemThreadsPerRow = kNWarps * 32 / (kBlockM / QRowsPerThread);
    static constexpr int QGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    using GmemLayoutAtomQ = Layout<Shape <Int<kBlockM / QRowsPerThread>, Int<QGmemThreadsPerRow>>,
                                  Stride<Int<QGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQ = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtomQ{}, // thr_layout
                        Layout<Shape<Int<QRowsPerThread>, Int<QGmemElemsPerLoad>>, Stride<Int<QGmemElemsPerLoad>, _1>>{})); // val_layout
    static constexpr int KVRowsPerThread = kBlockN / (kNWarps * 32 / KVGmemThreadsPerRow); // (kBlockN / (128/8))
    using GmemTiledCopyKVPaged = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<KVRowsPerThread>, Int<KVGmemThreadsPerRow>>, Stride<Int<KVGmemThreadsPerRow>, _1>>{}));
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<Warps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<kBlockM>, _16, _16>>;
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;
    static constexpr int ORowsPerThread = HRatio;
    static constexpr int OGmemThreadsPerRow = kNWarps * 32 / (kBlockM / ORowsPerThread);
    static constexpr int OGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    using GmemLayoutAtomO = Layout<Shape<Int<kBlockM / ORowsPerThread>, Int<OGmemThreadsPerRow>>,
                                  Stride<Int<OGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        GmemLayoutAtomO{},
                        Layout<Shape<Int<ORowsPerThread>, Int<OGmemElemsPerLoad>>, Stride<Int<OGmemElemsPerLoad>, _1>>{}));
};
